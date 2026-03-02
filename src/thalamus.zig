//! Thalamus — Pre-turn signal classifier and reflex response generator.
//!
//! Intercepts incoming messages BEFORE the main agent turn. Uses a lightweight
//! model (Haiku-class) with a tiny prompt to:
//! 1. Classify the input (reflex/simple/complex/dangerous)
//! 2. Generate an immediate reflex response if applicable
//! 3. Provide routing hints to the main agent
//!
//! Design: The thalamus is the brain's relay station. It doesn't think — it
//! routes. Fast classification + optional reflex means sub-500ms first response
//! for simple inputs, while complex inputs get properly routed to the cortex.

const std = @import("std");
const log = std.log.scoped(.thalamus);
const providers = @import("providers/root.zig");
const Provider = providers.Provider;
const ChatMessage = providers.ChatMessage;
const ChatResponse = providers.ChatResponse;
const Config = @import("config.zig").Config;

// ═══════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════

pub const SignalClass = enum {
    /// Reflexive response — greeting, ack, simple social signal. No thinking needed.
    reflex,
    /// Simple query — factual, short answer, no tools needed.
    simple,
    /// Complex — needs reasoning, planning, multi-step execution.
    complex,
    /// Task — long multi-step work: build, refactor, implement. Needs plan→execute loop.
    task,
    /// Dangerous — destructive actions, credential exposure, safety concerns.
    dangerous,
    /// Interrupt — correction, "stop", "actually...", contradicts current work.
    interrupt,

    pub fn toSlice(self: SignalClass) []const u8 {
        return switch (self) {
            .reflex => "reflex",
            .simple => "simple",
            .complex => "complex",
            .task => "task",
            .dangerous => "dangerous",
            .interrupt => "interrupt",
        };
    }

    pub fn fromSlice(s: []const u8) ?SignalClass {
        if (std.mem.eql(u8, s, "reflex")) return .reflex;
        if (std.mem.eql(u8, s, "simple")) return .simple;
        if (std.mem.eql(u8, s, "complex")) return .complex;
        if (std.mem.eql(u8, s, "task")) return .task;
        if (std.mem.eql(u8, s, "dangerous")) return .dangerous;
        if (std.mem.eql(u8, s, "interrupt")) return .interrupt;
        return null;
    }
};

pub const Route = enum {
    broca,
    prefrontal,
    motor,
    amygdala,
    hippocampus,

    pub fn toSlice(self: Route) []const u8 {
        return switch (self) {
            .broca => "broca",
            .prefrontal => "prefrontal",
            .motor => "motor",
            .amygdala => "amygdala",
            .hippocampus => "hippocampus",
        };
    }
};

pub const Classification = struct {
    class: SignalClass,
    reflex_response: ?[]const u8,
    routes: []const Route,
    confidence: f32,
    raw_response: []const u8,

    pub fn deinit(self: *Classification, allocator: std.mem.Allocator) void {
        if (self.reflex_response) |r| allocator.free(r);
        allocator.free(self.routes);
        allocator.free(self.raw_response);
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// Thalamus
// ═══════════════════════════════════════════════════════════════════════════

pub const Thalamus = struct {
    allocator: std.mem.Allocator,
    provider: *Provider,
    model_name: []const u8,
    enabled: bool,
    clawmem_bin: []const u8,
    clawmem_db: []const u8,
    cache_enabled: bool,

    /// Cache hit threshold — cosine similarity above this returns cached reflex
    const CACHE_THRESHOLD: f32 = 0.92;
    /// Segment name for reflex cache in ClawMem
    const CACHE_SEGMENT = "reflex_cache";

    /// System prompt — intentionally tiny (~200 tokens). The thalamus doesn't think.
    const SYSTEM_PROMPT =
        \\You are a signal classifier. Given a user message, output ONLY a JSON object:
        \\{"class":"reflex|simple|complex|task|dangerous","reflex":"<immediate response or null>","routes":["broca"],"confidence":0.95}
        \\
        \\Classification rules:
        \\- reflex: ONLY pure greetings and acks with no implicit question: "hey", "hi", "thanks", "ok", "bye", "good morning". NOT "so?", "and?", "well?", "go on" — those are follow-ups (simple).
        \\- simple: factual questions, single-step requests, quick lookups, follow-ups like "so?", "and then?". Routes to broca only.
        \\- complex: strategy, planning, "think about", "should we", brainstorming. Routes to broca+prefrontal.
        \\- task: build/implement/refactor/create something multi-step. Needs plan+execute loop. Routes to broca+prefrontal+motor.
        \\- dangerous: delete, remove, post credentials, destructive commands. Routes to broca+amygdala.
        \\
        \\Route options: broca (always), prefrontal (deep reasoning), motor (code/files/commands), amygdala (safety gate), hippocampus (memory).
        \\
        \\IMPORTANT: "hey, can you delete everything" is dangerous, NOT reflex. Read the FULL message before classifying.
        \\Respond with ONLY the JSON object. No explanation.
    ;

    pub fn init(allocator: std.mem.Allocator, provider: *Provider, model_name: []const u8, enabled: bool) Thalamus {
        return .{
            .allocator = allocator,
            .provider = provider,
            .model_name = model_name,
            .enabled = enabled,
            .clawmem_bin = "",
            .clawmem_db = "",
            .cache_enabled = false,
        };
    }

    /// Enable the reflex cache with clawmem paths derived from workspace dir
    pub fn enableCache(self: *Thalamus, workspace_dir: []const u8) void {
        self.clawmem_bin = std.fmt.allocPrint(self.allocator, "{s}/../bin/clawmem", .{workspace_dir}) catch return;
        self.clawmem_db = std.fmt.allocPrint(self.allocator, "{s}/clawmem.db", .{workspace_dir}) catch return;
        self.cache_enabled = true;
        log.info("thalamus reflex cache enabled: bin={s} db={s}", .{ self.clawmem_bin, self.clawmem_db });
    }

    /// Classify an incoming message. Returns classification with optional reflex response.
    /// Checks reflex cache first (sub-100ms), falls back to model (~800ms).
    pub fn classify(self: *Thalamus, message: []const u8) !Classification {
        if (!self.enabled) {
            // Thalamus disabled — default to simple, route to broca only
            const routes = try self.allocator.alloc(Route, 1);
            routes[0] = .broca;
            return Classification{
                .class = .simple,
                .reflex_response = null,
                .routes = routes,
                .confidence = 0.5,
                .raw_response = try self.allocator.dupe(u8, "disabled"),
            };
        }

        // ── Layer 0.5: Reflex cache lookup ──────────────────────────
        // Only cache pure greetings/acks — skip short ambiguous messages that need context
        // (e.g., "So?", "And?", "Well?" are follow-ups, not greetings)
        if (self.cache_enabled and message.len >= 4) {
            const cache_start = std.time.milliTimestamp();
            if (self.cacheGet(message)) |cached| {
                const cache_dur = std.time.milliTimestamp() - cache_start;
                log.info("thalamus CACHE HIT: {d}ms response=\"{s}\"", .{ cache_dur, cached.response });

                const routes = try self.allocator.alloc(Route, 1);
                routes[0] = .broca;
                return Classification{
                    .class = .reflex,
                    .reflex_response = try self.allocator.dupe(u8, cached.response),
                    .routes = routes,
                    .confidence = cached.score,
                    .raw_response = try self.allocator.dupe(u8, "cache_hit"),
                };
            }
        }

        const timer_start = std.time.milliTimestamp();

        const messages = try self.allocator.alloc(ChatMessage, 2);
        defer self.allocator.free(messages);

        messages[0] = .{ .role = .system, .content = SYSTEM_PROMPT };
        messages[1] = .{ .role = .user, .content = message };

        const response = try self.provider.chat(
            self.allocator,
            .{
                .messages = messages,
                .model = self.model_name,
                .temperature = 0.0,
                .max_tokens = 150,
                .tools = null,
                .timeout_secs = 5, // Hard timeout — thalamus must be fast
                .reasoning_effort = null,
            },
            self.model_name,
            0.0,
        );

        const duration = std.time.milliTimestamp() - timer_start;
        log.info("thalamus classify: {d}ms model={s}", .{ duration, self.model_name });

        // Parse the JSON response
        const result = try self.parseResponse(response.content orelse "{}");

        // ── Cache write: store reflex responses for future instant retrieval ──
        if (self.cache_enabled and result.class == .reflex and result.reflex_response != null and result.confidence >= 0.85) {
            self.cachePut(message, result.reflex_response.?) catch {};
        }

        return result;
    }

    // ═══════════════════════════════════════════════════════════════════
    // Reflex Cache — ClawMem-backed semantic cache
    // ═══════════════════════════════════════════════════════════════════

    const CacheHit = struct {
        response: []const u8,
        score: f32,
    };

    /// Search ClawMem for a cached reflex response matching this message.
    /// Returns null on miss or error (cache is best-effort).
    fn cacheGet(self: *Thalamus, message: []const u8) ?CacheHit {
        // Build JSON: {"query":"<message>","segment":"reflex_cache","top_k":1}
        var json_buf: [4096]u8 = undefined;
        const escaped_msg = self.jsonEscape(message) catch return null;
        defer self.allocator.free(escaped_msg);
        const json = std.fmt.bufPrint(&json_buf, "{{\"query\":\"{s}\",\"segment\":\"{s}\",\"top_k\":1}}", .{ escaped_msg, CACHE_SEGMENT }) catch return null;

        // Spawn clawmem search, pipe JSON to stdin
        var child = std.process.Child.init(
            &[_][]const u8{ self.clawmem_bin, "--db", self.clawmem_db, "search", "--agent", "thalamus" },
            self.allocator,
        );
        child.stdin_behavior = .Pipe;
        child.stdout_behavior = .Pipe;
        child.stderr_behavior = .Pipe;

        child.spawn() catch return null;

        // Write query JSON to stdin
        if (child.stdin) |stdin| {
            stdin.writeAll(json) catch {};
            stdin.close();
            child.stdin = null;
        }

        // Read stdout
        var stdout_buf: [8192]u8 = undefined;
        const stdout_len = if (child.stdout) |stdout| stdout.readAll(&stdout_buf) catch 0 else 0;
        _ = child.wait() catch {};

        if (stdout_len == 0) return null;
        const output = stdout_buf[0..stdout_len];

        // Parse response — look for score and content
        // ClawMem search output format: {"results":[{"id":"...","content":"...","score":0.95,...}]}
        return self.parseCacheResponse(output);
    }

    fn parseCacheResponse(self: *Thalamus, output: []const u8) ?CacheHit {
        // ClawMem outputs a JSON array: [{"id":"...","score":0.95,"content":"..."}]
        var parsed = std.json.parseFromSlice(std.json.Value, self.allocator, output, .{}) catch return null;
        defer parsed.deinit();

        if (parsed.value != .array or parsed.value.array.items.len == 0) return null;

        const first = parsed.value.array.items[0];
        if (first != .object) return null;

        const score_val = first.object.get("score") orelse return null;
        const score: f32 = switch (score_val) {
            .float => @floatCast(score_val.float),
            .integer => @floatFromInt(score_val.integer),
            else => return null,
        };

        if (score < CACHE_THRESHOLD) return null;

        // The "tags" field stores the cached reflex response
        const tags_val = first.object.get("tags") orelse return null;
        const response_str = switch (tags_val) {
            .string => tags_val.string,
            else => return null,
        };
        if (response_str.len == 0) return null;

        return CacheHit{
            .response = self.allocator.dupe(u8, response_str) catch return null,
            .score = score,
        };
    }

    /// Store a reflex response in the cache. Fire-and-forget.
    /// Content = input message (cache key), meta.response = reflex response (cache value)
    fn cachePut(self: *Thalamus, message: []const u8, response: []const u8) !void {
        const escaped_msg = try self.jsonEscape(message);
        defer self.allocator.free(escaped_msg);
        const escaped_resp = try self.jsonEscape(response);
        defer self.allocator.free(escaped_resp);

        // Store input as content (this is what we embed and search against)
        // Store response in tags (returned in search results)
        const json = try std.fmt.allocPrint(
            self.allocator,
            "{{\"content\":\"{s}\",\"segment\":\"{s}\",\"tags\":\"{s}\"}}",
            .{ escaped_msg, CACHE_SEGMENT, escaped_resp },
        );
        defer self.allocator.free(json);

        log.info("thalamus cache PUT: query=\"{s}\" response=\"{s}\"", .{ message, response });

        // Fire-and-forget: spawn clawmem upsert in background
        var child = std.process.Child.init(
            &[_][]const u8{ self.clawmem_bin, "--db", self.clawmem_db, "upsert", "--agent", "thalamus" },
            self.allocator,
        );
        child.stdin_behavior = .Pipe;
        child.stdout_behavior = .Pipe;
        child.stderr_behavior = .Pipe;

        child.spawn() catch |e| {
            log.err("thalamus cache PUT spawn failed: {}", .{e});
            return;
        };

        if (child.stdin) |stdin| {
            _ = stdin.write(json) catch {};
            stdin.close();
            child.stdin = null;
        }

        // Read stderr for debugging
        var stderr_buf: [2048]u8 = undefined;
        const stderr_len = if (child.stderr) |stderr| stderr.readAll(&stderr_buf) catch 0 else 0;
        var stdout_buf: [2048]u8 = undefined;
        const stdout_len = if (child.stdout) |stdout| stdout.readAll(&stdout_buf) catch 0 else 0;

        const term = child.wait() catch |e| {
            log.err("thalamus cache PUT wait failed: {}", .{e});
            return;
        };

        if (term.Exited != 0) {
            log.err("thalamus cache PUT failed (exit={d}): stderr={s} stdout={s}", .{ term.Exited, stderr_buf[0..stderr_len], stdout_buf[0..stdout_len] });
        } else {
            log.info("thalamus cache PUT success: stdout={s}", .{stdout_buf[0..stdout_len]});
        }
    }

    fn jsonEscape(self: *Thalamus, input: []const u8) ![]const u8 {
        var buf: std.ArrayListUnmanaged(u8) = .empty;
        for (input) |c| {
            switch (c) {
                '"' => try buf.appendSlice(self.allocator, "\\\""),
                '\\' => try buf.appendSlice(self.allocator, "\\\\"),
                '\n' => try buf.appendSlice(self.allocator, "\\n"),
                '\r' => try buf.appendSlice(self.allocator, "\\r"),
                '\t' => try buf.appendSlice(self.allocator, "\\t"),
                else => try buf.append(self.allocator, c),
            }
        }
        return try buf.toOwnedSlice(self.allocator);
    }

    fn parseResponse(self: *Thalamus, raw: []const u8) !Classification {
        const raw_owned = try self.allocator.dupe(u8, raw);

        // Find JSON object boundaries
        const start = std.mem.indexOf(u8, raw, "{") orelse {
            const routes = try self.allocator.alloc(Route, 1);
            routes[0] = .broca;
            return Classification{
                .class = .simple,
                .reflex_response = null,
                .routes = routes,
                .confidence = 0.5,
                .raw_response = raw_owned,
            };
        };
        const end = std.mem.lastIndexOf(u8, raw, "}") orelse {
            const routes = try self.allocator.alloc(Route, 1);
            routes[0] = .broca;
            return Classification{
                .class = .simple,
                .reflex_response = null,
                .routes = routes,
                .confidence = 0.5,
                .raw_response = raw_owned,
            };
        };

        const json_str = raw[start .. end + 1];

        // Parse with std.json
        var parsed = std.json.parseFromSlice(std.json.Value, self.allocator, json_str, .{}) catch {
            const routes = try self.allocator.alloc(Route, 1);
            routes[0] = .broca;
            return Classification{
                .class = .simple,
                .reflex_response = null,
                .routes = routes,
                .confidence = 0.5,
                .raw_response = raw_owned,
            };
        };
        defer parsed.deinit();

        const root = parsed.value.object;

        // Extract class
        const class_str = if (root.get("class")) |v| switch (v) {
            .string => |s| s,
            else => "simple",
        } else "simple";
        const class = SignalClass.fromSlice(class_str) orelse .simple;

        // Extract reflex response
        const reflex_response: ?[]const u8 = if (root.get("reflex")) |v| switch (v) {
            .string => |s| if (s.len > 0 and !std.mem.eql(u8, s, "null")) try self.allocator.dupe(u8, s) else null,
            else => null,
        } else null;

        // Extract routes
        var route_list: std.ArrayListUnmanaged(Route) = .empty;
        if (root.get("routes")) |v| {
            switch (v) {
                .array => |arr| {
                    for (arr.items) |item| {
                        switch (item) {
                            .string => |s| {
                                if (std.mem.eql(u8, s, "broca")) try route_list.append(self.allocator, .broca)
                                else if (std.mem.eql(u8, s, "prefrontal")) try route_list.append(self.allocator, .prefrontal)
                                else if (std.mem.eql(u8, s, "motor")) try route_list.append(self.allocator, .motor)
                                else if (std.mem.eql(u8, s, "amygdala")) try route_list.append(self.allocator, .amygdala)
                                else if (std.mem.eql(u8, s, "hippocampus")) try route_list.append(self.allocator, .hippocampus);
                            },
                            else => {},
                        }
                    }
                },
                else => {},
            }
        }
        if (route_list.items.len == 0) try route_list.append(self.allocator, .broca);

        // Extract confidence
        const confidence: f32 = if (root.get("confidence")) |v| switch (v) {
            .float => |f| @as(f32, @floatCast(f)),
            .integer => |i| @as(f32, @floatFromInt(i)),
            else => 0.8,
        } else 0.8;

        return Classification{
            .class = class,
            .reflex_response = reflex_response,
            .routes = try route_list.toOwnedSlice(self.allocator),
            .confidence = confidence,
            .raw_response = raw_owned,
        };
    }

    /// Format classification as a header string for injection into the agent's context.
    /// Example: "[Thalamus: class=complex, routes=[broca,prefrontal], confidence=0.92]"
    pub fn formatHeader(self: *Thalamus, classification: *const Classification) ![]const u8 {
        var buf: std.ArrayListUnmanaged(u8) = .empty;
        const writer = buf.writer(self.allocator);

        try writer.print("[Thalamus: class={s}, routes=[", .{classification.class.toSlice()});
        for (classification.routes, 0..) |route, i| {
            if (i > 0) try writer.writeAll(",");
            try writer.writeAll(route.toSlice());
        }
        try writer.print("], confidence={d:.2}]", .{classification.confidence});

        return try buf.toOwnedSlice(self.allocator);
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

test "parse reflex response" {
    const allocator = std.testing.allocator;
    var thalamus = Thalamus.init(allocator, undefined, "test", true);

    var result = try thalamus.parseResponse(
        \\{"class":"reflex","reflex":"Hey!","routes":["broca"],"confidence":0.97}
    );
    defer result.deinit(allocator);

    try std.testing.expectEqual(SignalClass.reflex, result.class);
    try std.testing.expect(result.reflex_response != null);
    try std.testing.expectEqualStrings("Hey!", result.reflex_response.?);
    try std.testing.expectEqual(@as(usize, 1), result.routes.len);
    try std.testing.expect(result.confidence > 0.96);
}

test "parse complex response" {
    const allocator = std.testing.allocator;
    var thalamus = Thalamus.init(allocator, undefined, "test", true);

    var result = try thalamus.parseResponse(
        \\{"class":"complex","reflex":null,"routes":["broca","prefrontal"],"confidence":0.91}
    );
    defer result.deinit(allocator);

    try std.testing.expectEqual(SignalClass.complex, result.class);
    try std.testing.expect(result.reflex_response == null);
    try std.testing.expectEqual(@as(usize, 2), result.routes.len);
}

test "parse dangerous response" {
    const allocator = std.testing.allocator;
    var thalamus = Thalamus.init(allocator, undefined, "test", true);

    var result = try thalamus.parseResponse(
        \\{"class":"dangerous","reflex":null,"routes":["broca","amygdala"],"confidence":0.99}
    );
    defer result.deinit(allocator);

    try std.testing.expectEqual(SignalClass.dangerous, result.class);
    try std.testing.expectEqual(@as(usize, 2), result.routes.len);
}

test "parse malformed json falls back gracefully" {
    const allocator = std.testing.allocator;
    var thalamus = Thalamus.init(allocator, undefined, "test", true);

    var result = try thalamus.parseResponse("not json at all");
    defer result.deinit(allocator);

    try std.testing.expectEqual(SignalClass.simple, result.class);
    try std.testing.expect(result.reflex_response == null);
}
