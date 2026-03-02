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

    /// System prompt — intentionally tiny (~200 tokens). The thalamus doesn't think.
    const SYSTEM_PROMPT =
        \\You are a signal classifier. Given a user message, output ONLY a JSON object:
        \\{"class":"reflex|simple|complex|task|dangerous","reflex":"<immediate response or null>","routes":["broca"],"confidence":0.95}
        \\
        \\Classification rules:
        \\- reflex: greetings, acks, "hey", "thanks", "ok", "nice", casual chat. Include a short natural reflex response.
        \\- simple: factual questions, single-step requests, quick lookups. Routes to broca only.
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
        };
    }

    /// Classify an incoming message. Returns classification with optional reflex response.
    /// This should complete in <500ms with a fast model (Haiku-class).
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
        return self.parseResponse(response.content orelse "{}");
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
