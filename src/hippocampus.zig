//! Hippocampus — Memory consolidation and retrieval.
//!
//! WRITE: Fires async after Broca's turn. Analyzes conversation, extracts
//! key facts/decisions/learnings, stores them via clawmem.
//! READ: Before Broca's turn, retrieves relevant memories for context injection.

const std = @import("std");
const providers = @import("providers/root.zig");
const brain_events = @import("brain_events.zig");

const Provider = providers.Provider;
const ChatMessage = providers.ChatMessage;

pub const Hippocampus = struct {
    allocator: std.mem.Allocator,
    provider: *Provider,
    model_name: []const u8,
    workspace_dir: []const u8,
    clawmem_bin: []const u8,
    clawmem_db: []const u8,
    agent_id: []const u8,

    const CONSOLIDATION_PROMPT =
        \\You are the hippocampus — the memory consolidation center.
        \\Given a conversation exchange, extract ONLY information worth remembering long-term.
        \\
        \\Output ONLY a JSON object:
        \\{"worth_saving":true|false,"memories":[{"text":"fact","type":"episodic|semantic|procedural","priority":0.0-1.0}],"category":"decision|preference|fact|lesson|todo"}
        \\
        \\Memory types:
        \\- episodic: events that happened (conversations, decisions made today)
        \\- semantic: enduring facts (user preferences, names, technical knowledge)
        \\- procedural: how-to knowledge (workflows, commands, tool usage)
        \\
        \\Rules:
        \\- worth_saving=false for casual chat, greetings, simple Q&A with no lasting value
        \\- worth_saving=true for: decisions made, preferences stated, lessons learned, important facts, todos
        \\- Each memory should be a single concise sentence
        \\- Max 5 memories per exchange
        \\- priority: 0.0 (trivial) to 1.0 (critical, e.g. user's name, explicit preferences)
        \\Respond with ONLY the JSON object.
    ;

    const RETRIEVAL_PROMPT =
        \\You are the hippocampus — the memory retrieval center.
        \\Given a user message, generate a brief search query (1-2 sentences) that captures
        \\what context/memories would help the assistant respond well.
        \\Focus on: who, what topic, what decisions, what preferences are relevant.
        \\Output ONLY the search query text, nothing else.
    ;

    pub fn init(
        allocator: std.mem.Allocator,
        provider: *Provider,
        model_name: []const u8,
        workspace_dir: []const u8,
    ) Hippocampus {
        // Derive paths from workspace_dir
        const bin_path = std.fmt.allocPrint(allocator, "{s}/../bin/clawmem", .{workspace_dir}) catch "clawmem";
        const db_path = std.fmt.allocPrint(allocator, "{s}/clawmem.db", .{workspace_dir}) catch "clawmem.db";

        return .{
            .allocator = allocator,
            .provider = provider,
            .model_name = model_name,
            .workspace_dir = workspace_dir,
            .clawmem_bin = bin_path,
            .clawmem_db = db_path,
            .agent_id = "caramelo",
        };
    }

    // ---- WRITE PATH: Consolidation ----

    /// Consolidate memories from a conversation exchange. Fire-and-forget.
    pub fn consolidate(self: *Hippocampus, user_message: []const u8, agent_response: []const u8) void {
        brain_events.emit(.hippocampus_start, "{\"phase\":\"consolidate\"}");

        const context = std.fmt.allocPrint(self.allocator, "User: {s}\nAssistant: {s}", .{ user_message, agent_response }) catch return;
        defer self.allocator.free(context);

        const messages = self.allocator.alloc(ChatMessage, 2) catch return;
        defer self.allocator.free(messages);

        messages[0] = .{ .role = .system, .content = CONSOLIDATION_PROMPT };
        messages[1] = .{ .role = .user, .content = context };

        const chat_response = self.provider.chat(
            self.allocator,
            .{
                .messages = messages,
                .model = self.model_name,
                .temperature = 0.0,
                .max_tokens = 500,
                .tools = null,
                .timeout_secs = 10,
                .reasoning_effort = null,
            },
            self.model_name,
            0.0,
        ) catch |err| {
            const log = std.log.scoped(.hippocampus);
            log.warn("consolidation failed: {s}", .{@errorName(err)});
            brain_events.emit(.hippocampus_done, "{\"saved\":false,\"error\":\"provider_call\"}");
            return;
        };
        const response = chat_response.content orelse "";
        defer if (chat_response.content) |c| self.allocator.free(c);

        // Check if worth saving
        if (std.mem.indexOf(u8, response, "\"worth_saving\":true") == null and
            std.mem.indexOf(u8, response, "\"worth_saving\": true") == null)
        {
            brain_events.emit(.hippocampus_done, "{\"saved\":false,\"reason\":\"not_worth\"}");
            return;
        }

        // Extract and store memories via clawmem
        const count = self.storeMemories(response);
        var event_buf: [128]u8 = undefined;
        const event = std.fmt.bufPrint(&event_buf, "{{\"saved\":true,\"count\":{d}}}", .{count}) catch "{\"saved\":true}";
        brain_events.emit(.hippocampus_done, event);

        // Also write to markdown as fallback (preserves human-readable log)
        self.appendMarkdown(response);
    }

    fn storeMemories(self: *Hippocampus, response: []const u8) usize {
        // Parse memories from JSON response
        const start = std.mem.indexOf(u8, response, "\"memories\":[") orelse return 0;
        const arr_start = start + "\"memories\":[".len;
        const arr_end = std.mem.indexOfPos(u8, response, arr_start, "]") orelse return 0;
        const arr = response[arr_start..arr_end];

        var count: usize = 0;

        // Find each object in the array
        var pos: usize = 0;
        while (pos < arr.len) {
            const obj_start = std.mem.indexOfPos(u8, arr, pos, "{") orelse break;
            const obj_end = std.mem.indexOfPos(u8, arr, obj_start, "}") orelse break;
            const obj = arr[obj_start .. obj_end + 1];

            // Extract text field
            const text = extractJsonString(obj, "text") orelse {
                // Fallback: try plain quoted string (old format)
                const q_start = std.mem.indexOfPos(u8, arr, pos, "\"") orelse break;
                const q_end = std.mem.indexOfPos(u8, arr, q_start + 1, "\"") orelse break;
                pos = q_end + 1;
                continue;
            };
            const mem_type = extractJsonString(obj, "type") orelse "episodic";
            const priority = extractJsonFloat(obj, "priority") orelse 0.5;

            // Generate simple hash embedding for the text
            var embedding: [64]f32 = undefined;
            hashEmbed(text, &embedding);

            // Call clawmem upsert
            self.callClawmemUpsert(text, mem_type, priority, &embedding) catch {
                pos = obj_end + 1;
                continue;
            };
            count += 1;
            pos = obj_end + 1;
        }

        return count;
    }

    fn callClawmemUpsert(self: *Hippocampus, text: []const u8, mem_type: []const u8, priority: f64, embedding: []const f32) !void {
        // Build JSON input for clawmem
        var json_buf: [4096]u8 = undefined;

        // Build embedding array string
        var emb_buf: [2048]u8 = undefined;
        var emb_pos: usize = 0;
        emb_buf[emb_pos] = '[';
        emb_pos += 1;
        for (embedding, 0..) |val, i| {
            if (i > 0) {
                emb_buf[emb_pos] = ',';
                emb_pos += 1;
            }
            const printed = std.fmt.bufPrint(emb_buf[emb_pos..], "{d:.6}", .{val}) catch break;
            emb_pos += printed.len;
        }
        emb_buf[emb_pos] = ']';
        emb_pos += 1;

        const json = std.fmt.bufPrint(&json_buf,
            \\{{"type":"{s}","content":"{s}","embedding":{s},"priority":{d:.2}}}
        , .{ mem_type, text, emb_buf[0..emb_pos], priority }) catch return error.BufferTooSmall;

        // Shell out to clawmem
        var child = std.process.Child.init(.{
            .argv = &.{ self.clawmem_bin, "--db", self.clawmem_db, "upsert", "--agent", self.agent_id },
            .stdin_behavior = .pipe,
            .stdout_behavior = .pipe,
            .stderr_behavior = .pipe,
        }, self.allocator);

        child.spawn() catch return error.SpawnFailed;

        if (child.stdin) |stdin| {
            _ = stdin.write(json) catch {};
            stdin.close();
            child.stdin = null;
        }

        _ = child.wait() catch {};
    }

    // ---- READ PATH: Retrieval ----

    /// Retrieve relevant memories for a user message. Returns formatted context string.
    pub fn retrieve(self: *Hippocampus, user_message: []const u8) ?[]const u8 {
        brain_events.emit(.hippocampus_start, "{\"phase\":\"retrieve\"}");

        // Generate hash embedding for the query
        var query_emb: [64]f32 = undefined;
        hashEmbed(user_message, &query_emb);

        // Call clawmem search
        const results = self.callClawmemSearch(&query_emb, 5) catch {
            brain_events.emit(.hippocampus_done, "{\"retrieved\":0}");
            return null;
        };
        defer self.allocator.free(results);

        if (results.len == 0 or std.mem.eql(u8, results, "[]") or std.mem.eql(u8, results, "[\n]")) {
            brain_events.emit(.hippocampus_done, "{\"retrieved\":0}");
            return null;
        }

        // Parse results and format as context
        const formatted = self.formatMemories(results) catch {
            brain_events.emit(.hippocampus_done, "{\"retrieved\":0,\"error\":\"format\"}");
            return null;
        };

        brain_events.emit(.hippocampus_done, "{\"retrieved\":true}");
        return formatted;
    }

    fn callClawmemSearch(self: *Hippocampus, embedding: []const f32, k: usize) ![]const u8 {
        var emb_buf: [2048]u8 = undefined;
        var emb_pos: usize = 0;
        emb_buf[emb_pos] = '[';
        emb_pos += 1;
        for (embedding, 0..) |val, i| {
            if (i > 0) {
                emb_buf[emb_pos] = ',';
                emb_pos += 1;
            }
            const printed = std.fmt.bufPrint(emb_buf[emb_pos..], "{d:.6}", .{val}) catch break;
            emb_pos += printed.len;
        }
        emb_buf[emb_pos] = ']';
        emb_pos += 1;

        var json_buf: [2560]u8 = undefined;
        const json = std.fmt.bufPrint(&json_buf,
            \\{{"query_embedding":{s},"k":{d}}}
        , .{ emb_buf[0..emb_pos], k }) catch return error.BufferTooSmall;

        var child = std.process.Child.init(.{
            .argv = &.{ self.clawmem_bin, "--db", self.clawmem_db, "search", "--agent", self.agent_id },
            .stdin_behavior = .pipe,
            .stdout_behavior = .pipe,
            .stderr_behavior = .pipe,
        }, self.allocator);

        child.spawn() catch return error.SpawnFailed;

        if (child.stdin) |stdin| {
            _ = stdin.write(json) catch {};
            stdin.close();
            child.stdin = null;
        }

        var stdout_buf = std.ArrayList(u8).init(self.allocator);
        if (child.stdout) |stdout| {
            while (true) {
                var buf: [4096]u8 = undefined;
                const n = stdout.read(&buf) catch break;
                if (n == 0) break;
                stdout_buf.appendSlice(buf[0..n]) catch break;
            }
        }

        _ = child.wait() catch {};
        return stdout_buf.toOwnedSlice() catch return error.OutOfMemory;
    }

    fn formatMemories(self: *Hippocampus, results_json: []const u8) ![]const u8 {
        // Extract "content" fields from results JSON and format as context block
        var output = std.ArrayList(u8).init(self.allocator);
        try output.appendSlice("[Recalled memories]\n");

        var pos: usize = 0;
        var count: usize = 0;
        while (pos < results_json.len and count < 10) {
            const content_key = std.mem.indexOfPos(u8, results_json, pos, "\"content\":\"") orelse break;
            const val_start = content_key + "\"content\":\"".len;
            const val_end = std.mem.indexOfPos(u8, results_json, val_start, "\"") orelse break;
            const content = results_json[val_start..val_end];
            if (content.len > 0 and !std.mem.eql(u8, content, "null")) {
                try output.appendSlice("- ");
                try output.appendSlice(content);
                try output.appendSlice("\n");
                count += 1;
            }
            pos = val_end + 1;
        }

        if (count == 0) {
            output.deinit();
            return error.NoResults;
        }

        return output.toOwnedSlice();
    }

    // ---- Markdown fallback (human-readable log) ----

    fn appendMarkdown(self: *Hippocampus, response: []const u8) void {
        const ts = std.time.timestamp();
        // Proper date calculation
        const days_since_epoch = @divFloor(ts, 86400);
        const date = epochDayToDate(days_since_epoch);

        var path_buf: [256]u8 = undefined;
        const path = std.fmt.bufPrint(&path_buf, "{s}/memory/{d:0>4}-{d:0>2}-{d:0>2}.md", .{
            self.workspace_dir, date.year, date.month, date.day,
        }) catch return;

        var dir_buf: [256]u8 = undefined;
        const dir_path = std.fmt.bufPrint(&dir_buf, "{s}/memory", .{self.workspace_dir}) catch return;
        std.fs.cwd().makePath(dir_path) catch {};

        var file = std.fs.cwd().openFile(path, .{ .mode = .write_only }) catch
            std.fs.cwd().createFile(path, .{}) catch return;
        defer file.close();
        file.seekFromEnd(0) catch {};

        // Extract and write memories
        const start = std.mem.indexOf(u8, response, "\"memories\"") orelse return;
        _ = start;

        // Write each memory text
        var pos: usize = 0;
        while (pos < response.len) {
            const text = extractJsonStringFromPos(response, "text", pos) orelse break;
            _ = file.write("- ") catch {};
            _ = file.write(text) catch {};
            _ = file.write("\n") catch {};
            pos = std.mem.indexOfPos(u8, response, pos + 1, "\"text\"") orelse break;
            pos += 1;
        }
    }

    // ---- Utilities ----

    /// Simple hash-based embedding. Not semantic — just deterministic projection.
    /// Replace with real embedding model in v0.2.
    fn hashEmbed(text: []const u8, out: []f32) void {
        // Use multiple hash seeds to project text into a fixed-dim vector
        for (out, 0..) |*slot, i| {
            var h: u64 = 0xcbf29ce484222325; // FNV-1a
            h ^= @as(u64, @intCast(i));
            h *%= 0x100000001b3;
            for (text) |c| {
                h ^= @as(u64, c);
                h *%= 0x100000001b3;
            }
            // Map to [-1, 1]
            slot.* = @as(f32, @floatFromInt(@as(i32, @intCast(h & 0x7FFFFFFF)))) / @as(f32, 0x7FFFFFFF) * 2.0 - 1.0;
        }
        // Normalize
        var norm: f32 = 0;
        for (out) |v| norm += v * v;
        norm = @sqrt(norm);
        if (norm > 1e-8) {
            for (out) |*v| v.* /= norm;
        }
    }
};

// ---- Date utilities (fix the broken date math) ----

const Date = struct { year: i32, month: i32, day: i32 };

fn epochDayToDate(days: i64) Date {
    // Algorithm from Howard Hinnant's date library (civil_from_days)
    var z = days + 719468;
    const era: i64 = @divFloor(if (z >= 0) z else z - 146096, 146097);
    const doe: u64 = @intCast(z - era * 146097);
    const yoe: u64 = @divFloor(doe - @divFloor(doe, 1460) + @divFloor(doe, 36524) - @divFloor(doe, 146096), 365);
    const y: i64 = @as(i64, @intCast(yoe)) + era * 400;
    const doy: u64 = doe - (365 * yoe + @divFloor(yoe, 4) - @divFloor(yoe, 100));
    const mp: u64 = @divFloor(5 * doy + 2, 153);
    const d: u64 = doy - @divFloor(153 * mp + 2, 5) + 1;
    const m: u64 = if (mp < 10) mp + 3 else mp - 9;
    return .{
        .year = @intCast(if (m <= 2) y + 1 else y),
        .month = @intCast(m),
        .day = @intCast(d),
    };
}

fn extractJsonString(json: []const u8, key: []const u8) ?[]const u8 {
    return extractJsonStringFromPos(json, key, 0);
}

fn extractJsonStringFromPos(json: []const u8, key: []const u8, start: usize) ?[]const u8 {
    // Find "key":"value"
    var search_buf: [64]u8 = undefined;
    const needle = std.fmt.bufPrint(&search_buf, "\"{s}\":\"", .{key}) catch return null;
    const pos = std.mem.indexOfPos(u8, json, start, needle) orelse {
        // Try with space after colon
        const needle2 = std.fmt.bufPrint(&search_buf, "\"{s}\": \"", .{key}) catch return null;
        const pos2 = std.mem.indexOfPos(u8, json, start, needle2) orelse return null;
        const val_start = pos2 + needle2.len;
        const val_end = std.mem.indexOfPos(u8, json, val_start, "\"") orelse return null;
        return json[val_start..val_end];
    };
    const val_start = pos + needle.len;
    const val_end = std.mem.indexOfPos(u8, json, val_start, "\"") orelse return null;
    return json[val_start..val_end];
}

fn extractJsonFloat(json: []const u8, key: []const u8) ?f64 {
    var search_buf: [64]u8 = undefined;
    const needle = std.fmt.bufPrint(&search_buf, "\"{s}\":", .{key}) catch return null;
    const pos = std.mem.indexOfPos(u8, json, 0, needle) orelse return null;
    var val_start = pos + needle.len;
    // Skip whitespace
    while (val_start < json.len and json[val_start] == ' ') val_start += 1;
    var val_end = val_start;
    while (val_end < json.len and (json[val_end] == '.' or (json[val_end] >= '0' and json[val_end] <= '9'))) val_end += 1;
    if (val_start == val_end) return null;
    return std.fmt.parseFloat(f64, json[val_start..val_end]) catch null;
}
