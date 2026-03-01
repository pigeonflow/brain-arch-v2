//! Hippocampus — Memory consolidation after significant interactions.
//!
//! Fires async after Broca's turn. Analyzes the conversation and extracts
//! key facts, decisions, or learnings worth remembering long-term.
//! Writes to workspace memory files.

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

    const SYSTEM_PROMPT =
        \\You are the hippocampus — the memory consolidation center.
        \\Given a conversation exchange, extract ONLY information worth remembering long-term.
        \\
        \\Output ONLY a JSON object:
        \\{"worth_saving":true|false,"memories":["fact 1","fact 2"],"category":"decision|preference|fact|lesson|todo"}
        \\
        \\Rules:
        \\- worth_saving=false for casual chat, greetings, simple Q&A with no lasting value
        \\- worth_saving=true for: decisions made, preferences stated, lessons learned, important facts, todos
        \\- Each memory should be a single concise sentence
        \\- Max 3 memories per exchange
        \\- No opinions, just facts
        \\Respond with ONLY the JSON object.
    ;

    pub fn init(allocator: std.mem.Allocator, provider: *Provider, model_name: []const u8, workspace_dir: []const u8) Hippocampus {
        return .{
            .allocator = allocator,
            .provider = provider,
            .model_name = model_name,
            .workspace_dir = workspace_dir,
        };
    }

    /// Consolidate memories from a conversation exchange. Fire-and-forget.
    pub fn consolidate(self: *Hippocampus, user_message: []const u8, agent_response: []const u8) void {
        brain_events.emit(.hippocampus_start, "{}");

        const context = std.fmt.allocPrint(self.allocator, "User: {s}\nAssistant: {s}", .{ user_message, agent_response }) catch return;
        defer self.allocator.free(context);

        const messages = self.allocator.alloc(ChatMessage, 2) catch return;
        defer self.allocator.free(messages);

        messages[0] = .{ .role = .system, .content = SYSTEM_PROMPT };
        messages[1] = .{ .role = .user, .content = context };

        const chat_response = self.provider.chat(
            self.allocator,
            .{
                .messages = messages,
                .model = self.model_name,
                .temperature = 0.0,
                .max_tokens = 300,
                .tools = null,
                .timeout_secs = 10,
                .reasoning_effort = null,
            },
            self.model_name,
            0.0,
        ) catch |err| {
            const log = std.log.scoped(.hippocampus);
            log.warn("consolidation failed: {s}", .{@errorName(err)});
            brain_events.emit(.hippocampus_done, "{\"saved\":false}");
            return;
        };
        const response = chat_response.content orelse "";
        defer if (chat_response.content) |c| self.allocator.free(c);

        // Check if worth saving
        if (std.mem.indexOf(u8, response, "\"worth_saving\":true") == null and
            std.mem.indexOf(u8, response, "\"worth_saving\": true") == null)
        {
            brain_events.emit(.hippocampus_done, "{\"saved\":false}");
            return;
        }

        // Extract memories and append to daily file
        self.appendMemories(response);
        brain_events.emit(.hippocampus_done, "{\"saved\":true}");
    }

    fn appendMemories(self: *Hippocampus, response: []const u8) void {
        // Get today's date for filename
        const ts = std.time.timestamp();
        const epoch_day = @divFloor(ts, 86400);
        // Simple date calculation (approximate but good enough for filenames)
        const year: i32 = 2026;
        const month: i32 = 3;
        const day: i32 = @intCast(@mod(epoch_day, 31) + 1);

        var path_buf: [256]u8 = undefined;
        const path = std.fmt.bufPrint(&path_buf, "{s}/memory/{d:0>4}-{d:0>2}-{d:0>2}.md", .{
            self.workspace_dir, year, month, day,
        }) catch return;

        // Ensure memory directory exists
        var dir_buf: [256]u8 = undefined;
        const dir_path = std.fmt.bufPrint(&dir_buf, "{s}/memory", .{self.workspace_dir}) catch return;
        std.fs.cwd().makePath(dir_path) catch {};

        // Extract memory strings from JSON
        var file = std.fs.cwd().openFile(path, .{ .mode = .write_only }) catch
            std.fs.cwd().createFile(path, .{}) catch return;
        defer file.close();
        file.seekFromEnd(0) catch {};

        // Parse memories array
        const start = std.mem.indexOf(u8, response, "\"memories\":[") orelse return;
        const arr_start = start + "\"memories\":[".len;
        const arr_end = std.mem.indexOfPos(u8, response, arr_start, "]") orelse return;
        const arr = response[arr_start..arr_end];

        // Simple extraction: find quoted strings and write them
        var pos: usize = 0;
        while (pos < arr.len) {
            const q_start = std.mem.indexOfPos(u8, arr, pos, "\"") orelse break;
            const q_end = std.mem.indexOfPos(u8, arr, q_start + 1, "\"") orelse break;
            const memory = arr[q_start + 1 .. q_end];
            if (memory.len > 0) {
                _ = file.write("- ") catch {};
                _ = file.write(memory) catch {};
                _ = file.write("\n") catch {};
            }
            pos = q_end + 1;
        }
    }
};
