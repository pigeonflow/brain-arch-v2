//! Amygdala — Safety gate for dangerous operations.
//!
//! Activated when Thalamus classifies a message as `dangerous`.
//! Runs a fast Haiku check to assess risk BEFORE Broca's turn.
//! If truly dangerous, returns a warning. Otherwise, clears for Broca's.

const std = @import("std");
const providers = @import("providers/root.zig");
const brain_events = @import("brain_events.zig");

const Provider = providers.Provider;
const ChatMessage = providers.ChatMessage;

pub const SafetyVerdict = enum {
    safe,     // Proceed normally
    warn,     // Proceed but add warning context
    block,    // Block the action, return safety message
};

pub const SafetyResult = struct {
    verdict: SafetyVerdict,
    message: ?[]const u8, // Warning or block message
    allocator: std.mem.Allocator,

    pub fn deinit(self: *SafetyResult) void {
        if (self.message) |m| self.allocator.free(m);
    }
};

pub const Amygdala = struct {
    allocator: std.mem.Allocator,
    provider: *Provider,
    model_name: []const u8,

    const SYSTEM_PROMPT =
        \\You are a safety gate. A message was classified as potentially dangerous.
        \\Assess the ACTUAL risk level. Output ONLY a JSON object:
        \\{"verdict":"safe|warn|block","reason":"brief explanation"}
        \\
        \\Rules:
        \\- safe: looks dangerous but is actually fine (e.g. "delete my temp files", "rm -rf node_modules")
        \\- warn: risky but user likely knows what they're doing. Proceed with caution note.
        \\- block: genuinely dangerous (credentials exposure, irreversible destruction of important data, sending private info publicly)
        \\
        \\Be practical, not paranoid. Developers delete things. That's fine.
        \\Only block truly dangerous actions. Respond with ONLY the JSON object.
    ;

    pub fn init(allocator: std.mem.Allocator, provider: *Provider, model_name: []const u8) Amygdala {
        return .{
            .allocator = allocator,
            .provider = provider,
            .model_name = model_name,
        };
    }

    pub fn assess(self: *Amygdala, message: []const u8) !SafetyResult {
        brain_events.emit(.amygdala_assess, "{}");

        const messages = try self.allocator.alloc(ChatMessage, 2);
        defer self.allocator.free(messages);

        messages[0] = .{ .role = .system, .content = SYSTEM_PROMPT };
        messages[1] = .{ .role = .user, .content = message };

        const chat_response = try self.provider.chat(
            self.allocator,
            .{
                .messages = messages,
                .model = self.model_name,
                .temperature = 0.0,
                .max_tokens = 150,
                .tools = null,
                .timeout_secs = 5,
                .reasoning_effort = null,
            },
            self.model_name,
            0.0,
        );
        const response = chat_response.content orelse "";
        defer if (chat_response.content) |c| self.allocator.free(c);

        // Parse verdict
        const verdict = parseVerdict(response);
        const reason = parseJsonField(self.allocator, response, "reason");

        var ev_buf: [256]u8 = undefined;
        const ev_data = std.fmt.bufPrint(&ev_buf, "{{\"verdict\":\"{s}\"}}", .{
            @tagName(verdict),
        }) catch "{}";
        brain_events.emit(.amygdala_result, ev_data);

        return SafetyResult{
            .verdict = verdict,
            .message = reason,
            .allocator = self.allocator,
        };
    }

    fn parseVerdict(response: []const u8) SafetyVerdict {
        if (std.mem.indexOf(u8, response, "\"block\"") != null) return .block;
        if (std.mem.indexOf(u8, response, "\"warn\"") != null) return .warn;
        return .safe;
    }

    fn parseJsonField(allocator: std.mem.Allocator, json: []const u8, key: []const u8) ?[]const u8 {
        // Simple JSON field extraction
        var search_buf: [64]u8 = undefined;
        const search = std.fmt.bufPrint(&search_buf, "\"{s}\":\"", .{key}) catch return null;
        const start_idx = (std.mem.indexOf(u8, json, search) orelse return null) + search.len;
        const end_idx = std.mem.indexOfPos(u8, json, start_idx, "\"") orelse return null;
        return allocator.dupe(u8, json[start_idx..end_idx]) catch null;
    }
};
