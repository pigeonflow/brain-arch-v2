//! Prefrontal Cortex — Deep async reasoning for complex queries.
//!
//! Activated when Thalamus classifies a message as `complex`.
//! Fires AFTER Broca's initial response using a heavier model (Opus-class).
//! Returns a deeper analysis that gets appended as a follow-up.
//! Implements Hugo's principle: "A boca fala mais rápido que o cérebro."

const std = @import("std");
const providers = @import("providers/root.zig");
const brain_events = @import("brain_events.zig");

const Provider = providers.Provider;
const ChatMessage = providers.ChatMessage;

pub const PrefrontalResult = struct {
    analysis: []const u8,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *PrefrontalResult) void {
        self.allocator.free(self.analysis);
    }
};

pub const Prefrontal = struct {
    allocator: std.mem.Allocator,
    provider: *Provider,
    model_name: []const u8,

    const SYSTEM_PROMPT =
        \\You are the prefrontal cortex — the deep reasoning center.
        \\The user's message was already answered quickly by Broca's area.
        \\Your job: provide DEEPER analysis that the fast response missed.
        \\
        \\Rules:
        \\- Only add value. If the fast response was sufficient, say "No additional analysis needed."
        \\- Focus on: hidden risks, alternative approaches, long-term implications, edge cases.
        \\- Be concise. This is a follow-up, not a replacement.
        \\- Start with "🧠 **Deeper analysis:**" if you have something to add.
        \\- No preamble, no repeating what Broca already said.
    ;

    pub fn init(allocator: std.mem.Allocator, provider: *Provider, model_name: []const u8) Prefrontal {
        return .{
            .allocator = allocator,
            .provider = provider,
            .model_name = model_name,
        };
    }

    /// Run deep analysis. Called with both the user message and Broca's response.
    pub fn analyze(self: *Prefrontal, user_message: []const u8, broca_response: []const u8) !PrefrontalResult {
        brain_events.emit(.prefrontal_start, "{}");

        const context = try std.fmt.allocPrint(self.allocator, "User message: {s}\n\nBroca's quick response: {s}\n\nProvide deeper analysis if needed.", .{ user_message, broca_response });
        defer self.allocator.free(context);

        const messages = try self.allocator.alloc(ChatMessage, 2);
        defer self.allocator.free(messages);

        messages[0] = .{ .role = .system, .content = SYSTEM_PROMPT };
        messages[1] = .{ .role = .user, .content = context };

        const chat_response = try self.provider.chat(
            self.allocator,
            .{
                .messages = messages,
                .model = self.model_name,
                .temperature = 0.3,
                .max_tokens = 1000,
                .tools = null,
                .timeout_secs = 30,
                .reasoning_effort = null,
            },
            self.model_name,
            0.3,
        );
        const response = chat_response.content orelse "";

        brain_events.emit(.prefrontal_done, "{}");

        if (std.mem.indexOf(u8, response, "No additional analysis needed") != null) {
            if (chat_response.content) |c| self.allocator.free(c);
            return PrefrontalResult{
                .analysis = try self.allocator.dupe(u8, ""),
                .allocator = self.allocator,
            };
        }

        return PrefrontalResult{
            .analysis = if (chat_response.content) |c| c else try self.allocator.dupe(u8, ""),
            .allocator = self.allocator,
        };
    }
};
