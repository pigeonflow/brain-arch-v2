//! RAS (Reticular Activating System) — Always-on attention and interrupt signaling.
//!
//! Runs as a background thread that monitors for incoming messages while the
//! main agent is processing. When a high-priority signal arrives (correction,
//! "stop", urgent message), RAS writes an interrupt signal that the agent
//! checks between tool iterations.
//!
//! This solves the "deaf while working" problem — the brain's RAS never sleeps.

const std = @import("std");
const log = std.log.scoped(.ras);

// ═══════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════

pub const InterruptPriority = enum {
    /// Low — can wait until current turn finishes
    low,
    /// Medium — process at next natural break point
    medium,
    /// High — interrupt current work ASAP (correction, "stop", "actually...")
    high,
    /// Critical — drop everything (safety concern, system alert)
    critical,
};

pub const InterruptSignal = struct {
    priority: InterruptPriority,
    message: []const u8,
    source: []const u8,
    timestamp: i64,
};

// ═══════════════════════════════════════════════════════════════════════════
// RAS
// ═══════════════════════════════════════════════════════════════════════════

pub const Ras = struct {
    allocator: std.mem.Allocator,
    mutex: std.Thread.Mutex,

    /// Pending interrupt signal. Checked by the agent between tool iterations.
    pending_interrupt: ?InterruptSignal,

    /// Whether the agent is currently processing a turn.
    agent_busy: bool,

    /// Queue of messages that arrived while agent was busy.
    queued_messages: std.ArrayListUnmanaged(QueuedMessage),

    const QueuedMessage = struct {
        content: []const u8,
        session_key: []const u8,
        timestamp: i64,
    };

    pub fn init(allocator: std.mem.Allocator) Ras {
        return .{
            .allocator = allocator,
            .mutex = .{},
            .pending_interrupt = null,
            .agent_busy = false,
            .queued_messages = .empty,
        };
    }

    pub fn deinit(self: *Ras) void {
        if (self.pending_interrupt) |*sig| {
            self.allocator.free(sig.message);
            self.allocator.free(sig.source);
        }
        for (self.queued_messages.items) |msg| {
            self.allocator.free(msg.content);
            self.allocator.free(msg.session_key);
        }
        self.queued_messages.deinit(self.allocator);
    }

    /// Called by the session manager when a new message arrives and the agent is busy.
    /// Classifies the message priority and either queues it or raises an interrupt.
    pub fn onMessageWhileBusy(self: *Ras, content: []const u8, session_key: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const priority = classifyInterruptPriority(content);

        log.info("ras: message while busy, priority={s} session={s}", .{
            @tagName(priority),
            session_key,
        });

        switch (priority) {
            .high, .critical => {
                // Raise interrupt — agent should stop current work
                if (self.pending_interrupt) |*old| {
                    self.allocator.free(old.message);
                    self.allocator.free(old.source);
                }
                self.pending_interrupt = .{
                    .priority = priority,
                    .message = try self.allocator.dupe(u8, content),
                    .source = try self.allocator.dupe(u8, session_key),
                    .timestamp = std.time.timestamp(),
                };
            },
            .low, .medium => {
                // Queue for later processing
                try self.queued_messages.append(self.allocator, .{
                    .content = try self.allocator.dupe(u8, content),
                    .session_key = try self.allocator.dupe(u8, session_key),
                    .timestamp = std.time.timestamp(),
                });
            },
        }
    }

    /// Called by the agent between tool iterations to check for interrupts.
    /// Returns the interrupt signal if one is pending, and clears it.
    pub fn checkInterrupt(self: *Ras) ?InterruptSignal {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.pending_interrupt) |sig| {
            self.pending_interrupt = null;
            return sig;
        }
        return null;
    }

    /// Mark agent as busy (starting a turn).
    pub fn agentBusy(self: *Ras) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.agent_busy = true;
    }

    /// Mark agent as idle (turn complete). Returns any queued messages.
    pub fn agentIdle(self: *Ras) []QueuedMessage {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.agent_busy = false;

        const queued = self.queued_messages.toOwnedSlice(self.allocator) catch return &.{};
        self.queued_messages = .empty;
        return queued;
    }

    /// Check if agent is currently busy.
    pub fn isAgentBusy(self: *Ras) bool {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.agent_busy;
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// Interrupt Classification (fast, no LLM needed)
// ═══════════════════════════════════════════════════════════════════════════

/// Fast heuristic classification of interrupt priority.
/// This runs synchronously — no LLM call. Must be < 1ms.
///
/// For v2, this could use a pre-warmed Haiku call for nuanced classification.
/// For now, keyword heuristics are fast and good enough for common patterns.
fn classifyInterruptPriority(content: []const u8) InterruptPriority {
    // Normalize to lowercase for matching
    var lower_buf: [512]u8 = undefined;
    const len = @min(content.len, lower_buf.len);
    for (0..len) |i| {
        lower_buf[i] = std.ascii.toLower(content[i]);
    }
    const lower = lower_buf[0..len];

    // Critical: explicit stop/cancel commands
    if (std.mem.startsWith(u8, lower, "stop") or
        std.mem.startsWith(u8, lower, "cancel") or
        std.mem.startsWith(u8, lower, "abort") or
        std.mem.eql(u8, std.mem.trim(u8, lower, " \t\n"), "no"))
    {
        return .critical;
    }

    // High: corrections and redirections
    if (std.mem.startsWith(u8, lower, "actually") or
        std.mem.startsWith(u8, lower, "wait") or
        std.mem.startsWith(u8, lower, "oops") or
        std.mem.startsWith(u8, lower, "sorry") or
        std.mem.startsWith(u8, lower, "never mind") or
        std.mem.startsWith(u8, lower, "nevermind") or
        std.mem.indexOf(u8, lower, "i meant") != null or
        std.mem.indexOf(u8, lower, "not that") != null or
        std.mem.indexOf(u8, lower, "wrong") != null)
    {
        return .high;
    }

    // Medium: new questions or requests while busy
    if (std.mem.indexOf(u8, lower, "?") != null or
        std.mem.startsWith(u8, lower, "can you") or
        std.mem.startsWith(u8, lower, "please"))
    {
        return .medium;
    }

    // Low: everything else (probably follow-up context)
    return .low;
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

test "classify stop as critical" {
    const priority = classifyInterruptPriority("stop");
    try std.testing.expectEqual(InterruptPriority.critical, priority);
}

test "classify correction as high" {
    const priority = classifyInterruptPriority("oops, I meant phase 2 and 3");
    try std.testing.expectEqual(InterruptPriority.high, priority);
}

test "classify actually as high" {
    const priority = classifyInterruptPriority("Actually, forget that — do this instead");
    try std.testing.expectEqual(InterruptPriority.high, priority);
}

test "classify question as medium" {
    const priority = classifyInterruptPriority("can you also check the logs?");
    try std.testing.expectEqual(InterruptPriority.medium, priority);
}

test "classify follow-up as low" {
    const priority = classifyInterruptPriority("also here is some extra context for you");
    try std.testing.expectEqual(InterruptPriority.low, priority);
}

test "ras interrupt flow" {
    const allocator = std.testing.allocator;
    var ras = Ras.init(allocator);
    defer ras.deinit();

    // Agent starts working
    ras.agentBusy();
    try std.testing.expect(ras.isAgentBusy());

    // User sends correction while agent is busy
    try ras.onMessageWhileBusy("oops, I meant phase 2", "session-1");

    // Agent checks for interrupt between tool calls
    const interrupt = ras.checkInterrupt();
    try std.testing.expect(interrupt != null);
    try std.testing.expectEqual(InterruptPriority.high, interrupt.?.priority);
    try std.testing.expectEqualStrings("oops, I meant phase 2", interrupt.?.message);

    // Clean up the returned signal
    allocator.free(interrupt.?.message);
    allocator.free(interrupt.?.source);

    // Agent finishes
    const queued = ras.agentIdle();
    allocator.free(queued);
    try std.testing.expect(!ras.isAgentBusy());
}

test "ras queues low priority messages" {
    const allocator = std.testing.allocator;
    var ras = Ras.init(allocator);
    defer ras.deinit();

    ras.agentBusy();

    // Low priority message gets queued, not interrupted
    try ras.onMessageWhileBusy("also here is some context", "session-1");

    const interrupt = ras.checkInterrupt();
    try std.testing.expect(interrupt == null);

    // But it's in the queue when agent finishes
    const queued = ras.agentIdle();
    defer allocator.free(queued);
    try std.testing.expectEqual(@as(usize, 1), queued.len);
    allocator.free(queued[0].content);
    allocator.free(queued[0].session_key);
}
