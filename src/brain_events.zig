//! Brain Events — lightweight pub/sub for brain-arch observability.
//!
//! Components (Thalamus, RAS, Broca's) emit events here.
//! The gateway SSE endpoint streams them to the dashboard.

const std = @import("std");

pub const EventKind = enum {
    thalamus_classify,
    thalamus_reflex,
    ras_busy,
    ras_idle,
    ras_interrupt,
    ras_queue,
    broca_start,
    broca_chunk,
    broca_done,
    broca_tool_call,
    broca_tool_result,
};

pub const Event = struct {
    kind: EventKind,
    timestamp_ms: i64,
    data: []const u8, // JSON payload
};

const MAX_SUBSCRIBERS = 8;
const RING_SIZE = 256;

/// Global event bus for brain activity.
var ring: [RING_SIZE]Event = undefined;
var ring_head: std.atomic.Value(u64) = std.atomic.Value(u64).init(0);
var subscribers: [MAX_SUBSCRIBERS]?*Subscriber = .{null} ** MAX_SUBSCRIBERS;
var sub_lock: std.Thread.Mutex = .{};

pub const Subscriber = struct {
    cursor: u64,
    event: std.Thread.ResetEvent = .{},

    pub fn init() Subscriber {
        return .{ .cursor = ring_head.load(.acquire) };
    }

    /// Wait for next event, returns null on timeout.
    pub fn poll(self: *Subscriber, timeout_ns: u64) ?Event {
        const head = ring_head.load(.acquire);
        if (self.cursor < head) {
            const idx = self.cursor % RING_SIZE;
            self.cursor += 1;
            return ring[idx];
        }
        self.event.timedWait(timeout_ns) catch {};
        self.event.reset();
        const head2 = ring_head.load(.acquire);
        if (self.cursor < head2) {
            const idx = self.cursor % RING_SIZE;
            self.cursor += 1;
            return ring[idx];
        }
        return null;
    }
};

pub fn subscribe(sub: *Subscriber) void {
    sub_lock.lock();
    defer sub_lock.unlock();
    for (&subscribers) |*slot| {
        if (slot.* == null) {
            slot.* = sub;
            return;
        }
    }
}

pub fn unsubscribe(sub: *Subscriber) void {
    sub_lock.lock();
    defer sub_lock.unlock();
    for (&subscribers) |*slot| {
        if (slot.* == sub) {
            slot.* = null;
            return;
        }
    }
}

/// Emit a brain event. Lock-free for the producer (single writer assumed).
pub fn emit(kind: EventKind, data: []const u8) void {
    const ts = std.time.milliTimestamp();
    const idx = ring_head.load(.monotonic) % RING_SIZE;
    ring[idx] = .{
        .kind = kind,
        .timestamp_ms = ts,
        .data = data,
    };
    _ = ring_head.fetchAdd(1, .release);

    // Wake all subscribers
    sub_lock.lock();
    defer sub_lock.unlock();
    for (&subscribers) |*slot| {
        if (slot.*) |s| {
            s.event.set();
        }
    }
}

/// Format an event as SSE line.
pub fn formatSSE(buf: *[4096]u8, ev: Event) []const u8 {
    const kind_str = @tagName(ev.kind);
    const written = std.fmt.bufPrint(buf, "event: {s}\ndata: {{\"kind\":\"{s}\",\"ts\":{d},\"data\":{s}}}\n\n", .{
        kind_str,
        kind_str,
        ev.timestamp_ms,
        if (ev.data.len > 0) ev.data else "null",
    }) catch return "";
    return written;
}
