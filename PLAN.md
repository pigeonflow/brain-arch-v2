# Brain-Arch v2 — Implementation Plan

## Modification Points in NullClaw

### 1. Thalamus (Pre-turn classifier + reflex)
**Where:** New `src/thalamus.zig` + hook in `session.zig:processMessage()` BEFORE `agent.turn()`
**What:**
- Intercept message before it hits the agent
- Fast Haiku call with tiny prompt (~200 tokens): classify as reflex/simple/complex/dangerous
- If reflex: send immediate response via stream_sink, THEN continue to agent.turn() (which may add more)
- Pre-warmed provider connection (keep Haiku connection alive)

### 2. RAS Interrupt (Mid-turn attention)
**Where:** Inside `agent/root.zig` tool loop (line ~787), check between iterations
**What:**
- New `src/ras.zig` — background thread watching a message queue/signal file
- Between each tool iteration in the while loop, check: `if (ras.hasInterrupt()) { break with interrupt; }`
- Session manager changes: instead of blocking on mutex, queue message + signal RAS
- RAS classifies incoming message priority while agent is working
- If interrupt-worthy: write signal, agent checks on next iteration

### 3. Slim System Prompt (Broca's optimization)  
**Where:** `agent/prompt.zig` buildSystemPrompt()
**What:**
- Thalamus classification injected as a header: `[Thalamus: class=complex, route=[prefrontal]]`
- Agent's SOUL.md can be much smaller — no routing logic needed, Thalamus already decided
- System prompt goes from ~5KB to ~500 tokens

## File Changes Summary

```
NEW:
  src/thalamus.zig          — Signal classifier + reflex response
  src/ras.zig               — Always-on attention, interrupt signaling

MODIFY:
  src/session.zig           — Thalamus hook before turn(), message queue for RAS
  src/agent/root.zig        — Interrupt check in tool loop
  src/agent/prompt.zig      — Inject Thalamus classification into system prompt
```

## Build Order
1. Thalamus (biggest impact, simplest change)
2. Slim prompts (depends on Thalamus output format)
3. RAS interrupt (most complex, requires threading changes)
