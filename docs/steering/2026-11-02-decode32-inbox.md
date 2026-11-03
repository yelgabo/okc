# Decode32 inbox and waiting chat

November 2, 2026. Evidence pin: `bdbf35bf1ff8829dbe7f9eb6854afef6cee0dbd8`.
The user asked this chat to wait for tasks from "okc research" and create
`docs/steering/decode32/` so research can steer its direction.

## Setup and scope

The [task inbox](decode32/README.md) records the update convention and links the
[priority brief](2026-11-02-decode32-priority.md),
[owner directive](2026-11-02-decode32-owner-directive.md) and
[current captain allocation](2026-11-02-driver-d32-mid-comparison.md).
Root instructions and the steering index link the inbox so future sessions find it.

This chat uses `/Users/gretchengabo/Documents/ChatGPT/okc`, initialized as a
separate checkout on `codex/decode32-steering`. The existing checkout at
`/Users/gretchengabo/GitHub/okc` had local-only commits and untracked profiles;
its branch and files were preserved.

The chat is waiting for an assignment. OK Driver retains dispatch and the
existing OKC Developer retains the MID-at-T32 implementation. The captain's
current report records the direct-weight job `0b684a4065354fc5` as terminal and
parked. It must not be restarted because an older handoff still calls it pending.
This setup adds no kernel change, GPU job, benchmark or public submission.

## Monitoring contract

The five-minute heartbeat reads new local files in the two checkouts and new
remote files under `docs/steering/decode32/`. It fetches remote state without
resetting a checkout or overwriting local edits. It reads changed instructions
and the owning handoffs, acknowledges new assignments in this chat, and acts
within the assignment and current ownership. Conflicting instructions or
overlapping ownership require reconciliation before dependent edits.

The monitor keeps its last inspected commit and file hashes outside the repo at
`/Users/gretchengabo/.codex/decode32-steering-watch/state.json`. It stays quiet
when the folder is unchanged or has no actionable update. A failure to read the
folder or fetch remote changes must remain visible as an access limit.

Periodic local automation depends on the desktop app being able to run it.
Registering a schedule proves configuration, not that a future scheduled wake
has already occurred. Active work also reads the inbox at task boundaries.

## Validation and next action

Verify the new files and their relative links, the documentation-only diff,
the heartbeat configuration, its initial file manifest, and the published Git
blobs. Record the resulting commit in the chat after remote verification.
A read-only independent review checked the owning briefs and recommended
initializing this inbox while waiting, preserving the existing allocation.

Research can now add a dated task file. The first task should identify this
chat's work and file ownership if it differs from the current MID assignment.
Performance claims still require the project's normal source, correctness and
measurement evidence. Folder setup supplies no performance evidence.
