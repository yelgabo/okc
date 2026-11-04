# Lane checkpoints

One file per lane job name (moe, layer, attention-d16, ...), overwritten by the lane after every step: base file and SHA256, measurements with numbers, candidate files and SHA256s, box paths, next action. A relaunched lane starts from its checkpoint instead of from scratch. Rule: `.claude/agents/*.md`, "Context budget".
