# Project skills

Skills committed here travel with the repo, so they work in any
environment the repo is checked out into — a local Claude Code
session, a cloud session, a teammate's machine — with no per-machine
install.

## Layout

```
.claude/skills/<skill-name>/SKILL.md
```

The directory name is the invoked name: `.claude/skills/impeccable/`
is invoked as `/impeccable`. Subdirectories nest the name
(`.claude/skills/nfl/backtest/` → `/nfl/backtest`). Supporting files
(scripts, reference docs, templates) live next to `SKILL.md` in the
same directory and are addressable as `${CLAUDE_SKILL_DIR}/<file>`.

## SKILL.md format

YAML frontmatter, then Markdown instructions:

```markdown
---
name: evidence-gate
description: One line saying WHEN to use this. This is what decides
  whether the skill gets picked up automatically, so write it as a
  trigger, not a summary.
allowed-tools: Bash Read Write        # optional, space-separated
disable-model-invocation: true        # optional: manual /invoke only
arguments: [market, seasons]          # optional named positional args
---

Instructions in plain Markdown. Reference arguments as $market,
$seasons, or $ARGUMENTS for everything.
```

Useful frontmatter:

| Field | Effect |
|---|---|
| `description` | when to use it — drives automatic invocation |
| `disable-model-invocation: true` | only fires when typed, never picked up on its own; use for anything with side effects |
| `user-invocable: false` | background reference only, never typed |
| `allowed-tools` | pre-approves tools for that skill's turn only |
| `context: fork` + `agent: Explore` | runs isolated, read-only — good for review/audit skills |
| `paths: "model/**/*.py"` | only activates when working in matching files |

## Porting a skill from a local install

A command installed only on one machine lives under that machine's
personal Claude config, not in the repo — which is why it does not
appear in other environments. To port it: copy its body into
`.claude/skills/<name>/SKILL.md` here, add the frontmatter above, and
commit. Nothing else is required; project skills are discovered
automatically, with no setting to enable and no trust prompt.

## Caveat worth knowing

A session enumerates skills when it starts. Committing a skill does
not light it up in a session that is already running — start a new
session against the updated repo to pick it up.

Docs: https://code.claude.com/docs/en/skills
