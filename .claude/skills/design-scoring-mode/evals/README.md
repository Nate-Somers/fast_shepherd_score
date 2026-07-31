# Evaluations for `design-scoring-mode`

Three scenarios that exercise the common paths through this skill, in the data-driven format from
Anthropic's [Skill best-practices](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices).
There is no built-in runner — these are rubrics. To use one: give Claude (with this skill loaded)
the `query`, working in the `fast_shepherd_score` repo, and check the result against
`expected_behavior`.

The scenarios deliberately span the three shapes a design task takes:

| File | Path exercised |
|---|---|
| `01_reuse_scalar_field.json` | new per-atom data + **channel reuse** (`get_overlap_esp`) |
| `02_pure_reduction.json` | no new data, no new channel — **minimal additive** mode |
| `03_new_channel_field.json` | new per-atom data + **new `score/` channel** |

`files` is empty for every scenario: the working context is the `fast_shepherd_score` repository
itself, not per-eval input files.
