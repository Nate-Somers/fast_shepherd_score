# Evaluations for `accelerate-scoring-mode`

Three scenarios exercising the common paths through this skill, in the data-driven format from
Anthropic's [Skill best-practices](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices).
There is no built-in runner — these are rubrics **for whoever is grading**, not instructions
for doing the work. They state the expected answer, so do not read them while carrying out a
task this skill covers. To use one: give Claude (with this skill loaded)
the `query`, working in the `fast_shepherd_score` repo with the named reference mode already
present, and check the result against `expected_behavior`.

| File | Path exercised |
|---|---|
| `01_reuse_no_new_kernel.json` | **reuse existing kernels** (skip the kernel trio) + a NEW CHANNEL (Tier-B store) |
| `02_new_channel_kernel.json` | **new kernel pair** (numba first, dispatch, three engine branches) + a pair-level channel |
| `03_tier_a_screening_speed.json` | reduction-only mode: **one spec, zero screen edits**, and array-path honesty |

The queries name modes that do **not** exist in the tree. The library already ships 21, and
`vol_lipo` / `vol_tversky` / `vol_avoid` — which earlier versions of these evals used — are all
built, so grading against them tests recall of a diff rather than the skill. Each scenario assumes
`design-scoring-mode` has just produced the reference layer and nothing else.

Three things every scenario should surface, because they are the failures this skill exists to
prevent: reaching for a new kernel when an existing one already emits the channel; writing a
per-mode driver, aligner, array builder or `screen.py` branch that the ModeSpec already generates
(which silently takes the mode back off the derived path); and reporting a screening throughput
number from the object path.

`files` is empty for every scenario: the working context is the repository itself.
