# Evaluations for `accelerate-scoring-mode`

Three scenarios that exercise the common paths through this skill, in the data-driven format from
Anthropic's [Skill best-practices](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices).
There is no built-in runner — these are rubrics. To use one: give Claude (with this skill loaded)
the `query`, working in the `fast_shepherd_score` repo with the named reference mode already present,
and check the result against `expected_behavior`.

The scenarios span the three shapes an acceleration task takes:

| File | Path exercised |
|---|---|
| `01_reuse_no_new_kernel.json` | **reuse existing kernels** (skip steps 2-4) + Tier-B screen wiring |
| `02_new_channel_kernel.json` | **new kernel pair** (numba-first, parity gates) |
| `03_tier_a_screening_speed.json` | reduction mode + **Tier-A screen + fast-engine honesty** |

`files` is empty for every scenario: the working context is the `fast_shepherd_score` repository,
where the reference mode from `design-scoring-mode` already exists.
