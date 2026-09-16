# Evaluations for `design-scoring-mode`

Three scenarios exercising the common paths through this skill, in the data-driven format from
Anthropic's [Skill best-practices](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices).
There is no built-in runner — these are rubrics **for whoever is grading**, not instructions
for doing the work. They state the expected answer, so do not read them while carrying out a
task this skill covers. To use one: give Claude (with this skill loaded)
the `query`, working in the `fast_shepherd_score` repo, and check the result against
`expected_behavior`.

The scenarios span the three rungs of the reuse ladder in `../seams.md`:

| File | Rung exercised |
|---|---|
| `01_reuse_existing_optimizer.json` | new per-atom data, **existing optimizer reused outright** — no alignment code at all |
| `02_pure_reduction.json` | no new data, no new channel — **minimal additive** mode |
| `03_new_channel_field.json` | new per-atom data + **new `score/` channel** |

**Every query names a mode that does not exist in the tree**, deliberately: the library already
ships 21 modes, and a scenario restating one of them would test nothing but the duplicate check.
Confirm that is still true before running a scenario — if one of these has since been built, pick
a different absent field rather than grading against an existing implementation.

`files` is empty for every scenario: the working context is the `fast_shepherd_score` repository
itself, not per-eval input files.
