# Developing scoring modes with agent skills

Two skills support adding scoring modes to this checkout:

- **`design-scoring-mode`** develops and tests a PyTorch reference objective
  and single-pair alignment method from a mathematical or plain-language description.
- **`accelerate-scoring-mode`** adds the mode and any new features to the
  registries, reuses or implements Numba/Triton kernels, and checks the batched
  and screening paths against the reference.

The skills and their supporting references live in
`.claude/skills/design-scoring-mode/` and
`.claude/skills/accelerate-scoring-mode/` at the repository root. Each folder
contains a `SKILL.md` entry point. Read the relevant entry point or ask a
skill-capable coding agent to use it against this checkout. For another agent's
skill directory, copy the complete folder, including its supporting references.
The skills are development instructions, not Python scoring APIs, and do not
run automatically when `shepherd_score` is imported.

Both folders are included in source distributions. Wheels install them under
the installation's data directory at `share/shepherd-score/skills/`. Locate
that directory with the Python interpreter used for the installation:

```python
from pathlib import Path
import sysconfig

skills = Path(sysconfig.get_path("data")) / "share/shepherd-score/skills"
print(skills)
```

For editable installs, use the folders in the source checkout. Personal agent
settings and worktrees are not distributed. These locations describe builds
from this checkout; older published releases may not contain the skills.

The reference workflow tests the objective and gradients before acceleration.
The acceleration workflow checks fixed-pose values and gradients, effective
optimization settings, returned transforms, feature storage, and execution
paths. It distinguishes reference and accelerated seed sets, sub-batch early
stopping, and frozen Triton launch configurations. The included Python test
template must be completed for the new objective; its unfilled gates fail.

Agent output still requires review and validation. Neither a unit self-score
nor agreement on a few optimized pairs establishes correctness for all inputs.
See the [API guide](accelerated_api.md) for existing modes and preparation,
and [scoring theory](theory.md) for the implemented objectives.
