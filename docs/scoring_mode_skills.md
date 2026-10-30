# Scoring-mode development skills

The repository includes two coding-agent skills for adding scoring modes:

- **`design-scoring-mode`** develops and tests a PyTorch reference objective
  and single-pair alignment method from a mathematical or plain-language description.
- **`accelerate-scoring-mode`** adds the mode and any new features to the
  registries, reuses or implements Numba/Triton kernels, and checks the batched
  and screening paths against the reference.

The skills and their supporting references live in
`.claude/skills/design-scoring-mode/` and
`.claude/skills/accelerate-scoring-mode/` at the repository root. Each folder
contains a `SKILL.md` entry point. Read it directly, or ask a coding agent that
supports skills to use it in this repository. To install a skill for another
agent, copy the whole folder, including its supporting references. The skills
are instructions for development; nothing in them runs when `shepherd_score` is
imported.

Both folders are included in source distributions. Wheels install them under
the installation's data directory at `share/shepherd-score/skills/`. Locate
that directory with the Python interpreter used for the installation:

```python
from pathlib import Path
import sysconfig

skills = Path(sysconfig.get_path("data")) / "share/shepherd-score/skills"
print(skills)
```

For editable installs, use the folders in the repository. Older PyPI releases
do not include the skills.

The reference workflow tests the objective and its gradients before any kernel
work. The acceleration workflow compares fixed-pose values and gradients,
optimizer settings, returned transforms, stored features and each execution
path against the reference, and accounts for differing seed sets, sub-batch
early stopping and frozen Triton launch configurations. The included test
template fails until its gates are filled in for the new objective.

Review generated code as you would any other contribution: a self-score of one
and agreement on a few optimized pairs do not show the objective is correct.
See the [API guide](accelerated_api.md) for existing modes and preparation,
and [scoring theory](theory.md) for the implemented objectives.
