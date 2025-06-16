#!/usr/bin/env bash
set -euo pipefail           # fail fast if anything goes wrong

###############################################################################
# 1) bootstrap micromamba (≈2 MB static binary, no root needed)               #
###############################################################################
MAMBA_ROOT="${HOME}/.mamba"            # everything lives here, keeps $HOME tidy
MAMBA_BIN="${MAMBA_ROOT}/bin/micromamba"

if [[ ! -x "${MAMBA_BIN}" ]]; then
  echo "[startup] installing micromamba → ${MAMBA_ROOT}"
  curl -Ls https://micro.mamba.pm/install.sh | bash -s -- -b -p "${MAMBA_ROOT}"  # :contentReference[oaicite:0]{index=0}
fi

# put micromamba *function* into this shell
eval "$("${MAMBA_BIN}" shell hook -s bash -p "${MAMBA_ROOT}")"

###############################################################################
# 2) make sure Conda-Forge is first in line (fewer licence blocks, faster)    #
###############################################################################
cat <<'EOF' > "${HOME}/.condarc"
channels:
  - conda-forge
  - defaults
channel_priority: strict
# (feel free to add bioconda, pytorch, etc. if your YAML needs them)
EOF

###############################################################################
# 3) create / update the env from environment.yml                             #
###############################################################################
ENV_FILE="environment.yml"             # change if your file has another name
ENV_NAME="codex-env"                   # or read `name:` from the YAML

if [[ -f "${ENV_FILE}" ]]; then
  echo "[startup] creating/updating ${ENV_NAME} from ${ENV_FILE}"
  # first try “create”; if it already exists fall back to “update”
  if ! micromamba env create -n "${ENV_NAME}" -f "${ENV_FILE}" --yes; then
      micromamba env update  -n "${ENV_NAME}" -f "${ENV_FILE}" --yes
  fi
else
  echo "[startup] WARNING: ${ENV_FILE} not found – making bare python env"
  micromamba create -n "${ENV_NAME}" python=3.10 --yes
fi

###############################################################################
# 4) activate so the rest of the Codex run uses the environment               #
###############################################################################
micromamba activate "${ENV_NAME}"
echo "[startup] ✔ environment '${ENV_NAME}' is active"
