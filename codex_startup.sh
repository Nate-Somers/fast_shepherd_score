#!/usr/bin/env sh

###############################################################################
# 1) bootstrap micromamba (≈2 MB static binary, no root needed)               #
###############################################################################
# Install micromamba into ~/.local/bin if it isn't present.
if ! command -v micromamba >/dev/null 2>&1; then
  echo "[startup] installing micromamba → ${HOME}/.local/bin"
  curl -Ls https://micro.mamba.pm/install.sh | bash -s -- -b
fi

# Discover the actual binary path left by the installer.
MAMBA_BIN="$(command -v micromamba)"

# Bring the micromamba shell function into this POSIX shell.
eval "$("${MAMBA_BIN}" shell hook -s posix)"

###############################################################################
# 2) make sure Conda‑Forge is first in line (fewer licence blocks, faster)    #
###############################################################################
cat > "${HOME}/.condarc" <<'EOF'
channels:
  - conda-forge
  - defaults
channel_priority: strict
EOF

###############################################################################
# 3) create / update the env from environment.yml                             #
###############################################################################
ENV_FILE="environment.yml"   # change if your file has another name
ENV_NAME="codex-env"         # or read `name:` from the YAML

if [ -f "${ENV_FILE}" ]; then
  echo "[startup] creating/updating ${ENV_NAME} from ${ENV_FILE}"
  if ! micromamba env create -n "${ENV_NAME}" -f "${ENV_FILE}" -y; then
    micromamba env update  -n "${ENV_NAME}" -f "${ENV_FILE}" -y
  fi
else
  echo "[startup] WARNING: ${ENV_FILE} not found – making bare python env"
  micromamba create -n "${ENV_NAME}" python=3.10 -y
fi

###############################################################################
# 4) activate so the rest of the Codex run uses the environment               #
###############################################################################
micromamba activate "${ENV_NAME}"
echo "[startup] ✔ environment '${ENV_NAME}' is active"
