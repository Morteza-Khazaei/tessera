#!/usr/bin/env bash
# Setup script to install @openai/codex in a user-writable location on Béluga.
set -euo pipefail

# Choose a NodeJS module (defaults to StdEnv/2023 nodejs/20.16.0 if available)
if command -v module >/dev/null 2>&1; then
  module load StdEnv/2023 nodejs/20.16.0 2>/dev/null || true
fi

# Configure a user-local npm prefix to avoid read-only system paths
PREFIX="${HOME}/npm-global"
mkdir -p "${PREFIX}"
npm config set prefix "${PREFIX}"

# Export paths for this shell; add to ~/.bashrc if you want it permanent
export PATH="${PREFIX}/bin:${PATH}"
export NODE_PATH="${PREFIX}/lib/node_modules:${NODE_PATH:-}"

echo "Using npm prefix: ${PREFIX}"
node -v
npm -v

# Install the CLI
npm install -g @openai/codex

# Persist PATH/NODE_PATH for future shells (idempotent append)
RC_FILE="${HOME}/.bashrc"
if ! grep -q 'npm-global/bin' "${RC_FILE}" 2>/dev/null; then
  {
    echo ""
    echo "# Codex CLI (user npm prefix)"
    echo "export PATH=${PREFIX}/bin:\${PATH}"
    echo "export NODE_PATH=${PREFIX}/lib/node_modules:\${NODE_PATH}"
  } >> "${RC_FILE}"
  echo "Appended PATH/NODE_PATH to ${RC_FILE}"
else
  echo "PATH/NODE_PATH already present in ${RC_FILE}"
fi

echo "Done. Open a new shell or run:"
echo "  source ${RC_FILE}"
echo "Then invoke: codex --help"
