#!/usr/bin/env bash
# Boot the context-retriever MCP server from a bare plugin checkout.
# First run installs dependencies quietly next to the plugin; after that,
# startup is instant. All subsequent operation is fully local.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [ ! -d node_modules ]; then
  npm install --silent --no-audit --no-fund >&2
fi
exec npx tsx src/index.ts
