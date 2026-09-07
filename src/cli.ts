#!/usr/bin/env node
import { promises as fs, statSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { buildIndex, loadIndex, savingsFor, search } from "./store.js";
import { extractTranscripts, MEMORY_DIR } from "./memory.js";

/**
 * `crs` — search from a shell, with no MCP tool loading involved.
 *
 * Why this exists (v1.2.0): measurement showed the MCP tools sat unused for
 * days. On a machine with many MCP servers the tools are DEFERRED — the
 * agent must load their schemas before calling them — and in five large real
 * sessions the agent called ToolSearch 5-6 times each and never once loaded
 * these. Meanwhile it called Bash over a thousand times. So the reliable
 * path to retrieval is a binary, not a tool.
 *
 *   crs "<query>"                 search the current directory
 *   crs <path> "<query>" [k]      search a specific project
 *   crs index <path>              pre-index (optional; search self-indexes)
 *   crs memory "<query>"          search past Claude Code sessions
 *   crs stats                     savings + which projects are indexed
 *
 * Searches update the same lifetime savings counter as the MCP server.
 */
const STATS_FILE = path.join(os.homedir(), ".rooo-context-retriever", "stats.json");

interface Stats { searches: number; chunkTokens: number; fullFileTokens: number }

async function recordSavings(chunkTokens: number, fullFileTokens: number): Promise<void> {
  let stats: Stats = { searches: 0, chunkTokens: 0, fullFileTokens: 0 };
  try {
    stats = JSON.parse(await fs.readFile(STATS_FILE, "utf8")) as Stats;
  } catch {
    // first run — start from zero
  }
  stats.searches += 1;
  stats.chunkTokens += chunkTokens;
  stats.fullFileTokens += fullFileTokens;
  try {
    await fs.mkdir(path.dirname(STATS_FILE), { recursive: true });
    await fs.writeFile(STATS_FILE, JSON.stringify(stats));
  } catch {
    // accounting must never break a search
  }
}

async function runSearch(root: string, query: string, k: number): Promise<void> {
  const store = (await loadIndex(root)) ?? (await buildIndex(root));
  const hits = await search(store, query, k);
  for (const hit of hits) {
    console.log(`\n=== ${hit.file}:${hit.startLine}-${hit.endLine} (score ${hit.score.toFixed(3)}) ===`);
    console.log(hit.text.split("\n").slice(0, 12).join("\n"));
    if (hit.endLine - hit.startLine > 12) console.log("  …");
  }
  const { chunkTokens, fullFileTokens } = savingsFor(store, hits);
  await recordSavings(chunkTokens, fullFileTokens);
  console.log(
    `\nretrieval: ${chunkTokens} tokens vs ~${fullFileTokens} to inline the files — ` +
      `saved ~${fullFileTokens - chunkTokens} (${(fullFileTokens / Math.max(chunkTokens, 1)).toFixed(1)}×)`,
  );
}

/** The first argument is a project path only if it really is a directory;
 * otherwise it is the query and we search the working directory. */
function isDirectory(p: string): boolean {
  try { return statSync(p).isDirectory(); } catch { return false; }
}

const args = process.argv.slice(2);
const [first, second, third] = args;

if (first === "index" && second) {
  const store = await buildIndex(second);
  const total = Object.values(store.fileTokens).reduce((a, b) => a + b, 0);
  console.log(`Indexed ${store.root}: ${store.files} files, ${store.chunks.length} chunks, ~${total.toLocaleString()} tokens (${store.embedderKind})`);
} else if (first === "memory" && second) {
  await extractTranscripts();
  await runSearch(MEMORY_DIR, second, Number(third ?? 5));
} else if (first === "stats") {
  let stats: Stats = { searches: 0, chunkTokens: 0, fullFileTokens: 0 };
  try {
    stats = JSON.parse(await fs.readFile(STATS_FILE, "utf8")) as Stats;
  } catch {
    // none yet
  }
  const saved = stats.fullFileTokens - stats.chunkTokens;
  const ratio = stats.chunkTokens > 0 ? (stats.fullFileTokens / stats.chunkTokens).toFixed(1) : "—";
  console.log(`lifetime: ${stats.searches} searches | saved ~${saved.toLocaleString()} tokens (${ratio}×)`);
} else if (first === "search" && second && third) {
  await runSearch(second, third, Number(args[3] ?? 5)); // legacy form
} else if (first && second && isDirectory(first)) {
  await runSearch(first, second, Number(third ?? 5)); // crs <path> "<query>" [k]
} else if (first) {
  await runSearch(process.cwd(), first, Number(second ?? 5)); // crs "<query>"
} else {
  console.log('usage: crs "<query>" | crs <path> "<query>" [k] | crs index <path> | crs stats');
  process.exit(1);
}
