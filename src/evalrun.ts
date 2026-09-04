import { promises as fs } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { buildIndex, loadIndex, search } from "./store.js";

/**
 * Retrieval eval: hit@5 (an expected file appears in the top 5) and MRR@5
 * (reciprocal rank of the first expected file). The query set is frozen in
 * eval/queries.json — the same discipline as the QLLM runs: measure every
 * ranking change against the same registered benchmark, never against vibes.
 *
 * Usage: tsx src/evalrun.ts <root> [--rebuild] [--label name]
 */
interface EvalQuery {
  query: string;
  expect: string[];
}

const K = 5;
const args = process.argv.slice(2);
const root = args.find((a) => !a.startsWith("--"));
if (!root) {
  console.log("usage: evalrun <root> [--rebuild] [--label name]");
  process.exit(1);
}
const label = args.includes("--label") ? args[args.indexOf("--label") + 1] : "current";

const here = path.dirname(fileURLToPath(import.meta.url));
const raw = await fs.readFile(path.join(here, "..", "eval", "queries.json"), "utf8");
const { queries } = JSON.parse(raw) as { queries: EvalQuery[] };

const store = args.includes("--rebuild")
  ? await buildIndex(root)
  : ((await loadIndex(root)) ?? (await buildIndex(root)));

let hits = 0;
let mrrSum = 0;
const misses: string[] = [];
for (const q of queries) {
  // The query file itself is in the index and matches its own queries
  // verbatim — exclude the measurement instrument from the measurement.
  const results = (await search(store, q.query, K + 2)).filter(
    (hit) => !hit.file.endsWith("eval/queries.json"),
  ).slice(0, K);
  const rank = results.findIndex((hit) => q.expect.some((e) => hit.file.startsWith(e)));
  if (rank >= 0) {
    hits += 1;
    mrrSum += 1 / (rank + 1);
    console.log(`  hit@${rank + 1}  ${q.query}`);
  } else {
    misses.push(q.query);
    console.log(`  MISS   ${q.query}  (top: ${results[0]?.file})`);
  }
}

console.log(`\n== ${label} | embedder=${store.embedderKind} | ${store.chunks.length} chunks ==`);
console.log(`hit@${K}: ${hits}/${queries.length} (${((100 * hits) / queries.length).toFixed(0)}%)`);
console.log(`MRR@${K}: ${(mrrSum / queries.length).toFixed(3)}`);
if (misses.length > 0) console.log(`misses: ${misses.length}`);
