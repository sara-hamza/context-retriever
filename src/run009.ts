import { promises as fs } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { loadIndex } from "./store.js";
import { NeuralEmbedder } from "./neural.js";

/**
 * Run 009 — network centrality (PageRank over the import graph) as a
 * retrieval prior, on the frozen 24-query file-retrieval eval.
 *
 * Theory: files the import graph points at are "central" — more likely to
 * be the implementation a "where is X?" question wants than leaf files.
 * Graph: importer → imported, built from import/require specifiers with
 * ESM-quirk resolution ("./stats.js" resolves to stats.ts). PageRank with
 * damping 0.85.
 *
 * Rankers (file-level, vs the incumbent):
 * - max:     best-chunk neural cosine (88% / 0.736 in runs 007-008).
 * - pr-0.1:  score = cos_max · pagerank_norm^0.1  (gentle prior)
 * - pr-0.25: score = cos_max · pagerank_norm^0.25 (strong prior)
 * - pure-pr: PageRank alone, query-blind — sanity floor, expected terrible.
 *
 * REGISTERED PREDICTIONS (written before running):
 *  P14. Neither pr blend beats max on BOTH measures. Predicted failure
 *       mode: high in-degree concentrates on barrel files (index.ts) and
 *       type/util modules — heavily imported but rarely the answer — so
 *       the prior boosts exactly the wrong neighbors of the right answer.
 *  P15. pure-pr scores near zero — centrality without the query is noise.
 *  Decision rule (unchanged): ship only on beating max on BOTH hit@5 and
 *  MRR; ties keep the incumbent.
 *
 * Usage: tsx src/run009.ts <indexed-root>
 */
const K = 5;
const DAMPING = 0.85;
const root = process.argv[2];
if (!root) {
  console.log("usage: run009 <indexed-root>");
  process.exit(1);
}

interface EvalQuery {
  query: string;
  expect: string[];
}

const here = path.dirname(fileURLToPath(import.meta.url));
const { queries } = JSON.parse(
  await fs.readFile(path.join(here, "..", "eval", "queries.json"), "utf8"),
) as { queries: EvalQuery[] };

const store = await loadIndex(root);
if (!store) throw new Error(`No index for ${root} — run index first`);

const files = Object.keys(store.fileTokens).filter((f) => !f.endsWith("eval/queries.json"));
const fileSet = new Set(files);

// --- import graph ---------------------------------------------------------
const IMPORT_RE = /(?:from\s+|require\()\s*["']([^"']+)["']/g;

function resolveSpecifier(fromFile: string, spec: string): string | null {
  let candidates: string[] = [];
  if (spec.startsWith(".")) {
    const base = path.join(path.dirname(fromFile), spec);
    candidates = [
      base,
      `${base}.ts`, `${base}.tsx`, `${base}.js`,
      base.replace(/\.js$/, ".ts"), base.replace(/\.js$/, ".tsx"),
      path.join(base, "index.ts"), path.join(base, "index.tsx"),
    ];
  } else if (spec.startsWith("@rooo/")) {
    const pkg = spec.split("/")[1] as string;
    candidates = [`packages/${pkg}/src/index.ts`];
  } else {
    return null; // external dependency — outside the graph
  }
  for (const c of candidates) if (fileSet.has(c)) return c;
  return null;
}

const outgoing = new Map<string, Set<string>>();
const inDegree = new Map<string, number>();
for (const file of files) {
  if (!/\.(ts|tsx|js|jsx|mjs)$/.test(file)) continue;
  let content: string;
  try {
    content = await fs.readFile(path.join(store.root, file), "utf8");
  } catch {
    continue;
  }
  const targets = new Set<string>();
  for (const match of content.matchAll(IMPORT_RE)) {
    const resolved = resolveSpecifier(file, match[1] as string);
    if (resolved && resolved !== file) targets.add(resolved);
  }
  outgoing.set(file, targets);
  for (const t of targets) inDegree.set(t, (inDegree.get(t) ?? 0) + 1);
}

// --- PageRank -------------------------------------------------------------
let rank = new Map<string, number>(files.map((f) => [f, 1 / files.length]));
for (let iteration = 0; iteration < 40; iteration++) {
  const next = new Map<string, number>(files.map((f) => [f, (1 - DAMPING) / files.length]));
  let dangling = 0;
  for (const file of files) {
    const targets = outgoing.get(file);
    const mass = rank.get(file) as number;
    if (!targets || targets.size === 0) {
      dangling += mass;
      continue;
    }
    for (const t of targets) next.set(t, (next.get(t) as number) + (DAMPING * mass) / targets.size);
  }
  for (const file of files) next.set(file, (next.get(file) as number) + (DAMPING * dangling) / files.length);
  rank = next;
}
const maxRank = Math.max(...rank.values());
const prNorm = new Map(files.map((f) => [f, (rank.get(f) as number) / maxRank]));

const edgeCount = [...outgoing.values()].reduce((s, t) => s + t.size, 0);
const topCentral = [...prNorm.entries()].sort((a, b) => b[1] - a[1]).slice(0, 5);
console.log(`Run 009 — import graph: ${outgoing.size} source files, ${edgeCount} edges`);
console.log("most central:", topCentral.map(([f, r]) => `${f} (${r.toFixed(2)})`).join(", "), "\n");

// --- eval -----------------------------------------------------------------
const byFile = new Map<string, number[][]>();
for (const chunk of store.chunks) {
  if (chunk.file.endsWith("eval/queries.json")) continue;
  const list = byFile.get(chunk.file) ?? [];
  list.push(chunk.vector);
  byFile.set(chunk.file, list);
}

function dot(a: readonly number[], b: readonly number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] as number) * (b[i] as number);
  return s;
}

const embedder = new NeuralEmbedder();
const rankerNames = ["max", "pr-0.1", "pr-0.25", "pure-pr"] as const;
const results = Object.fromEntries(rankerNames.map((n) => [n, { hits: 0, mrr: 0 }])) as Record<
  string,
  { hits: number; mrr: number }
>;

for (const q of queries) {
  const [queryVector] = await embedder.embedBatch([q.query]);
  const query = Array.from(queryVector as Float32Array);
  for (const name of rankerNames) {
    const ranked = [...byFile.entries()]
      .map(([file, vectors]) => {
        const cos = Math.max(...vectors.map((v) => dot(query, v)));
        const pr = Math.max(prNorm.get(file) ?? 0, 1e-6);
        let score: number;
        if (name === "max") score = cos;
        else if (name === "pr-0.1") score = cos * Math.pow(pr, 0.1);
        else if (name === "pr-0.25") score = cos * Math.pow(pr, 0.25);
        else score = pr;
        return { file, score };
      })
      .sort((a, b) => b.score - a.score)
      .slice(0, K);
    const at = ranked.findIndex((r) => q.expect.some((e) => r.file.startsWith(e)));
    if (at >= 0) {
      (results[name] as { hits: number; mrr: number }).hits += 1;
      (results[name] as { hits: number; mrr: number }).mrr += 1 / (at + 1);
    }
  }
}

console.log("ranker      hit@5      MRR@5");
for (const name of rankerNames) {
  const r = results[name] as { hits: number; mrr: number };
  console.log(
    `${name.padEnd(11)} ${String(r.hits).padStart(2)}/${queries.length} (${((100 * r.hits) / queries.length).toFixed(0)}%)  ${(r.mrr / queries.length).toFixed(3)}`,
  );
}
console.log("\nDecision rule: a ranker ships only if it beats max on BOTH measures; ties keep max.");
