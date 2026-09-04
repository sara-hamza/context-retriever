import { promises as fs } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { loadIndex } from "./store.js";
import { NeuralEmbedder } from "./neural.js";

/**
 * Run 007 — quantum measurement theory vs RAG, on the frozen retrieval eval.
 *
 * Question: does any qubit-style scoring rule beat plain best-chunk cosine
 * for ranking FILES? Four rankers, identical vectors, identical queries:
 *
 * - max:        score(file) = max_c cos(q, c)          — what RAG does today.
 * - incoherent: score(file) = Σ_c cos²(q, c)           — probabilities add.
 * - coherent:   score(file) = |Σ_c cos(q, c)|          — amplitudes add;
 *               opposite-sign evidence cancels (interference).
 * - subspace:   score(file) = ‖P_file q‖²              — the query's
 *               projection onto the span of the file's chunk vectors: the
 *               Born probability of finding state |q⟩ in the file's subspace
 *               (van Rijsbergen, The Geometry of Information Retrieval).
 *
 * REGISTERED PREDICTIONS (written before running):
 *  P10. subspace TIES max on hit@5/MRR — program history says structured
 *       mechanisms reduce to simple baselines; but this is the genuinely
 *       open one: projection rewards files where the query is explained by
 *       a COMBINATION of chunks, which max cannot see. A win here would be
 *       the program's first mechanism that beats a deployed baseline.
 *  P11. coherent LOSES to incoherent and max — chunk evidence for a
 *       relevant file shares sign, so cancellation only destroys signal.
 *       (Same verdict as negation-by-interference in runs 002-003.)
 *  Decision rule: a ranker replaces max in the plugin ONLY if it beats max
 *  on BOTH hit@5 and MRR on this frozen set. Ties keep max (parsimony).
 *
 * Usage: tsx src/run007.ts <indexed-root>
 */
const K = 5;
const root = process.argv[2];
if (!root) {
  console.log("usage: run007 <indexed-root>");
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
if (store.embedderKind !== "neural") throw new Error("Run 007 needs the neural index");

// Group normalized chunk vectors by file (excluding the eval file itself).
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

/** Solve G x = b for symmetric positive-definite G (ridge added by caller). */
function solve(G: number[][], b: number[]): number[] {
  const n = b.length;
  const A = G.map((row, i) => [...row, b[i] as number]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs((A[row] as number[])[col] as number) > Math.abs((A[pivot] as number[])[col] as number)) pivot = row;
    }
    [A[col], A[pivot]] = [A[pivot] as number[], A[col] as number[]];
    const diag = (A[col] as number[])[col] as number;
    for (let row = col + 1; row < n; row++) {
      const factor = ((A[row] as number[])[col] as number) / diag;
      for (let c = col; c <= n; c++) (A[row] as number[])[c] = ((A[row] as number[])[c] as number) - factor * ((A[col] as number[])[c] as number);
    }
  }
  const x = new Array<number>(n).fill(0);
  for (let row = n - 1; row >= 0; row--) {
    let sum = (A[row] as number[])[n] as number;
    for (let c = row + 1; c < n; c++) sum -= ((A[row] as number[])[c] as number) * (x[c] as number);
    x[row] = sum / ((A[row] as number[])[row] as number);
  }
  return x;
}

type Ranker = (q: number[], chunks: number[][]) => number;
const rankers: Record<string, Ranker> = {
  max: (q, chunks) => Math.max(...chunks.map((c) => dot(q, c))),
  incoherent: (q, chunks) => chunks.reduce((s, c) => s + dot(q, c) ** 2, 0),
  coherent: (q, chunks) => Math.abs(chunks.reduce((s, c) => s + dot(q, c), 0)),
  subspace: (q, chunks) => {
    // ‖P q‖² = (Vq)ᵀ (VVᵀ + εI)⁻¹ (Vq), rows of V = chunk vectors.
    const vq = chunks.map((c) => dot(q, c));
    const G = chunks.map((a, i) =>
      chunks.map((b, j) => dot(a, b) + (i === j ? 1e-6 : 0)),
    );
    const a = solve(G, vq);
    return vq.reduce((s, v, i) => s + v * (a[i] as number), 0);
  },
};

const embedder = new NeuralEmbedder();
console.log(`Run 007 — ${byFile.size} files, ${queries.length} frozen queries, K=${K}\n`);

const results: Record<string, { hits: number; mrr: number }> = {};
for (const name of Object.keys(rankers)) results[name] = { hits: 0, mrr: 0 };

for (const q of queries) {
  const [queryVector] = await embedder.embedBatch([q.query]);
  const query = Array.from(queryVector as Float32Array);
  for (const [name, ranker] of Object.entries(rankers)) {
    const ranked = [...byFile.entries()]
      .map(([file, chunks]) => ({ file, score: ranker(query, chunks) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, K);
    const rank = ranked.findIndex((r) => q.expect.some((e) => r.file.startsWith(e)));
    if (rank >= 0) {
      (results[name] as { hits: number; mrr: number }).hits += 1;
      (results[name] as { hits: number; mrr: number }).mrr += 1 / (rank + 1);
    }
  }
}

console.log("ranker      hit@5      MRR@5");
for (const [name, r] of Object.entries(results)) {
  console.log(
    `${name.padEnd(11)} ${String(r.hits).padStart(2)}/${queries.length} (${((100 * r.hits) / queries.length).toFixed(0)}%)  ${(r.mrr / queries.length).toFixed(3)}`,
  );
}
console.log(
  "\nDecision rule: replace max only if a ranker beats it on BOTH measures; ties keep max.",
);
