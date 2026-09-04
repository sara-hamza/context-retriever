import { promises as fs } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { loadIndex } from "./store.js";
import { NeuralEmbedder } from "./neural.js";
import { tokenize } from "./embed.js";

/**
 * Run 008 — non-quantum theory candidates vs best-chunk cosine, on the
 * frozen 24-query file-retrieval eval. Same corpus, same queries as run 007.
 *
 * - max:        best-chunk neural cosine — the incumbent (88% in run 007).
 * - bm25:       Okapi BM25 (probabilistic IR theory), file = best chunk.
 *               The theory behind every classical search engine.
 * - lse-T:      statistical mechanics: file = T·ln Σ_c exp(cos_c / T).
 *               Temperature interpolates max (T→0) ↔ sum (T→∞); tests
 *               whether any point BETWEEN the run-007 endpoints wins.
 *               Two registered temperatures: T=0.05 and T=0.2.
 *
 * REGISTERED PREDICTIONS (written before running):
 *  P12. bm25 loses to max on this benchmark — a third of the queries are
 *       paraphrases with no keyword overlap, which lexical theory cannot
 *       bridge (the same reason the TF-IDF embedder lost in v2 testing).
 *       It should still beat run 007's quantum rankers comfortably.
 *  P13. lse-0.05 TIES max (it is nearly max); lse-0.2 sits between max and
 *       the incoherent sum. No temperature significantly beats max — the
 *       best single chunk is the signal, extra chunks of one file are
 *       mostly redundant, not corroborating.
 *  Decision rule (same as 007): a ranker ships only if it beats max on
 *  BOTH hit@5 and MRR. Ties keep the incumbent.
 *
 * Usage: tsx src/run008.ts <indexed-root>
 */
const K = 5;
const root = process.argv[2];
if (!root) {
  console.log("usage: run008 <indexed-root>");
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

interface ChunkView {
  file: string;
  vector: number[];
  terms: Map<string, number>;
  length: number;
}

const chunks: ChunkView[] = [];
const documentFrequency = new Map<string, number>();
for (const chunk of store.chunks) {
  if (chunk.file.endsWith("eval/queries.json")) continue;
  const terms = new Map<string, number>();
  for (const t of tokenize(chunk.text)) terms.set(t, (terms.get(t) ?? 0) + 1);
  for (const t of new Set(terms.keys())) documentFrequency.set(t, (documentFrequency.get(t) ?? 0) + 1);
  chunks.push({ file: chunk.file, vector: chunk.vector, terms, length: [...terms.values()].reduce((a, b) => a + b, 0) });
}
const averageLength = chunks.reduce((s, c) => s + c.length, 0) / chunks.length;
const N = chunks.length;

function dot(a: readonly number[], b: readonly number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] as number) * (b[i] as number);
  return s;
}

const K1 = 1.2;
const B = 0.75;
function bm25(queryTerms: string[], chunk: ChunkView): number {
  let score = 0;
  for (const term of queryTerms) {
    const tf = chunk.terms.get(term) ?? 0;
    if (tf === 0) continue;
    const df = documentFrequency.get(term) ?? 0;
    const idf = Math.log(1 + (N - df + 0.5) / (df + 0.5));
    score += (idf * tf * (K1 + 1)) / (tf + K1 * (1 - B + (B * chunk.length) / averageLength));
  }
  return score;
}

function logSumExp(scores: number[], T: number): number {
  const m = Math.max(...scores);
  return m + T * Math.log(scores.reduce((s, v) => s + Math.exp((v - m) / T), 0));
}

const embedder = new NeuralEmbedder();
const byFile = new Map<string, ChunkView[]>();
for (const c of chunks) {
  const list = byFile.get(c.file) ?? [];
  list.push(c);
  byFile.set(c.file, list);
}
console.log(`Run 008 — ${byFile.size} files, ${queries.length} frozen queries, K=${K}\n`);

const rankerNames = ["max", "bm25", "lse-0.05", "lse-0.2"] as const;
const results = Object.fromEntries(rankerNames.map((n) => [n, { hits: 0, mrr: 0 }])) as Record<
  string,
  { hits: number; mrr: number }
>;

for (const q of queries) {
  const [queryVector] = await embedder.embedBatch([q.query]);
  const query = Array.from(queryVector as Float32Array);
  const queryTerms = tokenize(q.query);
  for (const name of rankerNames) {
    const ranked = [...byFile.entries()]
      .map(([file, list]) => {
        const cosines = list.map((c) => dot(query, c.vector));
        let score: number;
        if (name === "max") score = Math.max(...cosines);
        else if (name === "bm25") score = Math.max(...list.map((c) => bm25(queryTerms, c)));
        else score = logSumExp(cosines, name === "lse-0.05" ? 0.05 : 0.2);
        return { file, score };
      })
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
for (const name of rankerNames) {
  const r = results[name] as { hits: number; mrr: number };
  console.log(
    `${name.padEnd(11)} ${String(r.hits).padStart(2)}/${queries.length} (${((100 * r.hits) / queries.length).toFixed(0)}%)  ${(r.mrr / queries.length).toFixed(3)}`,
  );
}
console.log("\nDecision rule: a ranker ships only if it beats max on BOTH measures; ties keep max.");
