import { promises as fs } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { loadIndex } from "./store.js";
import { MEMORY_DIR } from "./memory.js";
import { NeuralEmbedder } from "./neural.js";

/**
 * Run 010 — ACT-R base-level activation (cognitive science) for transcript
 * memory retrieval, on the frozen 11-query memory benchmark.
 *
 * Theory: in ACT-R, a memory's retrieval activation is
 *   A = B + similarity,  with base level B = −d · ln(age)   (decay d ≈ 0.5)
 * i.e. recency enters ADDITIVELY in log time. Human recall really does
 * follow this power law — the question is whether session retrieval should.
 *
 * Rankers (session-file level):
 * - max:        best-chunk neural cosine — the incumbent.
 * - actr-0.02:  cos − 0.02 · ln(age_days + 1)   (gentle decay)
 * - actr-0.05:  cos − 0.05 · ln(age_days + 1)   (ACT-R-ish strength; at 10
 *               days old this subtracts ~0.12 — same order as cosine gaps)
 * - pure-age:   newest first, query-blind — sanity floor.
 *
 * REGISTERED PREDICTIONS (written before running):
 *  P16. Neither ACT-R blend beats max on BOTH measures. The benchmark's
 *       queries are topically distinct, and the embedding already separates
 *       sessions by topic; when the target is an OLD session, recency decay
 *       actively pushes the right answer down. ACT-R models interference
 *       between similar memories — it should only pay when many sessions
 *       discuss the SAME topic, which this corpus mostly does not.
 *  P17. pure-age scores ~0 — recency without content is noise.
 *  Decision rule (unchanged): ship only on beating max on BOTH hit@5 and
 *  MRR; ties keep the incumbent.
 *
 * Usage: tsx src/run010.ts
 */
const K = 5;

interface EvalQuery {
  query: string;
  expect: string[];
}

const here = path.dirname(fileURLToPath(import.meta.url));
const { queries } = JSON.parse(
  await fs.readFile(path.join(here, "..", "eval", "memory-queries.json"), "utf8"),
) as { queries: EvalQuery[] };

const store = await loadIndex(MEMORY_DIR);
if (!store) throw new Error("No memory index — run warmup-memory first");

const byFile = new Map<string, number[][]>();
for (const chunk of store.chunks) {
  const list = byFile.get(chunk.file) ?? [];
  list.push(chunk.vector);
  byFile.set(chunk.file, list);
}

const today = Date.now();
function ageDays(file: string): number {
  const date = path.basename(file).slice(0, 10);
  const ms = today - new Date(`${date}T00:00:00Z`).getTime();
  return Math.max(0, ms / 86_400_000);
}

function dot(a: readonly number[], b: readonly number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] as number) * (b[i] as number);
  return s;
}

const ages = [...byFile.keys()].map(ageDays);
console.log(
  `Run 010 — ${byFile.size} sessions, age ${Math.min(...ages).toFixed(0)}-${Math.max(...ages).toFixed(0)} days, ${queries.length} frozen queries, K=${K}\n`,
);

const rankerNames = ["max", "actr-0.02", "actr-0.05", "pure-age"] as const;
const results = Object.fromEntries(rankerNames.map((n) => [n, { hits: 0, mrr: 0 }])) as Record<
  string,
  { hits: number; mrr: number }
>;

const embedder = new NeuralEmbedder();
for (const q of queries) {
  const [queryVector] = await embedder.embedBatch([q.query]);
  const query = Array.from(queryVector as Float32Array);
  for (const name of rankerNames) {
    const ranked = [...byFile.entries()]
      .map(([file, vectors]) => {
        const cos = Math.max(...vectors.map((v) => dot(query, v)));
        const decay = Math.log(ageDays(file) + 1);
        let score: number;
        if (name === "max") score = cos;
        else if (name === "actr-0.02") score = cos - 0.02 * decay;
        else if (name === "actr-0.05") score = cos - 0.05 * decay;
        else score = -decay;
        return { file, score };
      })
      .sort((a, b) => b.score - a.score)
      .slice(0, K);
    const at = ranked.findIndex((r) =>
      q.expect.some((e) => path.basename(r.file).startsWith(e)),
    );
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
