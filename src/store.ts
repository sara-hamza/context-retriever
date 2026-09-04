import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";
import crypto from "node:crypto";
import { chunkText, estimateTokens, readCorpus, type Chunk } from "./chunk.js";
import { cosine, TfIdfHashEmbedder, type EmbedderState } from "./embed.js";
import { createEmbedder, NeuralEmbedder } from "./neural.js";

export interface IndexedChunk extends Chunk {
  tokens: number;
  /** Semantic vector (neural when available, lexical in fallback mode). */
  vector: number[];
  /** Lexical vector for hybrid ranking (null in lexical fallback mode —
   * `vector` already is the lexical one). */
  lexVector: number[] | null;
}

/** Bumped whenever the index layout or embedding space changes. */
const INDEX_VERSION = 3;

export interface StoreData {
  version: number;
  root: string;
  builtAt: string;
  files: number;
  embedderKind: "lexical" | "neural";
  chunks: IndexedChunk[];
  /** Total estimated tokens per file — the "what you'd pay to inline it" side. */
  fileTokens: Record<string, number>;
  /** sha1 per file, so re-indexing only touches what changed. */
  fileHashes: Record<string, string>;
  lexicalState: EmbedderState | null;
}

export interface SearchHit {
  file: string;
  startLine: number;
  endLine: number;
  score: number;
  tokens: number;
  text: string;
}

const CACHE_DIR = path.join(os.homedir(), ".rooo-context-retriever");

function cachePathFor(root: string): string {
  const digest = crypto.createHash("sha256").update(path.resolve(root)).digest("hex").slice(0, 16);
  return path.join(CACHE_DIR, `${digest}.json`);
}

function sha1(text: string): string {
  return crypto.createHash("sha1").update(text).digest("hex");
}

export interface BuildOptions {
  /** Re-embed everything even if a previous index exists. */
  full?: boolean;
}

export async function buildIndex(root: string, options: BuildOptions = {}): Promise<StoreData> {
  const resolved = path.resolve(root);
  const corpus = await readCorpus(resolved);
  if (corpus.length === 0) throw new Error(`No indexable text files under ${resolved}`);

  const previous = options.full ? null : await loadIndex(resolved);
  const previousChunks = new Map<string, IndexedChunk[]>();
  if (previous) {
    for (const chunk of previous.chunks) {
      const list = previousChunks.get(chunk.file) ?? [];
      list.push(chunk);
      previousChunks.set(chunk.file, list);
    }
  }

  const fileHashes: Record<string, string> = {};
  const unchanged: IndexedChunk[] = [];
  const dirty: { relative: string; content: string }[] = [];
  for (const file of corpus) {
    const hash = sha1(file.content);
    fileHashes[file.relative] = hash;
    if (previous && previous.fileHashes[file.relative] === hash) {
      unchanged.push(...(previousChunks.get(file.relative) ?? []));
    } else {
      dirty.push(file);
    }
  }

  // Fast path: identical corpus (no dirty files, no additions or deletions)
  // → nothing to embed and nothing to write. This is what makes a
  // staleness check before every search affordable.
  if (previous && dirty.length === 0 && corpus.length === Object.keys(previous.fileHashes).length) {
    return previous;
  }

  // Lexical idf: refit on a full build; on incremental builds reuse the
  // frozen statistics so unchanged files' vectors stay valid. Slightly stale
  // idf on a drifted corpus is the documented price of fast re-indexing —
  // a full rebuild refreshes it.
  const dirtyChunks: Chunk[] = dirty.flatMap((f) => chunkText(f.relative, f.content));
  const lexical = previous?.lexicalState
    ? new TfIdfHashEmbedder(previous.lexicalState.dimensions, previous.lexicalState)
    : new TfIdfHashEmbedder();
  if (!previous?.lexicalState) {
    lexical.fit([...unchanged.map((c) => c.text), ...dirtyChunks.map((c) => c.text)]);
  }

  const semantic = await createEmbedder();
  await semantic.fit(dirtyChunks.map((c) => c.text));
  const semanticVectors = await semantic.embedBatch(dirtyChunks.map((c) => c.text));

  const rebuilt: IndexedChunk[] = dirtyChunks.map((chunk, i) => ({
    ...chunk,
    tokens: estimateTokens(chunk.text),
    vector: Array.from(semanticVectors[i] as Float32Array),
    lexVector: semantic.kind === "neural" ? Array.from(lexical.embed(chunk.text)) : null,
  }));

  const data: StoreData = {
    version: INDEX_VERSION,
    root: resolved,
    builtAt: new Date().toISOString(),
    files: corpus.length,
    embedderKind: semantic.kind,
    chunks: [...unchanged, ...rebuilt],
    fileTokens: Object.fromEntries(corpus.map((f) => [f.relative, estimateTokens(f.content)])),
    fileHashes,
    lexicalState: lexical.state(),
  };
  await fs.mkdir(CACHE_DIR, { recursive: true });
  await fs.writeFile(cachePathFor(resolved), JSON.stringify(data));
  return data;
}

export async function loadIndex(root: string): Promise<StoreData | null> {
  try {
    const raw = await fs.readFile(cachePathFor(path.resolve(root)), "utf8");
    const data = JSON.parse(raw) as StoreData;
    return data.version === INDEX_VERSION ? data : null; // stale layout → rebuild
  } catch {
    return null;
  }
}

/**
 * Tests, specs, and fixtures mention every identifier they exercise, so they
 * out-match the implementation. A mild penalty ranks implementation first
 * while leaving tests reachable when they are the only match.
 */
const TEST_PATH = /(^|\/)(tests?|e2e|__tests__|fixtures?|spec)(\/|$)|\.(test|spec)\.[a-z]+$/i;
const TEST_PENALTY = 0.75;

export function rerank(hits: readonly SearchHit[]): SearchHit[] {
  return hits
    .map((h) => (TEST_PATH.test(h.file) ? { ...h, score: h.score * TEST_PENALTY } : h))
    .sort((a, b) => b.score - a.score);
}

/** Positions in a descending-score ranking (0 = best). */
function ranksOf(scores: readonly number[]): number[] {
  const order = scores.map((score, i) => ({ score, i })).sort((a, b) => b.score - a.score);
  const ranks = new Array<number>(scores.length);
  order.forEach((entry, rank) => (ranks[entry.i] = rank));
  return ranks;
}

const RRF_K = 60;

/**
 * Ranking. Neural cosine is the default: on the frozen 24-query benchmark it
 * scored hit@5 21/24, MRR 0.747, and the hybrid RRF blend (adding lexical
 * rank) measured slightly WORSE (MRR 0.729) — it did not earn default
 * status. Hybrid stays available via CONTEXT_RETRIEVER_RANK=hybrid for
 * identifier-heavy corpora; re-judge it on the eval before promoting it.
 */
export async function search(store: StoreData, query: string, k: number): Promise<SearchHit[]> {
  const useHybrid =
    store.embedderKind === "neural" &&
    store.lexicalState !== null &&
    process.env["CONTEXT_RETRIEVER_RANK"] === "hybrid";

  const semantic =
    store.embedderKind === "neural"
      ? new NeuralEmbedder()
      : new TfIdfHashEmbedder(store.lexicalState?.dimensions ?? 512, store.lexicalState ?? undefined);
  const [semanticQuery] = await semantic.embedBatch([query]);
  const semanticScores = store.chunks.map((chunk) =>
    cosine(semanticQuery as Float32Array, Float32Array.from(chunk.vector)),
  );

  let fused: number[];
  if (useHybrid) {
    const lexical = new TfIdfHashEmbedder(
      (store.lexicalState as EmbedderState).dimensions,
      store.lexicalState as EmbedderState,
    );
    const lexicalQuery = lexical.embed(query);
    const lexicalScores = store.chunks.map((chunk) =>
      chunk.lexVector ? cosine(lexicalQuery, Float32Array.from(chunk.lexVector)) : -1,
    );
    const semanticRanks = ranksOf(semanticScores);
    const lexicalRanks = ranksOf(lexicalScores);
    fused = store.chunks.map(
      (_, i) => 1 / (RRF_K + (semanticRanks[i] as number)) + 1 / (RRF_K + (lexicalRanks[i] as number)),
    );
  } else {
    fused = semanticScores;
  }

  const scored: SearchHit[] = store.chunks.map((chunk, i) => ({
    file: chunk.file,
    startLine: chunk.startLine,
    endLine: chunk.endLine,
    tokens: chunk.tokens,
    text: chunk.text,
    score: fused[i] as number,
  }));
  return rerank(scored).slice(0, k);
}

/**
 * The honest savings measure: tokens Claude would have spent reading every
 * file the hits came from, versus the tokens in the returned chunks alone.
 */
export function savingsFor(store: StoreData, hits: readonly SearchHit[]): {
  chunkTokens: number;
  fullFileTokens: number;
} {
  const chunkTokens = hits.reduce((sum, h) => sum + h.tokens, 0);
  const files = new Set(hits.map((h) => h.file));
  let fullFileTokens = 0;
  for (const file of files) fullFileTokens += store.fileTokens[file] ?? 0;
  return { chunkTokens, fullFileTokens };
}
