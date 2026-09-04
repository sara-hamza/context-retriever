/**
 * Hashed TF-IDF embedder: a real vector space (cosine similarity over ℝ^D)
 * with zero model downloads and deterministic output. For code and docs,
 * lexical vectors are a strong baseline — identifiers are exact-match
 * signals. The interface is the contract; a neural embedder (e.g.
 * transformers.js MiniLM) can replace this without touching store or server.
 */
export interface Embedder {
  readonly kind: "lexical" | "neural";
  readonly dimensions: number;
  /** Build corpus statistics if the embedder needs them. Once, at index time. */
  fit(documents: readonly string[]): void | Promise<void>;
  embedBatch(texts: readonly string[]): Promise<Float32Array[]>;
  /** Serializable state so queries at serve time match index-time weighting. */
  state(): EmbedderState | null;
}

export interface EmbedderState {
  dimensions: number;
  documentCount: number;
  /** token → document frequency, kept only for tokens seen ≥ 2 times. */
  documentFrequency: Record<string, number>;
}

const DEFAULT_DIMENSIONS = 512;

/** Lowercase, split camelCase and snake_case, drop 1-char noise. */
export function tokenize(text: string): string[] {
  return text
    .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((t) => t.length > 1);
}

/** FNV-1a — stable across runs, unlike anything seeded per-process. */
function fnv1a(token: string): number {
  let hash = 0x811c9dc5;
  for (let i = 0; i < token.length; i++) {
    hash ^= token.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
  }
  return hash >>> 0;
}

export class TfIdfHashEmbedder implements Embedder {
  readonly kind = "lexical" as const;
  readonly dimensions: number;
  private documentCount = 0;
  private documentFrequency = new Map<string, number>();

  constructor(dimensions = DEFAULT_DIMENSIONS, restored?: EmbedderState) {
    this.dimensions = restored?.dimensions ?? dimensions;
    if (restored) {
      this.documentCount = restored.documentCount;
      this.documentFrequency = new Map(Object.entries(restored.documentFrequency));
    }
  }

  fit(documents: readonly string[]): void {
    this.documentCount = documents.length;
    this.documentFrequency.clear();
    for (const doc of documents) {
      for (const token of new Set(tokenize(doc))) {
        this.documentFrequency.set(token, (this.documentFrequency.get(token) ?? 0) + 1);
      }
    }
    // Hapax tokens carry no cross-document signal and bloat the saved state.
    for (const [token, df] of this.documentFrequency) {
      if (df < 2) this.documentFrequency.delete(token);
    }
  }

  private idf(token: string): number {
    const df = this.documentFrequency.get(token) ?? 1;
    return Math.log(1 + this.documentCount / df);
  }

  embed(text: string): Float32Array {
    const vector = new Float32Array(this.dimensions);
    const counts = new Map<string, number>();
    for (const token of tokenize(text)) counts.set(token, (counts.get(token) ?? 0) + 1);
    for (const [token, count] of counts) {
      const hash = fnv1a(token);
      const index = hash % this.dimensions;
      // Signed hashing: the next bit decides the sign, so collisions tend to
      // cancel instead of stacking into phantom similarity.
      const sign = (hash >>> 30) & 1 ? 1 : -1;
      vector[index]! += sign * (1 + Math.log(count)) * this.idf(token);
    }
    let norm = 0;
    for (let i = 0; i < vector.length; i++) norm += vector[i]! * vector[i]!;
    norm = Math.sqrt(norm);
    if (norm > 0) for (let i = 0; i < vector.length; i++) vector[i]! /= norm;
    return vector;
  }

  async embedBatch(texts: readonly string[]): Promise<Float32Array[]> {
    return texts.map((t) => this.embed(t));
  }

  state(): EmbedderState {
    return {
      dimensions: this.dimensions,
      documentCount: this.documentCount,
      documentFrequency: Object.fromEntries(this.documentFrequency),
    };
  }
}

export function cosine(a: Float32Array, b: Float32Array): number {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i]! * b[i]!;
  return dot; // vectors are L2-normalized at embed time
}
