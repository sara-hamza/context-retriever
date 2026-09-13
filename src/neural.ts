import type { Embedder, EmbedderState } from "./embed.js";
import { TfIdfHashEmbedder } from "./embed.js";

/**
 * Neural embedder: all-MiniLM-L6-v2 (384-dim sentence embeddings) via
 * transformers.js. Runs fully locally; the quantized model (~25MB) downloads
 * once into the transformers cache, then works offline. This is what closes
 * the paraphrase gap the lexical embedder can't: "statistics tests" should
 * reach code that says "Welch's t".
 */
const MODEL_ID = "Xenova/all-MiniLM-L6-v2";
const BATCH_SIZE = 32;

type Extractor = (
  texts: string[],
  options: { pooling: "mean"; normalize: boolean },
) => Promise<{ data: Float32Array; dims: number[] }>;

/** One model session per process — instantiating it per search would reload
 * the ONNX session every query. */
let sharedExtractor: Promise<Extractor> | null = null;

export class NeuralEmbedder implements Embedder {
  readonly kind = "neural" as const;
  readonly dimensions = 384;

  private load(): Promise<Extractor> {
    sharedExtractor ??= import("@huggingface/transformers").then(
      async ({ pipeline }) =>
        (await pipeline("feature-extraction", MODEL_ID)) as unknown as Extractor,
    );
    return sharedExtractor;
  }

  fit(): void {
    // Pretrained — no corpus statistics needed.
  }

  async embedBatch(texts: readonly string[]): Promise<Float32Array[]> {
    const extractor = await this.load();
    const vectors: Float32Array[] = [];
    for (let start = 0; start < texts.length; start += BATCH_SIZE) {
      const batch = texts.slice(start, start + BATCH_SIZE) as string[];
      const output = await extractor(batch, { pooling: "mean", normalize: true });
      const [, width] = output.dims as [number, number];
      for (let row = 0; row < batch.length; row++) {
        vectors.push(Float32Array.from(output.data.slice(row * width, (row + 1) * width)));
      }
    }
    return vectors;
  }

  state(): EmbedderState | null {
    return null; // nothing corpus-specific to persist
  }
}

/**
 * Prefer neural; fall back to lexical when the model can't load (no network
 * on first run, unsupported platform). The store records which one built the
 * index so queries always use the matching space.
 */
export async function createEmbedder(
  preferred: "neural" | "lexical" = "neural",
  restored?: EmbedderState | null,
): Promise<Embedder> {
  if (preferred === "neural" && process.env["CONTEXT_RETRIEVER_EMBEDDER"] !== "lexical") {
    try {
      const embedder = new NeuralEmbedder();
      await embedder.embedBatch(["warm-up"]);
      return embedder;
    } catch (error) {
      console.error(
        "context-retriever: neural model unavailable, using the lexical embedder " +
          "(install for better search: npm i -g @huggingface/transformers). " +
          `reason: ${String(error).slice(0, 120)}`,
      );
    }
  }
  return new TfIdfHashEmbedder(restored?.dimensions ?? 512, restored ?? undefined);
}
