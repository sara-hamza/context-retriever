import { describe, expect, it } from "vitest";
import { chunkText, estimateTokens } from "../src/chunk.js";
import { cosine, TfIdfHashEmbedder, tokenize } from "../src/embed.js";
import { rerank, type SearchHit } from "../src/store.js";

describe("reranker", () => {
  const hit = (file: string, score: number): SearchHit => ({
    file, score, startLine: 1, endLine: 10, tokens: 50, text: "",
  });

  it("ranks implementation above a slightly stronger test-file match", () => {
    const ranked = rerank([
      hit("test/pack.test.ts", 0.30),
      hit("src/pack.ts", 0.25),
    ]);
    expect(ranked[0]?.file).toBe("src/pack.ts");
  });

  it("still surfaces tests when they are clearly the best match", () => {
    const ranked = rerank([
      hit("test/pack.test.ts", 0.60),
      hit("src/other.ts", 0.20),
    ]);
    expect(ranked[0]?.file).toBe("test/pack.test.ts");
  });

  it("penalizes e2e and spec paths too", () => {
    const ranked = rerank([
      hit("e2e/privacy.spec.ts", 0.30),
      hit("src/main/window.ts", 0.28),
    ]);
    expect(ranked[0]?.file).toBe("src/main/window.ts");
  });
});

describe("chunking", () => {
  it("covers every line with overlap and 1-indexed ranges", () => {
    const lines = Array.from({ length: 120 }, (_, i) => `line ${i + 1}`).join("\n");
    const chunks = chunkText("a.ts", lines);
    expect(chunks[0]?.startLine).toBe(1);
    expect(chunks.at(-1)?.endLine).toBe(120);
    // Consecutive chunks overlap so no boundary context is lost.
    for (let i = 1; i < chunks.length; i++) {
      expect(chunks[i]!.startLine).toBeLessThanOrEqual(chunks[i - 1]!.endLine);
    }
  });

  it("estimates tokens at roughly chars/4", () => {
    expect(estimateTokens("a".repeat(400))).toBe(100);
  });
});

describe("tokenizer", () => {
  it("splits camelCase and snake_case identifiers", () => {
    expect(tokenize("slotPhaseTables phase_pack")).toEqual([
      "slot", "phase", "tables", "phase", "pack",
    ]);
  });
});

describe("embedder", () => {
  const docs = [
    "rotary position embeddings rotate query and key pairs",
    "the bank held money and financial records",
    "phase superposition packs tokens into one chunk vector",
    "herons wade near the river bank at low tide",
  ];

  it("is deterministic and L2-normalized", () => {
    const embedder = new TfIdfHashEmbedder();
    embedder.fit(docs);
    const a = embedder.embed(docs[0]!);
    const b = embedder.embed(docs[0]!);
    expect(Array.from(a)).toEqual(Array.from(b));
    expect(cosine(a, a)).toBeCloseTo(1, 5);
  });

  it("ranks the topically matching document highest", () => {
    const embedder = new TfIdfHashEmbedder();
    embedder.fit(docs);
    const query = embedder.embed("how does phase superposition pack a chunk?");
    const scores = docs.map((d) => cosine(query, embedder.embed(d)));
    expect(scores.indexOf(Math.max(...scores))).toBe(2);
  });

  it("round-trips its state so serve-time queries match index-time weighting", () => {
    const embedder = new TfIdfHashEmbedder();
    embedder.fit(docs);
    const restored = new TfIdfHashEmbedder(512, embedder.state());
    expect(Array.from(restored.embed("river bank"))).toEqual(
      Array.from(embedder.embed("river bank")),
    );
  });
});
