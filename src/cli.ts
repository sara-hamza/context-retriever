import { buildIndex, loadIndex, savingsFor, search } from "./store.js";

/**
 * Direct CLI for testing without an MCP client:
 *   pnpm --filter @rooo/context-retriever cli index <path>
 *   pnpm --filter @rooo/context-retriever cli search <path> "<query>" [k]
 */
const [command, root, query, kArg] = process.argv.slice(2);

if (command === "index" && root) {
  const store = await buildIndex(root);
  const total = Object.values(store.fileTokens).reduce((a, b) => a + b, 0);
  console.log(`Indexed ${store.root}`);
  console.log(`files=${store.files} chunks=${store.chunks.length} sourceTokens≈${total.toLocaleString()}`);
} else if (command === "search" && root && query) {
  const store = (await loadIndex(root)) ?? (await buildIndex(root));
  console.log(`[${store.embedderKind} embeddings]`);
  const hits = await search(store, query, Number(kArg ?? 5));
  for (const hit of hits) {
    console.log(`\n=== ${hit.file}:${hit.startLine}-${hit.endLine} (score ${hit.score.toFixed(3)}, ~${hit.tokens} tokens) ===`);
    console.log(hit.text.split("\n").slice(0, 8).join("\n"));
    if (hit.endLine - hit.startLine > 8) console.log("  …");
  }
  const { chunkTokens, fullFileTokens } = savingsFor(store, hits);
  console.log(
    `\nretrieval: ${chunkTokens} tokens vs ~${fullFileTokens} to inline the files — saved ~${fullFileTokens - chunkTokens} ` +
      `(${(fullFileTokens / Math.max(chunkTokens, 1)).toFixed(1)}× reduction)`,
  );
} else {
  console.log("usage: cli index <path> | cli search <path> \"<query>\" [k]");
  process.exit(1);
}
