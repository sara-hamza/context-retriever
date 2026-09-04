import { extractTranscripts, MEMORY_DIR } from "./memory.js";
import { buildIndex } from "./store.js";

/** One-off: extract transcripts and build the memory index so the first
 * search_memory call in a session is warm. Usage: tsx src/warmup-memory.ts */
const report = await extractTranscripts();
console.log("extraction:", JSON.stringify(report));
const store = await buildIndex(MEMORY_DIR);
const tokens = Object.values(store.fileTokens).reduce((a, b) => a + b, 0);
console.log(`index: ${store.files} sessions, ${store.chunks.length} chunks, ~${tokens.toLocaleString()} tokens`);
