# context-retriever-mcp

**Local vector retrieval for coding agents — semantic code search, session memory,
and a live token-savings receipt. 100% on your machine.**

An MCP server that gives Claude Code (or any MCP client) a `search_context` tool
backed by a local vector index, so the agent pulls the few relevant chunks of a
codebase instead of reading whole files — plus a `search_memory` tool over your
past session transcripts, so it can answer "what did we decide about X?" without
you re-explaining. Every search updates a persistent counter of tokens saved.

## Why this one, when semantic-search MCP servers already exist?

Because every design decision in this tool survived a measurement, and several
popular ideas did not:

| Claim | Evidence |
|---|---|
| Finds the right file | 88% hit@5 on a frozen 24-query code benchmark (MRR 0.747) |
| Finds the right past session | 82% hit@5 on a frozen 11-query memory benchmark |
| Saves tokens on code questions | measured 3.4× (chunks vs inlining the source files) |
| Saves tokens on memory questions | measured 19–36× (transcripts are huge) |
| Ranking rule | best-chunk cosine — it beat **8 alternatives from 6 theories** (BM25, hybrid RRF, quantum-style evidence rules, Born-rule subspace projection, PageRank import-graph priors, ACT-R recency decay) on the same frozen benchmarks with predictions registered before each run |
| Freshness | index auto-refreshes before every search; ~2s after a one-file edit, near-zero when nothing changed |

Notably: **hybrid RRF fusion and centrality priors — shipped as headline features
elsewhere — measured *worse* than plain cosine here and were demoted to opt-in.**
The benchmarks and every experiment script are in `eval/` and `src/run0*.ts`;
run them yourself. (Benchmark queries are corpus-specific by nature — treat ours
as a worked example of the method and freeze your own set for your repo.)

## Install

```bash
git clone https://github.com/sara-hamza/context-retriever.git
cd context-retriever
npm install
```

Register with Claude Code (available in every project):

```bash
claude mcp add context-retriever -s user -- npx tsx /path/to/context-retriever/src/index.ts
```

Optional but recommended — add to your `~/.claude/CLAUDE.md` so sessions use it
by default:

```markdown
When the context-retriever MCP tools are available and you need to understand
code — "where is X?", "how does Y work?" — call search_context FIRST, before
reading whole files. For "what did we decide?" questions, call search_memory.
Read a full file only when you are about to edit it or the chunks are not
enough. retrieval_stats shows what this saves.
```

## Tools

- **`index_path { path, full? }`** — index a directory. Incremental: only files
  whose content hash changed are re-embedded. First index of a mid-size repo
  takes about a minute; after that ~2s.
- **`search_context { path, query, k? }`** — semantic search over an indexed
  directory; returns top-k chunks with `file:line` references and a per-call
  token comparison vs inlining the files.
- **`search_memory { query, k?, days? }`** — semantic search over your past
  Claude Code session transcripts (extracted from `~/.claude/projects/`,
  message text only, tool noise skipped). `days` limits how far back to look.
- **`retrieval_stats {}`** — tokens saved: this session and lifetime.

## How it works

1. Files are chunked at declaration/heading boundaries (functions stay whole;
   measured +5 points hit@5 over fixed windows), embedded with
   all-MiniLM-L6-v2 via transformers.js (~25MB one-time model download, fully
   local afterwards; deterministic hashed TF-IDF fallback when the model can't
   load), and stored under `~/.rooo-context-retriever/`.
2. Queries embed the same way; chunks rank by cosine similarity, with a mild
   penalty on test/spec paths so implementation outranks its tests.
3. Before every search, a staleness guard re-hashes the corpus and
   incrementally re-embeds anything that changed.
4. Transcript memory extracts the newest sessions (default 120; raise with
   `CONTEXT_RETRIEVER_MEMORY_SESSIONS`) and indexes them with the same
   machinery.

Environment switches: `CONTEXT_RETRIEVER_EMBEDDER=lexical` forces the offline
fallback; `CONTEXT_RETRIEVER_RANK=hybrid` enables RRF neural+lexical fusion
(measured slightly worse on our benchmark — see `src/store.ts`).

## Privacy

Everything — indexing, embedding, search, transcripts, stats — runs on your
machine. No network calls after the one-time model download; nothing is sent
anywhere, ever.

## Development

```bash
npm test          # 12 unit tests
npm run typecheck
npx tsx src/cli.ts index /path/to/repo          # try it without an MCP client
npx tsx src/cli.ts search /path/to/repo "where is the retry logic"
npx tsx src/evalrun.ts /path/to/repo            # run the retrieval benchmark
```

## Acknowledgments

Built on [@modelcontextprotocol/sdk](https://github.com/modelcontextprotocol/typescript-sdk) (MIT),
[transformers.js](https://github.com/huggingface/transformers.js) (Apache-2.0),
[zod](https://github.com/colinhacks/zod) (MIT), and the
[all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
model (Apache-2.0). Not affiliated with or endorsed by Anthropic; "Claude Code"
is referenced only to describe compatibility via MCP.

## License

MIT © Hamza Abid
