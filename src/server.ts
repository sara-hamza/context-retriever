import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { buildIndex, loadIndex, savingsFor, search, type StoreData } from "./store.js";
import { extractTranscripts, filterByDays, MEMORY_DIR } from "./memory.js";

/**
 * MCP server: local vector retrieval so the model pulls the few relevant
 * chunks instead of reading whole files. The token-savings counter is the
 * point of the exercise — it makes the reduction measurable, not vibes.
 * Lifetime totals persist across sessions in ~/.rooo-context-retriever/.
 */
interface SessionStats {
  searches: number;
  chunkTokens: number;
  fullFileTokens: number;
}

const STATS_FILE = path.join(os.homedir(), ".rooo-context-retriever", "stats.json");

async function loadLifetime(): Promise<SessionStats> {
  try {
    return JSON.parse(await fs.readFile(STATS_FILE, "utf8")) as SessionStats;
  } catch {
    return { searches: 0, chunkTokens: 0, fullFileTokens: 0 };
  }
}

async function saveLifetime(stats: SessionStats): Promise<void> {
  try {
    await fs.mkdir(path.dirname(STATS_FILE), { recursive: true });
    await fs.writeFile(STATS_FILE, JSON.stringify(stats));
  } catch {
    // Stats are a convenience — never let them break a search.
  }
}

/**
 * Health section of retrieval_stats. Added in v1.1.0 after a day lost to
 * "is it even working?": the plugin was connected but the project was
 * unindexed, so every answer fell back to reading whole files. Reading the
 * index directory answers that in one call instead of an investigation.
 */
async function healthReport(): Promise<string> {
  const dir = path.dirname(STATS_FILE);
  let entries: string[];
  try {
    entries = await fs.readdir(dir);
  } catch {
    return "indexed projects: none yet — search_context indexes a directory on first use.";
  }
  const live: { line: string; builtAt: number }[] = [];
  let stale = 0;
  for (const entry of entries) {
    if (!entry.endsWith(".json") || entry === "stats.json") continue;
    try {
      const raw = await fs.readFile(path.join(dir, entry), "utf8");
      const store = JSON.parse(raw) as {
        root: string; files: number; chunks: unknown[]; embedderKind?: string; builtAt: string;
      };
      // An index whose project directory is gone (deleted repo, temp dir from
      // a test run) is noise in a health report — count it, don't list it.
      try {
        await fs.stat(store.root);
      } catch {
        stale++;
        continue;
      }
      const builtAt = Date.parse(store.builtAt);
      const ageHours = (Date.now() - builtAt) / 3_600_000;
      const age = ageHours < 1 ? `${Math.round(ageHours * 60)}m` : `${Math.round(ageHours)}h`;
      live.push({
        builtAt,
        line:
          `  ${store.root} — ${store.files} files, ${store.chunks.length} chunks, ` +
          `${store.embedderKind ?? "unknown embedder"}, indexed ${age} ago`,
      });
    } catch {
      stale++; // unreadable/corrupt — rebuilt automatically on next search
    }
  }
  if (live.length === 0) {
    return "indexed projects: none yet — search_context indexes a directory on first use.";
  }
  live.sort((a, b) => b.builtAt - a.builtAt); // freshest first
  const staleNote = stale > 0 ? ` (${stale} stale index file${stale === 1 ? "" : "s"} ignored)` : "";
  return `indexed projects (${live.length})${staleNote}:\n${live.map((r) => r.line).join("\n")}`;
}

function report(label: string, s: SessionStats): string {
  const saved = s.fullFileTokens - s.chunkTokens;
  const ratio = s.chunkTokens > 0 ? `${(s.fullFileTokens / s.chunkTokens).toFixed(1)}×` : "—";
  return (
    `${label}: ${s.searches} searches | chunks ${s.chunkTokens.toLocaleString()} tokens ` +
    `vs whole files ${s.fullFileTokens.toLocaleString()} | saved ~${saved.toLocaleString()} (${ratio})`
  );
}

export function createServer(): McpServer {
  const server = new McpServer({ name: "context-retriever", version: "0.1.0" });
  const indexes = new Map<string, StoreData>();
  const stats: SessionStats = { searches: 0, chunkTokens: 0, fullFileTokens: 0 };

  async function getIndex(root: string, rebuild = false, full = false): Promise<StoreData> {
    const cached = indexes.get(root);
    if (cached && !rebuild) return cached;
    const loaded = rebuild ? null : await loadIndex(root);
    const store = loaded ?? (await buildIndex(root, { full }));
    indexes.set(root, store);
    return store;
  }

  /** Staleness guard: before a search, incrementally refresh the index so
   * results never describe yesterday's code. buildIndex's identical-corpus
   * fast path makes the no-change case cheap; the throttle keeps rapid
   * successive searches from re-hashing the tree every time. */
  const REFRESH_MS = 30_000;
  const lastRefresh = new Map<string, number>();
  let memoryReport = { scanned: 0, extracted: 0, upToDate: 0 };
  async function getFreshIndex(root: string): Promise<StoreData> {
    const now = Date.now();
    const cached = indexes.get(root);
    if (cached && now - (lastRefresh.get(root) ?? 0) < REFRESH_MS) return cached;
    const store = await buildIndex(root, {});
    indexes.set(root, store);
    lastRefresh.set(root, now);
    return store;
  }

  server.tool(
    "index_path",
    "Index (or re-index) a directory for vector retrieval. Incremental: only changed files are re-embedded. Run once per project, and again after changes.",
    {
      path: z.string().describe("Absolute path of the directory to index"),
      full: z.boolean().default(false).describe("Force a full re-embed instead of incremental"),
    },
    async ({ path: root, full }) => {
      const store = await getIndex(root, true, full);
      const totalTokens = Object.values(store.fileTokens).reduce((a, b) => a + b, 0);
      return {
        content: [{
          type: "text" as const,
          text:
            `Indexed ${store.root}: ${store.files} files, ${store.chunks.length} chunks, ` +
            `~${totalTokens.toLocaleString()} tokens of source (${store.embedderKind} embeddings). ` +
            `Use search_context instead of reading whole files.`,
        }],
      };
    },
  );

  server.tool(
    "search_context",
    "Vector-search a codebase and return only the most relevant chunks (with file:line refs), " +
      "instead of reading whole files. Indexes the directory automatically on first use. Use it for:\n" +
      "• \"where is X?\", \"how does Y work?\", \"what calls Z?\" — before reading or grepping\n" +
      "• DEBUGGING: paste the failing symbol, error message, or stack-trace frame as the query " +
      "(e.g. \"RideDetailScreen null check ride history\") to jump straight to the code that produced it\n" +
      "• orienting in unfamiliar packages, or code you have not touched recently",
    {
      path: z.string().describe("Absolute path of the project directory (indexed on demand)"),
      query: z
        .string()
        .describe(
          "What you are looking for: natural language, identifiers, an error message, or a stack-trace frame",
        ),
      k: z.number().int().min(1).max(20).default(5).describe("Number of chunks to return"),
    },
    async ({ path: root, query, k }) => {
      let store: StoreData;
      try {
        store = await getFreshIndex(root);
      } catch (error) {
        return {
          content: [{
            type: "text" as const,
            text:
              `Could not index ${root}: ${String(error).slice(0, 200)}\n` +
              `Check the path exists and contains source files.`,
          }],
        };
      }
      const hits = await search(store, query, k);
      const { chunkTokens, fullFileTokens } = savingsFor(store, hits);
      stats.searches += 1;
      stats.chunkTokens += chunkTokens;
      stats.fullFileTokens += fullFileTokens;
      const lifetime = await loadLifetime();
      lifetime.searches += 1;
      lifetime.chunkTokens += chunkTokens;
      lifetime.fullFileTokens += fullFileTokens;
      await saveLifetime(lifetime);

      const body = hits
        .map(
          (h) =>
            `=== ${h.file}:${h.startLine}-${h.endLine} (score ${h.score.toFixed(3)}) ===\n${h.text}`,
        )
        .join("\n\n");
      const saved = fullFileTokens - chunkTokens;
      return {
        content: [{
          type: "text" as const,
          text:
            `${body}\n\n` +
            `--- retrieval: ${chunkTokens} tokens returned vs ~${fullFileTokens} to inline the ` +
            `${new Set(hits.map((h) => h.file)).size} source file(s) (saved ~${saved}) ---`,
        }],
      };
    },
  );

  server.tool(
    "search_memory",
    "Search past Claude Code session transcripts on this machine — use for questions like " +
      "\"what did we decide about X?\" or \"what happened in the session where Y?\" instead of asking the user " +
      "to re-explain. Local only. First call may take minutes while history is indexed.",
    {
      query: z.string().describe("What you are trying to remember, in natural language"),
      k: z.number().int().min(1).max(20).default(5).describe("Number of transcript chunks to return"),
      days: z.number().int().min(1).optional().describe("Only search sessions from the last N days"),
    },
    async ({ query, k, days }) => {
      // Extraction shares the refresh throttle: scanning ~2k transcript
      // stats every single memory query would be waste.
      if (Date.now() - (lastRefresh.get("memory-extract") ?? 0) >= REFRESH_MS) {
        memoryReport = await extractTranscripts();
        lastRefresh.set("memory-extract", Date.now());
      }
      const report = memoryReport;
      // Not every surface has Claude Code session history (a fresh machine,
      // or a host that stores sessions elsewhere). Degrade to a clear
      // explanation instead of an error.
      let store: StoreData;
      try {
        store = await getFreshIndex(MEMORY_DIR);
      } catch {
        return {
          content: [{
            type: "text" as const,
            text:
              "No session transcripts are available on this machine (looked in ~/.claude/projects). " +
              "Transcript memory requires Claude Code session history and becomes available automatically " +
              "once past sessions exist. Code search via search_context works independently of this.",
          }],
        };
      }
      let hits = await search(store, query, k + 15);
      if (days !== undefined) hits = filterByDays(hits, days);
      hits = hits.slice(0, k);
      const { chunkTokens, fullFileTokens } = savingsFor(store, hits);
      stats.searches += 1;
      stats.chunkTokens += chunkTokens;
      stats.fullFileTokens += fullFileTokens;
      const lifetime = await loadLifetime();
      lifetime.searches += 1;
      lifetime.chunkTokens += chunkTokens;
      lifetime.fullFileTokens += fullFileTokens;
      await saveLifetime(lifetime);

      const body = hits
        .map(
          (h) =>
            `=== ${h.file.replace(/\.md$/, "")} lines ${h.startLine}-${h.endLine} (score ${h.score.toFixed(3)}) ===\n${h.text}`,
        )
        .join("\n\n");
      return {
        content: [{
          type: "text" as const,
          text:
            `${body}\n\n` +
            `--- memory: ${report.scanned} sessions known, ${hits.length} chunks returned ` +
            `(${chunkTokens} tokens vs ~${fullFileTokens} to inline the sessions, saved ~${fullFileTokens - chunkTokens}) ---`,
        }],
      };
    },
  );

  server.tool(
    "retrieval_stats",
    "Token savings (this session and lifetime) plus health: which projects are indexed, " +
      "how fresh each index is, and which embedder built it. Use it to check whether " +
      "retrieval is actually working before concluding it is not helping.",
    {},
    async () => {
      const lifetime = await loadLifetime();
      return {
        content: [{
          type: "text" as const,
          text:
            `${report("this session", stats)}\n${report("lifetime", lifetime)}\n\n` +
            (await healthReport()),
        }],
      };
    },
  );

  return server;
}

export async function serveStdio(): Promise<void> {
  const server = createServer();
  await server.connect(new StdioServerTransport());
}
