import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";
import type { SearchHit } from "./store.js";

/**
 * Transcript memory: past Claude Code sessions as a searchable corpus.
 * Extraction reads ~/.claude/projects/<project>/<session>.jsonl and writes
 * one markdown file per session into MEMORY_DIR — human-readable message
 * text only, no tool payloads. Everything stays on this machine.
 *
 * Measured on 4 known-answer probes before building: 4/4 correct passages,
 * 19-36× token reduction vs inlining the session (transcripts are huge, so
 * memory search saves far more per query than code search does).
 */
export const MEMORY_DIR = path.join(os.homedir(), ".rooo-context-retriever", "memory");

/** Bounded first index: newest sessions first. Override via env. */
const MAX_SESSIONS = Number(process.env["CONTEXT_RETRIEVER_MEMORY_SESSIONS"] ?? 120);
const MAX_CHARS = 300_000;
const MIN_PARTS = 4; // fewer text turns than this = tool noise, skip

/** Pure extraction: JSONL transcript text → readable session text (or null). */
export function extractText(jsonl: string, header: string): string | null {
  const parts = [header];
  for (const line of jsonl.split("\n")) {
    try {
      const entry = JSON.parse(line) as {
        message?: { role?: string; content?: unknown };
      };
      const role = entry.message?.role;
      const content = entry.message?.content;
      if (!role || !content) continue;
      if (typeof content === "string") {
        if (content.trim()) parts.push(`${role}: ${content}`);
        continue;
      }
      if (Array.isArray(content)) {
        for (const block of content as { type?: string; text?: string }[]) {
          if (block?.type === "text" && block.text?.trim()) parts.push(`${role}: ${block.text}`);
        }
      }
    } catch {
      // partial/corrupt line — skip it, keep the session
    }
  }
  if (parts.length < MIN_PARTS) return null;
  return parts.join("\n\n").slice(0, MAX_CHARS);
}

/**
 * Filter hits to sessions from the last `days`. Extracted filenames start
 * with the session date (YYYY-MM-DD__…), so a string compare is exact.
 */
export function filterByDays(hits: readonly SearchHit[], days: number): SearchHit[] {
  const cutoff = new Date(Date.now() - days * 86_400_000).toISOString().slice(0, 10);
  return hits.filter((h) => path.basename(h.file).slice(0, 10) >= cutoff);
}

export interface ExtractReport {
  scanned: number;
  extracted: number;
  upToDate: number;
}

/** Incremental: a session is re-extracted only when its .jsonl is newer than
 * its extracted file. New sessions appear automatically on the next call. */
export async function extractTranscripts(): Promise<ExtractReport> {
  const root = path.join(os.homedir(), ".claude", "projects");
  const sources: { path: string; mtimeMs: number; project: string; id: string }[] = [];
  let dirs: string[];
  try {
    dirs = await fs.readdir(root);
  } catch {
    return { scanned: 0, extracted: 0, upToDate: 0 };
  }
  for (const dir of dirs) {
    let files: string[];
    try {
      files = await fs.readdir(path.join(root, dir));
    } catch {
      continue;
    }
    for (const file of files) {
      if (!file.endsWith(".jsonl")) continue;
      try {
        const full = path.join(root, dir, file);
        const stat = await fs.stat(full);
        sources.push({ path: full, mtimeMs: stat.mtimeMs, project: dir, id: file.slice(0, -6) });
      } catch {
        // unreadable — skip
      }
    }
  }
  sources.sort((a, b) => b.mtimeMs - a.mtimeMs);

  await fs.mkdir(MEMORY_DIR, { recursive: true });
  let extracted = 0;
  let upToDate = 0;
  for (const src of sources.slice(0, MAX_SESSIONS)) {
    const date = new Date(src.mtimeMs).toISOString().slice(0, 10);
    const target = path.join(
      MEMORY_DIR,
      `${date}__${src.project.slice(-24)}__${src.id.slice(0, 8)}.md`,
    );
    try {
      const existing = await fs.stat(target);
      if (existing.mtimeMs >= src.mtimeMs) {
        upToDate++;
        continue;
      }
    } catch {
      // not extracted yet
    }
    let raw: string;
    try {
      raw = await fs.readFile(src.path, "utf8");
    } catch {
      continue;
    }
    const text = extractText(raw, `session: ${src.project} / ${src.id} (${date})`);
    if (!text) continue;
    await fs.writeFile(target, text);
    extracted++;
  }
  return { scanned: sources.length, extracted, upToDate };
}
