import { describe, expect, it } from "vitest";
import { extractText, filterByDays } from "../src/memory.js";
import type { SearchHit } from "../src/store.js";

const line = (role: string, text: string): string =>
  JSON.stringify({ message: { role, content: [{ type: "text", text }] } });

describe("transcript extraction", () => {
  it("keeps message text with roles and drops tool noise", () => {
    const jsonl = [
      line("user", "how does the packer work?"),
      JSON.stringify({ message: { role: "assistant", content: [{ type: "tool_use", name: "Read" }] } }),
      line("assistant", "it binds slots with fixed phases"),
      line("user", "and the budget?"),
      "not json at all",
      JSON.stringify({ message: { role: "user", content: "plain string content" } }),
    ].join("\n");
    const text = extractText(jsonl, "session: test");
    expect(text).toContain("user: how does the packer work?");
    expect(text).toContain("assistant: it binds slots with fixed phases");
    expect(text).toContain("user: plain string content");
    expect(text).not.toContain("tool_use");
  });

  it("returns null for tool-heavy sessions with almost no prose", () => {
    const jsonl = [
      line("user", "run it"),
      JSON.stringify({ message: { role: "assistant", content: [{ type: "tool_use" }] } }),
    ].join("\n");
    expect(extractText(jsonl, "session: test")).toBeNull();
  });
});

describe("days filter", () => {
  const hit = (file: string): SearchHit => ({
    file, score: 1, startLine: 1, endLine: 2, tokens: 10, text: "",
  });

  it("keeps only sessions dated inside the window", () => {
    const today = new Date().toISOString().slice(0, 10);
    const old = "2020-01-01";
    const hits = [hit(`${today}__proj__abc.md`), hit(`${old}__proj__def.md`)];
    const filtered = filterByDays(hits, 30);
    expect(filtered.map((h) => h.file)).toEqual([`${today}__proj__abc.md`]);
  });
});
