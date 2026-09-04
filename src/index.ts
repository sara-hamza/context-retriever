#!/usr/bin/env node
import { serveStdio } from "./server.js";

serveStdio().catch((error) => {
  console.error("context-retriever failed:", error);
  process.exit(1);
});
