import { type Plugin, tool } from "@opencode-ai/plugin";

export const SgrepOpenCodePlugin: Plugin = async ({ $, directory }) => ({
  tool: {
    sgrepSearch: tool({
      description:
        "Semantic code search for this repo (sgrep) — use instead of grep; returns ranked snippets",
      args: {
        query: tool.schema.string().describe("Search query passed to the sgrep CLI"),
        limit: tool.schema.number().int().positive().optional(),
      },
      async execute(args) {
        return runSearch($, directory, args.query, args.limit);
      },
    }),
  },
});

type JsonMatch = {
  path: string;
  start_line: number;
  end_line: number;
  language: string;
  score: number;
  snippet: string;
};

type JsonResponse = {
  results: JsonMatch[];
  index: object;
};

function formatResults(results: JsonMatch[]): string {
  return results
    .map((r, i) => {
      const loc = `${r.start_line}-${r.end_line}`;
      const snippet = r.snippet.trim();
      return `${i + 1}. ${r.path}:${loc} [${r.language}] (score: ${r.score.toFixed(2)})\n${snippet}`;
    })
    .join("\n\n");
}

function isJsonResponse(value: unknown): value is JsonResponse {
  if (typeof value !== "object" || value === null) {
    return false;
  }
  const obj = value as Record<string, unknown>;
  return Array.isArray(obj.results) && typeof obj.index === "object";
}

async function runSearch(
  $: Parameters<Plugin>[0]["$"],
  directory: string,
  query: string,
  limit?: number,
): Promise<string> {
  const args = ["search", "--json", "--path", directory];
  if (limit !== undefined) {
    args.push("--limit", String(limit));
  }
  args.push(query);

  const proc = $`sgrep ${args}`.quiet();
  const { stdout, stderr, exitCode } = await proc;
  const out = stdout?.toString?.() ?? "";
  const err = stderr?.toString?.() ?? "";

  if (exitCode !== 0) {
    return `Error (exit ${exitCode}): ${err || out || "unknown error"}`.trim();
  }

  try {
    const parsed: unknown = JSON.parse(out);
    if (!isJsonResponse(parsed)) {
      return `Unexpected response format: ${out.slice(0, 200)}`;
    }
    if (parsed.results.length === 0) {
      return "(no results)";
    }
    return formatResults(parsed.results);
  } catch (e) {
    const message = e instanceof Error ? e.message : "parse error";
    return `Failed to parse output (${message}): ${out.slice(0, 200)}`;
  }
}
