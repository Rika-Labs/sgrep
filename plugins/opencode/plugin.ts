import { type Plugin, tool } from "@opencode-ai/plugin";

export const SgrepOpenCodePlugin: Plugin = async ({ $, directory }) => ({
  tool: {
    sgrepSearch: tool({
      description: "Semantic code search for this repo (sgrep) — use instead of grep; returns ranked snippets",
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

export default SgrepOpenCodePlugin;

type SearchItem = {
  path: string;
  start_line?: number;
  end_line?: number;
  score?: number;
  snippet?: string;
  language?: string;
};

function formatResults(results: SearchItem[]): string {
  return results
    .map((r, i) => {
      const score = typeof r.score === "number" ? r.score.toFixed(2) : String(r.score ?? "?");
      const loc = r.start_line && r.end_line ? `${r.start_line}-${r.end_line}` : "?";
      const snippet = typeof r.snippet === "string" ? r.snippet.replace(/\s+/g, " ").trim() : "";
      return `${i + 1}. ${r.path}:${loc} (score ${score}) ${snippet}`;
    })
    .join("\n");
}

async function runSearch(
  $: Parameters<Plugin>[0]["$"],
  directory: string,
  query: string,
  limit?: number,
): Promise<string> {
  const baseCmd = limit
    ? $`cd ${directory} && sgrep search --json --path ${directory} --limit ${limit} ${query}`
    : $`cd ${directory} && sgrep search --json --path ${directory} ${query}`;
  const proc = baseCmd.quiet();
  const { stdout, stderr } = await proc;
  const out = stdout?.toString?.() ?? "";
  const err = stderr?.toString?.() ?? "";

  try {
    const parsed = JSON.parse(out) as { results?: Array<SearchItem> };
    const results = Array.isArray(parsed.results)
      ? parsed.results.filter((r): r is SearchItem => typeof r?.path === "string")
      : [];
    if (!results.length) return "(no results)";
    return formatResults(results);
  } catch {
    const fallback = (out || err || "(no output)").replace(/\s+/g, " ").trim();
    return fallback.slice(0, 500);
  }
}
