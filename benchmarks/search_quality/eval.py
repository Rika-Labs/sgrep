#!/usr/bin/env python3
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


def run_search(sgrep: str, repo: str, query: str, limit: int) -> dict:
    proc = subprocess.run(
        [sgrep, "search", query, "--path", repo, "--json", "-n", str(limit), "--offline"],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(proc.stdout)


def find_rank(results: list[dict], accepted_paths: list[str]) -> int | None:
    accepted = set(accepted_paths)
    for i, result in enumerate(results, start=1):
        if result["path"] in accepted:
            return i
    return None


def reciprocal_rank(rank: int | None) -> float:
    return 0.0 if rank is None else 1.0 / rank


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sgrep", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--queries", default="benchmarks/search_quality/queries.json")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    repo = str(Path(args.repo).resolve())
    with open(args.queries) as f:
        queries = json.load(f)

    rows = []
    for item in queries:
        response = run_search(args.sgrep, repo, item["query"], args.limit)
        rank = find_rank(response.get("results", []), item["accepted_paths"])
        rr = reciprocal_rank(rank)
        top_paths = [r["path"] for r in response.get("results", [])[:3]]
        rows.append(
            {
                "id": item["id"],
                "intent": item["intent"],
                "query": item["query"],
                "rank": rank,
                "rr": rr,
                "duration_ms": response.get("duration_ms", 0),
                "top_paths": top_paths,
            }
        )

    def summarize(items: list[dict]) -> dict:
        count = len(items)
        return {
            "count": count,
            "mrr": (sum(item["rr"] for item in items) / count) if count else 0.0,
            "hit_at_1": (sum(1 for item in items if item["rank"] == 1) / count) if count else 0.0,
            "hit_at_3": (sum(1 for item in items if item["rank"] and item["rank"] <= 3) / count)
            if count
            else 0.0,
            "hit_at_10": (sum(1 for item in items if item["rank"] and item["rank"] <= 10) / count)
            if count
            else 0.0,
            "median_search_ms": statistics.median(item["duration_ms"] for item in items) if count else 0.0,
        }

    overall = summarize(rows)
    code = summarize([row for row in rows if row["intent"] == "code"])
    docs = summarize([row for row in rows if row["intent"] == "docs"])

    print("# Search quality benchmark")
    print(f"queries={overall['count']}")
    print(
        "overall "
        f"MRR={overall['mrr']:.4f} "
        f"hit@1={overall['hit_at_1']:.4f} "
        f"hit@3={overall['hit_at_3']:.4f} "
        f"hit@10={overall['hit_at_10']:.4f} "
        f"median_search_ms={overall['median_search_ms']:.1f}"
    )
    print(
        "code    "
        f"MRR={code['mrr']:.4f} "
        f"hit@1={code['hit_at_1']:.4f} "
        f"hit@3={code['hit_at_3']:.4f} "
        f"hit@10={code['hit_at_10']:.4f}"
    )
    print(
        "docs    "
        f"MRR={docs['mrr']:.4f} "
        f"hit@1={docs['hit_at_1']:.4f} "
        f"hit@3={docs['hit_at_3']:.4f} "
        f"hit@10={docs['hit_at_10']:.4f}"
    )
    print()
    print("Worst queries:")
    worst = sorted(rows, key=lambda row: (row["rr"], row["rank"] or 999))[:8]
    for row in worst:
        print(
            f"- {row['id']}: rank={row['rank']} rr={row['rr']:.3f} | {row['query']} | top3={row['top_paths']}"
        )

    print()
    print(f"METRIC quality_mrr={overall['mrr'] * 100:.4f}")
    print(f"METRIC code_mrr={code['mrr'] * 100:.4f}")
    print(f"METRIC docs_mrr={docs['mrr'] * 100:.4f}")
    print(f"METRIC hit_at_1={overall['hit_at_1'] * 100:.4f}")
    print(f"METRIC hit_at_3={overall['hit_at_3'] * 100:.4f}")
    print(f"METRIC hit_at_10={overall['hit_at_10'] * 100:.4f}")
    print(f"METRIC median_search_ms={overall['median_search_ms']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
