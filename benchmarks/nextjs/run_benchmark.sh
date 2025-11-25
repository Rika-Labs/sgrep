#!/bin/bash

SGREP="/Users/dallenpyrah/rika-labs/sgrep/target/release/sgrep"
NEXTJS_PATH="/Users/dallenpyrah/rika-labs/nextjs"
OUTPUT_FILE="/Users/dallenpyrah/rika-labs/sgrep/benchmarks/nextjs/benchmark_results.md"

cd "$NEXTJS_PATH"

queries=(
    "How does the app router handle dynamic routes?"
    "Where is server-side rendering implemented?"
    "How does Next.js handle API route requests?"
    "Where is the image optimization logic?"
    "How does middleware intercept requests?"
    "Where are server actions processed?"
    "How does the build process compile pages?"
    "Where is static site generation handled?"
    "How does incremental static regeneration work?"
    "Where is the router cache implemented?"
    "How does Next.js handle 404 errors?"
    "Where is the error boundary logic?"
    "How does prefetching work for links?"
    "Where is the metadata API implemented?"
    "How does Next.js handle redirects?"
    "Where is the webpack configuration?"
    "How does hot module replacement work?"
    "Where is the edge runtime implemented?"
    "How does the font optimization work?"
    "Where is streaming SSR implemented?"
    "How does parallel routing work?"
    "Where is the data cache implemented?"
    "How does Next.js handle cookies?"
    "Where is the turbopack integration?"
    "How does code splitting work?"
    "Where is the flight protocol implemented?"
    "How does suspense boundary work in app router?"
    "Where is request deduplication handled?"
    "How does the dev server compile on demand?"
    "Where is the route matching logic?"
)

echo "# Next.js Semantic Search Benchmark Results" > "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "Generated: $(date)" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "---" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

total_time=0
query_num=0

for query in "${queries[@]}"; do
    query_num=$((query_num + 1))

    start_time=$(python3 -c 'import time; print(time.time())')

    results=$("$SGREP" search "$query" --limit 5 2>&1)

    end_time=$(python3 -c 'import time; print(time.time())')
    duration=$(python3 -c "print(round(($end_time - $start_time) * 1000))")
    total_time=$((total_time + duration))

    echo "## Query $query_num: $query" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    echo "**Time:** ${duration}ms" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    echo "**Results:**" >> "$OUTPUT_FILE"
    echo "\`\`\`" >> "$OUTPUT_FILE"
    echo "$results" >> "$OUTPUT_FILE"
    echo "\`\`\`" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    echo "**Rating:** _/10" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    echo "**Notes:**" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    echo "---" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"

    echo "[$query_num/30] $query - ${duration}ms"
done

avg_time=$((total_time / 30))

echo "## Summary" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "- **Total Queries:** 30" >> "$OUTPUT_FILE"
echo "- **Total Time:** ${total_time}ms" >> "$OUTPUT_FILE"
echo "- **Average Time:** ${avg_time}ms" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "### Overall Score: _/10" >> "$OUTPUT_FILE"

echo ""
echo "Benchmark complete!"
echo "Total time: ${total_time}ms"
echo "Average: ${avg_time}ms"
echo "Results saved to: $OUTPUT_FILE"
