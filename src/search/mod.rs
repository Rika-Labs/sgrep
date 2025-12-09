mod binary;
mod bm25_cache;
pub mod config;
mod dedup;
mod engine;
pub mod file_type;
mod hnsw;
mod results;
mod scoring;

#[allow(unused_imports)]
pub use dedup::{suppress_near_duplicates, DedupOptions};
pub use engine::{SearchEngine, SearchOptions};
#[allow(unused_imports)]
pub use file_type::{classify_path, FileType, FileTypePriority};
#[allow(unused_imports)]
pub use results::{DirectorySearchResult, FileSearchResult, SearchResult};
#[allow(unused_imports)]
pub use scoring::cosine_similarity;

#[cfg(test)]
mod tests;
