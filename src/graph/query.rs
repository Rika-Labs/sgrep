use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryType {
    Structural,
    Semantic,
    Hybrid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SearchGranularity {
    Chunk,
    File,
    Directory,
}

impl Default for SearchGranularity {
    fn default() -> Self {
        SearchGranularity::Chunk
    }
}
