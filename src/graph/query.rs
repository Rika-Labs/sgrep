use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum SearchGranularity {
    #[default]
    Chunk,
    File,
    Directory,
}
