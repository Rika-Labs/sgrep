//! BM25F index caching for improved search performance.
//!
//! This module provides a single-entry cache for `Bm25FIndex` to avoid
//! rebuilding the index on every search query.

use crate::fts::Bm25FIndex;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Bm25CacheKey {
    pub repo_hash: String,
    pub chunk_count: usize,
    pub has_graph: bool,
}

impl Bm25CacheKey {
    pub fn new(repo_hash: &str, chunk_count: usize, has_graph: bool) -> Self {
        Self {
            repo_hash: repo_hash.to_string(),
            chunk_count,
            has_graph,
        }
    }
}

pub struct Bm25FCache {
    entry: Option<(Bm25CacheKey, Bm25FIndex)>,
    hit_count: usize,
    miss_count: usize,
}

impl Bm25FCache {
    pub fn new() -> Self {
        Self {
            entry: None,
            hit_count: 0,
            miss_count: 0,
        }
    }

    pub fn get(&mut self, key: &Bm25CacheKey) -> Option<&Bm25FIndex> {
        match &self.entry {
            Some((cached_key, index)) if cached_key == key => {
                self.hit_count += 1;
                Some(index)
            }
            _ => {
                self.miss_count += 1;
                None
            }
        }
    }

    pub fn insert(&mut self, key: Bm25CacheKey, index: Bm25FIndex) {
        self.entry = Some((key, index));
    }

    pub fn clear(&mut self) {
        self.entry = None;
    }

    pub fn hit_count(&self) -> usize {
        self.hit_count
    }

    pub fn miss_count(&self) -> usize {
        self.miss_count
    }

    pub fn entry_count(&self) -> usize {
        if self.entry.is_some() {
            1
        } else {
            0
        }
    }
}

impl Default for Bm25FCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_key_equals_with_same_values() {
        let key1 = Bm25CacheKey::new("hash123", 100, true);
        let key2 = Bm25CacheKey::new("hash123", 100, true);
        assert_eq!(key1, key2);
    }

    #[test]
    fn cache_key_differs_on_repo_hash_change() {
        let key1 = Bm25CacheKey::new("hash123", 100, true);
        let key2 = Bm25CacheKey::new("hash456", 100, true);
        assert_ne!(key1, key2);
    }

    #[test]
    fn cache_key_differs_on_chunk_count_change() {
        let key1 = Bm25CacheKey::new("hash123", 100, true);
        let key2 = Bm25CacheKey::new("hash123", 200, true);
        assert_ne!(key1, key2);
    }

    #[test]
    fn cache_key_differs_on_graph_presence() {
        let key1 = Bm25CacheKey::new("hash123", 100, true);
        let key2 = Bm25CacheKey::new("hash123", 100, false);
        assert_ne!(key1, key2);
    }

    #[test]
    fn bm25_cache_stores_and_retrieves() {
        let mut cache = Bm25FCache::new();
        let key = Bm25CacheKey::new("hash123", 1, false);
        let index = Bm25FIndex::default();

        cache.insert(key.clone(), index);

        assert!(cache.get(&key).is_some());
        assert_eq!(cache.hit_count(), 1);
    }

    #[test]
    fn bm25_cache_returns_none_for_missing_key() {
        let mut cache = Bm25FCache::new();
        let key = Bm25CacheKey::new("hash123", 1, false);

        assert!(cache.get(&key).is_none());
        assert_eq!(cache.miss_count(), 1);
    }

    #[test]
    fn bm25_cache_replaces_on_different_key() {
        let mut cache = Bm25FCache::new();
        let key1 = Bm25CacheKey::new("hash123", 1, false);
        let key2 = Bm25CacheKey::new("hash456", 1, false);

        cache.insert(key1.clone(), Bm25FIndex::default());
        cache.insert(key2.clone(), Bm25FIndex::default());

        // Old key should be gone
        assert!(cache.get(&key1).is_none());
        // New key should exist
        assert!(cache.get(&key2).is_some());
        // Only one entry at a time
        assert_eq!(cache.entry_count(), 1);
    }

    #[test]
    fn bm25_cache_clear_removes_entry() {
        let mut cache = Bm25FCache::new();
        let key = Bm25CacheKey::new("hash123", 1, false);
        cache.insert(key.clone(), Bm25FIndex::default());

        cache.clear();

        assert!(cache.get(&key).is_none());
        assert_eq!(cache.entry_count(), 0);
    }

    #[test]
    fn bm25_cache_tracks_hits_and_misses() {
        let mut cache = Bm25FCache::new();
        let key = Bm25CacheKey::new("hash123", 1, false);

        // Miss on empty cache
        cache.get(&key);
        assert_eq!(cache.miss_count(), 1);
        assert_eq!(cache.hit_count(), 0);

        // Insert and hit
        cache.insert(key.clone(), Bm25FIndex::default());
        cache.get(&key);
        assert_eq!(cache.hit_count(), 1);

        // Another hit
        cache.get(&key);
        assert_eq!(cache.hit_count(), 2);

        // Miss with different key
        let other_key = Bm25CacheKey::new("other", 1, false);
        cache.get(&other_key);
        assert_eq!(cache.miss_count(), 2);
    }

    #[test]
    fn cache_entry_count_reports_correctly() {
        let mut cache = Bm25FCache::new();
        assert_eq!(cache.entry_count(), 0);

        let key = Bm25CacheKey::new("hash", 1, false);
        cache.insert(key, Bm25FIndex::default());
        assert_eq!(cache.entry_count(), 1);

        cache.clear();
        assert_eq!(cache.entry_count(), 0);
    }
}
