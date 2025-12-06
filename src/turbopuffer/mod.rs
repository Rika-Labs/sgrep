//! Turbopuffer remote storage module.
//!
//! This module provides:
//! - `TurbopufferStore`: Stores and queries vectors in Turbopuffer
//! - Remote index management for serverless vector storage

#[allow(dead_code)]
mod store;

#[allow(unused_imports)]
pub use store::TurbopufferStore;
