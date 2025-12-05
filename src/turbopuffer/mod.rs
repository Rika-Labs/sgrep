//! Turbopuffer remote storage module.
//!
//! This module provides:
//! - `TurbopufferStore`: Stores and queries vectors in Turbopuffer
//! - Remote index management for serverless vector storage

mod store;

pub use store::TurbopufferStore;
