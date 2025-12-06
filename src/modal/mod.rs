//! Modal.dev offload module for embedding and reranking.
//!
//! This module provides:
//! - `ModalEmbedder`: Implements `BatchEmbedder` trait, calls Modal for embeddings
//! - `ModalReranker`: Calls Modal for reranking search results
//! - `ModalDeployer`: Auto-deploys the Modal service if needed

mod deployer;
mod embedder;
mod reranker;

pub use deployer::ModalDeployer;
pub use embedder::ModalEmbedder;
pub use reranker::ModalReranker;

/// The bundled Modal Python service code
pub const MODAL_SERVICE_PY: &str = include_str!("../../modal/service.py");
