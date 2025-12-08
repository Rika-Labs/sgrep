//! Modal.dev offload module for embedding.

mod deployer;
mod embedder;

pub use deployer::ModalDeployer;
pub use embedder::ModalEmbedder;

pub const MODAL_SERVICE_PY: &str = include_str!("../../modal/service.py");
