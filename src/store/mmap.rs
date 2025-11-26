use memmap2::Mmap;

use super::index::IndexMetadata;
use super::RepositoryIndex;
use crate::chunker::CodeChunk;

pub const VECTOR_HEADER_SIZE: usize = 8;
pub const BYTES_PER_F32: usize = 4;
pub const BYTES_PER_U64: usize = 8;

pub struct MmapIndex {
    pub metadata: IndexMetadata,
    pub chunks: Vec<CodeChunk>,
    pub(crate) mmap: Mmap,
    pub(crate) binary_mmap: Option<Mmap>,
    pub(crate) vector_dim: usize,
    pub(crate) binary_words: usize,
}

impl MmapIndex {
    #[inline]
    pub fn get_vector(&self, idx: usize) -> &[f32] {
        let vector_bytes = self.vector_dim * BYTES_PER_F32;
        let offset = VECTOR_HEADER_SIZE + idx * vector_bytes;
        let slice = &self.mmap[offset..offset + vector_bytes];
        unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const f32, self.vector_dim) }
    }

    #[inline]
    pub fn get_binary_vector(&self, idx: usize) -> Option<&[u64]> {
        let mmap = self.binary_mmap.as_ref()?;
        let binary_bytes = self.binary_words * BYTES_PER_U64;
        let offset = VECTOR_HEADER_SIZE + idx * binary_bytes;
        let slice = &mmap[offset..offset + binary_bytes];
        Some(unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u64, self.binary_words) })
    }

    #[inline]
    pub fn has_binary_vectors(&self) -> bool {
        self.binary_mmap.is_some()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    #[inline]
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    pub fn to_repository_index(&self) -> RepositoryIndex {
        let vectors: Vec<Vec<f32>> = (0..self.len())
            .map(|i| self.get_vector(i).to_vec())
            .collect();
        RepositoryIndex {
            metadata: self.metadata.clone(),
            chunks: self.chunks.clone(),
            vectors,
        }
    }
}

pub fn parse_vectors_header(mmap: &Mmap) -> anyhow::Result<(u32, usize)> {
    if mmap.len() < VECTOR_HEADER_SIZE {
        return Err(anyhow::anyhow!("Invalid vectors file: too small"));
    }
    let format_version = u32::from_le_bytes([mmap[0], mmap[1], mmap[2], mmap[3]]);
    let vector_dim = u32::from_le_bytes([mmap[4], mmap[5], mmap[6], mmap[7]]) as usize;
    Ok((format_version, vector_dim))
}

pub fn parse_binary_header(mmap: &Mmap) -> anyhow::Result<(u32, usize)> {
    if mmap.len() < VECTOR_HEADER_SIZE {
        return Err(anyhow::anyhow!("Invalid binary vectors file: too small"));
    }
    let format_version = u32::from_le_bytes([mmap[0], mmap[1], mmap[2], mmap[3]]);
    let num_words = u32::from_le_bytes([mmap[4], mmap[5], mmap[6], mmap[7]]) as usize;
    Ok((format_version, num_words))
}

pub fn validate_vectors_size(
    mmap: &Mmap,
    vector_dim: usize,
    num_vectors: usize,
) -> anyhow::Result<()> {
    let vector_bytes = vector_dim * BYTES_PER_F32;
    let expected_size = VECTOR_HEADER_SIZE + num_vectors * vector_bytes;
    if mmap.len() < expected_size {
        return Err(anyhow::anyhow!(
            "Invalid vectors file: expected {} bytes, got {}",
            expected_size,
            mmap.len()
        ));
    }
    Ok(())
}

pub fn read_vectors_from_mmap(mmap: &Mmap, vector_dim: usize, num_vectors: usize) -> Vec<Vec<f32>> {
    let vector_bytes = vector_dim * BYTES_PER_F32;
    (0..num_vectors)
        .map(|i| {
            let offset = VECTOR_HEADER_SIZE + i * vector_bytes;
            mmap[offset..offset + vector_bytes]
                .chunks_exact(BYTES_PER_F32)
                .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
                .collect()
        })
        .collect()
}
