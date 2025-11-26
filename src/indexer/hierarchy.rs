use std::collections::HashMap;
use std::path::PathBuf;

use crate::chunker::CodeChunk;
use crate::store::HierarchicalIndex;

pub fn mean_pool_vectors(vectors: &[Vec<f32>]) -> Vec<f32> {
    if vectors.is_empty() {
        return Vec::new();
    }

    let dim = vectors[0].len();
    let count = vectors.len() as f32;

    let mut result = vec![0.0_f32; dim];
    for vec in vectors {
        for (i, &val) in vec.iter().enumerate() {
            result[i] += val;
        }
    }

    for val in &mut result {
        *val /= count;
    }

    result
}

pub fn l2_normalize(vector: &mut [f32]) {
    let magnitude: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if magnitude > 1e-10 {
        for val in vector.iter_mut() {
            *val /= magnitude;
        }
    }
}

pub fn compute_file_embeddings(chunks: &[CodeChunk], vectors: &[Vec<f32>]) -> HierarchicalIndex {
    let mut hier = HierarchicalIndex::new();

    let mut file_to_chunks: HashMap<PathBuf, Vec<usize>> = HashMap::new();
    for (idx, chunk) in chunks.iter().enumerate() {
        file_to_chunks
            .entry(chunk.path.clone())
            .or_default()
            .push(idx);
    }

    let mut file_paths: Vec<_> = file_to_chunks.keys().cloned().collect();
    file_paths.sort();

    for file_path in file_paths {
        let chunk_indices = file_to_chunks.get(&file_path).unwrap();

        let file_chunk_vectors: Vec<&Vec<f32>> = chunk_indices
            .iter()
            .map(|&idx| &vectors[idx])
            .collect();

        let refs: Vec<Vec<f32>> = file_chunk_vectors.iter().map(|v| (*v).clone()).collect();
        let mut file_vector = mean_pool_vectors(&refs);
        l2_normalize(&mut file_vector);

        hier.add_file(file_path, chunk_indices.clone(), file_vector);
    }

    hier
}

pub fn compute_directory_embeddings(hier: &mut HierarchicalIndex) {
    if hier.files.is_empty() {
        return;
    }

    let mut dir_to_files: HashMap<PathBuf, Vec<usize>> = HashMap::new();
    for (file_idx, file_entry) in hier.files.iter().enumerate() {
        let parent = file_entry
            .path
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from(""));
        dir_to_files.entry(parent).or_default().push(file_idx);
    }

    let mut dir_paths: Vec<_> = dir_to_files.keys().cloned().collect();
    dir_paths.sort();

    for dir_path in dir_paths {
        let file_indices = dir_to_files.get(&dir_path).unwrap();

        let dir_file_vectors: Vec<Vec<f32>> = file_indices
            .iter()
            .filter_map(|&idx| hier.get_file_vector(idx).cloned())
            .collect();

        if dir_file_vectors.is_empty() {
            continue;
        }

        let mut dir_vector = mean_pool_vectors(&dir_file_vectors);
        l2_normalize(&mut dir_vector);

        hier.add_directory(dir_path, file_indices.clone(), vec![], dir_vector);
    }
}

pub fn build_hierarchical_index(chunks: &[CodeChunk], vectors: &[Vec<f32>]) -> HierarchicalIndex {
    let mut hier = compute_file_embeddings(chunks, vectors);
    compute_directory_embeddings(&mut hier);
    hier
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use uuid::Uuid;

    #[test]
    fn mean_pool_vectors_single_vector() {
        let vectors = vec![vec![1.0, 2.0, 3.0]];
        let result = mean_pool_vectors(&vectors);
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn mean_pool_vectors_multiple_vectors() {
        let vectors = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let result = mean_pool_vectors(&vectors);
        let expected: Vec<f32> = vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
        for (a, b) in result.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn mean_pool_vectors_empty_returns_empty() {
        let vectors: Vec<Vec<f32>> = vec![];
        let result = mean_pool_vectors(&vectors);
        assert!(result.is_empty());
    }

    #[test]
    fn l2_normalize_unit_vector() {
        let mut vec = vec![1.0, 0.0, 0.0];
        l2_normalize(&mut vec);
        assert_eq!(vec, vec![1.0, 0.0, 0.0]);
    }

    #[test]
    fn l2_normalize_scales_correctly() {
        let mut vec = vec![3.0, 4.0];
        l2_normalize(&mut vec);
        let magnitude: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((magnitude - 1.0).abs() < 1e-6);
        assert!((vec[0] - 0.6).abs() < 1e-6);
        assert!((vec[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn l2_normalize_zero_vector_unchanged() {
        let mut vec = vec![0.0, 0.0, 0.0];
        l2_normalize(&mut vec);
        assert_eq!(vec, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn compute_file_embeddings_groups_by_file() {
        let chunks = vec![
            CodeChunk {
                id: Uuid::new_v4(),
                path: PathBuf::from("a.rs"),
                language: "rust".to_string(),
                start_line: 1,
                end_line: 10,
                text: "fn a()".to_string(),
                hash: "h1".to_string(),
                modified_at: Utc::now(),
            },
            CodeChunk {
                id: Uuid::new_v4(),
                path: PathBuf::from("a.rs"),
                language: "rust".to_string(),
                start_line: 11,
                end_line: 20,
                text: "fn b()".to_string(),
                hash: "h2".to_string(),
                modified_at: Utc::now(),
            },
            CodeChunk {
                id: Uuid::new_v4(),
                path: PathBuf::from("b.rs"),
                language: "rust".to_string(),
                start_line: 1,
                end_line: 5,
                text: "fn c()".to_string(),
                hash: "h3".to_string(),
                modified_at: Utc::now(),
            },
        ];

        let vectors = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];

        let hier = compute_file_embeddings(&chunks, &vectors);

        assert_eq!(hier.files.len(), 2);
        assert_eq!(hier.file_vectors.len(), 2);

        let a_entry = hier.find_file_by_path(&PathBuf::from("a.rs"));
        assert!(a_entry.is_some());
        let (a_idx, a_file) = a_entry.unwrap();
        assert_eq!(a_file.chunk_indices.len(), 2);

        let a_vec = &hier.file_vectors[a_idx];
        let expected_norm = (0.5_f32.powi(2) + 0.5_f32.powi(2)).sqrt();
        assert!((a_vec[0] - 0.5 / expected_norm).abs() < 1e-5);
        assert!((a_vec[1] - 0.5 / expected_norm).abs() < 1e-5);
        assert!((a_vec[2] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn compute_directory_embeddings_groups_by_parent() {
        let mut hier = HierarchicalIndex::new();
        hier.add_file(PathBuf::from("src/a.rs"), vec![0], vec![1.0, 0.0, 0.0]);
        hier.add_file(PathBuf::from("src/b.rs"), vec![1], vec![0.0, 1.0, 0.0]);
        hier.add_file(PathBuf::from("lib.rs"), vec![2], vec![0.0, 0.0, 1.0]);

        compute_directory_embeddings(&mut hier);

        assert!(hier.directories.len() >= 1);

        let src_entry = hier.find_dir_by_path(&PathBuf::from("src"));
        assert!(src_entry.is_some());
        let (src_idx, src_dir) = src_entry.unwrap();
        assert_eq!(src_dir.file_indices.len(), 2);

        let src_vec = &hier.dir_vectors[src_idx];
        let expected_norm = (0.5_f32.powi(2) + 0.5_f32.powi(2)).sqrt();
        assert!((src_vec[0] - 0.5 / expected_norm).abs() < 1e-5);
        assert!((src_vec[1] - 0.5 / expected_norm).abs() < 1e-5);
    }
}
