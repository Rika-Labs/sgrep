use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileEntry {
    pub path: PathBuf,
    pub chunk_indices: Vec<usize>,
    pub vector_offset: usize,
}

impl FileEntry {
    pub fn chunk_count(&self) -> usize {
        self.chunk_indices.len()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectoryEntry {
    pub path: PathBuf,
    pub file_indices: Vec<usize>,
    pub child_dir_indices: Vec<usize>,
    pub vector_offset: usize,
}

#[allow(dead_code)]
impl DirectoryEntry {
    pub fn file_count(&self) -> usize {
        self.file_indices.len()
    }

    pub fn child_dir_count(&self) -> usize {
        self.child_dir_indices.len()
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct HierarchicalStats {
    pub file_count: usize,
    pub directory_count: usize,
    pub total_chunks_in_files: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HierarchicalIndex {
    pub files: Vec<FileEntry>,
    pub directories: Vec<DirectoryEntry>,
    #[serde(skip)]
    pub file_vectors: Vec<Vec<f32>>,
    #[serde(skip)]
    pub dir_vectors: Vec<Vec<f32>>,
}

#[allow(dead_code)]
impl HierarchicalIndex {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_file(&mut self, path: PathBuf, chunk_indices: Vec<usize>, vector: Vec<f32>) {
        let vector_offset = self.file_vectors.len();
        self.files.push(FileEntry {
            path,
            chunk_indices,
            vector_offset,
        });
        self.file_vectors.push(vector);
    }

    pub fn add_directory(
        &mut self,
        path: PathBuf,
        file_indices: Vec<usize>,
        child_dir_indices: Vec<usize>,
        vector: Vec<f32>,
    ) {
        let vector_offset = self.dir_vectors.len();
        self.directories.push(DirectoryEntry {
            path,
            file_indices,
            child_dir_indices,
            vector_offset,
        });
        self.dir_vectors.push(vector);
    }

    pub fn get_file_vector(&self, idx: usize) -> Option<&Vec<f32>> {
        self.file_vectors.get(idx)
    }

    pub fn get_dir_vector(&self, idx: usize) -> Option<&Vec<f32>> {
        self.dir_vectors.get(idx)
    }

    pub fn find_file_by_path(&self, path: &Path) -> Option<(usize, &FileEntry)> {
        self.files.iter().enumerate().find(|(_, f)| f.path == path)
    }

    pub fn find_dir_by_path(&self, path: &Path) -> Option<(usize, &DirectoryEntry)> {
        self.directories
            .iter()
            .enumerate()
            .find(|(_, d)| d.path == path)
    }

    pub fn stats(&self) -> HierarchicalStats {
        HierarchicalStats {
            file_count: self.files.len(),
            directory_count: self.directories.len(),
            total_chunks_in_files: self.files.iter().map(|f| f.chunk_count()).sum(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.files.is_empty() && self.directories.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_entry_creation() {
        let entry = FileEntry {
            path: PathBuf::from("src/lib.rs"),
            chunk_indices: vec![0, 1, 2],
            vector_offset: 0,
        };

        assert_eq!(entry.path, PathBuf::from("src/lib.rs"));
        assert_eq!(entry.chunk_indices.len(), 3);
        assert_eq!(entry.chunk_count(), 3);
    }

    #[test]
    fn directory_entry_creation() {
        let entry = DirectoryEntry {
            path: PathBuf::from("src/auth"),
            file_indices: vec![0, 1],
            child_dir_indices: vec![],
            vector_offset: 0,
        };

        assert_eq!(entry.path, PathBuf::from("src/auth"));
        assert_eq!(entry.file_count(), 2);
        assert_eq!(entry.child_dir_count(), 0);
    }

    #[test]
    fn directory_entry_with_nested_dirs() {
        let entry = DirectoryEntry {
            path: PathBuf::from("src"),
            file_indices: vec![0],
            child_dir_indices: vec![0, 1, 2],
            vector_offset: 0,
        };

        assert_eq!(entry.file_count(), 1);
        assert_eq!(entry.child_dir_count(), 3);
    }

    #[test]
    fn hierarchical_index_empty() {
        let hier = HierarchicalIndex::new();

        assert!(hier.files.is_empty());
        assert!(hier.directories.is_empty());
        assert!(hier.file_vectors.is_empty());
        assert!(hier.dir_vectors.is_empty());
    }

    #[test]
    fn hierarchical_index_add_file() {
        let mut hier = HierarchicalIndex::new();
        let vector = vec![0.1, 0.2, 0.3];

        hier.add_file(PathBuf::from("src/main.rs"), vec![0, 1], vector.clone());

        assert_eq!(hier.files.len(), 1);
        assert_eq!(hier.file_vectors.len(), 1);
        assert_eq!(hier.files[0].path, PathBuf::from("src/main.rs"));
        assert_eq!(hier.file_vectors[0], vector);
    }

    #[test]
    fn hierarchical_index_add_directory() {
        let mut hier = HierarchicalIndex::new();
        let vector = vec![0.5, 0.5, 0.5];

        hier.add_directory(PathBuf::from("src"), vec![0, 1], vec![], vector.clone());

        assert_eq!(hier.directories.len(), 1);
        assert_eq!(hier.dir_vectors.len(), 1);
        assert_eq!(hier.directories[0].path, PathBuf::from("src"));
        assert_eq!(hier.dir_vectors[0], vector);
    }

    #[test]
    fn hierarchical_index_get_file_vector() {
        let mut hier = HierarchicalIndex::new();
        hier.add_file(PathBuf::from("a.rs"), vec![0], vec![1.0, 0.0, 0.0]);
        hier.add_file(PathBuf::from("b.rs"), vec![1], vec![0.0, 1.0, 0.0]);

        assert_eq!(hier.get_file_vector(0), Some(&vec![1.0, 0.0, 0.0]));
        assert_eq!(hier.get_file_vector(1), Some(&vec![0.0, 1.0, 0.0]));
        assert_eq!(hier.get_file_vector(2), None);
    }

    #[test]
    fn hierarchical_index_get_dir_vector() {
        let mut hier = HierarchicalIndex::new();
        hier.add_directory(PathBuf::from("src"), vec![], vec![], vec![0.5, 0.5, 0.0]);

        assert_eq!(hier.get_dir_vector(0), Some(&vec![0.5, 0.5, 0.0]));
        assert_eq!(hier.get_dir_vector(1), None);
    }

    #[test]
    fn hierarchical_index_find_file_by_path() {
        let mut hier = HierarchicalIndex::new();
        hier.add_file(PathBuf::from("src/main.rs"), vec![0], vec![1.0, 0.0, 0.0]);
        hier.add_file(PathBuf::from("src/lib.rs"), vec![1], vec![0.0, 1.0, 0.0]);

        let found = hier.find_file_by_path(&PathBuf::from("src/lib.rs"));
        assert!(found.is_some());
        assert_eq!(found.unwrap().0, 1);

        let not_found = hier.find_file_by_path(&PathBuf::from("src/other.rs"));
        assert!(not_found.is_none());
    }

    #[test]
    fn hierarchical_index_stats() {
        let mut hier = HierarchicalIndex::new();
        hier.add_file(PathBuf::from("a.rs"), vec![0, 1, 2], vec![1.0, 0.0, 0.0]);
        hier.add_file(PathBuf::from("b.rs"), vec![3, 4], vec![0.0, 1.0, 0.0]);
        hier.add_directory(
            PathBuf::from("src"),
            vec![0, 1],
            vec![],
            vec![0.5, 0.5, 0.0],
        );

        let stats = hier.stats();
        assert_eq!(stats.file_count, 2);
        assert_eq!(stats.directory_count, 1);
        assert_eq!(stats.total_chunks_in_files, 5);
    }
}
