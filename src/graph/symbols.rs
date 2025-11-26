use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SymbolKind {
    Function,
    Method,
    Class,
    Struct,
    Interface,
    Trait,
    Enum,
    Type,
    Constant,
    Variable,
    Module,
    Namespace,
    Package,
    Field,
    Property,
}

impl SymbolKind {
    pub fn label(&self) -> &'static str {
        match self {
            SymbolKind::Function => "function",
            SymbolKind::Method => "method",
            SymbolKind::Class => "class",
            SymbolKind::Struct => "struct",
            SymbolKind::Interface => "interface",
            SymbolKind::Trait => "trait",
            SymbolKind::Enum => "enum",
            SymbolKind::Type => "type",
            SymbolKind::Constant => "constant",
            SymbolKind::Variable => "variable",
            SymbolKind::Module => "module",
            SymbolKind::Namespace => "namespace",
            SymbolKind::Package => "package",
            SymbolKind::Field => "field",
            SymbolKind::Property => "property",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EdgeKind {
    Imports,
    Exports,
    Calls,
    DefinedIn,
    References,
    Contains,
    Implements,
    Extends,
    TypeOf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Symbol {
    pub id: Uuid,
    pub name: String,
    pub qualified_name: String,
    pub kind: SymbolKind,
    pub file_path: PathBuf,
    pub start_line: usize,
    pub end_line: usize,
    pub language: String,
    pub signature: String,
    pub parent_id: Option<Uuid>,
    pub chunk_id: Option<Uuid>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub source_id: Uuid,
    pub target_id: Uuid,
    pub kind: EdgeKind,
    pub metadata: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportRelation {
    pub source_file: PathBuf,
    pub target_path: String,
    pub alias: Option<String>,
    pub line: usize,
    pub is_type_only: bool,
}

pub fn is_container_kind(kind: SymbolKind) -> bool {
    matches!(
        kind,
        SymbolKind::Class
            | SymbolKind::Struct
            | SymbolKind::Interface
            | SymbolKind::Trait
            | SymbolKind::Module
            | SymbolKind::Namespace
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbol_kind_labels() {
        assert_eq!(SymbolKind::Function.label(), "function");
        assert_eq!(SymbolKind::Method.label(), "method");
        assert_eq!(SymbolKind::Class.label(), "class");
        assert_eq!(SymbolKind::Struct.label(), "struct");
        assert_eq!(SymbolKind::Interface.label(), "interface");
        assert_eq!(SymbolKind::Trait.label(), "trait");
        assert_eq!(SymbolKind::Enum.label(), "enum");
        assert_eq!(SymbolKind::Type.label(), "type");
        assert_eq!(SymbolKind::Constant.label(), "constant");
        assert_eq!(SymbolKind::Variable.label(), "variable");
        assert_eq!(SymbolKind::Module.label(), "module");
        assert_eq!(SymbolKind::Namespace.label(), "namespace");
        assert_eq!(SymbolKind::Package.label(), "package");
        assert_eq!(SymbolKind::Field.label(), "field");
        assert_eq!(SymbolKind::Property.label(), "property");
    }

    #[test]
    fn test_is_container_kind() {
        assert!(is_container_kind(SymbolKind::Class));
        assert!(is_container_kind(SymbolKind::Struct));
        assert!(is_container_kind(SymbolKind::Interface));
        assert!(is_container_kind(SymbolKind::Trait));
        assert!(is_container_kind(SymbolKind::Module));
        assert!(is_container_kind(SymbolKind::Namespace));
        assert!(!is_container_kind(SymbolKind::Function));
        assert!(!is_container_kind(SymbolKind::Method));
        assert!(!is_container_kind(SymbolKind::Enum));
    }
}
