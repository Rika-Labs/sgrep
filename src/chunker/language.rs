use std::path::Path;

use tree_sitter::Language;

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub enum LanguageKind {
    Rust,
    Python,
    JavaScript,
    TypeScript,
    Tsx,
    Go,
    Java,
    C,
    Cpp,
    CSharp,
    Ruby,
    Markdown,
    Json,
    Yaml,
    Toml,
    Html,
    Css,
    Bash,
}

impl LanguageKind {
    pub fn label(&self) -> &'static str {
        match self {
            LanguageKind::Rust => "rust",
            LanguageKind::Python => "python",
            LanguageKind::JavaScript => "javascript",
            LanguageKind::TypeScript => "typescript",
            LanguageKind::Tsx => "tsx",
            LanguageKind::Go => "go",
            LanguageKind::Java => "java",
            LanguageKind::C => "c",
            LanguageKind::Cpp => "cpp",
            LanguageKind::CSharp => "csharp",
            LanguageKind::Ruby => "ruby",
            LanguageKind::Markdown => "markdown",
            LanguageKind::Json => "json",
            LanguageKind::Yaml => "yaml",
            LanguageKind::Toml => "toml",
            LanguageKind::Html => "html",
            LanguageKind::Css => "css",
            LanguageKind::Bash => "bash",
        }
    }

    pub fn language(&self) -> Option<Language> {
        match self {
            LanguageKind::Rust => Some(tree_sitter_rust::LANGUAGE.into()),
            LanguageKind::Python => Some(tree_sitter_python::LANGUAGE.into()),
            LanguageKind::JavaScript => Some(tree_sitter_javascript::LANGUAGE.into()),
            LanguageKind::TypeScript => Some(tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()),
            LanguageKind::Tsx => Some(tree_sitter_typescript::LANGUAGE_TSX.into()),
            LanguageKind::Go => Some(tree_sitter_go::LANGUAGE.into()),
            LanguageKind::Java => Some(tree_sitter_java::LANGUAGE.into()),
            LanguageKind::C => Some(tree_sitter_c::LANGUAGE.into()),
            LanguageKind::Cpp => Some(tree_sitter_cpp::LANGUAGE.into()),
            LanguageKind::CSharp => Some(tree_sitter_c_sharp::LANGUAGE.into()),
            LanguageKind::Ruby => Some(tree_sitter_ruby::LANGUAGE.into()),
            LanguageKind::Markdown => Some(tree_sitter_md::LANGUAGE.into()),
            LanguageKind::Json => Some(tree_sitter_json::LANGUAGE.into()),
            LanguageKind::Yaml => Some(tree_sitter_yaml::LANGUAGE.into()),
            LanguageKind::Toml => Some(tree_sitter_toml::LANGUAGE.into()),
            LanguageKind::Html => Some(tree_sitter_html::LANGUAGE.into()),
            LanguageKind::Css => Some(tree_sitter_css::LANGUAGE.into()),
            LanguageKind::Bash => Some(tree_sitter_bash::LANGUAGE.into()),
        }
    }
}

pub fn detect_language(path: &Path) -> Option<LanguageKind> {
    let ext = path.extension()?.to_string_lossy().to_ascii_lowercase();
    match ext.as_str() {
        "rs" => Some(LanguageKind::Rust),
        "py" => Some(LanguageKind::Python),
        "ts" => Some(LanguageKind::TypeScript),
        "tsx" => Some(LanguageKind::Tsx),
        "js" | "jsx" => Some(LanguageKind::JavaScript),
        "go" => Some(LanguageKind::Go),
        "java" => Some(LanguageKind::Java),
        "c" | "h" => Some(LanguageKind::C),
        "cpp" | "cc" | "cxx" | "hpp" | "hh" | "hxx" => Some(LanguageKind::Cpp),
        "cs" => Some(LanguageKind::CSharp),
        "rb" => Some(LanguageKind::Ruby),
        "md" | "markdown" => Some(LanguageKind::Markdown),
        "json" => Some(LanguageKind::Json),
        "yaml" | "yml" => Some(LanguageKind::Yaml),
        "toml" => Some(LanguageKind::Toml),
        "html" | "htm" => Some(LanguageKind::Html),
        "css" => Some(LanguageKind::Css),
        "sh" | "bash" => Some(LanguageKind::Bash),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_language_handles_rust_extension() {
        let lang = detect_language(Path::new("main.rs"));
        assert!(matches!(lang, Some(LanguageKind::Rust)));
    }

    #[test]
    fn detect_language_covers_common_extensions() {
        let cases = [
            ("file.ts", "typescript"),
            ("component.tsx", "tsx"),
            ("script.js", "javascript"),
            ("script.jsx", "javascript"),
            ("main.c", "c"),
            ("main.go", "go"),
            ("main.java", "java"),
            ("header.hpp", "cpp"),
            ("class.cs", "csharp"),
            ("tool.rb", "ruby"),
            ("doc.md", "markdown"),
            ("data.json", "json"),
            ("config.yaml", "yaml"),
            ("config.toml", "toml"),
            ("index.html", "html"),
            ("style.css", "css"),
            ("script.sh", "bash"),
        ];

        for (file, expected_label) in cases {
            let lang = detect_language(Path::new(file)).expect("language detected");
            assert_eq!(lang.label(), expected_label);
            assert!(lang.language().is_some());
        }
    }
}
