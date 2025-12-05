use std::cell::RefCell;
use std::collections::HashMap;
use tree_sitter::Parser;

use super::language::LanguageKind;

thread_local! {
    static PARSER_CACHE: RefCell<HashMap<LanguageKind, Parser>> = RefCell::new(HashMap::new());
}

pub fn with_parser<F, R>(lang: LanguageKind, f: F) -> Option<R>
where
    F: FnOnce(&mut Parser) -> R,
{
    let ts_lang = lang.language()?;

    PARSER_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        let parser = cache.entry(lang).or_insert_with(|| {
            let mut p = Parser::new();
            let _ = p.set_language(&ts_lang);
            p
        });
        Some(f(parser))
    })
}

#[cfg(test)]
pub fn clear_cache() {
    PARSER_CACHE.with(|cache| cache.borrow_mut().clear());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn returns_configured_parser_for_language() {
        clear_cache();
        let result = with_parser(LanguageKind::Rust, |parser| {
            parser.parse("fn main() {}", None).is_some()
        });
        assert_eq!(result, Some(true));
    }

    #[test]
    fn reuses_parser_for_same_language() {
        clear_cache();
        let result1 = with_parser(LanguageKind::Rust, |parser| {
            parser.parse("fn foo() {}", None).is_some()
        });
        let result2 = with_parser(LanguageKind::Rust, |parser| {
            parser.parse("fn bar() {}", None).is_some()
        });
        assert_eq!(result1, Some(true));
        assert_eq!(result2, Some(true));
    }

    #[test]
    fn provides_different_parsers_per_language() {
        clear_cache();
        let rust_result = with_parser(LanguageKind::Rust, |parser| {
            parser.parse("fn main() {}", None).is_some()
        });
        let python_result = with_parser(LanguageKind::Python, |parser| {
            parser.parse("def main(): pass", None).is_some()
        });
        assert_eq!(rust_result, Some(true));
        assert_eq!(python_result, Some(true));
    }

    #[test]
    fn survives_parse_errors() {
        clear_cache();
        let error_result = with_parser(LanguageKind::Rust, |parser| {
            parser.parse("{{{{", None).is_some()
        });
        let valid_result = with_parser(LanguageKind::Rust, |parser| {
            parser.parse("fn valid() {}", None).is_some()
        });
        assert!(error_result.is_some());
        assert_eq!(valid_result, Some(true));
    }

    #[test]
    fn thread_local_isolation() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        clear_cache();
        let success_count = Arc::new(AtomicUsize::new(0));

        let handles: Vec<_> = (0..4)
            .map(|_| {
                let counter = Arc::clone(&success_count);
                std::thread::spawn(move || {
                    let result = with_parser(LanguageKind::Rust, |parser| {
                        parser.parse("fn thread_test() {}", None).is_some()
                    });
                    if result == Some(true) {
                        counter.fetch_add(1, Ordering::SeqCst);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(success_count.load(Ordering::SeqCst), 4);
    }

    #[test]
    fn handles_all_supported_languages() {
        clear_cache();
        let languages = [
            (LanguageKind::Rust, "fn main() {}"),
            (LanguageKind::Python, "def main(): pass"),
            (LanguageKind::JavaScript, "function main() {}"),
            (LanguageKind::TypeScript, "function main(): void {}"),
            (LanguageKind::Tsx, "const App = () => <div/>"),
            (LanguageKind::Go, "func main() {}"),
            (LanguageKind::Java, "class Main {}"),
            (LanguageKind::C, "int main() {}"),
            (LanguageKind::Cpp, "int main() {}"),
            (LanguageKind::CSharp, "class Main {}"),
            (LanguageKind::Ruby, "def main; end"),
            (LanguageKind::Json, "{}"),
            (LanguageKind::Yaml, "key: value"),
            (LanguageKind::Toml, "key = \"value\""),
            (LanguageKind::Html, "<html></html>"),
            (LanguageKind::Css, "body {}"),
            (LanguageKind::Bash, "echo hello"),
            (LanguageKind::Markdown, "# Hello"),
        ];

        for (lang, code) in languages {
            let result = with_parser(lang, |parser| parser.parse(code, None).is_some());
            assert_eq!(result, Some(true), "Parser for {:?} failed", lang);
        }
    }
}
