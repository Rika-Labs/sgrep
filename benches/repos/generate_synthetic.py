#!/usr/bin/env python3
"""
Generate synthetic code repositories for benchmarking.

Usage:
    ./generate_synthetic.py 1000    # Generate 1K files
    ./generate_synthetic.py 10000   # Generate 10K files
    ./generate_synthetic.py 50000   # Generate 50K files
"""

import argparse
import random
import string
from pathlib import Path
from typing import List

# Language templates with realistic code patterns
LANGUAGE_TEMPLATES = {
    "rust": {
        "ext": ".rs",
        "weight": 0.3,
        "templates": [
            """// {module_name} module
use std::collections::HashMap;
use std::sync::Arc;

/// {doc_comment}
pub struct {StructName} {{
    id: u64,
    name: String,
    data: HashMap<String, String>,
}}

impl {StructName} {{
    /// Creates a new instance
    pub fn new(id: u64, name: String) -> Self {{
        Self {{
            id,
            name,
            data: HashMap::new(),
        }}
    }}

    /// Processes the data
    pub fn process(&mut self) -> Result<(), Error> {{
        // TODO: Implement processing logic
        Ok(())
    }}

    /// Returns the ID
    pub fn id(&self) -> u64 {{
        self.id
    }}
}}

#[cfg(test)]
mod tests {{
    use super::*;

    #[test]
    fn test_new() {{
        let instance = {StructName}::new(1, "test".to_string());
        assert_eq!(instance.id(), 1);
    }}
}}
""",
        ],
    },
    "python": {
        "ext": ".py",
        "weight": 0.3,
        "templates": [
            """\"\"\"
{module_name} module

This module provides {description}.
\"\"\"

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class {ClassName}:
    \"\"\"Represents a {entity}.\"\"\"

    id: int
    name: str
    metadata: Dict[str, str]

    def process(self) -> None:
        \"\"\"Process the {entity}.\"\"\"
        logger.info(f"Processing {{self.name}}")
        # TODO: Implement processing logic
        pass

    def validate(self) -> bool:
        \"\"\"Validate the {entity} data.\"\"\"
        if not self.name:
            return False
        return True


def create_{entity}(id: int, name: str) -> {ClassName}:
    \"\"\"Create a new {entity} instance.\"\"\"
    return {ClassName}(id=id, name=name, metadata={{}})


if __name__ == "__main__":
    # Example usage
    instance = create_{entity}(1, "example")
    instance.process()
""",
        ],
    },
    "javascript": {
        "ext": ".js",
        "weight": 0.2,
        "templates": [
            """/**
 * {module_name} module
 * @module {module_name}
 */

const logger = require('./logger');

/**
 * Represents a {entity}
 */
class {ClassName} {{
  constructor(id, name) {{
    this.id = id;
    this.name = name;
    this.metadata = {{}};
  }}

  /**
   * Process the {entity}
   */
  async process() {{
    logger.info(`Processing ${{this.name}}`);
    // TODO: Implement processing logic
  }}

  /**
   * Validate the {entity} data
   * @returns {{boolean}} True if valid
   */
  validate() {{
    return !!this.name;
  }}
}}

/**
 * Create a new {entity}
 * @param {{number}} id - The entity ID
 * @param {{string}} name - The entity name
 * @returns {{{ClassName}}}
 */
function create{ClassName}(id, name) {{
  return new {ClassName}(id, name);
}}

module.exports = {{ {ClassName}, create{ClassName} }};
""",
        ],
    },
    "go": {
        "ext": ".go",
        "weight": 0.1,
        "templates": [
            """package {package_name}

import (
\t"fmt"
\t"log"
)

// {StructName} represents a {entity}
type {StructName} struct {{
\tID       int64
\tName     string
\tMetadata map[string]string
}}

// New{StructName} creates a new instance
func New{StructName}(id int64, name string) *{StructName} {{
\treturn &{StructName}{{
\t\tID:       id,
\t\tName:     name,
\t\tMetadata: make(map[string]string),
\t}}
}}

// Process processes the {entity}
func (s *{StructName}) Process() error {{
\tlog.Printf("Processing %s", s.Name)
\t// TODO: Implement processing logic
\treturn nil
}}

// Validate validates the {entity} data
func (s *{StructName}) Validate() bool {{
\tif s.Name == "" {{
\t\treturn false
\t}}
\treturn true
}}

// String returns a string representation
func (s *{StructName}) String() string {{
\treturn fmt.Sprintf("{StructName}{{ID: %d, Name: %s}}", s.ID, s.Name)
}}
""",
        ],
    },
    "typescript": {
        "ext": ".ts",
        "weight": 0.1,
        "templates": [
            """/**
 * {module_name} module
 */

import {{ Logger }} from './logger';

/**
 * Represents a {entity}
 */
export interface I{ClassName} {{
  id: number;
  name: string;
  metadata: Record<string, string>;
}}

/**
 * {ClassName} implementation
 */
export class {ClassName} implements I{ClassName} {{
  public id: number;
  public name: string;
  public metadata: Record<string, string>;

  constructor(id: number, name: string) {{
    this.id = id;
    this.name = name;
    this.metadata = {{}};
  }}

  /**
   * Process the {entity}
   */
  async process(): Promise<void> {{
    Logger.info(`Processing ${{this.name}}`);
    // TODO: Implement processing logic
  }}

  /**
   * Validate the {entity} data
   */
  validate(): boolean {{
    return !!this.name;
  }}
}}

/**
 * Factory function to create a {entity}
 */
export function create{ClassName}(id: number, name: string): {ClassName} {{
  return new {ClassName}(id, name);
}}
""",
        ],
    },
}


def random_identifier(prefix: str = "") -> str:
    """Generate a random identifier."""
    suffix = "".join(random.choices(string.ascii_lowercase, k=8))
    return f"{prefix}{suffix}" if prefix else suffix


def generate_file_content(language: str) -> str:
    """Generate random file content for a given language."""
    config = LANGUAGE_TEMPLATES[language]
    template = random.choice(config["templates"])

    # Fill in template variables
    replacements = {
        "module_name": random_identifier("module_"),
        "StructName": random_identifier("Struct").title(),
        "ClassName": random_identifier("Class").title(),
        "entity": random.choice(["user", "task", "item", "record", "entity"]),
        "doc_comment": f"Documentation for {random_identifier()}",
        "description": f"functionality for {random_identifier()}",
        "package_name": random_identifier("pkg"),
    }

    content = template
    for key, value in replacements.items():
        content = content.replace(f"{{{key}}}", value)

    return content


def select_language() -> str:
    """Select a random language based on weights."""
    languages = list(LANGUAGE_TEMPLATES.keys())
    weights = [LANGUAGE_TEMPLATES[lang]["weight"] for lang in languages]
    return random.choices(languages, weights=weights)[0]


def generate_directory_structure(base_path: Path, file_count: int) -> None:
    """Generate a realistic directory structure with files."""

    # Create common directory structure
    directories = [
        "src",
        "src/core",
        "src/utils",
        "src/services",
        "src/models",
        "src/controllers",
        "src/middleware",
        "tests",
        "tests/unit",
        "tests/integration",
        "lib",
        "lib/helpers",
        "config",
        "scripts",
    ]

    for dir_name in directories:
        (base_path / dir_name).mkdir(parents=True, exist_ok=True)

    # Generate files distributed across directories
    files_per_dir = file_count // len(directories)
    remaining = file_count % len(directories)

    file_num = 0
    for i, dir_name in enumerate(directories):
        num_files = files_per_dir + (1 if i < remaining else 0)

        for _ in range(num_files):
            language = select_language()
            ext = LANGUAGE_TEMPLATES[language]["ext"]
            filename = f"file_{file_num:06d}{ext}"
            file_path = base_path / dir_name / filename

            content = generate_file_content(language)
            file_path.write_text(content)

            file_num += 1

            if file_num % 1000 == 0:
                print(f"Generated {file_num}/{file_count} files...")

    # Add some common config files
    (base_path / "README.md").write_text(f"# Synthetic Repository ({file_count} files)\n\nGenerated for benchmarking.\n")
    (base_path / ".gitignore").write_text("target/\nnode_modules/\n*.pyc\n")

    print(f"✅ Generated {file_count} files in {base_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic code repositories")
    parser.add_argument("file_count", type=int, help="Number of files to generate")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: benches/repos/synthetic_N)",
    )

    args = parser.parse_args()

    if args.output:
        output_dir = args.output
    else:
        script_dir = Path(__file__).parent
        output_dir = script_dir / f"synthetic_{args.file_count}"

    print(f"Generating {args.file_count} files in {output_dir}...")

    # Clean up existing directory
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    generate_directory_structure(output_dir, args.file_count)


if __name__ == "__main__":
    main()
