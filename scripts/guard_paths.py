#!/usr/bin/env python3
"""Path and lane ownership guard for the harness."""

import argparse
import fnmatch
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_ALLOW_LISTS = {
    "orchestrator": ["harness/state/**", "harness/logs/**", "AGENTS.md", ".mcp.json"],
    "po-spec": ["harness/artifacts/*/01-spec.md", "harness/artifacts/*/handoff-*.json"],
    "designer": [
        "harness/artifacts/*/02-design.md",
        "docs/adr/**",
        "harness/artifacts/*/handoff-*.json",
    ],
    "tech-lead": [
        "harness/artifacts/*/03-plan.md",
        "docs/adr/**",
        "harness/artifacts/*/handoff-*.json",
    ],
    "test-author": [
        "tests/**",
        "harness/artifacts/*/04-tests.md",
        "harness/artifacts/*/handoff-*.json",
    ],
    "builder": [
        "src/**",
        "config/**",
        "harness/artifacts/*/05-build*.md",
        "harness/artifacts/*/handoff-*.json",
    ],
    "reviewer": [
        "harness/artifacts/*/06-review.md",
        "harness/artifacts/*/handoff-*.json",
    ],
    "security": [
        "harness/artifacts/*/06b-security.md",
        "harness/artifacts/*/handoff-*.json",
    ],
    "qa": ["harness/artifacts/*/07-qa.md", "harness/artifacts/*/handoff-*.json"],
    "scribe": [
        "docs/**",
        "README.md",
        "CHANGELOG.md",
        "report/**",
        "notebooks/**",
        "harness/artifacts/*/08-docs.md",
        "harness/artifacts/*/handoff-*.json",
    ],
    "librarian": ["docs/memory/**", "AGENTS.md", "harness/artifacts/*/handoff-*.json"],
}


def load_role_allow_list(role: str, root=ROOT):
    role_file = root / f"harness/roles/{role}.md"
    if role_file.exists():
        text = role_file.read_text(encoding="utf-8")
        m = re.search(
            r"^#+\s*Allowed paths.*?\n(.*?)(?=^#+\s|\Z)",
            text,
            re.MULTILINE | re.DOTALL | re.IGNORECASE,
        )
        if m:
            patterns = []
            for line in m.group(1).splitlines():
                line = line.strip().lstrip("-*` ").rstrip("` ")
                if line and not line.startswith("#"):
                    patterns.append(line)
            if patterns:
                return patterns
    return DEFAULT_ALLOW_LISTS.get(role, [])


def check_path_allowed(path: str, allow_patterns):
    path_norm = path.replace("\\", "/")
    for pattern in allow_patterns:
        pat_norm = pattern.replace("\\", "/")
        if pat_norm.endswith("/**"):
            base = pat_norm[:-3]
            if path_norm == base or path_norm.startswith(base + "/"):
                return True
        elif fnmatch.fnmatch(path_norm, pat_norm):
            return True
    return False


def check_paths(role: str, changed_files, root=ROOT):
    allow_patterns = load_role_allow_list(role, root)
    violations = []
    for f in changed_files:
        if not f.strip():
            continue
        rel = f.strip().replace("\\", "/")
        if not check_path_allowed(rel, allow_patterns):
            violations.append(rel)
    return violations


def main():
    parser = argparse.ArgumentParser(
        description="Guard file modifications by agent role."
    )
    parser.add_argument("--role", required=True, help="Agent role name")
    parser.add_argument("--files", nargs="*", default=[], help="List of modified files")
    parser.add_argument(
        "--file-list", help="Path to file containing list of changed files"
    )
    args = parser.parse_args()

    files = list(args.files)
    if args.file_list and Path(args.file_list).exists():
        files.extend(Path(args.file_list).read_text(encoding="utf-8").splitlines())

    violations = check_paths(args.role, files)
    if violations:
        print(f"GUARD VIOLATION: Role '{args.role}' is not allowed to modify:")
        for v in violations:
            print(f"  - {v}")
        return 1
    print(f"GUARD OK: All {len(files)} file modifications permitted for '{args.role}'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
