#!/usr/bin/env python3
"""Deterministic linter for the harness. Exit 0 = healthy."""

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQUIRED = [
    "AGENTS.md",
    "docs/raw.md",
    "docs/harness.md",
    "docs/prompt.md",
    "harness/state/state.json",
    "harness/state/state.schema.json",
    "harness/state/budgets.json",
    "harness/tools/registry.yaml",
    "harness/tools/sandbox.md",
    "harness/workflows/pipeline.md",
    "harness/standards/handoff-contract.md",
    "harness/standards/definition-of-done.md",
    "harness/logs/events.jsonl",
]
HARNESS_HEADINGS = [
    "Problem Statement",
    "Goals & Non-Goals",
    "Users & Use Cases",
    "Requirements",
    "Acceptance Criteria",
    "Architecture & Tech Stack",
    "Project Structure",
    "Commands",
    "Coding & Testing Standards",
    "Tools, APIs & MCP Servers",
    "Autonomy & Approval Matrix",
    "Sprint Plan",
    "Risks & Mitigations",
    "Assumptions",
    "Open Questions",
    "Definition of Done",
]
ROLE_SECTIONS = [
    "Purpose",
    "Responsibilities",
    "Prerequisites",
    "Inputs",
    "Outputs",
    "Allowed paths",
    "Forbidden",
    "Process",
    "Iteration policy",
    "Completion criteria",
    "Checks to run",
    "Escalation",
]
BREAKER_KEYS = [
    "max_stage_retries",
    "max_identical_failures",
    "max_diff_lines_per_lane",
    "max_files_per_lane",
    "max_parallel_agents",
    "max_session_minutes",
    "max_tool_calls_per_stage",
    "no_progress_window",
]
STATE_KEYS = [
    "harness_version",
    "mode",
    "sprint",
    "current_stage",
    "attempts",
    "breaker",
    "updated_at",
]
SECRET_RE = re.compile(
    r"(AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{30,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)"
)


def check_required_files(root=ROOT):
    return [f"missing: {p}" for p in REQUIRED if not (root / p).exists()]


def check_harness_md(root=ROOT):
    p = root / "docs/harness.md"
    if not p.exists():
        return ["docs/harness.md missing"]
    text = p.read_text(encoding="utf-8")
    errs = [
        f"harness.md missing heading: {h}"
        for h in HARNESS_HEADINGS
        if not re.search(
            rf"^#+\s*(\d+\.\s*)?{re.escape(h)}", text, re.MULTILINE | re.IGNORECASE
        )
    ]
    m = re.search(
        r"^#+\s*(\d+\.\s*)?Open Questions\s*\n(.*?)(?=^#+\s|\Z)",
        text,
        re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )
    state = _load(root / "harness/state/state.json")
    if (
        m
        and state.get("sprint", 0) >= 1
        and re.search(
            r"\S", re.sub(r"<!--.*?-->|\(.*?\)", "", m.group(2), flags=re.DOTALL)
        )
    ):
        errs.append("harness.md Open Questions must be empty when sprint >= 1")
    return errs


def check_agents_and_roles(root=ROOT):
    errs, agents, roles = [], root / "harness/agents", root / "harness/roles"
    a_names = {p.stem for p in agents.glob("*.md")} if agents.exists() else set()
    r_names = {p.stem for p in roles.glob("*.md")} if roles.exists() else set()
    errs += [f"agent without role: {n}" for n in sorted(a_names - r_names)]
    errs += [f"role without agent: {n}" for n in sorted(r_names - a_names)]
    for n in sorted(a_names & r_names):
        a = (agents / f"{n}.md").read_text(encoding="utf-8")
        if len(a.splitlines()) > 40:
            errs.append(f"agent too long (>40 lines): {n}")
        if f"role_file: harness/roles/{n}.md" not in a:
            errs.append(f"agent {n} missing role_file front-matter")
        r = (roles / f"{n}.md").read_text(encoding="utf-8")
        errs += [
            f"role {n} missing section: {s}"
            for s in ROLE_SECTIONS
            if not re.search(rf"^#+\s*{re.escape(s)}", r, re.MULTILINE | re.IGNORECASE)
        ]
    return errs


def check_agents_md(root=ROOT):
    p = root / "AGENTS.md"
    if not p.exists():
        return ["AGENTS.md missing"]
    text = p.read_text(encoding="utf-8")
    errs = ["AGENTS.md exceeds 150 lines"] if len(text.splitlines()) > 150 else []
    for link in re.findall(r"\]\((?!https?://|#|mailto:)([^)\s]+)\)", text):
        if not (root / link.split("#")[0]).exists():
            errs.append(f"AGENTS.md broken link: {link}")
    return errs


def check_state(root=ROOT):
    s = _load(root / "harness/state/state.json")
    if not s:
        return ["state.json missing or invalid JSON"]
    errs = [f"state.json missing key: {k}" for k in STATE_KEYS if k not in s]
    if s.get("mode") not in {"DISCOVERY", "CURATION", "SCAFFOLD", "EXECUTE", "RETRO"}:
        errs.append("state.json invalid mode")
    return errs


def check_budgets(root=ROOT):
    b = _load(root / "harness/state/budgets.json")
    return [
        f"budgets.json bad/missing: {k}"
        for k in BREAKER_KEYS
        if not isinstance(b.get(k), int) or b[k] <= 0
    ]


def check_locks(root=ROOT):
    owners, errs = {}, []
    d = root / "harness/state/locks"
    for f in d.glob("*.json") if d.exists() else []:
        for path in _load(f).get("owns", []):
            if path in owners and owners[path] != f.name:
                errs.append(f"path {path} owned by both {owners[path]} and {f.name}")
            owners[path] = f.name
    return errs


def check_secrets(root=ROOT):
    errs = []
    for base in ("harness", "docs", "AGENTS.md", ".mcp.json"):
        p = root / base
        files = [p] if p.is_file() else list(p.rglob("*")) if p.exists() else []
        for f in files:
            if f.is_file() and f.suffix in {
                ".md",
                ".json",
                ".yaml",
                ".yml",
                ".jsonl",
                "",
            }:
                if f.name == "prompt.md":
                    continue
                if SECRET_RE.search(f.read_text(encoding="utf-8", errors="ignore")):
                    errs.append(f"possible secret in {f.relative_to(root)}")
    return errs


PLACEHOLDER_RE = re.compile(r"\b(TODO|TBD|FIXME)\b|<describe here>|\{\{[^}]*\}\}")
CONFIGURABLE = [
    "AGENTS.md",
    "harness/agents",
    "harness/roles",
    "harness/standards",
    "harness/workflows",
    "harness/tools",
    "harness/backlog.md",
]


def check_unconfigured(root=ROOT):
    """Existing-but-not-configured files: blank or still holding template placeholders."""
    errs = []
    for base in CONFIGURABLE:
        p = root / base
        files = [p] if p.is_file() else sorted(p.rglob("*")) if p.exists() else []
        for f in files:
            if not f.is_file():
                continue
            txt = f.read_text(encoding="utf-8", errors="ignore")
            rel = f.relative_to(root)
            if not txt.strip():
                errs.append(f"unconfigured: {rel} is blank")
            elif PLACEHOLDER_RE.search(txt):
                errs.append(f"unconfigured: {rel} still contains placeholders")
    return errs


def check_stale(root=ROOT):
    """Derived files older than the problem statement / plan they were generated from."""
    d = _load(root / "harness/state/state.json").get("derived_from") or {}
    errs = []
    for key, rel in (
        ("raw_md_sha", "docs/raw.md"),
        ("harness_md_sha", "docs/harness.md"),
    ):
        f = root / rel
        if (
            key in d
            and f.exists()
            and d[key] != hashlib.sha256(f.read_bytes()).hexdigest()
        ):
            errs.append(
                f"stale: {rel} changed since harness was generated (re-run reconcile)"
            )
    return errs


def _load(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


CHECKS = [
    check_required_files,
    check_harness_md,
    check_agents_and_roles,
    check_agents_md,
    check_state,
    check_budgets,
    check_locks,
    check_secrets,
    check_unconfigured,
    check_stale,
]


def classify(msg):
    head = msg.split(":", 1)[0].lower()
    return {"missing": "MISSING", "unconfigured": "UNCONFIGURED", "stale": "STALE"}.get(
        head, "INVALID"
    )


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    failures = [e for c in CHECKS for e in c()]
    if (
        "--json" in argv
    ):  # machine-readable gap list consumed by the Orchestrator's reconcile step
        print(
            json.dumps(
                [{"status": classify(e), "detail": e} for e in failures], indent=2
            )
        )
        return 1 if failures else 0
    for e in failures:
        print(f"FAIL  {e}")
    print(
        f"{'OK' if not failures else 'HARNESS UNHEALTHY'} — {len(failures)} problem(s)"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
