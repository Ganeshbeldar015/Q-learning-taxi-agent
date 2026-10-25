import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import check_harness as ch

ROLE = "\n".join(f"## {s}\n-" for s in ch.ROLE_SECTIONS)
AGENT = "---\nname: builder\nrole_file: harness/roles/builder.md\n---\nPointer.\n"


def make(root: Path):
    (root / "harness/agents").mkdir(parents=True)
    (root / "harness/roles").mkdir(parents=True)
    (root / "harness/state/locks").mkdir(parents=True)
    (root / "docs").mkdir(parents=True)
    (root / "harness/agents/builder.md").write_text(AGENT, encoding="utf-8")
    (root / "harness/roles/builder.md").write_text(ROLE, encoding="utf-8")
    (root / "harness/state/budgets.json").write_text(
        json.dumps({k: 1 for k in ch.BREAKER_KEYS}), encoding="utf-8"
    )
    (root / "harness/state/state.json").write_text(
        json.dumps(
            {
                "harness_version": "1",
                "mode": "EXECUTE",
                "sprint": 1,
                "current_stage": "build",
                "attempts": {},
                "breaker": {"tripped": False},
                "updated_at": "now",
            }
        ),
        encoding="utf-8",
    )
    (root / "AGENTS.md").write_text(
        "# Index\n[roles](harness/roles/builder.md)\n", encoding="utf-8"
    )
    (root / "docs/harness.md").write_text(
        "\n".join(
            f"## {i}. {h}\n" + ("" if h == "Open Questions" else "x")
            for i, h in enumerate(ch.HARNESS_HEADINGS, 1)
        ),
        encoding="utf-8",
    )
    return root


def test_valid_minimal_harness_passes_targeted_checks(tmp_path):
    r = make(tmp_path)
    for c in (
        ch.check_harness_md,
        ch.check_agents_and_roles,
        ch.check_agents_md,
        ch.check_state,
        ch.check_budgets,
        ch.check_locks,
    ):
        assert c(r) == [], c.__name__


def test_missing_required_files_detected(tmp_path):
    assert any("missing" in e for e in ch.check_required_files(tmp_path))


def test_agent_without_role_detected(tmp_path):
    r = make(tmp_path)
    (r / "harness/roles/builder.md").unlink()
    assert any("agent without role" in e for e in ch.check_agents_and_roles(r))


def test_role_missing_section_detected(tmp_path):
    r = make(tmp_path)
    (r / "harness/roles/builder.md").write_text("## Purpose\n", encoding="utf-8")
    assert any("missing section" in e for e in ch.check_agents_and_roles(r))


def test_fat_agent_rejected(tmp_path):
    r = make(tmp_path)
    (r / "harness/agents/builder.md").write_text(AGENT + "\n" * 50, encoding="utf-8")
    assert any("too long" in e for e in ch.check_agents_and_roles(r))


def test_agents_md_broken_link_and_length(tmp_path):
    r = make(tmp_path)
    (r / "AGENTS.md").write_text("[x](nope.md)\n" + "l\n" * 160, encoding="utf-8")
    errs = ch.check_agents_md(r)
    assert any("broken link" in e for e in errs) and any("150" in e for e in errs)


def test_open_questions_block_sprint_one(tmp_path):
    r = make(tmp_path)
    txt = (
        (r / "docs/harness.md")
        .read_text(encoding="utf-8")
        .replace("## 15. Open Questions\n", "## 15. Open Questions\nWhich DB?\n")
    )
    (r / "docs/harness.md").write_text(txt, encoding="utf-8")
    assert any("Open Questions" in e for e in ch.check_harness_md(r))


def test_bad_state_mode_detected(tmp_path):
    r = make(tmp_path)
    s = json.loads((r / "harness/state/state.json").read_text(encoding="utf-8"))
    s["mode"] = "YOLO"
    (r / "harness/state/state.json").write_text(json.dumps(s), encoding="utf-8")
    assert any("invalid mode" in e for e in ch.check_state(r))


def test_budget_must_be_positive_ints(tmp_path):
    r = make(tmp_path)
    b = {k: 1 for k in ch.BREAKER_KEYS}
    b["max_stage_retries"] = 0
    (r / "harness/state/budgets.json").write_text(json.dumps(b), encoding="utf-8")
    assert any("max_stage_retries" in e for e in ch.check_budgets(r))


def test_overlapping_lane_locks_detected(tmp_path):
    r = make(tmp_path)
    for n in ("a", "b"):
        (r / f"harness/state/locks/{n}.json").write_text(
            json.dumps({"owns": ["src/x.py"]}), encoding="utf-8"
        )
    assert ch.check_locks(r)


def test_secret_detection(tmp_path):
    r = make(tmp_path)
    (r / "docs/leak.md").write_text("key=AKIAABCDEFGHIJKLMNOP", encoding="utf-8")
    assert ch.check_secrets(r)


def test_placeholder_and_blank_files_flagged_unconfigured(tmp_path):
    r = make(tmp_path)
    (r / "AGENTS.md").write_text("# Index\nTODO fill commands\n", encoding="utf-8")
    (r / "harness/roles/builder.md").write_text("", encoding="utf-8")
    errs = ch.check_unconfigured(r)
    assert any("AGENTS.md" in e for e in errs) and any("blank" in e for e in errs)


def test_configured_files_not_flagged(tmp_path):
    assert ch.check_unconfigured(make(tmp_path)) == []


def test_stale_when_problem_statement_changes(tmp_path):
    import hashlib

    r = make(tmp_path)
    (r / "docs/raw.md").write_text("v1", encoding="utf-8")
    s = json.loads((r / "harness/state/state.json").read_text(encoding="utf-8"))
    s["derived_from"] = {"raw_md_sha": hashlib.sha256(b"v1").hexdigest()}
    (r / "harness/state/state.json").write_text(json.dumps(s), encoding="utf-8")
    assert ch.check_stale(r) == []
    (r / "docs/raw.md").write_text("v2 — requirements changed", encoding="utf-8")
    assert any(e.startswith("stale") for e in ch.check_stale(r))


def test_classify_maps_messages_to_reconcile_statuses():
    assert ch.classify("missing: AGENTS.md") == "MISSING"
    assert ch.classify("unconfigured: AGENTS.md is blank") == "UNCONFIGURED"
    assert ch.classify("stale: docs/raw.md changed") == "STALE"
    assert ch.classify("state.json invalid mode") == "INVALID"


def test_json_gap_list_for_reconcile(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ch, "ROOT", tmp_path)
    monkeypatch.setattr(ch, "CHECKS", [lambda: ch.check_required_files(tmp_path)])
    assert ch.main(["--json"]) == 1
    out = json.loads(capsys.readouterr().out)
    assert out and all(o["status"] == "MISSING" for o in out)
