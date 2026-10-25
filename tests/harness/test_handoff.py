import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import validate_handoff as vh


def test_valid_handoff_passes(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/harness.md").write_text("AC-001-a\nAC-001-b\n", encoding="utf-8")
    (tmp_path / "art.md").write_text("# Spec\n", encoding="utf-8")

    data = {
        "task_id": "T-0001",
        "from": "po-spec",
        "to": "designer",
        "artifact": "art.md",
        "status": "complete",
        "summary": "Short summary of work done.",
        "inputs_consumed": ["task"],
        "acceptance_refs": ["AC-001-a"],
        "open_questions": [],
        "assumptions": [],
        "files_touched": ["art.md"],
        "checks_run": [],
        "next_expected_artifact": "02-design.md",
    }
    ok, errs = vh.validate_handoff(data, root=tmp_path)
    assert ok, errs


def test_handoff_rejected_if_status_not_complete(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/harness.md").write_text("AC-001-a\n", encoding="utf-8")
    (tmp_path / "art.md").write_text("spec", encoding="utf-8")

    data = {
        "task_id": "T-0001",
        "from": "po-spec",
        "to": "designer",
        "artifact": "art.md",
        "status": "in_progress",
        "summary": "Almost done",
        "inputs_consumed": [],
        "acceptance_refs": ["AC-001-a"],
        "open_questions": [],
        "assumptions": [],
        "files_touched": [],
        "checks_run": [],
        "next_expected_artifact": "02-design.md",
    }
    ok, errs = vh.validate_handoff(data, root=tmp_path)
    assert not ok
    assert any("must be 'complete'" in e for e in errs)


def test_handoff_rejected_if_missing_ac_in_harness(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/harness.md").write_text("AC-001-a\n", encoding="utf-8")
    (tmp_path / "art.md").write_text("spec", encoding="utf-8")

    data = {
        "task_id": "T-0001",
        "from": "po-spec",
        "to": "designer",
        "artifact": "art.md",
        "status": "complete",
        "summary": "Done",
        "inputs_consumed": [],
        "acceptance_refs": ["AC-NONEXISTENT"],
        "open_questions": [],
        "assumptions": [],
        "files_touched": [],
        "checks_run": [],
        "next_expected_artifact": "02-design.md",
    }
    ok, errs = vh.validate_handoff(data, root=tmp_path)
    assert not ok
    assert any("not defined" in e for e in errs)


def test_builder_must_have_passing_checks(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/harness.md").write_text("AC-001-a\n", encoding="utf-8")
    (tmp_path / "art.md").write_text("build", encoding="utf-8")

    data = {
        "task_id": "T-0001",
        "from": "builder",
        "to": "reviewer",
        "artifact": "art.md",
        "status": "complete",
        "summary": "Built",
        "inputs_consumed": [],
        "acceptance_refs": ["AC-001-a"],
        "open_questions": [],
        "assumptions": [],
        "files_touched": [],
        "checks_run": [],
        "next_expected_artifact": "06-review.md",
    }
    ok, errs = vh.validate_handoff(data, root=tmp_path)
    assert not ok
    assert any("must record at least one check" in e for e in errs)
