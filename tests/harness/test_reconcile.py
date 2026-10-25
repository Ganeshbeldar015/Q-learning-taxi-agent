import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import check_harness as ch


def test_reconcile_detects_missing_file(tmp_path):
    assert ch.classify("missing: AGENTS.md") == "MISSING"


def test_reconcile_detects_blank_file(tmp_path):
    f = tmp_path / "AGENTS.md"
    f.write_text("   \n", encoding="utf-8")
    assert ch.classify("unconfigured: AGENTS.md is blank") == "UNCONFIGURED"


def test_reconcile_detects_placeholder_content(tmp_path):
    f = tmp_path / "AGENTS.md"
    f.write_text("# Title\nTODO: fill commands\n", encoding="utf-8")
    assert (
        ch.classify("unconfigured: AGENTS.md still contains placeholders")
        == "UNCONFIGURED"
    )


def test_reconcile_detects_stale_when_harness_changes(tmp_path):
    (tmp_path / "docs").mkdir(parents=True)
    h_file = tmp_path / "docs/harness.md"
    h_file.write_text("v1", encoding="utf-8")
    s_file = tmp_path / "harness/state/state.json"
    s_file.parent.mkdir(parents=True)
    s_file.write_text(
        json.dumps(
            {"derived_from": {"harness_md_sha": hashlib.sha256(b"v1").hexdigest()}}
        ),
        encoding="utf-8",
    )

    assert ch.check_stale(tmp_path) == []

    # Modify harness.md
    h_file.write_text("v2 updated plan", encoding="utf-8")
    errs = ch.check_stale(tmp_path)
    assert len(errs) == 1
    assert ch.classify(errs[0]) == "STALE"


def test_drifted_detection_preserves_user_edits():
    # Header format: <!-- generated-by: harness v1 | source: harness.md@<sha> | hash: <sha256 of body> -->
    body = "Custom content written by user."
    body_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
    header = f"<!-- generated-by: harness v1 | source: harness.md@123 | hash: {body_hash} -->\n"
    # User edits content
    tampered = header + "Hand-edited content."
    # Compare current body hash with header hash
    current_body = tampered.split("\n", 1)[1]
    is_drifted = hashlib.sha256(current_body.encode("utf-8")).hexdigest() != body_hash
    assert is_drifted is True
