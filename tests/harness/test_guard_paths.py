import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import guard_paths as gp


def test_allowed_path_passes():
    files = [
        "src/agent.py",
        "config/default.yaml",
        "harness/artifacts/T-0001/05-build-main.md",
    ]
    violations = gp.check_paths("builder", files)
    assert violations == []


def test_disallowed_path_violates():
    files = ["docs/raw.md", "src/agent.py"]
    violations = gp.check_paths("builder", files)
    assert "docs/raw.md" in violations
    assert "src/agent.py" not in violations


def test_po_spec_allowed_scope():
    files = [
        "harness/artifacts/T-0001/01-spec.md",
        "harness/artifacts/T-0001/handoff-po-spec-to-designer.json",
    ]
    violations = gp.check_paths("po-spec", files)
    assert violations == []


def test_po_spec_cannot_touch_source():
    violations = gp.check_paths("po-spec", ["src/agent.py"])
    assert "src/agent.py" in violations


def test_test_author_cannot_touch_builder_src():
    violations = gp.check_paths("test-author", ["src/train.py"])
    assert "src/train.py" in violations
