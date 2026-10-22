from pathlib import Path

import pandas as pd
import pytest

from src.data.loader import load_assignments, materialize_assignments, validate_assignment_selection


def assignment(tmp_path, rows, header="protocol,fold,split,project,commit_hash,target"):
    path = tmp_path / "assignments.csv"
    path.write_text(header + "\n" + "\n".join(rows) + "\n")
    return path


def test_filter_and_target_alignment(tmp_path):
    a = load_assignments(assignment(tmp_path, ["holdout,0,train,p,a,fix", "holdout,0,test,p,b,feat"]))
    data = pd.DataFrame([dict(project="p", commit_hash="a", target="fix"),
                         dict(project="p", commit_hash="b", target="feat")])
    selected = materialize_assignments(data, a, split="test")
    assert selected.commit_hash.tolist() == ["b"]
    data.loc[1, "target"] = "fix"
    with pytest.raises(RuntimeError, match="Target mismatch"): materialize_assignments(data, a)
    with pytest.raises(RuntimeError, match="absent"): materialize_assignments(data.iloc[:1], a)


@pytest.mark.parametrize("rows", [
    ["holdout,0,test,p,a,fix", "holdout,0,test,p,a,fix"],
    ["holdout,0,test,p,a,fix", "holdout,0,train,p,a,fix"],
    ["cross_project,0,test,p,a,fix", "cross_project,0,train,p,b,fix"],
    ["cross_project,0,test,p,a,fix", "cross_project,1,test,p,b,fix"],
])
def test_duplicate_and_leakage_rejection(tmp_path, rows):
    with pytest.raises(RuntimeError): load_assignments(assignment(tmp_path, rows))


def test_duplicate_selection():
    selected = pd.DataFrame([dict(project="p", commit_hash="a")] * 2)
    assignments = pd.DataFrame(columns=["protocol", "fold", "split", "project", "commit_hash"])
    with pytest.raises(RuntimeError): validate_assignment_selection(selected, assignments)


@pytest.mark.parametrize("name,count", [("flaccs", 2573), ("hqcm", 2132)])
def test_packaged_artifact_schema(name, count):
    root = Path(__file__).resolve().parents[1]
    assignments = load_assignments(root / f"data/rq4_2/artifact_compatible/{name}/assignments.csv")
    assert len(assignments) == count and assignments.project.nunique() == 30
    assert (assignments.target == assignments.canonical_target).all()


def test_artifact_target_mismatch(tmp_path):
    path = assignment(tmp_path, ["hqcm_compatible,p,a,fix,feat"],
                      "protocol,project,commit_hash,canonical_target,protocol_target")
    with pytest.raises(RuntimeError, match="protocol target"): load_assignments(path)
