"""Tests for benchmark result parsing."""
import json
from pathlib import Path
import pytest

RESULTS_DIR = Path(__file__).parent.parent / "results" / "parsed"

def test_normalized_result_exists():
    path = RESULTS_DIR / "qwen36_27b_normalized.json"
    assert path.exists(), f"Missing {path}"

def test_normalized_result_schema():
    path = RESULTS_DIR / "qwen36_27b_normalized.json"
    data = json.loads(path.read_text())
    assert "model" in data
    assert "tasks" in data
    assert "mean" in data
    assert "hardware" in data

def test_task_results_have_speedup():
    path = RESULTS_DIR / "qwen36_27b_normalized.json"
    data = json.loads(path.read_text())
    for task_name, task in data["tasks"].items():
        assert "ar_tps" in task, f"{task_name} missing ar_tps"
        assert "dflash_tps" in task, f"{task_name} missing dflash_tps"
        assert "speedup" in task, f"{task_name} missing speedup"
        assert task["speedup"] > 1.0, f"{task_name} speedup should be > 1.0"

def test_mean_speedup_above_2x():
    path = RESULTS_DIR / "qwen36_27b_normalized.json"
    data = json.loads(path.read_text())
    assert data["mean"]["speedup"] >= 2.0, "Mean speedup should be >= 2.0x"

def test_all_three_tasks_present():
    path = RESULTS_DIR / "qwen36_27b_normalized.json"
    data = json.loads(path.read_text())
    expected_tasks = {"HumanEval", "GSM8K", "Math500"}
    assert set(data["tasks"].keys()) == expected_tasks
