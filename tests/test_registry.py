"""Tests for model registry and baseline data."""
import json
from pathlib import Path
import pytest

REGISTRY_DIR = Path(__file__).parent.parent / "registry"
BASELINES_DIR = Path(__file__).parent.parent / "data" / "baselines"

def test_registry_exists():
    path = REGISTRY_DIR / "dflash_models.json"
    assert path.exists()

def test_registry_has_drafts():
    data = json.loads((REGISTRY_DIR / "dflash_models.json").read_text())
    assert "dflash_drafts" in data
    assert len(data["dflash_drafts"]) > 0

def test_registry_drafts_have_required_fields():
    data = json.loads((REGISTRY_DIR / "dflash_models.json").read_text())
    for draft in data["dflash_drafts"]:
        assert "target_pattern" in draft
        assert "draft_repo" in draft
        assert "target_arch" in draft

def test_qwen35_draft_in_registry():
    data = json.loads((REGISTRY_DIR / "dflash_models.json").read_text())
    patterns = [d["target_pattern"] for d in data["dflash_drafts"]]
    assert "Qwen3.5-27B" in patterns
    assert "Qwen3.6-27B" in patterns

def test_baselines_exist():
    path = BASELINES_DIR / "rtx3090_baselines.json"
    assert path.exists()

def test_baselines_have_four_models():
    data = json.loads((BASELINES_DIR / "rtx3090_baselines.json").read_text())
    assert len(data["models"]) == 4

def test_baselines_gen512_values():
    data = json.loads((BASELINES_DIR / "rtx3090_baselines.json").read_text())
    models = data["models"]
    assert models["Qwen3.6-27B-Q4_K_M"]["gen512"] == 40.4
    assert models["Qwen3.6-35B-A3B-UD-Q4_K_M"]["gen512"] == 146.6
    assert models["gemma-4-26B-A4B-UD-Q4_K_M"]["gen512"] == 131.4
    assert models["gemma-4-31B-it-Q4_K_M"]["gen512"] == 36.21
