"""
Model merge data-quality (DQ) validation utilities.

This module validates a merged Hugging Face model directory BEFORE GGUF
conversion. It focuses on quick, file-system based checks that are
independent of GPU/CPU heavy ops.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List


@dataclass
class MergeDQThresholds:
    min_size_ratio: float = 0.10   # >= 10% of expected size
    max_size_ratio: float = 2.50   # <= 250% of expected size
    min_adapters: int = 1
    min_categories: int = 3


def _calc_dir_size_mb(path: Path) -> float:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total / (1024 * 1024)


def validate_merged_artifact(
    merged_dir: Path,
    expected_size_mb: float,
    adapters: List[Dict[str, Any]],
    base_model: str,
    thresholds: MergeDQThresholds | None = None,
) -> Dict[str, Any]:
    """
    Validate merged model before GGUF conversion.

    Returns a dict with keys:
      - success: bool
      - quality_score: float (0..100)
      - checks: Dict[str, Any]
      - recommendations: List[str]
    """
    th = thresholds or MergeDQThresholds()

    checks: Dict[str, Any] = {}
    recommendations: List[str] = []
    score_parts: List[float] = []

    # Check 1: directory existence and tokenizer presence
    if not merged_dir.exists() or not merged_dir.is_dir():
        return {
            "success": False,
            "quality_score": 0.0,
            "checks": {"dir_exists": False},
            "recommendations": [f"Merged directory not found: {merged_dir}"]
        }

    tokenizer_ok = any((merged_dir / name).exists() for name in ["tokenizer.json", "tokenizer.model", "tokenizer_config.json"])
    checks["tokenizer_present"] = tokenizer_ok
    score_parts.append(1.0 if tokenizer_ok else 0.0)
    if not tokenizer_ok:
        recommendations.append("Tokenizer files missing; save_pretrained(tokenizer) may have failed")

    # Check 2: size sanity vs expected
    actual_mb = _calc_dir_size_mb(merged_dir)
    ratio = (actual_mb / expected_size_mb) if expected_size_mb > 0 else 0.0
    size_ok = th.min_size_ratio <= ratio <= th.max_size_ratio
    checks["size_validation"] = {
        "passed": size_ok,
        "actual_mb": actual_mb,
        "expected_mb": expected_size_mb,
        "ratio": ratio,
        "bounds": [th.min_size_ratio, th.max_size_ratio],
    }
    score_parts.append(1.0 if size_ok else 0.0)
    if ratio < th.min_size_ratio:
        recommendations.append("Model size too small; merge may be incomplete or corrupted")
    elif ratio > th.max_size_ratio:
        recommendations.append("Model size unusually large; consider shard size or duplicate weights")

    # Check 3: adapters present
    adapters_ok = len(adapters) >= th.min_adapters
    checks["adapter_count"] = {"passed": adapters_ok, "count": len(adapters), "min": th.min_adapters}
    score_parts.append(1.0 if adapters_ok else 0.0)
    if not adapters_ok:
        recommendations.append("No adapters were merged; universal model quality will be poor")

    # Check 4: category diversity (for universal builds)
    categories = {a.get("category", "unknown") for a in adapters}
    diversity_ok = len(categories) >= th.min_categories
    checks["category_diversity"] = {"passed": diversity_ok, "count": len(categories), "min": th.min_categories, "categories": list(categories)}
    score_parts.append(1.0 if diversity_ok else 0.0)
    if not diversity_ok:
        recommendations.append("Low domain diversity; add adapters from more categories")

    # Overall
    quality_score = 100.0 * sum(score_parts) / max(len(score_parts), 1)
    success = all(v.get("passed", False) if isinstance(v, dict) else v for v in checks.values())

    return {
        "success": success,
        "quality_score": quality_score,
        "checks": checks,
        "recommendations": recommendations,
        "base_model": base_model,
        "merged_dir": str(merged_dir),
    }


