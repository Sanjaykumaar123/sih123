"""Scenario discovery, artifact validation, and data loader for multi-scenario evaluation.

Discovers HDF5 scenario recordings in dataset/scan/ and precomputed operational
evaluation JSON artifacts in results/, validating integrity and schema compliance.
Zero fabricated statistics.
"""

from __future__ import annotations
from dataclasses import dataclass
import glob
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RESULTS_DIR = os.path.join(_BASE_DIR, "results")
DEFAULT_SCAN_DIR = os.path.join(_BASE_DIR, "dataset", "scan", "test_scan")

RESULTS_DIR = r"D:\sih\results" if os.path.exists(r"D:\sih\results") else DEFAULT_RESULTS_DIR
SCAN_DIR = r"D:\sih\dataset\scan\test_scan" if os.path.exists(r"D:\sih\dataset\scan\test_scan") else DEFAULT_SCAN_DIR


@dataclass
class ScenarioDescriptor:
    scenario_id: str
    scenario_name: str
    h5_path: Optional[str]
    json_path: Optional[str]
    status: str  # "VALIDATED", "EVALUATION NOT AVAILABLE", "ARTIFACT VALIDATION FAILED"
    data_present: bool
    evaluation_present: bool
    tested: bool
    num_steps: int = 600
    duration_s: float = 30.0
    channels: int = 5
    num_bands: int = 50
    metrics_summary: Optional[Dict[str, Any]] = None


def validate_operational_artifact(json_path: str) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """Validate that an operational JSON artifact exists, has valid schema, and contains no NaNs."""
    if not os.path.exists(json_path):
        return False, "File does not exist", None

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return False, f"JSON parse error: {str(e)}", None

    required_keys = ["scenario", "num_steps", "channels", "metrics_summary", "time_series", "emitter_interceptions"]
    for k in required_keys:
        if k not in data:
            return False, f"Missing required key: {k}", None

    summary = data.get("metrics_summary", {})
    if "baseline" not in summary or "smart_scan" not in summary:
        return False, "Missing baseline or smart_scan in metrics_summary", None

    # Verify no NaN in essential metrics
    for strat in ["baseline", "smart_scan"]:
        m = summary[strat]
        for num_field in ["true_detections", "unique_emitters_intercepted", "sensor_pd", "scenario_coverage"]:
            val = m.get(num_field)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                return False, f"Invalid value for {strat}.{num_field}: {val}", None

    return True, None, data


def discover_scenarios(
    results_dir: str = RESULTS_DIR,
    scan_dir: str = SCAN_DIR,
) -> Dict[str, ScenarioDescriptor]:
    """Discover all available H5 scenarios and operational evaluation JSONs."""
    scenarios: Dict[str, ScenarioDescriptor] = {}

    # 1. Discover H5 files in scan_dir
    if os.path.exists(scan_dir):
        h5_files = glob.glob(os.path.join(scan_dir, "config_*.h5"))
        for h5_p in sorted(h5_files):
            fname = os.path.basename(h5_p)
            cfg_id = os.path.splitext(fname)[0]
            scenarios[cfg_id] = ScenarioDescriptor(
                scenario_id=cfg_id,
                scenario_name=fname,
                h5_path=h5_p,
                json_path=None,
                status="EVALUATION NOT AVAILABLE",
                data_present=True,
                evaluation_present=False,
                tested=False,
            )

    # 2. Discover and Validate Operational Evaluation JSONs
    if os.path.exists(results_dir):
        json_files = glob.glob(os.path.join(results_dir, "operational_evaluation_config_*.json"))
        for j_p in sorted(json_files):
            fname = os.path.basename(j_p)
            # Extract config ID e.g. operational_evaluation_config_1.json -> config_1
            cfg_id = fname.replace("operational_evaluation_", "").replace(".json", "")
            
            is_valid, err_msg, loaded_data = validate_operational_artifact(j_p)
            
            h5_path = os.path.join(scan_dir, f"{cfg_id}.h5") if os.path.exists(os.path.join(scan_dir, f"{cfg_id}.h5")) else None
            
            if is_valid and loaded_data is not None:
                scenarios[cfg_id] = ScenarioDescriptor(
                    scenario_id=cfg_id,
                    scenario_name=loaded_data.get("scenario", f"{cfg_id}.h5"),
                    h5_path=h5_path,
                    json_path=j_p,
                    status="VALIDATED",
                    data_present=bool(h5_path),
                    evaluation_present=True,
                    tested=True,
                    num_steps=loaded_data.get("num_steps", 600),
                    duration_s=loaded_data.get("num_steps", 600) * 0.05,
                    channels=loaded_data.get("channels", 5),
                    num_bands=50,
                    metrics_summary=loaded_data.get("metrics_summary"),
                )
            else:
                if cfg_id in scenarios:
                    scenarios[cfg_id].status = "ARTIFACT VALIDATION FAILED"
                else:
                    scenarios[cfg_id] = ScenarioDescriptor(
                        scenario_id=cfg_id,
                        scenario_name=f"{cfg_id}.h5",
                        h5_path=h5_path,
                        json_path=j_p,
                        status="ARTIFACT VALIDATION FAILED",
                        data_present=bool(h5_path),
                        evaluation_present=True,
                        tested=False,
                    )

    return scenarios


def get_validated_scenarios(
    results_dir: str = RESULTS_DIR,
    scan_dir: str = SCAN_DIR,
) -> Dict[str, Dict[str, Any]]:
    """Load full JSON data for all scenarios with VALIDATED status."""
    all_discovered = discover_scenarios(results_dir, scan_dir)
    validated: Dict[str, Dict[str, Any]] = {}
    
    for cfg_id, desc in all_discovered.items():
        if desc.status == "VALIDATED" and desc.json_path:
            with open(desc.json_path, "r", encoding="utf-8") as f:
                validated[cfg_id] = json.load(f)
                
    return validated
