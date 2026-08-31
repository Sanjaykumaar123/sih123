"""Data ingestion, TSRD HDF5 processing, and scenario discovery."""

from data_adapter.tsrd_loader import TSRDLoader, TSRDRawData
from data_adapter.pdw_processor import PDWProcessor, TimeStepActivity, BinnedBandActivity
from data_adapter.frequency_mapper import FrequencyMapper
from data_adapter.truth_manager import TruthManager
from data_adapter.scenario_builder import TSRDEnvironment
from dashboard.scenario_loader import (
    discover_scenarios,
    get_validated_scenarios,
    validate_operational_artifact,
    ScenarioDescriptor,
)

__all__ = [
    "TSRDLoader",
    "TSRDRawData",
    "PDWProcessor",
    "TimeStepActivity",
    "BinnedBandActivity",
    "FrequencyMapper",
    "TruthManager",
    "TSRDEnvironment",
    "discover_scenarios",
    "get_validated_scenarios",
    "validate_operational_artifact",
    "ScenarioDescriptor",
]
