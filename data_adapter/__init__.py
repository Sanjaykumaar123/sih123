"""TSRD Data Adapter Package.

Provides seamless integration of the Turing Synthetic Radar Dataset (TSRD)
with the Cognitive Smart Scan Electronic Warfare prototype.
"""

from .frequency_mapper import BandInfo, FrequencyMapper
from .pdw_processor import BinnedBandActivity, PDWProcessor, TimeStepActivity
from .scenario_builder import EnvironmentSource, TSRDEnvironment, create_environment
from .truth_manager import TruthManager
from .tsrd_loader import TSRDLoader, TSRDRawData, TSRDReceiverMetadata

__all__ = [
    "BandInfo",
    "FrequencyMapper",
    "PDWProcessor",
    "BinnedBandActivity",
    "TimeStepActivity",
    "TruthManager",
    "TSRDLoader",
    "TSRDRawData",
    "TSRDReceiverMetadata",
    "EnvironmentSource",
    "TSRDEnvironment",
    "create_environment",
]
