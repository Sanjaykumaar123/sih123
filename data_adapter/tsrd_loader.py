"""TSRD Raw HDF5 Data Loader.

Reads single HDF5 radar scan/stare scenario files on-demand,
decoding feature names, labels, and scenario metadata.
"""

from dataclasses import dataclass
import os
from typing import Any, Dict, List, Optional
import h5py
import numpy as np


@dataclass
class TSRDReceiverMetadata:
    freq_range_mhz: List[float]
    bandwidth_mhz: float
    dwell_centres_mhz: List[float]
    dwell_times_s: List[float]
    collection_time_s: float
    sensitivity_dbm: float
    scan_mode: str
    start_position_km: List[float]


@dataclass
class TSRDRawData:
    file_path: str
    features: List[str]
    pulse_data: np.ndarray  # (N_pulses, 5) float32
    labels: np.ndarray      # (N_pulses,) int32
    receiver_metadata: TSRDReceiverMetadata
    transmitter_metadata: List[Dict[str, Any]]
    num_pulses: int
    num_emitter_classes: int
    duration_s: float


class TSRDLoader:
    """Read-only on-demand loader for Turing Synthetic Radar Dataset HDF5 files."""

    DEFAULT_FEATURE_NAMES = ["ToA", "Frequency", "PulseWidth", "AoA", "Amplitude"]

    @staticmethod
    def _decode_bytes(val: Any) -> Any:
        if isinstance(val, (bytes, bytearray)):
            return val.decode("utf-8", errors="replace")
        if isinstance(val, (np.bytes_, np.str_)):
            return str(val)
        if isinstance(val, np.ndarray) and val.dtype.kind in ("S", "U"):
            return [TSRDLoader._decode_bytes(item) for item in val]
        return val

    @classmethod
    def _read_hdf5_item(cls, item: Any) -> Any:
        if isinstance(item, h5py.Dataset):
            val = item[()]
            return cls._decode_bytes(val)
        elif isinstance(item, h5py.Group):
            res = {k: cls._decode_bytes(v) for k, v in item.attrs.items()}
            for k, v in item.items():
                res[k] = cls._read_hdf5_item(v)
            return res
        return cls._decode_bytes(item)

    @classmethod
    def load_file(cls, file_path: str) -> TSRDRawData:
        """Load a single TSRD .h5 file and return structured dataset representation."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"TSRD file not found: {file_path}")

        with h5py.File(file_path, "r") as h5f:
            # 1. Load pulse data matrix
            if "data" not in h5f:
                raise ValueError(f"HDF5 file {file_path} missing '/data' dataset.")
            data_ds = h5f["data"]
            pulse_data = np.array(data_ds[:], dtype=np.float32)

            # Read features attribute if available
            feature_names = cls.DEFAULT_FEATURE_NAMES
            if "features" in data_ds.attrs:
                raw_feats = data_ds.attrs["features"]
                feature_names = [cls._decode_bytes(f) for f in raw_feats]

            # 2. Load ground-truth labels
            if "labels" in h5f:
                labels = np.array(h5f["labels"][:], dtype=np.int32).squeeze()
            else:
                labels = np.zeros(len(pulse_data), dtype=np.int32)

            # 3. Load Receiver Metadata
            rec_meta_grp = h5f.get("metadata/receiver")
            if rec_meta_grp is not None:
                rec_meta_dict = cls._read_hdf5_item(rec_meta_grp)
                freq_range = rec_meta_dict.get("freq_range_mhz", [500.0, 18000.0])
                bw = float(rec_meta_dict.get("bandwith_mhz", 500.0))
                dwell_centres = rec_meta_dict.get("dwell_centres_mhz", [])
                dwell_times = rec_meta_dict.get("dwell_times_s", [])
                col_time = float(rec_meta_dict.get("collection_time_s", 30.0))
                sens = float(rec_meta_dict.get("sensitivity_dbm", -110.0))
                scan_mode = str(rec_meta_dict.get("scan_mode", "Scanning"))
                start_pos = rec_meta_dict.get("start_position_km", [0.0, 0.0, 0.0])

                rec_metadata = TSRDReceiverMetadata(
                    freq_range_mhz=[float(x) for x in freq_range],
                    bandwidth_mhz=bw,
                    dwell_centres_mhz=[float(x) for x in dwell_centres],
                    dwell_times_s=[float(x) for x in dwell_times],
                    collection_time_s=col_time,
                    sensitivity_dbm=sens,
                    scan_mode=scan_mode,
                    start_position_km=[float(x) for x in start_pos],
                )
            else:
                rec_metadata = TSRDReceiverMetadata(
                    freq_range_mhz=[500.0, 18000.0],
                    bandwidth_mhz=500.0,
                    dwell_centres_mhz=[],
                    dwell_times_s=[],
                    collection_time_s=30.0,
                    sensitivity_dbm=-110.0,
                    scan_mode="Scanning",
                    start_position_km=[0.0, 0.0, 0.0],
                )

            # 4. Load Transmitter Metadata
            transmitters = []
            trans_meta_grp = h5f.get("metadata/transmitters")
            if trans_meta_grp is not None:
                trans_dict = cls._read_hdf5_item(trans_meta_grp)
                for k in sorted(trans_dict.keys(), key=lambda x: int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else x):
                    t_item = trans_dict[k]
                    if isinstance(t_item, dict):
                        t_item["id"] = k
                        transmitters.append(t_item)
                    else:
                        transmitters.append({"id": k, "data": t_item})

        # 5. Ensure pulses are sorted by Time of Arrival (ToA is column 0)
        num_pulses = len(pulse_data)
        if num_pulses > 0:
            toa_col = pulse_data[:, 0]
            # Check if sorted
            if not np.all(toa_col[:-1] <= toa_col[1:]):
                sort_idx = np.argsort(toa_col)
                pulse_data = pulse_data[sort_idx]
                labels = labels[sort_idx]

        unique_emitters = int(len(np.unique(labels))) if num_pulses > 0 else 0
        duration_s = rec_metadata.collection_time_s

        return TSRDRawData(
            file_path=os.path.abspath(file_path),
            features=feature_names,
            pulse_data=pulse_data,
            labels=labels,
            receiver_metadata=rec_metadata,
            transmitter_metadata=transmitters,
            num_pulses=num_pulses,
            num_emitter_classes=unique_emitters,
            duration_s=duration_s,
        )
