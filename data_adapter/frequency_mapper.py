"""Frequency mapping module for the TSRD dataset.

Maps continuous RF frequencies (500 - 18000 MHz) into 50 discrete bands (F01 - F50)
while preserving exact continuous frequency values in metadata.
"""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class BandInfo:
    band_id: str
    band_index: int
    freq_min_mhz: float
    freq_max_mhz: float
    freq_center_mhz: float
    bandwidth_mhz: float


class FrequencyMapper:
    """Deterministic mapper between continuous frequency (MHz) and discrete band channels."""

    def __init__(self, f_min_mhz: float = 500.0, f_max_mhz: float = 18000.0, num_bands: int = 50):
        self.f_min_mhz = float(f_min_mhz)
        self.f_max_mhz = float(f_max_mhz)
        self.num_bands = int(num_bands)
        self.total_bandwidth = self.f_max_mhz - self.f_min_mhz
        self.band_width = self.total_bandwidth / float(self.num_bands)

        # Precalculate band info lookup
        self._band_info_list: List[BandInfo] = []
        self._band_id_to_info: dict[str, BandInfo] = {}

        for i in range(self.num_bands):
            band_id = f"F{i + 1:02d}"
            b_min = self.f_min_mhz + i * self.band_width
            b_max = b_min + self.band_width
            b_center = (b_min + b_max) / 2.0
            info = BandInfo(
                band_id=band_id,
                band_index=i,
                freq_min_mhz=b_min,
                freq_max_mhz=b_max,
                freq_center_mhz=b_center,
                bandwidth_mhz=self.band_width,
            )
            self._band_info_list.append(info)
            self._band_id_to_info[band_id] = info

        self.all_band_ids = [info.band_id for info in self._band_info_list]

    def freq_to_band_index(self, freq_mhz: float) -> int:
        """Map a frequency in MHz to 0-indexed band number [0, num_bands - 1]."""
        if freq_mhz <= self.f_min_mhz:
            return 0
        if freq_mhz >= self.f_max_mhz:
            return self.num_bands - 1
        idx = int((freq_mhz - self.f_min_mhz) / self.band_width)
        return min(self.num_bands - 1, max(0, idx))

    def freq_to_band_id(self, freq_mhz: float) -> str:
        """Map a frequency in MHz to band ID string ('F01' to 'F50')."""
        idx = self.freq_to_band_index(freq_mhz)
        return self._band_info_list[idx].band_id

    def band_id_to_index(self, band_id: str) -> int:
        """Convert 'F01' -> 0."""
        if band_id in self._band_id_to_info:
            return self._band_id_to_info[band_id].band_index
        # Fallback numeric parsing
        num = int(band_id.replace("F", ""))
        return num - 1

    def band_index_to_id(self, index: int) -> str:
        """Convert 0 -> 'F01'."""
        idx = min(self.num_bands - 1, max(0, int(index)))
        return self._band_info_list[idx].band_id

    def get_band_info(self, band_id: str) -> BandInfo:
        """Retrieve full frequency bounds for a band ID."""
        if band_id not in self._band_id_to_info:
            raise KeyError(f"Invalid band ID: {band_id}. Available: F01-F{self.num_bands:02d}")
        return self._band_id_to_info[band_id]

    def get_band_frequency_range(self, band_id: str) -> Tuple[float, float, float]:
        """Return (min_mhz, max_mhz, center_mhz) for a given band ID."""
        info = self.get_band_info(band_id)
        return (info.freq_min_mhz, info.freq_max_mhz, info.freq_center_mhz)
