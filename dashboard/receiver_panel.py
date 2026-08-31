"""Receiver Hardware Panel: Real-time 5-channel RF state objects and instantaneous sensing telemetry.

Duck-typed over whichever runtime is driving the workstation (PlaybackController,
SimulationEngine, or OperationalEngine) - all expose get_snapshot()["channel_telemetry"].
"""

from typing import Any, Dict, List
import streamlit as st
from core.state import ChannelState
from dashboard import theme


def render_receiver_panel(engine: Any, compact: bool = False) -> None:
    """Render 5 spacious receiver channel cards with clean high-contrast metrics."""
    snap = engine.get_snapshot()
    ch_telemetry = snap.get("channel_telemetry", [])
    k_ch = snap.get("k_channels", len(ch_telemetry) if ch_telemetry else 5)

    st.markdown(
        """
        <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem;'>
            <div>
                <div style='font-size:1.3rem; font-weight:800; color:#F3F4F6; font-family:"Outfit"; letter-spacing:0.04em;'>
                    RECEIVER HARDWARE ARRAY (K=5 CHANNELS)
                </div>
                <div style='font-size:0.8rem; color:#9CA3AF; margin-top:0.1rem;'>
                    Real-time RF Hardware Channel Allocations & Sensing Telemetry
                </div>
            </div>
            <span style='font-size:0.75rem; font-weight:700; color:#00F0FF; background:rgba(0,240,255,0.1); padding:0.35rem 0.75rem; border-radius:4px; font-family:monospace;'>
                CHANNELS 01 - 05
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    ch_cols = st.columns(k_ch)

    for i in range(k_ch):
        ch_data = ch_telemetry[i] if i < len(ch_telemetry) else {}
        b_name = ch_data.get("band", f"F{i+1:02d}")
        f_range = ch_data.get("frequency_range_ghz", "N/A")
        status_txt = ch_data.get("status", ch_data.get("state", ChannelState.IDLE))
        amp = ch_data.get("amplitude_dbm")
        snr = ch_data.get("snr_db")
        aoa = ch_data.get("aoa_deg")

        if status_txt in (ChannelState.SIGNAL_DETECTED, "TRUE INTERCEPTION", "DETECT"):
            status_badge = "<span style='color:#00FF9D; font-weight:700;'>★ SIGNAL DETECTED</span>"
            border_col = "#00FF9D"
        elif status_txt == ChannelState.FALSE_ALARM:
            status_badge = "<span style='color:#FFB800; font-weight:700;'>▲ FALSE ALARM</span>"
            border_col = "#FFB800"
        elif status_txt == ChannelState.SCANNING:
            status_badge = "<span style='color:#00F0FF;'>○ SCANNING</span>"
            border_col = "#00F0FF"
        elif status_txt == ChannelState.TRACKING:
            status_badge = "<span style='color:#A855F7; font-weight:700;'>◆ TRACKING</span>"
            border_col = "#A855F7"
        else:
            status_badge = "<span style='color:#6B7280;'>— MONITORING</span>"
            border_col = "rgba(255,255,255,0.15)"

        amp_str = f"{amp:.1f} dBm" if amp is not None else "N/A"
        snr_str = f"{snr:.1f} dB" if snr is not None else "N/A"
        aoa_str = f"{aoa:.1f}°" if aoa is not None else "N/A"
        snr_color = "#6B7280" if snr is None else "#00FF9D"

        with ch_cols[i]:
            st.markdown(
                f"""
                <div class='glass-card' style='padding:1rem 0.9rem; border-top:4px solid {border_col} !important;'>
                    <div style='display:flex; justify-content:space-between; align-items:center;'>
                        <span style='font-size:0.78rem; font-weight:800; color:#9CA3AF; font-family:monospace;'>CH0{i+1}</span>
                        <span style='font-size:0.72rem;'>{status_badge}</span>
                    </div>
                    <div style='font-size:1.8rem; font-weight:900; color:#F3F4F6; font-family:"Outfit"; margin-top:0.2rem; letter-spacing:0.02em;'>
                        {b_name}
                    </div>
                    <div style='font-size:0.85rem; color:#9CA3AF; font-family:"Inter"; margin-bottom:0.6rem;'>
                        {f_range} GHz
                    </div>
                    <div style='background:rgba(0,0,0,0.35); padding:0.5rem 0.6rem; border-radius:6px; font-family:monospace;'>
                        <div style='font-size:0.95rem; font-weight:700; color:{snr_color};'>SNR: {snr_str}</div>
                        <div style='display:flex; justify-content:space-between; font-size:0.75rem; color:#D1D5DB; margin-top:0.25rem;'>
                            <span>AMP: {amp_str}</span>
                            <span>AoA: {aoa_str}</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
