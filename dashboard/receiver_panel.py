"""Receiver Hardware Panel: Real-time 5-channel RF state objects and instantaneous sensing telemetry.

Duck-typed over whichever runtime is driving the workstation (PlaybackController,
SimulationEngine, or OperationalEngine) - all expose get_snapshot()["channel_telemetry"].
"""

from typing import Any, Dict, List
import streamlit as st
from core.state import ChannelState
from dashboard import theme


def render_receiver_panel(engine: Any, compact: bool = False) -> None:
    """Render the 5 live receiver channel objects. Fields with no real data (e.g. SNR/
    amplitude/AoA on a replay run) are shown as N/A rather than invented.

    `compact=False` (default): the original 9-field card, used by the standalone
    RECEIVER ARRAY view - unchanged, byte-for-byte, from before this parameter
    existed.

    `compact=True`: Mission Control's reduced-hierarchy card (redesign section
    11) - CHANNEL/STATE, then FREQUENCY, then ONE highlighted primary
    measurement (SNR - the field an operator actually acts on), then a quieter
    secondary row (AMP/AoA/DWELL). Exact same computed values either way; only
    which ones are visually prominent differs. No data path changed."""
    snap = engine.get_snapshot()
    ch_telemetry = snap.get("channel_telemetry", [])

    k_ch = snap.get("k_channels", len(ch_telemetry) if ch_telemetry else 5)
    ch_cols = st.columns(k_ch)

    for i in range(k_ch):
        ch_data = ch_telemetry[i] if i < len(ch_telemetry) else {}
        b_name = ch_data.get("band", f"F{i+1:02d}")
        f_range = ch_data.get("frequency_range_ghz", "N/A")
        f_center = ch_data.get("frequency_mhz", 0.0)
        status_txt = ch_data.get("status", ch_data.get("state", ChannelState.IDLE))
        role_txt = ch_data.get("scheduler_role", f"CHANNEL 0{i+1}")
        dwell = ch_data.get("dwell_time_ms", 50.0)
        amp = ch_data.get("amplitude_dbm")
        snr = ch_data.get("snr_db")
        aoa = ch_data.get("aoa_deg")
        pw = ch_data.get("pulse_width_us")

        if status_txt in (ChannelState.SIGNAL_DETECTED, "TRUE INTERCEPTION", "DETECT"):
            status_badge = "<span style='color:#00c853; font-weight:800;'>★ SIGNAL DETECTED</span>"
            border_col = "#00c853"
        elif status_txt == ChannelState.FALSE_ALARM:
            status_badge = "<span style='color:#ffab00; font-weight:800;'>▲ FALSE ALARM</span>"
            border_col = "#ffab00"
        elif status_txt == ChannelState.SCANNING:
            status_badge = "<span style='color:#00e5ff;'>○ SCANNING</span>"
            border_col = "#1f6feb88"
        elif status_txt == ChannelState.TRACKING:
            status_badge = "<span style='color:#a371f7; font-weight:800;'>◆ TRACKING</span>"
            border_col = "#a371f7"
        elif status_txt == ChannelState.QUIET:
            status_badge = "<span style='color:#8b949e;'>— QUIET</span>"
            border_col = "#2d2d30"
        else:
            status_badge = f"<span style='color:#8b949e;'>○ {status_txt}</span>"
            border_col = "#2d2d30"

        amp_str = f"{amp:.1f} dBm" if amp is not None else "N/A"
        snr_str = f"{snr:.1f} dB" if snr is not None else "N/A"
        aoa_str = f"{aoa:.1f}°" if aoa is not None else "N/A"
        pw_str = f"{pw:.2f} µs" if pw is not None else "N/A"

        with ch_cols[i]:
            if compact:
                # Mission Control redesign (section 11): reduced field hierarchy -
                # CHANNEL/STATE -> FREQUENCY -> ONE highlighted primary
                # measurement (SNR, the field an operator actually acts on) ->
                # a quieter secondary row (AMP/AoA/DWELL). Exact same computed
                # values as the standalone card below - nothing recalculated,
                # nothing hidden, just de-emphasized.
                snr_color = "#c9d1d9" if snr is None else theme.COLOR_PRIMARY
                st.markdown(
                    f"""
                    <div class='channel-card' style='border-color:{border_col};'>
                        <div style='display:flex; justify-content:space-between; align-items:center;'>
                            <span class='channel-header' style='font-size:0.6rem;'>CH0{i+1}</span>
                            <span style='font-size:0.62rem;'>{status_badge}</span>
                        </div>
                        <div class='channel-band' style='margin-top:0.1rem; font-size:0.85rem;'>{b_name}</div>
                        <div class='channel-freq' style='font-size:0.58rem;'>{f_range}</div>
                        <div style='margin-top:0.3rem; font-family:monospace;'>
                            <span style='font-size:0.55rem; color:#8b949e; letter-spacing:0.04em;'>SNR</span><br/>
                            <span style='font-size:0.95rem; font-weight:700; color:{snr_color};'>{snr_str}</span>
                        </div>
                        <div style='display:flex; gap:0.5rem; font-size:0.55rem; font-family:monospace; margin-top:0.3rem; color:#6B7280;'>
                            <span>AMP {amp_str}</span><span>AoA {aoa_str}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                continue

            # Standalone card (RECEIVER ARRAY view - unchanged from before this
            # `compact` parameter existed). Same 9 fields (Channel/Frequency/
            # Center freq/Role/Monitoring state/Dwell/SNR/AMP/AoA).
            st.markdown(
                f"""
                <div class='channel-card' style='border-color:{border_col};'>
                    <div style='display:flex; justify-content:space-between; align-items:center;'>
                        <span class='channel-header' style='font-size:0.6rem;'>CH0{i+1}</span>
                        <span style='font-size:0.64rem;'>{status_badge}</span>
                    </div>
                    <div class='channel-band' style='margin-top:0.1rem; font-size:0.88rem;'>{b_name}</div>
                    <div class='channel-freq' style='font-size:0.6rem;'>{f_range} (fc: {f_center:.0f} MHz)</div>
                    <div style='font-size:0.58rem; color:#00e5ff; font-family:monospace; margin-top:0.15rem;'>
                        ROLE: {role_txt}
                    </div>
                    <div style='display:grid; grid-template-columns:1fr 1fr; gap:0.15rem; font-size:0.6rem; font-family:monospace; margin-top:0.3rem; color:#8b949e; background-color:#0a0a0b; padding:0.3rem; border-radius:3px;'>
                        <div>DWELL: <strong style='color:#c9d1d9;'>{dwell:.0f} ms</strong></div>
                        <div>SNR: <strong style='color:#c9d1d9;'>{snr_str}</strong></div>
                        <div>AMP: <strong style='color:#c9d1d9;'>{amp_str}</strong></div>
                        <div>AoA: <strong style='color:#c9d1d9;'>{aoa_str}</strong></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
