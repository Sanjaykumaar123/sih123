"""Live real-time RF spectrum time-frequency waterfall display."""

from typing import Any, Dict, List, Optional
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.tracker import get_band_freq_range
from dashboard import theme


from dashboard import visualizations


def render_live_spectrum_map(engine: Any, show_ground_truth: bool = False, time_series_override: Optional[List[Dict[str, Any]]] = None) -> None:
    """Render live updating RF spectrum display using the Dual-Panel Spectrum Analyzer."""
    snap = engine.get_snapshot()
    time_series = time_series_override if time_series_override is not None else snap.get("time_series", [])
    current_t = snap.get("timestep", snap.get("current_step", 0))

    fig = visualizations.spectrum_activity_map(time_series, current_t, strategy_view="smart_scan")
    st.plotly_chart(fig, use_container_width=True)


def render_band_inspector(engine: Any) -> None:
    """Section 16: BAND DETAILS. Streamlit has no reliable click-to-inspect on Plotly
    scatter charts across versions, so band selection is a plain, robust selectbox —
    functionally the same operator capability, without a brittle click-event dependency.
    Only real, currently-available values are shown; anything not tracked is N/A.
    """
    st.markdown("<div class='channel-header' style='font-size:0.85rem; margin-top:0.6rem;'>BAND DETAILS</div>", unsafe_allow_html=True)
    snap = engine.get_snapshot()
    all_bands = [f"F{i:02d}" for i in range(1, snap.get("n_bands", 50) + 1)]
    chosen = st.selectbox("INSPECT BAND", options=all_bands, key="spectrum_band_inspector")

    f_low, f_high, f_center = get_band_freq_range(chosen)
    sel = snap.get("selected_bands", [])
    is_selected = chosen in sel
    step_td = set(snap.get("step_true_detections", []))
    step_fa = set(snap.get("step_false_alarms", []))

    if chosen in step_td:
        state_txt, state_col = "SIGNAL DETECTED (this step)", "#00c853"
    elif chosen in step_fa:
        state_txt, state_col = "FALSE ALARM (this step)", "#ffab00"
    elif is_selected:
        state_txt, state_col = "SCANNING (this step)", "#00e5ff"
    else:
        state_txt, state_col = "NOT SCANNED (this step)", "#8b949e"

    score_row = next((r for r in snap.get("band_scores_table", []) if r["Band"] == chosen), None)
    final_score = score_row.get("Final Score") if score_row else None

    recent = engine.get_recent_band_activity(30) if hasattr(engine, "get_recent_band_activity") else []
    recent_row = next((r for r in recent if r.get("Band") == chosen), None)

    d_c1, d_c2 = st.columns(2)
    with d_c1:
        st.markdown(f"**Band ID:** `{chosen}`")
        st.markdown(f"**Frequency range:** `{f_low/1000:.2f} – {f_high/1000:.2f} GHz`")
        st.markdown(f"**Center frequency:** `{f_center:.0f} MHz`")
        st.markdown(f"**Current selection state:** <span style='color:{state_col}; font-weight:700;'>{state_txt}</span>", unsafe_allow_html=True)
    with d_c2:
        assign_txt = f"CH0{sel.index(chosen)+1}" if is_selected else "None"
        score_txt = f"{final_score:.3f}" if final_score is not None else "N/A"
        activity_txt = f"{recent_row['Hits']} hits, {recent_row['False Alarms']} false alarms" if recent_row else "No activity observed"
        last_seen_txt = str(recent_row["Last Seen Step"]) if recent_row else "Never (in this window)"
        st.markdown(f"**Current receiver assignment:** `{assign_txt}`")
        st.markdown(f"**Score this step:** `{score_txt}`")
        st.markdown(f"**Recent activity (last 30 steps):** `{activity_txt}`")

        ch_row = next((c for c in snap.get("channel_telemetry", []) if c.get("band") == chosen), None)
        if ch_row:
            snr = ch_row.get("snr_db")
            amp = ch_row.get("amplitude_dbm")
            aoa = ch_row.get("aoa_deg")
            pw = ch_row.get("pulse_width_us")
            telem_txt = (
                f"SNR: {snr:.1f} dB, Amp: {amp:.1f} dBm" if snr is not None and amp is not None
                else "N/A — no real detection on this band this step"
            )
            if aoa is not None or pw is not None:
                telem_txt += f", AoA: {aoa:.1f}°" if aoa is not None else ", AoA: N/A"
                telem_txt += f", PW: {pw:.2f} µs" if pw is not None else ", PW: N/A"
        else:
            telem_txt = "N/A — band not currently assigned to a receiver channel"
        st.markdown(f"**Latest telemetry:** `{telem_txt}`")
        st.markdown(f"**Last observed step:** `{last_seen_txt}`")


def render_spectrum_analyzer(engine: Any) -> None:
    """Stitch-inspired engineering spectrum-analyzer view: CF / SPAN / RBW readout +
    power-vs-frequency + Detected Signals table.

    This receiver has no continuously-tunable analog front end - it channelizes the
    500 MHz-18 GHz spectrum into N=50 fixed bands (see PROJECT_SPEC.md). CF/SPAN/RBW
    below are real, honestly-derived static architecture constants, not a swept
    instrument's live readings - labelled precisely so as not to imply a sweep
    capability this system doesn't have. Power is plotted ONLY for bands the receiver
    actually measured this step (real amplitude_dbm from channel_telemetry, populated
    by simulation/engine.py only on a real hit/false-alarm - see core/state.py's
    ChannelTelemetry). Every other band is left as a gap - never interpolated,
    never a fabricated noise floor. REPLAY VERIFIED RUN artifacts do not record
    per-pulse amplitude at all (see core/playback_controller.py), so this view is
    honestly all-N/A there; only LIVE SIMULATION can show real power.
    """
    snap = engine.get_snapshot()
    ch_tel = snap.get("channel_telemetry", [])
    n_bands = snap.get("n_bands", 50)
    f_min_mhz, f_max_mhz = 500.0, 18000.0
    span_mhz = f_max_mhz - f_min_mhz
    cf_mhz = (f_min_mhz + f_max_mhz) / 2.0
    rbw_mhz = (span_mhz / n_bands) if n_bands else None

    st.markdown(
        f"<div class='channel-header' style='font-size:0.85rem; margin-top:0.6rem;'>SPECTRUM ANALYZER — RECEIVER ARCHITECTURE SWEEP</div> {theme.provenance_badge('STATIC')}",
        unsafe_allow_html=True,
    )
    st.caption(
        "This receiver channelizes the spectrum into 50 fixed bands rather than "
        "continuously sweeping — CF/SPAN describe that fixed architecture (real, "
        "static). Power is plotted only for bands the receiver actually measured "
        "this step; nothing is invented for the other bands."
    )
    r_c1, r_c2, r_c3 = st.columns(3)
    with r_c1:
        st.markdown(
            f"<div class='metric-card'><div class='metric-lbl'>CF (Full-Span Center)</div>"
            f"<div class='metric-val' style='font-size:1.0rem;'>{cf_mhz/1000.0:.3f} GHz</div></div>",
            unsafe_allow_html=True,
        )
    with r_c2:
        st.markdown(
            f"<div class='metric-card'><div class='metric-lbl'>SPAN</div>"
            f"<div class='metric-val' style='font-size:1.0rem;'>{span_mhz/1000.0:.3f} GHz</div></div>",
            unsafe_allow_html=True,
        )
    with r_c3:
        rbw_txt = f"{rbw_mhz:.1f} MHz" if rbw_mhz else "N/A"
        st.markdown(
            f"<div class='metric-card'><div class='metric-lbl'>Channel Width (N={n_bands})</div>"
            f"<div class='metric-val' style='font-size:1.0rem;'>{rbw_txt}</div></div>",
            unsafe_allow_html=True,
        )

    # Power vs frequency - plot only channels with a REAL measured amplitude this step.
    freqs, powers, labels, colors, hover = [], [], [], [], []
    for ch in ch_tel:
        amp = ch.get("amplitude_dbm")
        if amp is None:
            continue
        band = ch.get("band", "F01")
        _, _, f_center = get_band_freq_range(band)
        status = str(ch.get("status", ch.get("state", "")))
        if "TRUE" in status or "DETECTED" in status:
            col = theme.COLOR_NOMINAL
        elif "FALSE" in status:
            col = theme.COLOR_CAUTION
        else:
            col = theme.COLOR_PRIMARY
        freqs.append(f_center)
        powers.append(amp)
        labels.append(f"{band}/CH0{ch.get('channel_idx', '?')}")
        colors.append(col)
        hover.append(f"{band} (CH0{ch.get('channel_idx','?')}) — {amp:.1f} dBm @ {f_center:.0f} MHz — {status}")

    fig = go.Figure()
    if freqs:
        fig.add_trace(go.Scatter(
            x=freqs, y=powers, mode="markers+text",
            marker=dict(size=12, color=colors, line=dict(width=1, color="#ffffff")),
            text=labels, textposition="top center",
            textfont=dict(size=9, color=theme.COLOR_TEXT_MUTED),
            hovertext=hover, hoverinfo="text", name="Measured Power",
        ))
    fig.update_layout(
        height=300,
        margin=dict(l=45, r=20, t=20, b=40),
        paper_bgcolor=theme.COLOR_PANEL,
        plot_bgcolor=theme.COLOR_BASE,
        font=dict(color=theme.COLOR_TEXT, family="monospace", size=10),
        xaxis=dict(title="Frequency (MHz)", range=[f_min_mhz, f_max_mhz], gridcolor="#00e5ff1a", zeroline=False),  # Stitch: 10%-opacity cyan grid
        yaxis=dict(title="Power (dBm)", range=[-130, 0], gridcolor="#00e5ff1a", zeroline=False),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
    if not freqs:
        reason = (
            "REPLAY VERIFIED RUN artifacts do not record per-pulse amplitude (see SYSTEM view)."
            if not hasattr(engine, "tracker") else "no active detection on any tuned channel this step."
        )
        st.info(f"No real power measurement available for the current step — {reason}")

    st.markdown(
        f"<div class='channel-header' style='font-size:0.8rem; margin-top:0.6rem;'>DETECTED SIGNALS</div> {theme.provenance_badge('REAL' if freqs else 'NA')}",
        unsafe_allow_html=True,
    )
    rows = []
    for ch in ch_tel:
        band = ch.get("band", "N/A")
        _, _, f_center = get_band_freq_range(band) if band != "N/A" else (0.0, 0.0, 0.0)
        amp = ch.get("amplitude_dbm")
        rows.append({
            "ID": band,
            "Frequency (MHz)": f"{f_center:.1f}" if band != "N/A" else "N/A",
            "Power (dBm)": f"{amp:.1f}" if amp is not None else "N/A",
            "Status": ch.get("status", ch.get("state", "N/A")),
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=190)
    else:
        st.info("No channels currently tuned.")
