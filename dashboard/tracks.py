"""Track / interception view.

Two genuinely different runtimes render through this one view, and they are NOT the
same capability, so they are never presented as if they were:

- Live SimulationEngine/OperationalEngine path (has `.tracker`): a real autonomous
  TrackManager clusters live per-pulse Observations into signal tracks with confidence,
  estimated PRI, etc. This is real live tracking.
- Replay PlaybackController path (no `.tracker`): there is no live per-pulse clustering
  available - only the artifact's post-hoc, ground-truth-derived `emitter_interceptions`
  record. This is shown as exactly that: a verified post-hoc record, never dressed up
  as a live track with an invented confidence value.
"""

from typing import Any, Dict, List
import pandas as pd
import streamlit as st


def render_tracks_view(engine: Any) -> None:
    if hasattr(engine, "tracker"):
        _render_live_track_manager_view(engine)
    else:
        _render_replay_interception_view(engine)


# -----------------------------------------------------------------------------
# Live path: real autonomous TrackManager (unsupervised clustering of observations)
# -----------------------------------------------------------------------------
def _render_live_track_manager_view(engine: Any) -> None:
    """Render autonomous signal tracks created entirely from observable pulse measurements."""
    snap = engine.get_snapshot()
    tracks = snap.get("tracks", [])

    st.markdown("<div class='system-title'>AUTONOMOUS EMITTER SIGNAL TRACK MANAGER</div>", unsafe_allow_html=True)
    st.markdown("<div class='system-subtitle'>LIVE UNSUPERVISED PULSE CLUSTERING & TRACK STATE ESTIMATION (ZERO GROUND-TRUTH LEAKAGE)</div>", unsafe_allow_html=True)

    t_c1, t_c2, t_c3, t_c4 = st.columns(4)
    active_cnt = snap.get("active_tracks_count", 0)
    total_cnt = snap.get("total_tracks_count", 0)
    confirmed_cnt = sum(1 for t in tracks if t.get("State") == "CONFIRMED")
    lost_cnt = sum(1 for t in tracks if t.get("State") == "LOST")

    for col, (lbl, val, color, sub) in zip(
        (t_c1, t_c2, t_c3, t_c4),
        [
            ("Total Tracks Formed", total_cnt, "#e6edee", "Autonomous Clusters"),
            ("Confirmed Tracks", confirmed_cnt, "#00c853", "≥ 2 Observed Hits"),
            ("Active Tracks Tracked", active_cnt, "#e6edee", "Current PRI Tracking"),
            ("Lost / Stale Tracks", lost_cnt, "#ffab00", "Unseen > 8 Steps"),
        ],
    ):
        with col:
            st.markdown(
                f"""<div class='metric-card'><div class='metric-lbl'>{lbl}</div>
                <div class='metric-val' style='color:{color};'>{val}</div>
                <div class='metric-imp imp-neutral'>{sub}</div></div>""",
                unsafe_allow_html=True,
            )

    flt_col1, flt_col2 = st.columns([7, 3])
    with flt_col1:
        st_filter = st.radio(
            "FILTER TRACK STATE",
            options=["ALL", "ACTIVE", "CONFIRMED", "LOST", "EXPIRED"],
            horizontal=True,
            key="tracks_state_filter",
        )
    with flt_col2:
        tracks_csv = engine.export_tracks_csv() if hasattr(engine, "export_tracks_csv") else (
            engine.tracker.export_tracks_csv() if hasattr(engine, "tracker") else ""
        )
        st.download_button(
            "📥 EXPORT TRACK HISTORY (CSV)", data=tracks_csv,
            file_name=f"rf_tracks_{snap['scenario_name'].replace('.h5','')}_{snap['timestep']}steps.csv",
            mime="text/csv", use_container_width=True, key="export_tracks_csv_btn",
        )

    filtered_tracks = tracks
    if st_filter == "ACTIVE":
        filtered_tracks = [t for t in tracks if t.get("State") in ("CONFIRMED", "ACTIVE", "NEW", "TENTATIVE")]
    elif st_filter in ("CONFIRMED", "LOST", "EXPIRED"):
        filtered_tracks = [t for t in tracks if t.get("State") == st_filter]

    st.markdown("<div class='channel-header' style='font-size:0.85rem; margin-top:0.4rem;'>SIGNAL TRACK TABLE</div>", unsafe_allow_html=True)
    if filtered_tracks:
        st.dataframe(filtered_tracks, height=300, use_container_width=True)
    else:
        st.info("No signal tracks match the selected filter.")

    if hasattr(engine, "tracker") and engine.tracker.tracks:
        all_t_ids = list(engine.tracker.tracks.keys())
        st.markdown("<div class='channel-header' style='font-size:0.85rem; margin-top:0.8rem;'>🔬 DETAILED TRACK INSPECTOR (OBSERVABLE PULSE HISTORY)</div>", unsafe_allow_html=True)
        selected_t_id = st.selectbox("SELECT TRACK ID TO INSPECT", options=all_t_ids, key="inspect_track_select")
        if selected_t_id and selected_t_id in engine.tracker.tracks:
            tr_obj = engine.tracker.tracks[selected_t_id]
            insp_c1, insp_c2, insp_c3, insp_c4 = st.columns(4)
            with insp_c1:
                st.markdown(f"**Track ID:** `{tr_obj.track_id}`")
                st.markdown(f"**Band:** `{tr_obj.band_id}`")
                st.markdown(f"**State:** `{tr_obj.state}`")
            with insp_c2:
                st.markdown(f"**Est. Frequency:** `{tr_obj.estimated_frequency_mhz:.2f} MHz`")
                st.markdown(f"**Est. Pulse Width:** `{tr_obj.estimated_pulse_width_us:.2f} µs`")
                st.markdown(f"**Est. AoA:** `{tr_obj.estimated_aoa_deg:.1f}°`")
            with insp_c3:
                st.markdown(f"**Est. Amplitude:** `{tr_obj.estimated_amplitude_dbm:.1f} dBm`")
                st.markdown(f"**Est. SNR:** `{tr_obj.estimated_snr_db:.1f} dB`")
                st.markdown(f"**Confidence:** `{tr_obj.confidence_pct:.1f}%`")
            with insp_c4:
                pri_txt = f"{tr_obj.estimated_pri_timesteps * 0.05 * 1000:.1f} ms" if tr_obj.estimated_pri_timesteps else "Estimating..."
                st.markdown(f"**Est. PRI:** `{pri_txt}`")
                st.markdown(f"**Total Hits:** `{tr_obj.hit_count}`")
                st.markdown(f"**First Seen:** `t={tr_obj.first_seen_timestep * 0.05:.2f}s`")

            if tr_obj.observable_history:
                with st.expander(f"📋 Observable Pulse Measurements History ({len(tr_obj.observable_history)} pulses)", expanded=True):
                    st.dataframe(pd.DataFrame(tr_obj.observable_history), use_container_width=True)

    if hasattr(engine, "tracker") and engine.tracker.track_event_log:
        with st.expander("📋 Autonomous Track Transition & Maintenance Event History", expanded=False):
            st.dataframe(pd.DataFrame(engine.tracker.track_event_log), use_container_width=True)


# -----------------------------------------------------------------------------
# Replay path: no live tracker exists - show the real post-hoc interception record
# -----------------------------------------------------------------------------
def _render_replay_interception_view(engine: Any) -> None:
    snap = engine.get_snapshot()
    t = snap.get("timestep", 0)

    st.markdown("<div class='system-title'>EMITTER INTERCEPTION RECORD</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='system-subtitle'>POST-HOC VERIFIED ARTIFACT — NOT LIVE PER-PULSE TRACKING</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "This replay run has no live signal-track clustering. The rows below are the "
        "scenario's PHYSICAL EMITTER records (ground truth, used only post-hoc / never "
        "fed to the scheduler) and when each was first intercepted. Autonomous live "
        "track clustering with confidence estimation only exists in the separate "
        "SimulationEngine runtime path — see the SYSTEM view for details."
    )

    records = engine.get_emitter_interception_records() if hasattr(engine, "get_emitter_interception_records") else []
    is_smart = "smart" in getattr(engine, "strategy_type", "smart_scan")
    step_key = "first_intercept_step_ss" if is_smart else "first_intercept_step_ol"
    lat_s_key = "intercept_latency_s_ss" if is_smart else "intercept_latency_s_ol"

    total = len(records)
    intercepted = [r for r in records if r.get(step_key) is not None and r[step_key] <= t]
    pending = total - len(intercepted)

    m_c1, m_c2, m_c3 = st.columns(3)
    for col, (lbl, val, color) in zip(
        (m_c1, m_c2, m_c3),
        [
            ("Physical Emitters in Scenario", total, "#e6edee"),
            ("Intercepted So Far", len(intercepted), "#00c853"),
            ("Not Yet Intercepted", pending, "#ffab00"),
        ],
    ):
        with col:
            st.markdown(
                f"""<div class='metric-card'><div class='metric-lbl'>{lbl}</div>
                <div class='metric-val' style='color:{color};'>{val}</div></div>""",
                unsafe_allow_html=True,
            )

    st.markdown("<div class='channel-header' style='font-size:0.85rem; margin-top:0.6rem;'>INTERCEPTION HISTORY (BY EMITTER)</div>", unsafe_allow_html=True)
    if records:
        time_s_key = "first_intercept_time_s_ss" if is_smart else "first_intercept_time_s_ol"
        rows = []
        for r in records:
            step_val = r.get(step_key)
            lat_val = r.get(lat_s_key)
            intercept_time_val = r.get(time_s_key, 0.0)
            rows.append({
                "Emitter ID": r["emitter_id"],
                "First Activity (s)": f"{r.get('first_activity_time_s', 0.0):.2f}",
                f"First {'Smart Scan' if is_smart else 'Open Loop'} Intercept (s)": (
                    f"{intercept_time_val:.2f}" if step_val is not None else "Not yet intercepted"
                ),
                "Latency (s)": f"{lat_val:.2f}" if lat_val is not None else "—",
                "Status": "INTERCEPTED" if (step_val is not None and step_val <= t) else "PENDING",
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, height=300)
    else:
        st.info("No emitter interception records available for this scenario.")

    st.markdown("<div class='channel-header' style='font-size:0.85rem; margin-top:0.8rem;'>RECENT OBSERVED BAND ACTIVITY (LAST 30 STEPS)</div>", unsafe_allow_html=True)
    st.caption("Real hit/false-alarm counts per band from the replay, not a confidence-scored track.")
    recent = engine.get_recent_band_activity(30) if hasattr(engine, "get_recent_band_activity") else []
    if recent:
        st.dataframe(pd.DataFrame(recent), use_container_width=True, height=220)
    else:
        st.info("No band activity observed in the last 30 steps.")
