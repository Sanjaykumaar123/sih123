"""Live Event Console: Real-time scrolling telemetry stream with search, filtering, and CSV export."""

from typing import Any, Dict, List
import pandas as pd
import streamlit as st
from simulation.engine import SimulationEngine


def render_event_console(engine: SimulationEngine) -> None:
    """Render the live scrolling event console with search, filters, and CSV export."""
    if hasattr(engine, "get_event_console_rows"):
        _render_structured_console(engine)
        return
    snap = engine.get_snapshot()
    events = snap.get("recent_events", [])

    st.markdown(
        """
        <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;'>
            <div class='channel-header' style='font-size:0.85rem;'>LIVE SIGNAL INTERCEPTION & OPERATIONAL EVENT CONSOLE</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    ev_c1, ev_c2, ev_c3 = st.columns([5, 3, 2])
    with ev_c1:
        flt = st.radio(
            "FILTER EVENTS",
            options=["ALL EVENTS", "INTERCEPTIONS ONLY", "FALSE ALARMS ONLY", "TRACK UPDATES ONLY"],
            horizontal=True,
            label_visibility="collapsed",
            key="event_console_filter",
        )
    with ev_c2:
        search_query = st.text_input("SEARCH LOG", placeholder="Filter by Band / Freq / Channel...", label_visibility="collapsed", key="event_console_search")
    with ev_c3:
        csv_data = engine.export_events_csv()
        st.download_button(
            label="📥 EXPORT CSV",
            data=csv_data,
            file_name=f"rf_events_{snap['scenario_name'].replace('.h5','')}_{snap['timestep']}steps.csv",
            mime="text/csv",
            use_container_width=True,
            key="export_console_events_btn",
        )

    # Filter events by type
    if flt == "INTERCEPTIONS ONLY":
        filtered_events = [e for e in events if e.get("event_type") in ("INTERCEPTION", "DETECTION")]
    elif flt == "FALSE ALARMS ONLY":
        filtered_events = [e for e in events if e.get("event_type") == "FALSE ALARM"]
    elif flt == "TRACK UPDATES ONLY":
        filtered_events = [e for e in events if e.get("event_type") in ("TRACK UPDATE", "TRACK_CREATED", "TRACK_CONFIRMED")]
    else:
        filtered_events = events

    # Search filter
    if search_query:
        q = search_query.lower()
        filtered_events = [
            e for e in filtered_events
            if q in str(e.get("band", "")).lower()
            or q in str(e.get("channel", "")).lower()
            or q in str(e.get("event_type", "")).lower()
            or q in str(e.get("frequency_mhz", "")).lower()
        ]

    if not filtered_events:
        st.markdown(
            """
            <div style='background-color:#0a0a0b; border:1px solid #2d2d30; border-radius:4px; padding:1.0rem; text-align:center; color:#8b949e; font-size:0.85rem;'>
                No signal interception events match the selected criteria.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Render Event Rows
    html_rows = []
    for ev in filtered_events:
        ev_type = ev.get("event_type", "")
        if ev_type in ("INTERCEPTION", "DETECTION"):
            badge_col = "#00c853"
            badge_txt = "★ INTERCEPT"
        elif ev_type == "FALSE ALARM":
            badge_col = "#ffab00"
            badge_txt = "▲ FALSE ALARM"
        elif "TRACK" in ev_type:
            badge_col = "#00e5ff"
            badge_txt = "◆ TRACK"
        else:
            badge_col = "#8b949e"
            badge_txt = ev_type
        
        def _na(key: str) -> str:
            v = ev.get(key)
            return "N/A" if v is None else str(v)

        html_rows.append(
            f"""
            <div style='display:flex; justify-content:space-between; align-items:center; padding:0.3rem 0.5rem; border-bottom:1px solid #2d2d30; font-family:monospace; font-size:0.75rem;'>
                <div style='display:flex; gap:0.6rem; align-items:center;'>
                    <span style='color:#8b949e;'>[{ev.get('time_s')}]</span>
                    <span style='color:#00e5ff; font-weight:700;'>{ev.get('channel')}</span>
                    <span style='color:#e6edee; font-weight:700;'>→ {ev.get('band')}</span>
                    <span style='color:{badge_col}; font-weight:800; background-color:{badge_col}22; padding:0.1rem 0.35rem; border-radius:3px;'>{badge_txt}</span>
                </div>
                <div style='display:flex; gap:0.8rem; color:#8b949e;'>
                    <span>fc: <strong style='color:#c9d1d9;'>{_na('frequency_mhz')}</strong></span>
                    <span>PW: <strong style='color:#c9d1d9;'>{_na('pulse_width_us')}</strong></span>
                    <span>AoA: <strong style='color:#c9d1d9;'>{_na('aoa_deg')}</strong></span>
                    <span>Amp: <strong style='color:#c9d1d9;'>{_na('amplitude_dbm')}</strong></span>
                    <span>SNR: <strong style='color:#c9d1d9;'>{_na('snr_db')}</strong></span>
                </div>
            </div>
            """
        )

    st.markdown(
        f"""
        <div style='background-color:#0a0a0b; border:1px solid #2d2d30; border-radius:4px; max-height:260px; overflow-y:auto;'>
            {''.join(html_rows)}
        </div>
        """,
        unsafe_allow_html=True,
    )


LEVEL_COLORS = {
    "INFO": "#8b949e", "COG": "#a371f7", "DETECT": "#00c853",
    "TRACK": "#00e5ff", "ALERT": "#d50000",
}


def _render_structured_console(engine: Any) -> None:
    """Section 6: TIME / LEVEL / SOURCE / EVENT console for the LIVE runtime, built
    entirely from LiveMissionRuntime.get_event_console_rows() (real lifecycle,
    strategy-change, and detection events - see core/live_mission.py). Supports
    newest-first ordering, filtering by level and source, and CSV export."""
    st.markdown("<div class='channel-header' style='font-size:0.85rem;'>OPERATIONAL EVENT CONSOLE</div>", unsafe_allow_html=True)
    rows = engine.get_event_console_rows(limit=200)

    c1, c2, c3, c4 = st.columns([3, 3, 3, 2])
    with c1:
        level_filter = st.multiselect("FILTER BY LEVEL", options=["INFO", "COG", "DETECT", "TRACK", "ALERT"], default=[], key="ec_level_filter")
    with c2:
        sources = sorted(set(r["source"] for r in rows))
        source_filter = st.multiselect("FILTER BY SOURCE", options=sources, default=[], key="ec_source_filter")
    with c3:
        search_query = st.text_input("SEARCH", placeholder="Search event text...", key="ec_search")
    with c4:
        if st.button("🗑 CLEAR LOG", use_container_width=True, key="ec_clear_log"):
            engine.op_event_log.clear()
            st.rerun()

    filtered = rows
    if level_filter:
        filtered = [r for r in filtered if r["level"] in level_filter]
    if source_filter:
        filtered = [r for r in filtered if r["source"] in source_filter]
    if search_query:
        q = search_query.lower()
        filtered = [r for r in filtered if q in r["event"].lower() or q in r["source"].lower()]

    if not filtered:
        st.info("No operational events match the selected criteria." if rows else "No events recorded yet — mission not started.")
    else:
        df = pd.DataFrame([{"Time (s)": r["time_s"], "Level": r["level"], "Source": r["source"], "Event": r["event"]} for r in filtered])
        st.dataframe(df, use_container_width=True, height=280)

    csv_lines = ["Time,Level,Source,Event"]
    for r in filtered:
        csv_lines.append(f"{r['time_s']},{r['level']},{r['source']},\"{r['event']}\"")
    st.download_button(
        "📥 EXPORT EVENT LOG (CSV)", data="\n".join(csv_lines),
        file_name=f"event_log_{getattr(engine, 'mission_id', 'session')}.csv",
        mime="text/csv", use_container_width=True, key="ec_export_structured_csv",
    )
