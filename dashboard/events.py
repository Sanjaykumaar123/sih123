"""Real-time signal interception and event telemetry log with CSV export."""

from typing import Any, Dict, List
import streamlit as st
from simulation.engine import SimulationEngine


def render_event_log(engine: SimulationEngine) -> None:
    """Render the real-time event telemetry log and CSV download."""
    snap = engine.get_snapshot()
    events = snap.get("recent_events", [])

    st.markdown(
        """
        <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;'>
            <div class='channel-header' style='font-size:0.85rem;'>SIGNAL INTERCEPTION & RF EVENT CONSOLE</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    ev_c1, ev_c2 = st.columns([7, 3])
    with ev_c1:
        flt = st.radio(
            "FILTER EVENTS",
            options=["ALL EVENTS", "CONFIRMED HITS ONLY", "FALSE ALARMS ONLY"],
            horizontal=True,
            label_visibility="collapsed",
            key="event_log_filter",
        )
    with ev_c2:
        csv_data = engine.export_events_csv()
        st.download_button(
            label="📥 EXPORT EVENT TELEMETRY (CSV)",
            data=csv_data,
            file_name=f"rf_events_{snap['scenario_name'].replace('.h5','')}_{snap['timestep']}steps.csv",
            mime="text/csv",
            use_container_width=True,
            key="export_events_btn",
        )

    # Filter events
    if flt == "CONFIRMED HITS ONLY":
        filtered_events = [e for e in events if e.get("event_type") == "CONFIRMED HIT"]
    elif flt == "FALSE ALARMS ONLY":
        filtered_events = [e for e in events if e.get("event_type") == "FALSE ALARM"]
    else:
        filtered_events = events

    if not filtered_events:
        st.markdown(
            """
            <div style='background-color:#0a0a0b; border:1px solid #2d2d30; border-radius:4px; padding:1.2rem; text-align:center; color:#8b949e; font-size:0.85rem;'>
                No signal interception events recorded yet. Press <strong>▶ START</strong> or <strong>⏭ STEP</strong> to begin scanning.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Render Event Rows
    html_rows = []
    for ev in filtered_events:
        is_hit = ev.get("event_type") == "CONFIRMED HIT"
        badge_col = "#00c853" if is_hit else "#ffab00"
        badge_txt = "★ HIT" if is_hit else "▲ FA"
        
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
                    <span>fc: <strong style='color:#c9d1d9;'>{ev.get('frequency_mhz')}</strong></span>
                    <span>PW: <strong style='color:#c9d1d9;'>{ev.get('pulse_width_us')}</strong></span>
                    <span>AoA: <strong style='color:#c9d1d9;'>{ev.get('aoa_deg')}</strong></span>
                    <span>Amp: <strong style='color:#c9d1d9;'>{ev.get('amplitude_dbm')}</strong></span>
                    <span>SNR: <strong style='color:#c9d1d9;'>{ev.get('snr_db')}</strong></span>
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
