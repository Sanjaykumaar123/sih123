"""Operator Alert System (section 7).

Alerts are real, severity-tagged operational notifications derived entirely from
actual runtime state transitions (see core.live_mission.LiveMissionRuntime.
_observe_step_deltas / _sync_completion / pause_mission / etc.) - never generated for
visual effect. For the replay runtime (no live alert stream), a small set of alerts
is derived the same honest way, directly from the current snapshot.
"""

import csv
import io
import re
from typing import Any, Dict, List
import pandas as pd
import streamlit as st

SEVERITY_COLORS = {
    "INFO": "#00e5ff", "NOTICE": "#00c853", "WARNING": "#ffab00", "CRITICAL": "#d50000",
}


def _replay_alerts(engine: Any) -> List[Dict[str, Any]]:
    """Honest, minimal alert derivation for PlaybackController (no persistent alert
    log exists there - only this step's real snapshot is available)."""
    snap = engine.get_snapshot()
    alerts: List[Dict[str, Any]] = []
    step_td = snap.get("step_true_detections", [])
    step_fa = snap.get("step_false_alarms", [])
    status = snap.get("mission_status", "READY")
    t = snap.get("timestep", 0)

    if snap.get("total_scans", 0) == 0:
        alerts.append({"time_s": "0.00", "severity": "INFO", "event": "NO DATA — mission not started"})
    if step_td:
        alerts.append({"time_s": f"{t*0.05:.2f}", "severity": "NOTICE", "event": f"NEW TRUE INTERCEPTION on {', '.join(step_td)}"})
    if len(step_td) >= 2:
        alerts.append({"time_s": f"{t*0.05:.2f}", "severity": "NOTICE", "event": f"MULTIPLE DETECTIONS this step ({len(step_td)} channels)"})
    if step_fa:
        alerts.append({"time_s": f"{t*0.05:.2f}", "severity": "WARNING", "event": f"FALSE ALARM on {', '.join(step_fa)}"})
    if status == "PAUSED":
        alerts.append({"time_s": f"{t*0.05:.2f}", "severity": "NOTICE", "event": "MISSION PAUSED"})
    if status == "COMPLETE":
        alerts.append({"time_s": f"{t*0.05:.2f}", "severity": "NOTICE", "event": "MISSION COMPLETED"})
    return alerts


def render_alerts_panel(engine: Any, max_items: int = 8) -> None:
    """Compact, newest-first alert strip. No flashing animations - severity is
    communicated by color and an explicit label only."""
    if hasattr(engine, "get_alerts"):
        alerts = engine.get_alerts(limit=max_items)
    else:
        alerts = _replay_alerts(engine)[:max_items]

    st.markdown("<div class='channel-header' style='font-size:0.75rem;'>OPERATOR ALERTS</div>", unsafe_allow_html=True)
    if not alerts:
        st.markdown(
            "<div style='background-color:#0a0a0b; border:1px solid #2d2d30; border-radius:4px; padding:0.5rem; color:#8b949e; font-size:0.75rem;'>"
            "No alerts — TELEMETRY UNAVAILABLE until the mission starts.</div>",
            unsafe_allow_html=True,
        )
        return

    rows = "".join(
        f"<div style='display:flex; gap:0.6rem; align-items:center; padding:0.2rem 0.5rem; font-family:monospace; font-size:0.72rem; border-bottom:1px solid #2d2d30;'>"
        f"<span style='color:#8b949e;'>[{a.get('time_s','0.00')}s]</span>"
        f"<span style='color:{SEVERITY_COLORS.get(a.get('severity','INFO'), '#8b949e')}; font-weight:800;'>{a.get('severity','INFO')}</span>"
        f"<span style='color:#c9d1d9;'>{a.get('event','')}</span></div>"
        for a in alerts
    )
    st.markdown(f"<div style='background-color:#0a0a0b; border:1px solid #2d2d30; border-radius:4px;'>{rows}</div>", unsafe_allow_html=True)


def render_alert_summary_counts(engine: Any) -> None:
    """Compact severity-count row (Mission Control redesign section H: "OPERATOR
    ALERTS / CRITICAL 0 / WARNING 2 / INFO 4"). Purely additive - does not touch
    render_alerts_panel or render_attention_required; counts are computed from
    the exact same real alert stream those two already use (engine.get_alerts()
    or _replay_alerts()), so this can never disagree with them. Intended to be
    called immediately before render_alerts_panel(engine) so the existing
    "OPERATOR ALERTS" header serves both."""
    if hasattr(engine, "get_alerts"):
        all_alerts = engine.get_alerts(limit=200)
    else:
        all_alerts = _replay_alerts(engine)

    counts = {sev: 0 for sev in SEVERITY_COLORS}
    for a in all_alerts:
        sev = a.get("severity", "INFO")
        if sev in counts:
            counts[sev] += 1

    chips = "".join(
        f"<div style='display:flex; align-items:center; gap:0.3rem; font-family:monospace; font-size:0.72rem;'>"
        f"<span style='color:{SEVERITY_COLORS[sev]};'>●</span>"
        f"<span style='color:#8b949e;'>{sev}</span>"
        f"<strong style='color:#e6edee;'>{counts[sev]}</strong></div>"
        for sev in ("CRITICAL", "WARNING", "NOTICE", "INFO")
    )
    st.markdown(f"<div style='display:flex; gap:1.1rem; margin-bottom:0.35rem; flex-wrap:wrap;'>{chips}</div>", unsafe_allow_html=True)


ACTIONABLE_SEVERITIES = ("WARNING", "CRITICAL")


def render_attention_required(engine: Any, max_items: int = 5) -> None:
    """Section 8: OPERATOR ATTENTION - only actionable (WARNING/CRITICAL) items, a
    strict subset of the full alert stream above. When nothing needs attention this
    says so explicitly rather than showing an empty box."""
    if hasattr(engine, "get_alerts"):
        alerts = engine.get_alerts(limit=50)
    else:
        alerts = _replay_alerts(engine)

    actionable = [a for a in alerts if a.get("severity") in ACTIONABLE_SEVERITIES][:max_items]

    st.markdown("<div class='channel-header' style='font-size:0.75rem;'>OPERATOR ATTENTION</div>", unsafe_allow_html=True)
    if not actionable:
        st.markdown(
            "<div style='background-color:#0b0f1e; border:1px solid #00c853; border-radius:4px; padding:0.5rem; color:#00c853; font-size:0.78rem; font-weight:700;'>"
            "✓ NO ACTIVE OPERATOR ACTION REQUIRED</div>",
            unsafe_allow_html=True,
        )
        return

    rows = "".join(
        f"<div style='display:flex; gap:0.6rem; align-items:center; padding:0.25rem 0.5rem; font-family:monospace; font-size:0.75rem; border-bottom:1px solid #2d2d30;'>"
        f"<span style='color:{SEVERITY_COLORS.get(a.get('severity','WARNING'), '#ffab00')}; font-weight:800;'>⚠ {a.get('severity','WARNING')}</span>"
        f"<span style='color:#c9d1d9;'>{a.get('event','')}</span></div>"
        for a in actionable
    )
    st.markdown(f"<div style='background-color:#0a0a0b; border:1px solid #ffab00; border-radius:4px;'>{rows}</div>", unsafe_allow_html=True)


def _extract_source(event_text: str) -> str:
    """Best-effort SOURCE derived from the real alert/event text itself (e.g. an
    alert already reads "... on F12 (CH03)") - parses an existing real string, never
    fabricates a new field. Falls back to SYSTEM for lifecycle-only alerts (paused /
    resumed / mission completed / etc.) that don't reference a band or channel."""
    m = re.search(r"\bon (F\d{2})", event_text)
    if m:
        return m.group(1)
    m = re.search(r"\((CH\d+)\)", event_text)
    if m:
        return m.group(1)
    return "SYSTEM"


def render_alerts_view(engine: Any) -> None:
    """Full ALERTS & EVENTS page (Stitch screen 'Alerts & Events').

    Reuses the existing real alert stream verbatim - LiveMissionRuntime.get_alerts()
    (built in core/live_mission.py from real observed per-step deltas) or, in
    REPLAY VERIFIED RUN, this module's own _replay_alerts() (derived the same honest
    way from the current snapshot). No new event-generation logic is added here;
    this function only filters, tabulates, and exports what those two functions
    already produce. ACKNOWLEDGED/STATUS is a pure operator UI action (tracked in
    st.session_state), not a fabricated telemetry field.
    """
    st.markdown("<div class='system-title'>ALERTS & EVENTS</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='system-subtitle'>REAL OPERATOR ALERT STREAM — SEVERITY-TAGGED STATE TRANSITIONS ONLY, NEVER GENERATED FOR VISUAL EFFECT</div>",
        unsafe_allow_html=True,
    )

    is_live = hasattr(engine, "get_alerts")
    raw_alerts = engine.get_alerts(limit=200) if is_live else _replay_alerts(engine)

    if "acknowledged_alert_keys" not in st.session_state:
        st.session_state.acknowledged_alert_keys = set()
    ack_keys = st.session_state.acknowledged_alert_keys

    def _row_key(a: Dict[str, Any]) -> str:
        return f"{a.get('time_s')}|{a.get('severity')}|{a.get('event')}"

    rows: List[Dict[str, Any]] = []
    for a in raw_alerts:
        k = _row_key(a)
        rows.append({
            "_key": k,
            "TIME": a.get("time_s", "0.00"),
            "SEVERITY": a.get("severity", "INFO"),
            "SOURCE": _extract_source(a.get("event", "")),
            "EVENT": a.get("event", ""),
            "STATUS": "ACKNOWLEDGED" if k in ack_keys else "NEW",
        })

    # Real severities actually present in this stream (INFO/NOTICE/WARNING/CRITICAL -
    # see SEVERITY_COLORS above), not a hardcoded guess at what the data contains.
    real_severities = [s for s in ("CRITICAL", "WARNING", "NOTICE", "INFO") if any(r["SEVERITY"] == s for r in rows)]
    filter_options = ["ALL"] + real_severities + ["ACKNOWLEDGED"]

    f_c1, f_c2, f_c3, f_c4 = st.columns([5, 2, 2, 2])
    with f_c1:
        chosen_filter = st.radio(
            "FILTER", options=filter_options, horizontal=True,
            key="alerts_filter", label_visibility="collapsed",
        )
    with f_c2:
        if st.button("✓ ACKNOWLEDGE ALL", use_container_width=True, key="btn_ack_all_alerts"):
            for r in rows:
                ack_keys.add(r["_key"])
            st.rerun()
    with f_c3:
        if st.button(
            "🗑 CLEAR LOG", use_container_width=True, key="btn_clear_alerts_log",
            disabled=not is_live,
            help=None if is_live else (
                "REPLAY VERIFIED RUN alerts are derived fresh from the artifact each "
                "step — there is no persistent log to clear in this mode."
            ),
        ):
            engine.alert_log.clear()
            st.rerun()
    with f_c4:
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["Time", "Severity", "Source", "Event", "Status"])
        for r in rows:
            writer.writerow([r["TIME"], r["SEVERITY"], r["SOURCE"], r["EVENT"], r["STATUS"]])
        st.download_button(
            "📥 EXPORT CSV", data=buf.getvalue(),
            file_name=f"alerts_events_{getattr(engine, 'mission_id', 'session')}.csv",
            mime="text/csv", use_container_width=True, key="btn_export_alerts_csv",
        )

    if chosen_filter == "ACKNOWLEDGED":
        filtered = [r for r in rows if r["STATUS"] == "ACKNOWLEDGED"]
    elif chosen_filter != "ALL":
        filtered = [r for r in rows if r["SEVERITY"] == chosen_filter]
    else:
        filtered = rows

    st.markdown("<div class='channel-header' style='font-size:0.8rem; margin-top:0.4rem;'>ALERT & EVENT LOG</div>", unsafe_allow_html=True)
    if not filtered:
        if not rows:
            # Consistent with render_attention_required's "no action needed" state
            # elsewhere in this module - same visual language for the same real fact.
            st.markdown(
                "<div style='background-color:#0a0a0b; border:1px solid #00c853; border-radius:4px; padding:0.6rem; "
                "color:#00c853; font-size:0.8rem; font-weight:700;'>✓ NO ACTIVE ALERTS</div>",
                unsafe_allow_html=True,
            )
        else:
            st.info("No alerts match the selected filter.")
    else:
        display_df = pd.DataFrame([{k: v for k, v in r.items() if k != "_key"} for r in filtered])
        st.dataframe(display_df, use_container_width=True, height=320)

    if filtered:
        st.markdown("<div class='channel-header' style='font-size:0.8rem; margin-top:0.6rem;'>🔬 EVENT INSPECTOR</div>", unsafe_allow_html=True)
        opts = [f"[{r['TIME']}s] {r['SEVERITY']} — {r['EVENT'][:60]}" for r in filtered]
        idx = st.selectbox(
            "SELECT EVENT", options=list(range(len(opts))),
            format_func=lambda i: opts[i], key="alerts_inspect_select",
        )
        r = filtered[idx]
        i_c1, i_c2 = st.columns(2)
        with i_c1:
            st.markdown(f"**Time:** `{r['TIME']}s`")
            st.markdown(f"**Severity:** `{r['SEVERITY']}`")
            st.markdown(f"**Source:** `{r['SOURCE']}`")
        with i_c2:
            st.markdown(f"**Status:** `{r['STATUS']}`")
            st.markdown(f"**Full event text:** {r['EVENT']}")
