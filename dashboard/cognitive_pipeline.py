"""Cognitive decision pipeline status strip: OBSERVE -> ... -> LEARN.

Every stage runs once, in order, each timestep - that is architectural fact, not
something inferred from timing. What is highlighted per node is real per-step data
(this step's selected bands, hits, false alarms, reward, chosen strategy), never an
animation invented for effect.
"""

from __future__ import annotations
from typing import Any, Dict
import streamlit as st


def render_pipeline(snap: Dict[str, Any], compact: bool = False) -> None:
    sel = snap.get("selected_bands", [])
    step_td = snap.get("step_true_detections", [])
    step_fa = snap.get("step_false_alarms", [])
    reward = snap.get("latest_reward", 0.0)
    strat = snap.get("current_strategy", "BALANCED")
    k = snap.get("k_channels", 5)

    if step_td:
        detect_txt, detect_col = f"{len(step_td)} hit(s)", "#00c853"
    elif step_fa:
        detect_txt, detect_col = f"{len(step_fa)} false alarm(s)", "#ffab00"
    else:
        detect_txt, detect_col = "quiet", "#8b949e"

    reward_col = "#00c853" if reward > 0 else ("#ffab00" if reward < 0 else "#8b949e")

    nodes = [
        ("1. OBSERVE", f"{k} channels", "#00e5ff"),
        ("2. UPDATE BELIEF", "per-band P(active)", "#00e5ff"),
        ("3. ANALYZE TEMPORAL", "recurrence check", "#00e5ff"),
        ("4. SCORE BANDS", "rank candidates", "#00e5ff"),
        ("5. SELECT BANDS", f"K={k}", "#a371f7"),
        ("6. SCAN", ", ".join(sel) if sel and not compact else f"{len(sel)} bands", "#a371f7"),
        ("7. DETECT", detect_txt, detect_col),
        ("8. REWARD", f"{reward:+.2f}", reward_col),
        ("9. LEARN", strat, "#a371f7"),
    ]

    html = ["<div class='trace-container' style='padding:0.45rem; margin-top:0.4rem;'>"]
    for label, detail, col in nodes:
        html.append(
            f"<div class='trace-step trace-active' style='color:{col}; border-color:{col}88;'>"
            f"{label}<div style='font-size:0.6rem; color:#8b949e; font-weight:500;'>{detail}</div></div>"
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)

    if not compact:
        st.caption(
            "Every stage executes once per timestep, in this order — highlighted labels "
            "show real values from the current step, not a simulated timing animation."
        )
