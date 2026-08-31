"""Centralized Stitch design-system tokens for the operator workstation UI.

Presentation layer ONLY: color/typography/spacing/radius/border constants and the
single CSS block app.py injects once per render. No backend, data, or scheduling
logic lives here - this module is never imported by rf_env/, core/, simulation/,
data_adapter/, evaluation/, or experiments/.

Source: Stitch project "Cognitive RF Mission Control Console"
(projects/13287453408275741277), design system "Cognitive RF Operations Workstation" -
inspected read-only via StitchMCP (list_projects/get_project/list_screens/get_screen/
list_design_systems), never modified.

Six semantic roles were given explicit hex values by the operator brief: Base, Panels,
Primary, Nominal, Caution, Critical, Borders. Everything else (body/muted text, the
existing purple "cognitive/learning" accent used for PREDICT/BALANCED/tracks) is left
close to its prior value rather than invented, since the Stitch brief did not specify
those roles and forcing them into one of the four semantic colors above would make
PREDICT/BALANCED/track-state colors misleading (e.g. reusing CAUTION amber for a
strategy that is not a warning).

Enterprise Design System layer (added on top of the above): a fuller design-token/
component system (surface hierarchy, typography scale, spacing scale, radius scale,
restrained shadows, canonical STATUS/SEVERITY vocabularies, KPI/panel/alert-row/
empty-state/loading-skeleton component helpers) modeled on a reference enterprise
operations dashboard screenshot the user supplied, reconciled against the real
status/severity mappings already live in dashboard/live_operations.py and
dashboard/alerts.py so nothing here contradicts them.

Global Shell redesign (second pass, on top of the above): a second reference
dashboard screenshot + an explicit, prescriptive design brief asked for a more
restrained "modern enterprise SaaS" palette/geometry, replacing the earlier
Stitch-sourced cyan-heavy palette and flat 4px control radius outright - the six
core color constants, the three-tier text hierarchy, and the radius scale below
were updated to that brief's exact values. This is a deliberate, requested global
retint, not a mistake: the earlier Stitch project remains the historical origin of
this module's structure (typography roles, spacing unit, component shapes), but the
CURRENT color/radius values below are the second brief's, not Stitch's. Scope for
this pass was the GLOBAL SHELL only (this file's shared CSS + dashboard/
live_operations.py's header/sidebar-adjacent chrome) - per-view modules
(dashboard/spectrum.py, receiver_panel.py, decision_panel.py, alerts.py, tracks.py,
performance.py, system.py, help.py, event_console.py, etc.) still contain their own
hardcoded hex literals from the earlier cyan palette and were deliberately left
untouched (redesigning their content is explicitly out of scope for a "global shell
only" phase) - so those views will look visually inconsistent with the new shell
until a later phase migrates them to these tokens. Not a bug; a known, reported
seam.
"""

# --- Core palette (Global Shell redesign: modern enterprise SaaS operations
# dashboard reference, restrained blue accent - replaces the earlier Stitch
# cyan-heavy palette per explicit new design brief. See module docstring.) ---
COLOR_BASE = "#0B0D12"           # Level 0 - app shell background
COLOR_SIDEBAR = "#11141C"        # Sidebar - distinct, slightly elevated vs base
COLOR_PANEL = "#171A23"          # Level 1 - primary panels/cards
COLOR_PANEL_RAISED = "#1D212C"   # Level 2 - overlays / nested blocks
COLOR_BORDER = "rgba(255,255,255,0.08)"  # 1px hairline border, used everywhere
COLOR_PRIMARY = "#4F8CFF"        # interactive / focus / active telemetry - restrained blue
COLOR_PRIMARY_SOFT = "#8FB4FF"   # primary-on-dark hover/text variant
COLOR_NOMINAL = "#22C55E"        # success / confirmed / healthy
COLOR_CAUTION = "#F59E0B"        # warning / false-alarm / degraded
COLOR_CRITICAL = "#EF4444"       # error / critical / destructive

# Accent for a 5th real semantic category (cognitive / learning / PREDICT-
# BALANCED strategy / track state) that the 4 core semantic colors above do not
# cover - see module docstring.
COLOR_COGNITIVE = "#A78BFA"

# Text - three-tier hierarchy per the global-shell brief.
COLOR_TEXT = "#F3F4F6"           # Primary text
COLOR_TEXT_MUTED = "#9CA3AF"     # Secondary text
COLOR_TEXT_FAINT = "#6B7280"     # Muted text - placeholders, disabled labels, "N/A", timestamps
COLOR_OVERLAY = COLOR_PANEL_RAISED  # Level 3 surface - tooltips/popovers/dropdown menus (alias, same tone)

# --- Typography (Stitch: Inter for chrome, JetBrains Mono for telemetry) ---
FONT_UI = '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif'
FONT_MONO = '"JetBrains Mono", ui-monospace, SFMono-Regular, "Roboto Mono", Consolas, monospace'

# Typography scale (role -> font/size/weight/line-height/letter-spacing). Additive:
# existing classes (.system-title, .metric-val, etc.) are untouched; this is the
# named scale new/refactored components should draw from.
TYPE_SCALE = {
    "display":      {"size": "1.75rem", "weight": 700, "line": "2.1rem",  "letter": "-0.01em"},
    "headline":     {"size": "1.1rem",  "weight": 700, "line": "1.4rem",  "letter": "0"},
    "body":         {"size": "0.8rem",  "weight": 400, "line": "1.2rem",  "letter": "0"},
    "label_caps":   {"size": "0.63rem", "weight": 700, "line": "0.9rem",  "letter": "0.06em"},
    "telemetry_lg": {"size": "1.25rem", "weight": 700, "line": "1.5rem",  "letter": "0"},
    "telemetry_md": {"size": "0.8rem",  "weight": 500, "line": "1.1rem",  "letter": "0"},
    "telemetry_sm": {"size": "0.68rem", "weight": 400, "line": "0.9rem",  "letter": "0.02em"},
}

# --- Spacing / sizing (Stitch tokens) ---
SPACE_UNIT = "4px"
SPACE_GUTTER = "12px"
SPACE_PANEL_PADDING = "16px"
SPACE_PAGE_MARGIN = "24px"
CONTROL_HEIGHT_SM = "28px"
CONTROL_HEIGHT_MD = "36px"
# Numbered spacing scale (4/8/12/16/20/24/32px) for programmatic use in components.
SPACE = {1: "4px", 2: "8px", 3: "12px", 4: "16px", 5: "20px", 6: "24px", 7: "32px"}

# Radius scale. Global-shell redesign moves off the old flat-military 4px scale
# toward a modern SaaS-dashboard geometry: compact controls get a gentle round,
# cards/panels get a soft 12-16px round per the new design brief's "Cards /
# Surfaces" section. RADIUS is still the name used by control-level CSS
# (buttons/inputs/badges) below; RADIUS_CARD is used by panels/KPI cards/chart
# containers.
RADIUS = "8px"        # controls: buttons / inputs / tech-badges
RADIUS_SM = "6px"      # chips / severity pills / status pips
RADIUS_CARD = "14px"   # KPI cards, panels, chart containers
RADIUS_XL = "20px"     # hero / major containers
BORDER = f"1px solid {COLOR_BORDER}"

# Shadow scale - deliberately restrained per the global-shell brief ("very
# restrained shadows... use elevation primarily for KPI cards, important
# operational panels, dialogs, popovers, highlighted states"). SHADOW_SM is
# now applied to KPI cards/surface panels/the header bar (a soft lift, not a
# glow); SHADOW_MD/LG stay reserved for genuinely elevated overlays (dropdown
# menus, tooltips, popovers).
SHADOW_SM = "0 1px 2px rgba(0,0,0,0.24)"
SHADOW_MD = "0 4px 12px rgba(0,0,0,0.32)"
SHADOW_LG = "0 8px 24px rgba(0,0,0,0.40)"

# Nav rail icon prefixes (Stitch's 8-item icon rail approximated with emoji glyphs -
# Streamlit's sidebar radio has no native icon-rail widget). Order matches the
# operator-oriented 8-item navigation from the implementation brief.
NAV_ICONS = {
    "SOLUTION EXPLAINER": "",
    "MISSION CONTROL": "",
    "SPECTRUM": "",
    "COGNITIVE ENGINE": "",
    "RECEIVER ARRAY": "",
    "TRACKS": "",
    "ALERTS": "",
    "ANALYTICS": "",
    "SYSTEM": "",
}


# Canonical operational-status vocabulary (color + label) for receiver channels,
# subsystem health, mission status, track state, etc. Reconciled against the
# real status_map already live in dashboard/live_operations.py's
# render_top_status_bar (IDLE/READY=primary cyan, RUNNING=nominal green,
# PAUSED=caution amber, STOPPED=muted, COMPLETE=cognitive purple, ERROR=critical
# red) and core.state's ChannelState values - not a competing definition. No
# module reads this dict yet (each still has its own local status_map/color
# lookup); this is the shared source those migrate to in a later phase.
STATUS = {
    "IDLE": (COLOR_PRIMARY, "IDLE"),
    "READY": (COLOR_PRIMARY, "READY"),
    "SCANNING": (COLOR_PRIMARY, "SCANNING"),
    "RUNNING": (COLOR_NOMINAL, "RUNNING"),
    "ACTIVE": (COLOR_NOMINAL, "ACTIVE"),
    "HEALTHY": (COLOR_NOMINAL, "HEALTHY"),
    "ONLINE": (COLOR_NOMINAL, "ONLINE"),
    "SIGNAL_DETECTED": (COLOR_NOMINAL, "SIGNAL DETECTED"),
    "TRACKING": (COLOR_COGNITIVE, "TRACKING"),
    "PAUSED": (COLOR_CAUTION, "PAUSED"),
    "FALSE_ALARM": (COLOR_CAUTION, "FALSE ALARM"),
    "DEGRADED": (COLOR_CAUTION, "DEGRADED"),
    "STOPPED": (COLOR_TEXT_MUTED, "STOPPED"),
    "STANDBY": (COLOR_TEXT_MUTED, "STANDBY"),
    "QUIET": (COLOR_TEXT_MUTED, "QUIET"),
    "COMPLETE": (COLOR_COGNITIVE, "MISSION COMPLETE"),
    "ERROR": (COLOR_CRITICAL, "ERROR"),
    "OFFLINE": (COLOR_CRITICAL, "OFFLINE"),
    "CRITICAL": (COLOR_CRITICAL, "CRITICAL"),
    "N/A": (COLOR_TEXT_FAINT, "N/A"),
}

# Canonical 4-severity alert/event vocabulary. Reconciled against the real
# SEVERITY_COLORS already live in dashboard/alerts.py (INFO=cyan, NOTICE=green,
# WARNING=amber, CRITICAL=red) with HIGH/MEDIUM/LOW added as aliases matching
# the reference dashboard's own severity-chip vocabulary (HIGH~=WARNING amber,
# MEDIUM~=INFO cyan, LOW~=muted). Same "shared source, not yet wired in" status
# as STATUS above.
SEVERITY = {
    "CRITICAL": (COLOR_CRITICAL, "CRITICAL"),
    "HIGH": (COLOR_CAUTION, "HIGH"),
    "WARNING": (COLOR_CAUTION, "WARNING"),
    "MEDIUM": (COLOR_PRIMARY, "MEDIUM"),
    "NOTICE": (COLOR_NOMINAL, "NOTICE"),
    "INFO": (COLOR_PRIMARY, "INFO"),
    "LOW": (COLOR_TEXT_MUTED, "LOW"),
}


# Data-provenance taxonomy (Step 17 section 4/6): every value shown anywhere in this
# workstation is exactly one of these four things. Naming it explicitly, in one place,
# lets any view mark a value's provenance with a single consistent badge instead of
# ad-hoc captions that drift out of sync with each other.
PROVENANCE = {
    "REAL": ("REAL RUNTIME DATA", COLOR_NOMINAL, "Computed this run, this step, by the actual scheduler/detector/tracker."),
    "POST_HOC": ("POST-HOC VERIFIED DATA", COLOR_PRIMARY, "Ground-truth-derived, used only for display/evaluation after the fact - never fed to the scheduler."),
    "STATIC": ("STATIC ARCHITECTURE CONSTANT", COLOR_COGNITIVE, "A fixed, real configuration fact (e.g. 500 MHz-18 GHz, N=50 bands) - not a live measurement."),
    "NA": ("N/A", COLOR_TEXT_MUTED, "No real value exists for this field right now - never filled with a placeholder."),
}


def provenance_badge(kind: str) -> str:
    """Return a small inline HTML badge naming a value's data provenance. `kind` is
    one of PROVENANCE's keys (REAL / POST_HOC / STATIC / NA)."""
    label, color, tooltip = PROVENANCE.get(kind, PROVENANCE["NA"])
    return (
        f"<span class='tech-badge' style='background-color:{color}18; color:{color}; "
        f"border-color:{color}66; font-size:0.6rem;' title='{tooltip}'>{label}</span>"
    )


def stat_tile(label: str, value: str, color: str = None) -> str:
    """One compact telemetry stat tile (label + mono value) for the header's
    live-telemetry zone. Pure string formatting - no data computed here."""
    style = f" color:{color};" if color else ""
    return (
        f"<div class='stat-tile'><span class='stat-lbl'>{label}</span>"
        f"<span class='stat-val' style='{style}'>{value}</span></div>"
    )


def stat_row(tiles: list) -> str:
    """Wrap a list of stat_tile() HTML strings into one aligned row."""
    return f"<div style='display:flex; align-items:flex-start; flex-wrap:wrap;'>{''.join(tiles)}</div>"


# -----------------------------------------------------------------------------
# Enterprise Design System — component layer (additive). Pure string-formatting
# helpers, same shape/spirit as stat_tile()/stat_row()/provenance_badge() above.
# None of these are called by any dashboard/*.py view yet — that adoption is a
# later phase; this phase only builds the shared vocabulary/component layer so
# every view can eventually draw from one definition instead of N hand-copied
# HTML f-strings. No backend/data logic here — pure presentation over whatever
# a caller passes in.
# -----------------------------------------------------------------------------

def status_pip(status_key: str) -> str:
    """8px status pip + colored label from the canonical STATUS vocabulary.
    Unknown keys fall back to N/A rather than guessing a color."""
    color, label = STATUS.get(str(status_key).upper(), STATUS["N/A"])
    pulse = " status-pip-critical" if str(status_key).upper() in ("ERROR", "CRITICAL", "OFFLINE") else ""
    return (
        f"<span class='status-pip{pulse}' style='background-color:{color};'></span>"
        f"<span style='color:{color}; font-weight:700;'>{label}</span>"
    )


def severity_badge(severity: str) -> str:
    """Small rounded severity chip (CRITICAL/HIGH/WARNING/MEDIUM/NOTICE/INFO/LOW)
    from the canonical SEVERITY vocabulary - matches the reference dashboard's
    alert-list severity pills."""
    color, label = SEVERITY.get(str(severity).upper(), SEVERITY["INFO"])
    return f"<span class='severity-chip' style='background-color:{color}1f; color:{color}; border-color:{color}55;'>{label}</span>"


def panel(title: str, body_html: str, badge_html: str = "", subtitle: str = "") -> str:
    """One bordered, tonal Level-1 content panel with a header bar - the shared
    replacement for the ad-hoc '---' section dividers and hand-copied panel HTML
    currently repeated across dashboard/*.py views."""
    sub = f"<div class='panel-subtitle'>{subtitle}</div>" if subtitle else ""
    return (
        f"<div class='surface-panel'>"
        f"<div class='surface-panel-header'><span class='panel-title'>{title}</span>{badge_html}</div>"
        f"{sub}<div class='surface-panel-body'>{body_html}</div></div>"
    )


def kpi_card(label: str, value: str, delta: str = "", delta_dir: str = "neutral", icon: str = "",
             icon_color: str = None, size: str = "standard") -> str:
    """Compact KPI stat card: icon chip + label + big value + colored delta line
    (matches the reference dashboard's top KPI row). delta_dir is 'up' (nominal
    green), 'down' (critical red), or 'neutral' (muted) - caller decides which,
    since only the caller knows whether the underlying value going up is good.
    size='hero' renders a visually louder variant (larger value, bigger icon
    chip, colored left accent) for the handful of KPIs that should dominate the
    row (Mission Control redesign: "Do NOT make every KPI equally visually
    loud"); size='standard' (default) is the original, quieter treatment."""
    icon_color = icon_color or COLOR_PRIMARY
    delta_color = {"up": COLOR_NOMINAL, "down": COLOR_CRITICAL, "neutral": COLOR_TEXT_MUTED}.get(delta_dir, COLOR_TEXT_MUTED)
    arrow = {"up": "▲", "down": "▼", "neutral": ""}.get(delta_dir, "")
    is_hero = size == "hero"
    card_cls = "kpi-card kpi-card-hero" if is_hero else "kpi-card"
    value_cls = "kpi-value-hero" if is_hero else "kpi-value"
    accent = f" style='border-left:3px solid {icon_color};'" if is_hero else ""
    icon_html = f"<div class='kpi-icon' style='background-color:{icon_color}1f; color:{icon_color};'>{icon}</div>" if icon else ""
    delta_html = f"<div class='kpi-delta' style='color:{delta_color};'>{arrow} {delta}</div>" if delta else ""
    return (
        f"<div class='{card_cls}'{accent}>{icon_html}<div class='kpi-body'>"
        f"<div class='kpi-label'>{label}</div><div class='{value_cls}'>{value}</div>{delta_html}</div></div>"
    )


def mission_progress_bar(pct: float, step_label: str, elapsed_label: str, remaining_label: str = None) -> str:
    """Compact, restrained mission-progress bar (Mission Control redesign section
    D) - a thin filled track, not a "huge neon progress bar". `remaining_label`
    is omitted entirely (not shown as a fabricated placeholder) when the caller
    has no honest value for it."""
    pct_clamped = max(0.0, min(100.0, pct))
    remaining_html = (
        f"<div class='mp-stat'><span class='mp-stat-lbl'>REMAINING</span><span class='mp-stat-val'>{remaining_label}</span></div>"
        if remaining_label else ""
    )
    return (
        f"<div class='mission-progress'>"
        f"<div class='mission-progress-header'><span class='panel-title'>MISSION PROGRESS</span>"
        f"<span class='mission-progress-pct'>{pct:.0f}%</span></div>"
        f"<div class='mission-progress-track'><div class='mission-progress-fill' style='width:{pct_clamped:.1f}%;'></div></div>"
        f"<div class='mission-progress-stats'>"
        f"<div class='mp-stat'><span class='mp-stat-lbl'>STEP</span><span class='mp-stat-val'>{step_label}</span></div>"
        f"<div class='mp-stat'><span class='mp-stat-lbl'>ELAPSED</span><span class='mp-stat-val'>{elapsed_label}</span></div>"
        f"{remaining_html}</div></div>"
    )


def section_divider(label: str = "") -> str:
    """Thin, subtle zone divider - the replacement for a literal '---' rule
    between Mission Control's major zones (KPI Overview / Workspace / Secondary
    Information), per "create visual hierarchy... use grouping and whitespace"
    rather than another bordered box."""
    lbl_html = f"<div class='section-divider-label'>{label}</div>" if label else ""
    return f"<div class='section-divider-wrap'>{lbl_html}<div class='section-divider'></div></div>"


def alert_row(title: str, subtitle: str, time_ago: str, severity: str, icon: str = "!") -> str:
    """One alert-list row: severity-colored icon chip + title/subtitle + right-
    aligned time + severity chip - matches the reference dashboard's ACTIVE
    ALERTS panel layout."""
    color, _ = SEVERITY.get(str(severity).upper(), SEVERITY["INFO"])
    return (
        f"<div class='alert-row'><div class='alert-row-icon' style='background-color:{color}1f; color:{color};'>{icon}</div>"
        f"<div class='alert-row-body'><div class='alert-row-title'>{title}</div>"
        f"<div class='alert-row-subtitle'>{subtitle}</div></div>"
        f"<div class='alert-row-meta'><div class='alert-row-time'>{time_ago}</div>{severity_badge(severity)}</div></div>"
    )


def empty_state(message: str, icon: str = "○") -> str:
    """Centered empty-state block - the shared replacement for the various
    hand-written 'No X yet' blocks across dashboard/*.py. Presentation only:
    callers still decide, from real state, when there is genuinely nothing to
    show - this never manufactures a message about data it hasn't checked."""
    return f"<div class='empty-state'><div class='empty-state-icon'>{icon}</div><div class='empty-state-msg'>{message}</div></div>"


def loading_skeleton(rows: int = 3) -> str:
    """Shimmering placeholder bars for a panel that has no real data yet. Purely
    visual - never a substitute for real data, and callers must still gate this
    behind an honest 'no data yet' check rather than showing it over stale or
    fabricated values."""
    bars = "".join(f"<div class='skeleton-bar' style='width:{85 - i * 12}%;'></div>" for i in range(max(1, rows)))
    return f"<div class='skeleton-block'>{bars}</div>"


def get_custom_css() -> str:
    """Return the single <style> block app.py injects once per render. Reuses every
    existing class name already referenced across dashboard/*.py's inline HTML
    (system-title, tech-badge, channel-card, metric-card, decision-card,
    trace-container/step, channel-header/band/freq, metric-lbl/val/imp) so restyling
    is a token swap here rather than a rewrite of every view module."""
    return f"""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@500;700;800&family=Inter:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        html, body, .stApp {{
            background-color: {COLOR_BASE};
            color: {COLOR_TEXT};
            font-family: {FONT_UI};
        }}
        .stApp * {{
            border-radius: 0;
        }}
        /* ---- Glassmorphism Card Styling ---- */
        .glass-card, .channel-card, .metric-card, .decision-card, div[data-testid="stExpander"] {{
            background: rgba(23, 26, 35, 0.8) !important;
            backdrop-filter: blur(12px) !important;
            -webkit-backdrop-filter: blur(12px) !important;
            border: 1px solid rgba(79, 140, 255, 0.2) !important;
            border-radius: {RADIUS_CARD} !important;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35) !important;
            transition: all 0.25s ease-in-out !important;
        }}
        .glass-card:hover, .channel-card:hover, .metric-card:hover, .decision-card:hover {{
            border-color: rgba(0, 240, 255, 0.45) !important;
            box-shadow: 0 6px 24px rgba(0, 240, 255, 0.15) !important;
        }}
        /* ---- Hide default clunky Streamlit radio input squares in sidebar navigation ---- */
        div[data-testid="stSidebar"] div[data-testid="stRadio"] label > div:first-child {{
            display: none !important;
        }}
        /* ---- Sleek modern navigation item styling ---- */
        div[data-testid="stSidebar"] div[data-testid="stRadio"] label {{
            padding: 0.5rem 0.8rem !important;
            border-radius: 6px !important;
            margin-bottom: 0.2rem !important;
            border-left: 3px solid transparent !important;
            transition: all 0.2s ease-in-out !important;
            background: transparent !important;
            color: #9CA3AF !important;
            font-weight: 500 !important;
            font-size: 0.85rem !important;
        }}
        div[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {{
            background: rgba(79, 140, 255, 0.1) !important;
            color: #4F8CFF !important;
            border-left: 3px solid rgba(79, 140, 255, 0.5) !important;
        }}
        /* Active navigation item styling with glowing cyan left border & highlight */
        div[data-testid="stSidebar"] div[data-testid="stRadio"] label[aria-checked="true"],
        div[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) {{
            background: rgba(0, 240, 255, 0.12) !important;
            border-left: 3px solid #00F0FF !important;
            color: #00F0FF !important;
            font-weight: 700 !important;
        }}
        /* ---- Headings / labels (Outfit & Inter) ---- */
        h1, h2, h3, h4 {{
            font-family: "Outfit", {FONT_UI} !important;
        }}
        .system-title {{
            font-family: "Outfit", {FONT_UI};
            font-size: 1.15rem; font-weight: 800; letter-spacing: 0.05em;
            color: {COLOR_PRIMARY}; margin-bottom: 0.1rem; text-transform: uppercase;
        }}
        .system-subtitle {{
            font-family: {FONT_UI};
            font-size: 0.68rem; color: {COLOR_TEXT_MUTED}; letter-spacing: 0.06em;
            font-weight: 700; margin-bottom: 0.5rem; text-transform: uppercase;
        }}
        .channel-header {{
            font-family: {FONT_UI};
            font-size: 0.68rem; font-weight: 700; color: {COLOR_TEXT_MUTED};
            letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 0.25rem;
        }}
        .metric-lbl {{
            font-family: {FONT_UI};
            font-size: 0.63rem; font-weight: 700; color: {COLOR_TEXT_MUTED};
            text-transform: uppercase; letter-spacing: 0.05em;
            word-break: keep-all; overflow-wrap: normal;
        }}
        /* At narrower desktop widths (~1366px) equal-width button/KPI columns can
           get tighter than a label's longest word - keep-all prevents mid-word
           breaks ("STOP MISSIO/N") in favor of wrapping at whitespace only. */
        .stButton button p, .metric-card, .channel-card {{
            word-break: keep-all !important; overflow-wrap: normal !important;
        }}
        /* ---- Telemetry / numeric (JetBrains Mono) ---- */
        .channel-band, .channel-freq, .metric-val, .metric-imp, .trace-step,
        .stDataFrame, .stDataFrame * {{
            font-family: {FONT_MONO} !important;
        }}
        .channel-band {{ font-size: 1.15rem; font-weight: 700; color: {COLOR_TEXT}; }}
        .channel-freq {{ font-size: 0.85rem; color: {COLOR_TEXT_MUTED}; }}
        .metric-val {{ font-size: 2.2rem !important; font-weight: 800 !important; color: {COLOR_TEXT} !important; line-height: 1.2 !important; font-family: "Outfit", {FONT_UI} !important; }}
        .metric-val-lg {{ font-size: 2.6rem !important; font-weight: 800 !important; color: {COLOR_TEXT} !important; line-height: 1.2 !important; font-family: "Outfit", {FONT_UI} !important; }}
        .metric-card-quiet {{ opacity: 0.85; }}
        .metric-imp {{ font-size: 0.88rem !important; font-weight: 600 !important; margin-top: 0.25rem !important; }}
        .imp-good {{ color: {COLOR_NOMINAL}; }}
        .imp-bad {{ color: {COLOR_CRITICAL}; }}
        .imp-neutral {{ color: {COLOR_TEXT_MUTED}; }}

        /* ---- Tech badge / status chip ---- */
        .tech-badge {{
            display: inline-block; padding: 0.18rem 0.5rem; border: {BORDER};
            border-radius: {RADIUS}; font-family: {FONT_MONO};
            font-size: 0.68rem; font-weight: 600;
            margin-right: {SPACE_UNIT}; margin-bottom: {SPACE_UNIT};
        }}
        .badge-primary {{ background-color: {COLOR_PRIMARY}18; color: {COLOR_PRIMARY}; border-color: {COLOR_PRIMARY}66; }}
        .badge-success {{ background-color: {COLOR_NOMINAL}18; color: {COLOR_NOMINAL}; border-color: {COLOR_NOMINAL}66; }}
        .badge-warning {{ background-color: {COLOR_CAUTION}18; color: {COLOR_CAUTION}; border-color: {COLOR_CAUTION}66; }}
        .badge-danger  {{ background-color: {COLOR_CRITICAL}18; color: {COLOR_CRITICAL}; border-color: {COLOR_CRITICAL}66; }}
        .badge-neutral {{ background-color: {COLOR_PANEL}; color: {COLOR_TEXT_MUTED}; border-color: {COLOR_BORDER}; }}
        .badge-live    {{ background-color: {COLOR_COGNITIVE}18; color: {COLOR_COGNITIVE}; border-color: {COLOR_COGNITIVE}66; }}

        /* ---- Panels / cards - tonal layering, soft card radius (Global Shell:
           "12-16px border radius... very restrained shadows... generous internal
           padding"). Cards use RADIUS_CARD (14px), not the smaller control-level
           RADIUS. ---- */
        .channel-card, .metric-card, .decision-card, .trace-container,
        div[data-testid="stExpander"], div[data-testid="stForm"] {{
            background-color: {COLOR_PANEL} !important;
            border: {BORDER} !important;
            border-radius: {RADIUS_CARD} !important;
            box-shadow: none !important;
        }}
        .channel-card {{ padding: 0.6rem 0.75rem; margin-bottom: {SPACE_UNIT}; }}
        .metric-card {{ padding: 0.65rem 0.85rem; text-align: center; margin-bottom: {SPACE_UNIT}; }}
        .decision-card {{ padding: 0.65rem 0.8rem; margin-bottom: {SPACE_UNIT}; }}
        .trace-container {{
            display: flex; gap: {SPACE_UNIT}; padding: 0.4rem 0.6rem;
            overflow-x: auto; margin-bottom: 0.45rem;
        }}
        .trace-step {{
            padding: 0.25rem 0.55rem; border-radius: {RADIUS}; font-size: 0.66rem;
            font-weight: 700; white-space: nowrap;
            background-color: {COLOR_PANEL_RAISED}; color: {COLOR_TEXT_MUTED};
            border: {BORDER};
        }}
        .trace-active {{ background-color: {COLOR_PRIMARY}12; }}

        /* ---- Controls: 28/36px heights, 4px radius, cyan focus ----
           Descendant selectors (not ">"-scoped) - confirmed via a real rendered-DOM
           inspection that Streamlit 1.55 wraps <button> deeply enough that a direct-
           child ".stButton > button" combinator never matches at all (background-
           color/border-color silently never applied, buttons stayed Streamlit's own
           default white/light-gray the whole time). Explicit background-color was
           also missing from this rule entirely before - buttons were never actually
           dark. Both fixed here, applying to every plain button project-wide. */
        .stButton button, .stDownloadButton button {{
            background-color: {COLOR_PANEL} !important;
            border-radius: 6px !important;
            border: 1px solid rgba(79, 140, 255, 0.25) !important;
            font-family: {FONT_UI} !important;
            font-weight: 700 !important;
            font-size: 0.8rem !important;
            color: {COLOR_TEXT} !important;
            height: 38px !important;
            min-height: 38px !important;
            max-height: 38px !important;
            line-height: 1 !important;
            white-space: nowrap !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            padding: 0 0.5rem !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2) !important;
            transition: all 0.2s ease-in-out !important;
        }}
        .stButton button:hover, .stDownloadButton button:hover {{
            border-color: #00F0FF !important;
            color: #00F0FF !important;
            box-shadow: 0 0 12px rgba(0, 240, 255, 0.3) !important;
        }}
        .stButton button:disabled, .stDownloadButton button:disabled {{
            opacity: 0.3 !important;
            border-color: rgba(255, 255, 255, 0.08) !important;
            box-shadow: none !important;
        }}
            border-color: {COLOR_BORDER} !important;
            color: {COLOR_TEXT_MUTED} !important;
            background-color: {COLOR_PANEL} !important;
        }}

        /* ---- Button hierarchy (Stitch: solid-cyan primary / bordered-cyan
           secondary). Streamlit's native type="primary" gets the solid-cyan
           treatment; a few specific, semantically distinct mission-control
           buttons are targeted by their existing `key=` via Streamlit's
           `st-key-<key>` class (stable since Streamlit 1.31) - START stays the
           default secondary look until it becomes primary (handled in Python by
           passing type="primary" only when enabled), STOP reads as destructive,
           RESET reads as a muted, deliberately de-emphasized secondary action,
           and the two STEP buttons share a visual grouping. ---- */
        /* START/RESUME: solid-cyan "primary" fill only while actually enabled
           (:not(:disabled) - Streamlit sets the real disabled attribute on the
           native <button>, so this needs no Python-side state duplication).
           Deliberately NOT using Streamlit's own type="primary" here - its
           built-in stock-red (#FF4B4B) stylesheet won the cascade over a plain
           color override in testing (confirmed via a real rendered-DOM check),
           so the button stays default-kind and gets its color purely from this
           key-scoped rule instead. */
        .st-key-btn_ops_start button:not(:disabled) {{
            background-color: {COLOR_PRIMARY} !important;
            border-color: {COLOR_PRIMARY} !important;
            color: #00161a !important;
        }}
        .st-key-btn_ops_start button:not(:disabled):hover {{
            background-color: {COLOR_PRIMARY_SOFT} !important;
            border-color: {COLOR_PRIMARY_SOFT} !important;
            color: #00161a !important;
        }}
        /* RESUME (Mission Control redesign): button hierarchy is Primary=START
           only, everything else secondary - RESUME used to share START's solid
           cyan fill; it now gets a cyan OUTLINE instead (still reads as "the
           next available action", without competing with START for primary
           visual weight, per "do not make every button cyan"). */
        .st-key-btn_ops_resume button:not(:disabled) {{
            background-color: {COLOR_BASE} !important;
            border-color: {COLOR_PRIMARY} !important;
            color: {COLOR_PRIMARY} !important;
        }}
        .st-key-btn_ops_resume button:not(:disabled):hover {{
            background-color: {COLOR_PRIMARY}18 !important;
            color: {COLOR_PRIMARY_SOFT} !important;
        }}
        .st-key-btn_ops_stop button:not(:disabled) {{
            background-color: {COLOR_BASE} !important;
            border-color: {COLOR_CRITICAL}88 !important;
            color: {COLOR_CRITICAL} !important;
        }}
        .st-key-btn_ops_stop button:not(:disabled):hover {{
            background-color: {COLOR_CRITICAL}18 !important;
            border-color: {COLOR_CRITICAL} !important;
            color: {COLOR_CRITICAL} !important;
        }}
        .st-key-btn_ops_reset button:not(:disabled) {{
            background-color: {COLOR_BASE} !important;
            border-color: {COLOR_BORDER} !important;
            color: {COLOR_TEXT_MUTED} !important;
        }}
        .st-key-btn_ops_reset button:not(:disabled):hover {{
            color: {COLOR_TEXT} !important;
            border-color: {COLOR_TEXT_MUTED} !important;
        }}
        .st-key-btn_ops_step1 button:not(:disabled), .st-key-btn_ops_step10 button:not(:disabled) {{
            background-color: {COLOR_BASE} !important;
            border-color: {COLOR_COGNITIVE}55 !important;
            color: {COLOR_TEXT} !important;
        }}
        .st-key-btn_ops_step1 button:not(:disabled):hover, .st-key-btn_ops_step10 button:not(:disabled):hover {{
            border-color: {COLOR_COGNITIVE} !important;
            color: {COLOR_COGNITIVE} !important;
        }}

        /* ---- Segmented-control look for specific filter/toggle radios only
           (Alerts severity filter, Spectrum view toggle) - Stitch's filled-pill
           selected state vs. muted unselected, not applied to the sidebar nav or
           mode radios, which are a different control (icon rail / mode select,
           not a filter). ---- */
        .st-key-alerts_filter div[role="radiogroup"],
        .st-key-spectrum_view_mode div[role="radiogroup"] {{
            gap: 0 !important;
            background-color: {COLOR_BASE};
            border: {BORDER};
            border-radius: {RADIUS};
            padding: 2px;
            display: inline-flex;
        }}
        .st-key-alerts_filter div[role="radiogroup"] label,
        .st-key-spectrum_view_mode div[role="radiogroup"] label {{
            border-radius: 2px;
            padding: 0.2rem 0.6rem;
            margin: 0 !important;
            font-family: {FONT_MONO};
            font-size: 0.7rem;
        }}
        /* Hide BaseWeb's native radio dot (the label's first child div) so this
           reads as a pill/segment, not a radio button. */
        .st-key-alerts_filter div[role="radiogroup"] label > div:first-child,
        .st-key-spectrum_view_mode div[role="radiogroup"] label > div:first-child {{
            display: none !important;
        }}
        /* Selected-pill background: BaseWeb marks the selected option via a native
           checked <input>, not a data-checked attribute on the label - :has() is
           supported on the Chromium versions this app is verified against. */
        .st-key-alerts_filter div[role="radiogroup"] label:has(input:checked),
        .st-key-spectrum_view_mode div[role="radiogroup"] label:has(input:checked) {{
            background-color: {COLOR_PRIMARY}22 !important;
        }}
        .st-key-alerts_filter div[role="radiogroup"] div[data-testid="stMarkdownContainer"] p,
        .st-key-spectrum_view_mode div[role="radiogroup"] div[data-testid="stMarkdownContainer"] p {{
            font-family: {FONT_MONO} !important;
        }}

        /* ---- Compact telemetry stat tile (header live-telemetry zone) ---- */
        .stat-tile {{
            display: inline-flex; flex-direction: column; align-items: flex-start;
            padding: 0.15rem 0.6rem; border-left: 1px solid {COLOR_BORDER};
        }}
        .stat-tile:first-child {{ border-left: none; padding-left: 0; }}
        .stat-tile .stat-lbl {{
            font-family: {FONT_UI}; font-size: 0.58rem; font-weight: 700;
            color: {COLOR_TEXT_MUTED}; text-transform: uppercase; letter-spacing: 0.05em;
        }}
        .stat-tile .stat-val {{
            font-family: {FONT_MONO}; font-size: 0.85rem; font-weight: 600;
            color: {COLOR_TEXT}; white-space: nowrap;
        }}
        .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"],
        .stMultiSelect div[data-baseweb="select"] {{
            background-color: {COLOR_BASE} !important;
            border: {BORDER} !important;
            border-radius: {RADIUS} !important;
            font-family: {FONT_MONO} !important;
        }}
        .stTextInput input:focus, .stNumberInput input:focus {{
            border-color: {COLOR_PRIMARY} !important;
        }}

        /* ---- Tables: no zebra, horizontal dividers only, mono right-aligned ---- */
        [data-testid="stDataFrame"] {{
            border: {BORDER};
            border-radius: {RADIUS};
        }}
        [data-testid="stTable"] table {{ font-family: {FONT_MONO}; }}
        [data-testid="stTable"] thead tr th {{
            font-family: {FONT_UI} !important; font-size: 0.65rem !important;
            font-weight: 700 !important; text-transform: uppercase; letter-spacing: 0.05em;
            color: {COLOR_TEXT_MUTED} !important; background-color: transparent !important;
            border-bottom: {BORDER} !important; border-top: none !important;
        }}
        [data-testid="stTable"] tbody tr td {{
            background-color: transparent !important;
            border-bottom: 1px solid {COLOR_BORDER} !important; border-top: none !important;
        }}

        /* ---- Sidebar: distinct, slightly elevated surface (Global Shell) - its
           own token (COLOR_SIDEBAR), not the same tone as content panels. ---- */
        section[data-testid="stSidebar"] {{
            background-color: {COLOR_SIDEBAR};
            border-right: {BORDER};
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] label {{
            font-family: {FONT_UI};
        }}
        /* Subtle dividers, not full-strength <hr>s - "no excessive borders". */
        section[data-testid="stSidebar"] hr {{
            border-color: {COLOR_BORDER}; opacity: 0.6; margin: 0.7rem 0;
        }}

        /* ---- 8px status pip (Stitch) ---- */
        .status-pip {{
            display: inline-block; width: 8px; height: 8px; border-radius: 50%;
            margin-right: 0.35rem; vertical-align: middle;
        }}
        .status-pip-critical {{ background-color: {COLOR_CRITICAL}; animation: pip-pulse 1.4s infinite; }}
        @keyframes pip-pulse {{
            0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.35; }}
        }}

        /* ==================================================================
           ENTERPRISE DESIGN SYSTEM — additive component layer. Every rule
           below is a NEW selector (new classes, or an attribute-scoped pattern
           like [class*="_segmented"]) — nothing above this line is redefined
           or overridden, so every previously browser-verified color/selector
           is unchanged. These classes are not yet referenced by any
           dashboard/*.py view; that adoption is a later phase.
           ================================================================== */

        /* ---- Typography scale utilities ---- */
        .text-display {{ font-family:{FONT_UI}; font-size:{TYPE_SCALE['display']['size']}; font-weight:{TYPE_SCALE['display']['weight']}; line-height:{TYPE_SCALE['display']['line']}; letter-spacing:{TYPE_SCALE['display']['letter']}; color:{COLOR_TEXT}; }}
        .text-headline {{ font-family:{FONT_UI}; font-size:{TYPE_SCALE['headline']['size']}; font-weight:{TYPE_SCALE['headline']['weight']}; line-height:{TYPE_SCALE['headline']['line']}; color:{COLOR_TEXT}; }}
        .text-body {{ font-family:{FONT_UI}; font-size:{TYPE_SCALE['body']['size']}; font-weight:{TYPE_SCALE['body']['weight']}; line-height:{TYPE_SCALE['body']['line']}; color:{COLOR_TEXT}; }}
        .text-label-caps {{ font-family:{FONT_UI}; font-size:{TYPE_SCALE['label_caps']['size']}; font-weight:{TYPE_SCALE['label_caps']['weight']}; letter-spacing:{TYPE_SCALE['label_caps']['letter']}; text-transform:uppercase; color:{COLOR_TEXT_MUTED}; }}
        .text-telemetry-lg {{ font-family:{FONT_MONO}; font-size:{TYPE_SCALE['telemetry_lg']['size']}; font-weight:{TYPE_SCALE['telemetry_lg']['weight']}; color:{COLOR_TEXT}; }}
        .text-telemetry-md {{ font-family:{FONT_MONO}; font-size:{TYPE_SCALE['telemetry_md']['size']}; font-weight:{TYPE_SCALE['telemetry_md']['weight']}; color:{COLOR_TEXT}; }}
        .text-telemetry-sm {{ font-family:{FONT_MONO}; font-size:{TYPE_SCALE['telemetry_sm']['size']}; font-weight:{TYPE_SCALE['telemetry_sm']['weight']}; letter-spacing:{TYPE_SCALE['telemetry_sm']['letter']}; color:{COLOR_TEXT_MUTED}; }}

        /* ---- Density: Global Shell wants "structured density with breathing
           room" (not the tightly-packed-boxes look, not oversized gaps either) -
           a touch looser than a pure engineering console's zero-gap grids. Card
           padding itself is separately set per-class below. ---- */
        div[data-testid="stHorizontalBlock"] {{ gap: {SPACE[3]} !important; }}

        /* ---- Surface panel (Level 1): bordered panel + header bar + a soft
           lift (Global Shell: "use elevation primarily for... important
           operational panels"). Shared replacement for '---' section dividers
           (Phase 2 adoption). ---- */
        .surface-panel {{
            background-color: {COLOR_PANEL}; border: {BORDER}; border-radius: {RADIUS_CARD};
            padding: {SPACE[4]}; margin-bottom: {SPACE[3]}; box-shadow: {SHADOW_SM};
        }}
        .surface-panel-header {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:{SPACE[2]}; }}
        .surface-panel .panel-title {{ font-family:{FONT_UI}; font-size:0.78rem; font-weight:700; letter-spacing:0.04em; text-transform:uppercase; color:{COLOR_TEXT}; }}
        .surface-panel .panel-subtitle {{ font-family:{FONT_UI}; font-size:0.68rem; color:{COLOR_TEXT_MUTED}; margin-bottom:{SPACE[2]}; }}

        /* ---- KPI / stat cards: icon chip + label + big value + delta, with a
           soft lift - "Primary KPIs should visually dominate". ---- */
        .kpi-card {{
            display:flex; align-items:flex-start; gap:{SPACE[3]};
            background-color:{COLOR_PANEL}; border:{BORDER}; border-radius:{RADIUS_CARD};
            padding:{SPACE[4]}; box-shadow: {SHADOW_SM};
        }}
        .kpi-icon {{
            width:32px; height:32px; border-radius:{RADIUS_SM}; display:flex; align-items:center;
            justify-content:center; font-size:1rem; flex-shrink:0;
        }}
        .kpi-label {{ font-family: "Outfit", {FONT_UI}; font-size: 0.85rem !important; font-weight: 700 !important; letter-spacing: 0.05em; text-transform: uppercase; color: {COLOR_TEXT_MUTED}; }}
        .kpi-value {{ font-family: "Outfit", {FONT_UI}; font-size: 2.1rem !important; font-weight: 800 !important; color: {COLOR_TEXT}; line-height: 1.2; margin-top: 0.2rem; }}
        .kpi-delta {{ font-family: {FONT_UI}; font-size: 0.9rem !important; font-weight: 600 !important; margin-top: 0.25rem; }}
        .kpi-card-hero {{ padding-left: calc({SPACE[4]} - 2px); }}
        .kpi-card-hero .kpi-icon {{ width: 38px; height: 38px; font-size: 1.2rem; }}
        .kpi-value-hero {{ font-family: "Outfit", {FONT_UI}; font-size: 2.5rem !important; font-weight: 800 !important; color: {COLOR_TEXT}; line-height: 1.2; margin-top: 0.2rem; }}

        /* ---- Mission progress bar: thin filled track, not a "huge neon
           progress bar". ---- */
        .mission-progress {{
            background-color:{COLOR_PANEL}; border:{BORDER}; border-radius:{RADIUS_CARD};
            padding:{SPACE[4]}; margin-bottom:{SPACE[3]};
        }}
        .mission-progress-header {{ display:flex; justify-content:space-between; align-items:baseline; margin-bottom:{SPACE[2]}; }}
        .mission-progress-pct {{ font-family:{FONT_MONO}; font-size:0.95rem; font-weight:700; color:{COLOR_PRIMARY}; }}
        .mission-progress-track {{
            height:6px; border-radius:{RADIUS_SM}; background-color:{COLOR_BASE}; overflow:hidden;
        }}
        .mission-progress-fill {{
            height:100%; border-radius:{RADIUS_SM}; background-color:{COLOR_PRIMARY};
            transition:width 300ms ease;
        }}
        .mission-progress-stats {{ display:flex; gap:{SPACE[6]}; margin-top:{SPACE[3]}; flex-wrap:wrap; }}
        .mp-stat {{ display:flex; flex-direction:column; gap:2px; }}
        .mp-stat-lbl {{ font-family:{FONT_UI}; font-size:0.58rem; font-weight:700; letter-spacing:0.05em; text-transform:uppercase; color:{COLOR_TEXT_MUTED}; }}
        .mp-stat-val {{ font-family:{FONT_MONO}; font-size:0.8rem; font-weight:600; color:{COLOR_TEXT}; }}

        /* ---- Section divider: subtle zone separator, replaces a literal '---'
           rule between Mission Control's major zones. ---- */
        .section-divider-wrap {{ margin:{SPACE[5]} 0 {SPACE[3]}; }}
        .section-divider-label {{
            font-family:{FONT_UI}; font-size:0.63rem; font-weight:700; letter-spacing:0.08em;
            text-transform:uppercase; color:{COLOR_TEXT_FAINT}; margin-bottom:{SPACE[2]};
        }}
        .section-divider {{ height:1px; background-color:{COLOR_BORDER}; }}

        /* ---- Chart container: consistent frame for Plotly figures ---- */
        .chart-container {{ background-color:{COLOR_PANEL}; border:{BORDER}; border-radius:{RADIUS_CARD}; padding:{SPACE[3]}; }}

        /* ---- Alert list row ---- */
        .alert-row {{ display:flex; align-items:flex-start; gap:{SPACE[3]}; padding:{SPACE[2]} 0; border-bottom:1px solid {COLOR_BORDER}; }}
        .alert-row:last-child {{ border-bottom:none; }}
        .alert-row-icon {{ width:28px; height:28px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:0.8rem; flex-shrink:0; }}
        .alert-row-body {{ flex:1; }}
        .alert-row-title {{ font-family:{FONT_UI}; font-size:0.78rem; font-weight:700; color:{COLOR_TEXT}; }}
        .alert-row-subtitle {{ font-family:{FONT_MONO}; font-size:0.68rem; color:{COLOR_TEXT_MUTED}; margin-top:0.05rem; }}
        .alert-row-meta {{ text-align:right; flex-shrink:0; }}
        .alert-row-time {{ font-family:{FONT_MONO}; font-size:0.63rem; color:{COLOR_TEXT_FAINT}; margin-bottom:0.2rem; }}
        .severity-chip {{
            display:inline-block; padding:0.1rem 0.4rem; border-radius:{RADIUS_SM}; border:1px solid;
            font-family:{FONT_MONO}; font-size:0.6rem; font-weight:700; letter-spacing:0.03em;
        }}

        /* ---- Empty / loading states ---- */
        .empty-state {{ text-align:center; padding:{SPACE[6]} {SPACE[4]}; color:{COLOR_TEXT_MUTED}; }}
        .empty-state-icon {{ font-size:1.3rem; color:{COLOR_TEXT_FAINT}; margin-bottom:{SPACE[1]}; }}
        .empty-state-msg {{ font-family:{FONT_UI}; font-size:0.75rem; }}
        .skeleton-block {{ display:flex; flex-direction:column; gap:{SPACE[2]}; padding:{SPACE[2]} 0; }}
        .skeleton-bar {{
            height:10px; border-radius:{RADIUS_SM};
            background:linear-gradient(90deg, {COLOR_PANEL_RAISED} 25%, {COLOR_BORDER} 50%, {COLOR_PANEL_RAISED} 75%);
            background-size:200% 100%; animation: skeleton-shimmer 1.6s ease-in-out infinite;
        }}
        @keyframes skeleton-shimmer {{ 0% {{ background-position:200% 0; }} 100% {{ background-position:-200% 0; }} }}

        /* ---- Sidebar nav rail (Global Shell section 12): active item gets a
           slightly raised surface + accent left-bar + full-strength text;
           inactive items are muted with a clear but subtle hover state; scoped
           to the NAVIGATION radiogroup via its st-key- class only - does not
           touch the OPERATING MODE radio or any other control. ---- */
        .st-key-nav_view_radio div[role="radiogroup"] label {{
            border-radius:{RADIUS}; padding:0.4rem 0.6rem; margin:1px 0 !important;
            border-left:2px solid transparent; transition: background-color 120ms ease, border-color 120ms ease;
        }}
        .st-key-nav_view_radio div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] p {{
            color:{COLOR_TEXT_MUTED}; font-weight:500;
        }}
        .st-key-nav_view_radio div[role="radiogroup"] label:hover:not(:has(input:checked)) {{
            background-color:{COLOR_PANEL_RAISED};
        }}
        .st-key-nav_view_radio div[role="radiogroup"] label:hover:not(:has(input:checked)) div[data-testid="stMarkdownContainer"] p {{
            color:{COLOR_TEXT};
        }}
        .st-key-nav_view_radio div[role="radiogroup"] label:has(input:checked) {{
            background-color:{COLOR_PANEL_RAISED}; border-left:2px solid {COLOR_PRIMARY};
        }}
        .st-key-nav_view_radio div[role="radiogroup"] label:has(input:checked) div[data-testid="stMarkdownContainer"] p {{
            color:{COLOR_TEXT}; font-weight:700;
        }}

        /* ---- Generic segmented-control pattern: any radio widget whose key
           ends in "_segmented" gets the filled-pill treatment, generalizing
           the two hardcoded alerts_filter / spectrum_view_mode rules above
           (kept as-is) for future widgets without a new CSS rule per widget. ---- */
        [class*="st-key-"][class*="_segmented"] div[role="radiogroup"] {{
            gap:0 !important; background-color:{COLOR_BASE}; border:{BORDER}; border-radius:{RADIUS}; padding:2px; display:inline-flex;
        }}
        [class*="st-key-"][class*="_segmented"] div[role="radiogroup"] label {{ border-radius:2px; padding:0.2rem 0.6rem; margin:0 !important; font-family:{FONT_MONO}; font-size:0.7rem; }}
        [class*="st-key-"][class*="_segmented"] div[role="radiogroup"] label > div:first-child {{ display:none !important; }}
        [class*="st-key-"][class*="_segmented"] div[role="radiogroup"] label:has(input:checked) {{ background-color:{COLOR_PRIMARY}22 !important; }}

        /* ---- Sliders: track/thumb in the primary accent. Supplements
           .streamlit/config.toml's primaryColor (which retints Streamlit's
           native slider/checkbox/radio accent from stock red to this design
           system's cyan) with an explicit override for the slider specifically. ---- */
        .stSlider [data-baseweb="slider"] div[role="slider"] {{ background-color:{COLOR_PRIMARY} !important; border-color:{COLOR_PRIMARY} !important; }}
        .stSlider [data-baseweb="slider"] > div > div {{ background-color:{COLOR_PRIMARY}55 !important; }}

        /* ---- Tooltip (Streamlit's native help= popover): best-effort restyle.
           Streamlit's tooltip DOM is not a stable public API across versions —
           this degrades harmlessly to the native tooltip if the selector ever
           stops matching a future Streamlit version. ---- */
        div[data-testid="stTooltipContent"] {{
            background-color:{COLOR_PANEL_RAISED} !important; border:{BORDER} !important;
            color:{COLOR_TEXT} !important; font-family:{FONT_UI} !important; font-size:0.7rem !important;
            border-radius:{RADIUS} !important; box-shadow:{SHADOW_MD} !important;
        }}
    </style>
    """
