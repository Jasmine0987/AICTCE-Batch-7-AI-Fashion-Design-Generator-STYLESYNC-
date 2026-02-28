"""
theme.py — Luxury Minimalist CSS Theme
Inspired by Doji.com: High-contrast, editorial, zero visual noise.
Palette: Pure White · Jet Black · Cloud Gray · Warm Taupe accent
"""

LUXURY_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ─── RESET & BASE ─────────────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

/* Force white backgrounds everywhere */
div[data-testid="stAppViewContainer"],
div[data-testid="stAppViewContainer"] > section,
div[data-testid="stMainBlockContainer"],
.main, .block-container,
div[data-testid="block-container"] {
    background-color: #FFFFFF !important;
}

/* ─── COLLAPSED SIDEBAR CONTROL FIX ───────────────────────── */

/* Target the collapsed control container */
div[data-testid="collapsedControl"] {
    background: transparent !important;
}

/* Target the button */
button[data-testid="baseButton-headerNoPadding"] {
    background: transparent !important;
    color: #FFFFFF !important; 
    border: 1px solid rgba(255,255,255,0.25) !important;
    border-radius: 2px !important;
    padding: 6px !important;
}

/* Force the SVG icon white */
button[data-testid="baseButton-headerNoPadding"] svg {
    fill: #FFFFFF !important;
    color: #FFFFFF !important;
}

/* Optional: subtle hover */
button[data-testid="baseButton-headerNoPadding"]:hover {
    opacity: 0.7 !important;
}

/* Remove Streamlit's default padding */
.block-container { 
    padding-top: 0rem !important; 
    padding-bottom: 4rem !important;
    max-width: 1280px !important;
}

/* ─── TYPOGRAPHY BASE ───────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    color: #0A0A0A !important;
    -webkit-font-smoothing: antialiased;
}

h1, h2, h3, h4 {
    font-family: 'Cormorant Garamond', serif !important;
    color: #0A0A0A !important;
    letter-spacing: 0.5px;
    font-weight: 400 !important;
}

/* All Streamlit text elements default to black */
p, span, label, div, li,
.stMarkdown p, .stMarkdown span,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span,
[data-testid="stWidgetLabel"] p,
.stRadio label, .stSelectbox label,
.stCheckbox label, .stTextArea label,
.stTextInput label, .stSlider label,
.stToggle label {
    color: #0A0A0A !important;
}

/* Caption / helper text */
.stCaption, small, .stCaption p { color: #888888 !important; font-size: 0.78rem !important; }

/* Code */
code { 
    font-family: 'DM Sans', monospace !important;
    font-size: 0.82rem !important;
    background: #F2F2F2 !important;
    color: #0A0A0A !important;
    padding: 1px 6px !important;
    border-radius: 2px !important;
}

/* ─── SIDEBAR ───────────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E8E8E8 !important;
    padding-top: 1.5rem !important;
}

section[data-testid="stSidebar"] * { color: #0A0A0A !important; }

/* Sidebar header */
section[data-testid="stSidebar"] h3 {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 1.3rem !important;
    font-weight: 400 !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    color: #0A0A0A !important;
}

/* Radio buttons — mode selector */
section[data-testid="stSidebar"] .stRadio label {
    font-size: 0.82rem !important;
    font-weight: 400 !important;
    letter-spacing: 0.5px !important;
    padding: 6px 0 !important;
    color: #444444 !important;
    transition: color 0.2s !important;
}

section[data-testid="stSidebar"] .stRadio label:hover { color: #0A0A0A !important; }

/* Selected radio item — bold + black */
section[data-testid="stSidebar"] [data-baseweb="radio"] [aria-checked="true"] ~ div p {
    color: #0A0A0A !important;
    font-weight: 600 !important;
}


/* Radio indicator dot */
section[data-testid="stSidebar"] [data-baseweb="radio"] span {
    border-color: #0A0A0A !important;
}
section[data-testid="stSidebar"] [data-baseweb="radio"] [aria-checked="true"] span::before {
    background-color: #0A0A0A !important;
}

/* Sidebar dividers */
section[data-testid="stSidebar"] hr { border-color: #E8E8E8 !important; }

/* Sidebar section labels */
section[data-testid="stSidebar"] strong,
section[data-testid="stSidebar"] b {
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: #888888 !important;
}

/* Selectbox, dropdowns in sidebar */
section[data-testid="stSidebar"] [data-baseweb="select"] {
    border: 1px solid #E8E8E8 !important;
    border-radius: 2px !important;
    background: #FFFFFF !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] div {
    background: #FFFFFF !important;
    color: #0A0A0A !important;
}
/* Dropdown menu container (portal layer) */
[data-baseweb="popover"] {
    background: #FFFFFF !important;
}

/* Dropdown list */
ul[role="listbox"] {
    background: #FFFFFF !important;
    border: 1px solid #E8E8E8 !important;
    border-radius: 2px !important;
}

/* Dropdown options */
li[role="option"] {
    background: #FFFFFF !important;
    color: #0A0A0A !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.82rem !important;
}

/* Hover state */
li[role="option"]:hover {
    background: #F2F2F2 !important;
    color: #0A0A0A !important;
}

/* Selected item */
li[role="option"][aria-selected="true"] {
    background: #F9F9FB !important;
    color: #0A0A0A !important;
    font-weight: 500 !important;
}
/* Toggle */
section[data-testid="stSidebar"] [data-baseweb="toggle"] span {
    background-color: #E8E8E8 !important;
}

/* ─── SIDEBAR BUTTONS ───────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] .stButton > button {
    background: #0A0A0A !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 2px !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    padding: 10px 16px !important;
    width: 100% !important;
    transition: background 0.2s !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: #333333 !important;
}

/* ─── SIDEBAR SECONDARY BUTTON FIX ───────────────────────── */

/* Target only secondary buttons inside sidebar */
section[data-testid="stSidebar"] 
button[data-testid="baseButton-secondary"] {
    background: #0A0A0A !important;
    color: #FFFFFF !important;
    border: 1px solid #0A0A0A !important;
}

/* Force inner markdown text white */
section[data-testid="stSidebar"] 
button[data-testid="baseButton-secondary"] p {
    color: #FFFFFF !important;
}

/* Hover invert */
section[data-testid="stSidebar"] 
button[data-testid="baseButton-secondary"]:hover {
    background: #FFFFFF !important;
    color: #0A0A0A !important;
    border-color: #0A0A0A !important;
}

section[data-testid="stSidebar"] 
button[data-testid="baseButton-secondary"]:hover p {
    color: #0A0A0A !important;
}

/* ─── MAIN BUTTONS ──────────────────────────────────────────────────────────── */
.stButton > button {
    background: #0A0A0A !important;
    color: #FFFFFF !important;
    border: 1px solid #0A0A0A !important;
    border-radius: 2px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    padding: 12px 24px !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    background: #FFFFFF !important;
    color: #0A0A0A !important;
}

/* Quick prompt buttons — ghost style */
.stButton > button[kind="secondary"] {
    background: #FFFFFF !important;
    color: #0A0A0A !important;
    border: 1px solid #E8E8E8 !important;
}
.stButton > button[kind="secondary"]:hover {
    border-color: #0A0A0A !important;
}

/* ─── INPUTS ────────────────────────────────────────────────────────────────── */
.stTextArea textarea, .stTextInput input {
    background: #F9F9FB !important;
    border: 1px solid #E8E8E8 !important;
    border-radius: 2px !important;
    color: #0A0A0A !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    padding: 14px !important;
    transition: border-color 0.2s !important;
}

.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: #0A0A0A !important;
    box-shadow: none !important;
    outline: none !important;
}

.stTextArea textarea::placeholder, .stTextInput input::placeholder {
    color: #BBBBBB !important;
    font-style: italic !important;
}

/* ─── SELECTBOX ─────────────────────────────────────────────────────────────── */
[data-baseweb="select"] {
    border-radius: 2px !important;
}
[data-baseweb="select"] > div {
    background: #F9F9FB !important;
    border: 1px solid #E8E8E8 !important;
    border-radius: 2px !important;
    color: #0A0A0A !important;
}

/* ─── CHECKBOX ──────────────────────────────────────────────────────────────── */
.stCheckbox label { font-size: 0.82rem !important; color: #444444 !important; }
.stCheckbox [data-baseweb="checkbox"] span { 
    border-color: #CCCCCC !important;
    border-radius: 2px !important;
}
.stCheckbox [data-baseweb="checkbox"] [aria-checked="true"] span {
    background: #0A0A0A !important;
    border-color: #0A0A0A !important;
}

/* ─── METRICS ───────────────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #F9F9FB !important;
    border: 1px solid #E8E8E8 !important;
    border-radius: 2px !important;
    padding: 20px !important;
}
[data-testid="stMetricLabel"] { color: #888888 !important; font-size: 0.72rem !important; text-transform: uppercase !important; letter-spacing: 1.5px !important; }
[data-testid="stMetricValue"] { color: #0A0A0A !important; font-family: 'Cormorant Garamond', serif !important; font-size: 1.8rem !important; font-weight: 400 !important; }
[data-testid="stMetricDelta"] { font-size: 0.75rem !important; color: #888888 !important; }

/* ─── DATAFRAME ─────────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid #E8E8E8 !important;
    border-radius: 2px !important;
}
.dvn-scroller { background: #FFFFFF !important; }
th { 
    background: #F9F9FB !important; 
    color: #0A0A0A !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid #E8E8E8 !important;
}
td { 
    color: #0A0A0A !important; 
    font-size: 0.82rem !important;
    border-bottom: 1px solid #F2F2F2 !important;
}

/* ─── PROGRESS BAR ──────────────────────────────────────────────────────────── */
.stProgress > div > div {
    background: #E8E8E8 !important;
    border-radius: 1px !important;
    height: 2px !important;
}
.stProgress > div > div > div {
    background: #0A0A0A !important;
    border-radius: 1px !important;
}

/* Progress text */
[data-testid="stProgressText"] { 
    color: #888888 !important; 
    font-size: 0.78rem !important;
    letter-spacing: 0.5px !important;
}

/* ─── SPINNER ───────────────────────────────────────────────────────────────── */
.stSpinner > div { border-color: #0A0A0A transparent transparent !important; }

/* ─── EXPANDER ──────────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid #E8E8E8 !important;
    border-radius: 2px !important;
    background: #FFFFFF !important;
}
[data-testid="stExpander"] summary {
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.5px !important;
    color: #0A0A0A !important;
    padding: 14px 18px !important;
}

/* ─── FILE UPLOADER ─────────────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    border: 1px dashed #CCCCCC !important;
    border-radius: 2px !important;
    background: #F9F9FB !important;
    padding: 32px !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploader"]:hover { border-color: #0A0A0A !important; }
[data-testid="stFileUploader"] button {
    background: #0A0A0A !important;
    color: #FFFFFF !important;
    border-radius: 2px !important;
    border: none !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 1px !important;
}

/* ─── SUCCESS / WARNING / INFO MESSAGES ─────────────────────────────────────── */
.stSuccess { background: #F9F9FB !important; border: 1px solid #0A0A0A !important; border-radius: 2px !important; color: #0A0A0A !important; }
.stWarning { background: #F9F9FB !important; border: 1px solid #CCCCCC !important; border-radius: 2px !important; color: #0A0A0A !important; }
.stSuccess p, .stWarning p { color: #0A0A0A !important; }

/* ─── SCROLLBAR ─────────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #F9F9FB; }
::-webkit-scrollbar-thumb { background: #CCCCCC; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #0A0A0A; }

/* ─── HIDE STREAMLIT CHROME ─────────────────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden !important; }
.stDeployButton { display: none !important; }

/* ─── CUSTOM COMPONENTS ─────────────────────────────────────────────────────── */

/* HERO — editorial, full-bleed black */
.hero-wrap {
    position: relative;

    width: 100vw;
    margin-left: calc(50% - 50vw);
    margin-right: calc(50% - 50vw);

    margin-top: -2rem;   /* cancel block-container padding */
    padding: 90px 48px 80px;

    background: #0A0A0A;
    text-align: center;
}
.hero-wrap::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
}
.hero-eyebrow {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: rgba(255,255,255,0.4);
    margin-bottom: 20px;
}
.hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: clamp(3rem, 7vw, 5.5rem);
    font-weight: 300;
    color: #FFFFFF;
    letter-spacing: 4px;
    line-height: 1;
    margin: 0 0 8px;
    text-transform: uppercase;
}
.hero-title em {
    font-style: italic;
    font-weight: 300;
}
.hero-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.78rem;
    color: rgba(255,255,255,0.45);
    letter-spacing: 3px;
    text-transform: uppercase;
    margin: 20px 0 32px;
}
.hero-badges {
    display: flex;
    gap: 8px;
    justify-content: center;
    flex-wrap: wrap;
}
.hero-badge {
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: rgba(255,255,255,0.55);
    padding: 5px 14px;
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 1px;
    transition: all 0.2s;
}
.hero-badge:hover { 
    color: #FFFFFF; 
    border-color: rgba(255,255,255,0.5); 
}
.hero-wrap,
.hero-wrap p,
.hero-wrap span,
.hero-wrap div {
    color: #FFFFFF !important;
}

/* TREND ROW — understated, monochrome */
.trend-row {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    justify-content: center;
    margin: 0 0 48px;
    padding: 0 24px;
}
.trend-tag {
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #888888;
    padding: 6px 14px;
    border: 1px solid #E8E8E8;
    border-radius: 1px;
    transition: all 0.2s;
    cursor: default;
}
.trend-tag:hover { color: #0A0A0A; border-color: #0A0A0A; }

/* SECTION LABEL — editorial overline */
.section-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #888888;
    margin-bottom: 12px;
    padding-bottom: 12px;
    border-bottom: 1px solid #E8E8E8;
}

/* CARD — no shadow, 1px border, sharp corners */
.lux-card {
    background: #FFFFFF;
    border: 1px solid #E8E8E8;
    border-radius: 2px;
    padding: 28px;
    margin-bottom: 20px;
    transition: border-color 0.2s;
}
.lux-card:hover { border-color: #CCCCCC; }

.lux-card-sm {
    background: #F9F9FB;
    border: 1px solid #E8E8E8;
    border-radius: 2px;
    padding: 20px 24px;
    margin-bottom: 16px;
}

/* CARD HEADING */
.card-heading {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.4rem;
    font-weight: 400;
    color: #0A0A0A;
    letter-spacing: 1px;
    margin: 0 0 4px;
}
.card-subheading {
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #888888;
    margin-bottom: 20px;
}

/* ML INFO BOX — minimal, monochrome */
.ml-info {
    background: #F9F9FB;
    border-left: 2px solid #0A0A0A;
    padding: 14px 18px;
    margin: 16px 0;
    border-radius: 0 2px 2px 0;
}
.ml-info-title {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #0A0A0A;
    margin-bottom: 6px;
}
.ml-info-body {
    font-size: 0.8rem;
    color: #555555;
    line-height: 1.6;
}
.ml-info-body code {
    background: #EEEEEE !important;
    color: #0A0A0A !important;
    font-size: 0.75rem !important;
    padding: 1px 5px !important;
}

/* CONFIDENCE BARS — thin, editorial */
.conf-wrap { margin: 12px 0; }
.conf-meta {
    display: flex;
    justify-content: space-between;
    margin-bottom: 5px;
}
.conf-label {
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.5px;
    color: #0A0A0A;
    text-transform: capitalize;
}
.conf-pct {
    font-size: 0.72rem;
    color: #888888;
    font-family: 'DM Sans', monospace;
}
.conf-bar-bg {
    background: #F0F0F0;
    height: 2px;
    border-radius: 1px;
}
.conf-bar-fg {
    height: 2px;
    border-radius: 1px;
    background: #0A0A0A;
    transition: width 0.8s ease;
}

/* COLOR SWATCHES */
.swatch-row {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin: 14px 0;
}
.swatch {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 14px 8px 10px;
    border: 1px solid #E8E8E8;
    border-radius: 2px;
    background: #FFFFFF;
    transition: border-color 0.2s;
}
.swatch:hover { border-color: #0A0A0A; }
.swatch-dot {
    width: 22px;
    height: 22px;
    border-radius: 50%;
    border: 1px solid rgba(0,0,0,0.08);
    flex-shrink: 0;
}
.swatch-name {
    font-size: 0.78rem;
    color: #0A0A0A;
    font-weight: 500;
}
.swatch-hex {
    font-size: 0.68rem;
    color: #888888;
    font-family: 'DM Sans', monospace;
    letter-spacing: 0.5px;
}

/* STYLE TAGS */
.tag-row { display: flex; gap: 6px; flex-wrap: wrap; margin: 10px 0; background: white; text-color: black }
.tag {
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #0A0A0A !important;
    padding: 4px 12px;
    border: 1px solid #444;
    border-radius: 2px;
    background: #ffffff !important;
    transition: border-color 0.2s ease;
    cursor: pointer;
}
.tag:hover { border-color: #FFFFFF;}
.tag-green { color: #2E7D32; border-color: #C8E6C9; background: #F9FBF9; }
.tag-muted { color: #888888; border-color: #E8E8E8; }

/* PRODUCT CARD — Doji style, no shadow */
.prod-card {
    background: #FFFFFF;
    border: 1px solid #E8E8E8;
    border-radius: 2px;
    overflow: hidden;
    transition: border-color 0.2s;
    height: 100%;
}
.prod-card:hover { border-color: #0A0A0A; }
.prod-img {
    width: 100%;
    height: 220px;
    object-fit: cover;
    display: block;
    background: #F9F9FB;
}
.prod-body { padding: 16px; }
.prod-title {
    font-size: 0.82rem;
    font-weight: 500;
    color: #0A0A0A;
    line-height: 1.4;
    margin-bottom: 8px;
    letter-spacing: 0.2px;
}
.prod-price {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.2rem;
    font-weight: 400;
    color: #0A0A0A;
    letter-spacing: 0.5px;
}
.prod-source {
    font-size: 0.68rem;
    color: #888888;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-top: 2px;
}
.prod-score {
    font-size: 0.68rem;
    color: #888888;
    margin-top: 6px;
    padding-top: 6px;
    border-top: 1px solid #F2F2F2;
}
.prod-btn {
    display: block;
    text-align: center;
    background: #0A0A0A;
    color: #FFFFFF !important;
    padding: 10px;
    text-decoration: none !important;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 12px;
    border-radius: 1px;
    transition: background 0.2s;
}
.prod-btn:hover { background: #333333; color: #FFFFFF !important; }

/* DIVIDER */
.lux-divider {
    text-align: center;
    margin: 40px 0;
    position: relative;
}
.lux-divider::before {
    content: '';
    position: absolute;
    top: 50%;
    left: 0; right: 0;
    height: 1px;
    background: #E8E8E8;
}
.lux-divider-text {
    display: inline-block;
    position: relative;
    background: #FFFFFF;
    padding: 0 20px;
    font-size: 0.68rem;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: #CCCCCC;
}

/* RECENT SAVE ITEM */
.save-item {
    padding: 14px 0;
    border-bottom: 1px solid #F2F2F2;
}
.save-item:last-child { border-bottom: none; }
.save-name {
    font-size: 0.85rem;
    font-weight: 500;
    color: #0A0A0A;
    letter-spacing: 0.2px;
}
.save-date {
    font-size: 0.72rem;
    color: #AAAAAA;
    margin-top: 2px;
    letter-spacing: 0.5px;
}

/* ML MODELS LIST */
.ml-model-list {
    margin-top: 8px;
}
.ml-model-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 7px 0;
    border-bottom: 1px solid #F2F2F2;
    font-size: 0.78rem;
    color: #0A0A0A;
}
.ml-model-item:last-child { border-bottom: none; }
.ml-dot {
    width: 6px;
    height: 6px;
    background: #0A0A0A;
    border-radius: 50%;
    flex-shrink: 0;
}

/* MODE SECTION HEADING */
.mode-heading {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2rem;
    font-weight: 300;
    letter-spacing: 2px;
    color: #0A0A0A;
    margin-bottom: 4px;
}
.mode-sub {
    font-size: 0.72rem;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #888888;
    margin-bottom: 32px;
}

/* FOOTER */
.lux-footer {
    text-align: center;
    padding: 40px 0 20px;
    margin-top: 60px;
    border-top: 1px solid #E8E8E8;
}
.lux-footer-text {
    font-size: 0.68rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #CCCCCC;
}
.active_filters{ color: #FFFFFF !important; }


</style>
"""