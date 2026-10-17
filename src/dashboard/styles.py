"""Enterprise dashboard styling using CSS-only Streamlit injection."""

import streamlit as st

ENTERPRISE_CSS = """
<style>
:root {
    --rp-bg: #0B1220;
    --rp-panel: #142237;
    --rp-border: #29405A;
    --rp-accent: #38BDF8;
    --rp-text: #F1F5F9;
    --rp-muted: #94A3B8;
}

/* Main application background */
.stApp {
    background:
        radial-gradient(
            ellipse at 82% 0%,
            rgba(56, 189, 248, 0.07),
            transparent 38%
        ),
        var(--rp-bg);
    color: var(--rp-text);
}

/* Content width and breathing room */
.block-container {
    max-width: 1500px;
    padding-top: 2rem;
    padding-bottom: 3rem;
    padding-left: 2.5rem;
    padding-right: 2.5rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0E192B;
    border-right: 1px solid var(--rp-border);
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem;
}

/* Headings */
h1, h2, h3 {
    color: var(--rp-text);
    letter-spacing: -0.035em;
}

h1 {
    font-size: clamp(1.9rem, 3vw, 2.65rem);
    font-weight: 750;
    line-height: 1.18;
    margin-bottom: 0.5rem;
}

h2, h3 {
    font-weight: 680;
}

/* Supporting text */
[data-testid="stCaptionContainer"] {
    color: var(--rp-muted);
}

/* KPI cards */
div[data-testid="stMetric"] {
    background:
        linear-gradient(
            145deg,
            rgba(25, 46, 70, 0.98),
            rgba(17, 32, 52, 0.98)
        );
    border: 1px solid var(--rp-border);
    border-radius: 16px;
    padding: 1.25rem 1.35rem;
    min-height: 122px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
    transition:
        border-color 180ms ease,
        transform 180ms ease,
        box-shadow 180ms ease;
}

div[data-testid="stMetric"]:hover {
    border-color: rgba(56, 189, 248, 0.55);
    transform: translateY(-2px);
    box-shadow: 0 12px 28px rgba(0, 0, 0, 0.2);
}

div[data-testid="stMetricLabel"] {
    color: var(--rp-muted);
    font-size: 0.83rem;
    font-weight: 550;
    letter-spacing: 0.01em;
}

div[data-testid="stMetricValue"] {
    color: #F8FAFC;
    font-size: clamp(1.45rem, 2vw, 2.05rem);
    font-weight: 730;
    letter-spacing: -0.045em;
    font-variant-numeric: tabular-nums;
}

/* Charts */
div[data-testid="stPlotlyChart"] {
    background: rgba(20, 34, 55, 0.6);
    border: 1px solid var(--rp-border);
    border-radius: 16px;
    padding: 0.55rem;
    overflow: hidden;
}

/* Tables */
div[data-testid="stDataFrame"] {
    border: 1px solid var(--rp-border);
    border-radius: 14px;
    overflow: hidden;
}

/* Native input controls */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    border-radius: 10px;
}

/* Native buttons */
.stButton > button,
.stDownloadButton > button {
    border-radius: 10px;
    font-weight: 600;
    transition:
        border-color 180ms ease,
        background-color 180ms ease;
}

.stDownloadButton > button {
    border: 1px solid rgba(56, 189, 248, 0.45);
    background: rgba(56, 189, 248, 0.10);
    color: #E0F2FE;
}

.stDownloadButton > button:hover {
    border-color: var(--rp-accent);
    background: rgba(56, 189, 248, 0.18);
}

/* Dividers */
hr {
    border-color: var(--rp-border);
    margin-top: 1.7rem;
    margin-bottom: 1.7rem;
}

/* Expander */
div[data-testid="stExpander"] {
    border: 1px solid var(--rp-border);
    border-radius: 12px;
    overflow: hidden;
}

/* Focus visibility */
button:focus-visible,
input:focus-visible {
    outline: 2px solid var(--rp-accent);
    outline-offset: 2px;
}

/* Mobile layout */
@media (max-width: 768px) {
    .block-container {
        padding: 1.25rem 1rem 2rem;
    }

    div[data-testid="stMetric"] {
        padding: 1rem;
        min-height: 100px;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.45rem;
    }
}

/* Respect reduced-motion accessibility settings */
@media (prefers-reduced-motion: reduce) {
    div[data-testid="stMetric"],
    .stButton > button,
    .stDownloadButton > button {
        transition: none;
        transform: none;
    }
}
</style>
"""


def apply_enterprise_styles():
    """Inject trusted static CSS without rendering custom HTML markup."""
    st.html(ENTERPRISE_CSS)