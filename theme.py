import streamlit as st


def apply_theme(dark_mode: bool = False):
    bg_color = "#121214" if dark_mode else "#F7F4EC"
    dot_color = "#242428" if dark_mode else "#E5DFD0"
    card_bg = "#1C1C20" if dark_mode else "#FFFDF7"
    card_border = "#2E2E34" if dark_mode else "#E8E2D3"
    text_color = "#F4F4F6" if dark_mode else "#141414"
    muted_color = "#9898A0" if dark_mode else "#5B574D"
    input_bg = "#16161A" if dark_mode else "#FFFDF7"
    btn_bg = "#24242A" if dark_mode else "#FFFFFF"
    btn_text = "#F4F4F6" if dark_mode else "#141414"
    header_bg = "#24242A" if dark_mode else "#EFEADF"
    sidebar_bg = "#0C0C0E" if dark_mode else "#141414"
    sidebar_border = "#222226" if dark_mode else "#282828"
    accent = "#FFC21A"

    theme_css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: {text_color};
    }}

    h1 {{
        font-family: 'Inter', sans-serif !important;
        font-size: 48px !important;
        font-weight: 900 !important;
        letter-spacing: -0.02em !important;
        color: {text_color} !important;
        line-height: 1.15 !important;
    }}

    h2 {{
        font-family: 'Inter', sans-serif !important;
        font-size: 28px !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em !important;
        color: {text_color} !important;
    }}

    h3, h4, h5, h6 {{
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
        color: {text_color} !important;
    }}

    #MainMenu, header[data-testid="stHeader"], footer {{
        visibility: hidden !important;
        height: 0 !important;
    }}

    .block-container {{
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: 1200px !important;
    }}

    .stApp {{
        background-color: {bg_color} !important;
        background-image: radial-gradient({dot_color} 1.5px, transparent 1.5px) !important;
        background-size: 24px 24px !important;
        color: {text_color} !important;
    }}

    [data-testid="stSidebar"] {{
        background-color: {sidebar_bg} !important;
        color: #FFFFFF !important;
        border-right: 1px solid {sidebar_border} !important;
    }}

    [data-testid="stSidebar"] * {{
        color: #E0E0E0 !important;
    }}

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {{
        color: #FFFFFF !important;
    }}

    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] small {{
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 11px !important;
        letter-spacing: 0.15em !important;
        text-transform: uppercase !important;
        color: #8A8577 !important;
    }}

    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label[data-checked="true"],
    [data-testid="stSidebar"] div[aria-selected="true"] {{
        background-color: {accent} !important;
        color: #141414 !important;
        border-radius: 12px !important;
    }}

    [data-testid="stMetric"] {{
        background: {card_bg} !important;
        border: 1px solid {card_border} !important;
        border-radius: 20px !important;
        padding: 20px 24px !important;
        box-shadow: none !important;
    }}

    [data-testid="stMetricLabel"] {{
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 12px !important;
        letter-spacing: 0.15em !important;
        text-transform: uppercase !important;
        color: {muted_color} !important;
        font-weight: 600 !important;
    }}

    [data-testid="stMetricValue"] {{
        font-family: 'Inter', sans-serif !important;
        font-size: 44px !important;
        font-weight: 800 !important;
        color: {text_color} !important;
        letter-spacing: -0.02em !important;
    }}

    [data-testid="stMetricDelta"] {{
        color: {muted_color} !important;
        font-size: 13px !important;
    }}

    [data-testid="stForm"],
    div[data-testid="stVerticalBlockBorderWrapper"],
    details[data-testid="stExpander"] {{
        background: {card_bg} !important;
        border: 1px solid {card_border} !important;
        border-radius: 20px !important;
        padding: 24px !important;
        box-shadow: none !important;
    }}

    details[data-testid="stExpander"] summary {{
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        color: {text_color} !important;
    }}

    div.stButton > button {{
        border-radius: 12px !important;
        border: 1px solid {card_border} !important;
        background-color: {btn_bg} !important;
        color: {btn_text} !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        padding: 10px 20px !important;
        box-shadow: none !important;
        transition: background-color 0.15s ease, transform 0.1s ease !important;
    }}

    div.stButton > button:hover {{
        background-color: {accent} !important;
        color: #141414 !important;
        border-color: #141414 !important;
    }}

    .theme-toggle-container {{
        display: flex;
        justify-content: flex-end;
        align-items: center;
        padding-top: 14px;
    }}

    .theme-toggle-container div[data-testid="stToggle"] {{
        display: flex;
        justify-content: flex-end;
        align-items: center;
        width: auto !important;
        margin: 0 !important;
    }}

    .theme-toggle-container div[data-testid="stToggle"] label {{
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 11px !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        color: {muted_color} !important;
        font-weight: 700 !important;
        margin-right: 8px !important;
        cursor: pointer !important;
    }}

    .theme-toggle-container div[data-testid="stToggle"] div[role="switch"] {{
        transform: scale(0.75);
        transform-origin: right center;
    }}

    .theme-toggle-container div[data-testid="stToggle"] div[role="switch"][aria-checked="true"] {{
        background-color: {accent} !important;
    }}

    button[kind="primary"],
    div.stButton > button[kind="primaryFormSubmit"] {{
        background-color: {accent} !important;
        color: #141414 !important;
        border: 1px solid #141414 !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
    }}

    button[kind="primary"]:hover,
    div.stButton > button[kind="primaryFormSubmit"]:hover {{
        background-color: #E5AD12 !important;
        color: #141414 !important;
        border-color: #141414 !important;
    }}

    button[data-baseweb="tab"] {{
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 12px !important;
        letter-spacing: 0.15em !important;
        text-transform: uppercase !important;
        color: {muted_color} !important;
        font-weight: 600 !important;
        padding: 12px 18px !important;
        border: none !important;
    }}

    button[data-baseweb="tab"][aria-selected="true"] {{
        color: {text_color} !important;
        font-weight: 700 !important;
        border-bottom: 3px solid {accent} !important;
    }}

    div[data-baseweb="tab-highlight"] {{
        background-color: {accent} !important;
    }}

    div[data-baseweb="input"],
    div[data-baseweb="base-input"],
    div[data-baseweb="select"] > div {{
        border-radius: 12px !important;
        border: 1px solid {card_border} !important;
        background-color: {input_bg} !important;
        color: {text_color} !important;
    }}

    div[data-baseweb="input"]:focus-within,
    div[data-baseweb="base-input"]:focus-within,
    div[data-baseweb="select"] > div:focus-within {{
        border-color: {accent} !important;
    }}

    input, select, textarea {{
        color: {text_color} !important;
        font-family: 'Inter', sans-serif !important;
    }}

    label {{
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 12px !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
        color: {muted_color} !important;
        font-weight: 600 !important;
    }}

    [data-testid="stProgress"] > div > div > div {{
        background-color: {accent} !important;
        border-radius: 9999px !important;
    }}

    [data-testid="stProgress"] > div > div {{
        background-color: {card_border} !important;
        border-radius: 9999px !important;
    }}

    [data-testid="stDataFrame"],
    [data-testid="stTable"] {{
        background-color: {card_bg} !important;
        border: 1px solid {card_border} !important;
        border-radius: 16px !important;
        overflow: hidden !important;
    }}

    [data-testid="stDataFrame"] th,
    [data-testid="stTable"] th {{
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 12px !important;
        letter-spacing: 0.15em !important;
        text-transform: uppercase !important;
        background-color: {header_bg} !important;
        color: {muted_color} !important;
    }}

    [data-testid="stAlert"] {{
        border-radius: 16px !important;
        border-width: 1px !important;
        border-style: solid !important;
    }}

    div[data-baseweb="notification"][kind="positive"],
    .stAlert:has([data-testid="stNotificationContentSuccess"]) {{
        background-color: #E3F2EA !important;
        border-color: #BFE0D0 !important;
        color: #1F6B4F !important;
    }}
    .stAlert:has([data-testid="stNotificationContentSuccess"]) * {{
        color: #1F6B4F !important;
    }}

    div[data-baseweb="notification"][kind="warning"],
    .stAlert:has([data-testid="stNotificationContentWarning"]) {{
        background-color: #FFF1CC !important;
        border-color: #F5D26B !important;
        color: #7A5A00 !important;
    }}
    .stAlert:has([data-testid="stNotificationContentWarning"]) * {{
        color: #7A5A00 !important;
    }}

    div[data-baseweb="notification"][kind="info"],
    .stAlert:has([data-testid="stNotificationContentInfo"]) {{
        background-color: {header_bg} !important;
        border-color: {card_border} !important;
        color: {muted_color} !important;
    }}
    .stAlert:has([data-testid="stNotificationContentInfo"]) * {{
        color: {muted_color} !important;
    }}

    div[data-baseweb="notification"][kind="negative"],
    .stAlert:has([data-testid="stNotificationContentError"]) {{
        background-color: #FFF0EB !important;
        border-color: #F8C3B1 !important;
        color: #B84A00 !important;
    }}
    .stAlert:has([data-testid="stNotificationContentError"]) * {{
        color: #B84A00 !important;
    }}

    .stCaption, small {{
        color: {muted_color} !important;
    }}
    </style>
    """
    st.markdown(theme_css, unsafe_allow_html=True)
