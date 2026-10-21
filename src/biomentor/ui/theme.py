"""Visual identity: modern biomedical clinical workstation.

Designed for precision, legibility, and high-contrast scholarly readability.
Supports Auto, Light, and Dark modes smoothly across Streamlit and browser themes.
Zero visual artifacts, no blur overlays, and clean professional typography.
"""

from __future__ import annotations

LIGHT_VARS = """
  --ink: #0F172A;
  --muted: #475569;
  --paper: #F8FAFC;
  --card-bg: #FFFFFF;
  --sidebar-bg: #F1F5F9;
  --rule: #E2E8F0;
  --indigo: #3B82F6;
  --indigo-soft: #EFF6FF;
  --eosin: #BE185D;
  --eosin-soft: #FDF2F8;
  --amber: #B45309;
  --amber-soft: #FFFBEB;
"""

DARK_VARS = """
  --ink: #F8FAFC;
  --muted: #94A3B8;
  --paper: #090D16;
  --card-bg: #111827;
  --sidebar-bg: #070A10;
  --rule: #1E293B;
  --indigo: #60A5FA;
  --indigo-soft: rgba(96, 165, 250, 0.12);
  --eosin: #F472B6;
  --eosin-soft: rgba(244, 114, 182, 0.12);
  --amber: #FBBF24;
  --amber-soft: rgba(251, 191, 36, 0.12);
"""


def get_theme_css(mode: str = "auto") -> str:
    """Generate dynamic CSS based on active theme mode (auto, light, or dark)."""
    mode = (mode or "auto").lower()

    if mode == "dark":
        theme_rules = f"""
        :root, .stApp {{
          {DARK_VARS}
        }}
        """
    elif mode == "light":
        theme_rules = f"""
        :root, .stApp {{
          {LIGHT_VARS}
        }}
        """
    else:  # auto
        theme_rules = f"""
        :root {{
          {LIGHT_VARS}
        }}
        @media (prefers-color-scheme: dark) {{
          :root, .stApp {{
            {DARK_VARS}
          }}
        }}
        [data-theme="dark"], [data-theme="dark"] .stApp,
        html[theme="dark"], html[data-theme="dark"],
        body[data-theme="dark"], .stApp[data-theme="dark"],
        [data-base="dark"], .dark-theme {{
          {DARK_VARS}
        }}
        """

    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&display=swap');

:root {{
  --serif: 'Source Serif 4', Georgia, 'Times New Roman', serif;
  --sans: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}}

{theme_rules}

/* Global app container */
.stApp {{
  background-color: var(--paper) !important;
  color: var(--ink) !important;
  font-family: var(--sans);
}}

/* Eradicate any blur across the entire application */
*, *::before, *::after {{
  backdrop-filter: none !important;
  -webkit-backdrop-filter: none !important;
}}

/* Top header: completely transparent, ZERO blur, non-blocking */
[data-testid="stDecoration"] {{
  display: none !important;
}}

header[data-testid="stHeader"],
.stAppHeader,
[data-testid="stHeader"],
[data-testid="stHeader"] * {{
  background: transparent !important;
  background-color: transparent !important;
  backdrop-filter: none !important;
  -webkit-backdrop-filter: none !important;
  box-shadow: none !important;
  border: none !important;
}}

[data-testid="stToolbar"] {{
  right: 1.5rem;
  top: 0.5rem;
}}
[data-testid="stToolbar"] * {{
  color: var(--muted) !important;
}}

.block-container {{
  max-width: 1080px;
  padding-top: 3.2rem !important;
  padding-bottom: 4rem;
}}

/* Sidebar styling */
[data-testid="stSidebar"] {{
  background-color: var(--sidebar-bg) !important;
  border-right: 1px solid var(--rule) !important;
}}
[data-testid="stSidebar"] .block-container {{
  padding-top: 2rem !important;
}}
[data-testid="stSidebarNav"],
[data-testid="stSidebarNavItems"],
[data-testid="stSidebarNavSeparator"],
ul[data-testid="stSidebarNavItems"] {{
  display: none !important;
}}
[data-testid="stSidebar"] * {{
  color: var(--ink) !important;
}}
[data-testid="stSidebar"] .stMarkdown p {{
  color: var(--muted) !important;
}}
[data-testid="stSidebar"] a {{
  color: var(--ink) !important;
  text-decoration: none;
  font-weight: 500;
  font-size: 0.92rem;
  padding: 0.4rem 0.65rem;
  border-radius: 6px;
  display: flex;
  align-items: center;
  transition: background 0.15s ease, color 0.15s ease;
}}
[data-testid="stSidebar"] a:hover {{
  background-color: var(--indigo-soft) !important;
  color: var(--indigo) !important;
}}
[data-testid="stSidebar"] a[aria-current="page"],
[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"][aria-current="page"] {{
  background-color: var(--indigo-soft) !important;
  color: var(--indigo) !important;
  font-weight: 600;
  border-left: 3px solid var(--indigo);
}}

/* Typography */
h1, h2, h3, h4, h5, h6 {{
  font-family: var(--serif);
  color: var(--ink) !important;
  letter-spacing: -0.015em;
}}
h1 {{ font-weight: 600; font-size: 2.1rem; line-height: 1.2; margin-top: 0.2rem; margin-bottom: 0.5rem; }}
h2 {{ font-weight: 600; font-size: 1.4rem; margin-top: 1.5rem; }}
h3 {{ font-weight: 600; font-size: 1.15rem; }}

.hero {{ padding: 0.5rem 0 1rem; }}
.hero h1 {{ font-size: 2.5rem; line-height: 1.15; max-width: 22ch; margin: 0 0 .6rem; }}
.hero p  {{ font-size: 1.1rem; line-height: 1.6; color: var(--muted) !important; max-width: 62ch; margin: 0; }}

.prose {{ font-family: var(--serif); font-size: 1.05rem; line-height: 1.7; max-width: 70ch; margin: .35rem 0 .9rem; color: var(--ink); }}
.lead  {{ font-family: var(--serif); font-size: 1.22rem; line-height: 1.55; max-width: 64ch;
          border-left: 3px solid var(--indigo); padding: .2rem 0 .2rem 1rem; margin: .4rem 0 1.1rem; color: var(--ink); }}

/* Citations & badges */
sup.cite {{
  font-family: var(--sans);
  font-size: .68rem;
  font-weight: 600;
  color: var(--eosin) !important;
  background: var(--eosin-soft) !important;
  border-radius: 4px;
  padding: 0.1em .35em;
  margin-left: .22em;
  vertical-align: super;
  line-height: 1;
}}
.basis {{
  font-family: var(--sans);
  font-size: .72rem;
  color: var(--muted) !important;
  border: 1px solid var(--rule);
  border-radius: 999px;
  padding: 0 .5em;
  margin-left: .35em;
  white-space: nowrap;
}}

.chip {{
  display: inline-flex;
  align-items: center;
  font-size: .76rem;
  font-weight: 500;
  border-radius: 999px;
  padding: .15rem .65rem;
  margin: 0 .25rem .25rem 0;
  border: 1px solid var(--rule);
  background: var(--card-bg) !important;
  color: var(--ink) !important;
}}
.chip.src-pubmed {{ background: var(--indigo-soft) !important; border-color: var(--rule); color: var(--indigo) !important; }}
.chip.src-synth  {{ background: var(--amber-soft) !important; border-color: var(--rule); color: var(--amber) !important; }}
.chip.src-upload {{ background: var(--eosin-soft) !important; border-color: var(--rule); color: var(--eosin) !important; }}

/* Provenance Ribbon */
.rail {{
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: .3rem .7rem;
  padding: .55rem .85rem;
  border: 1px solid var(--rule);
  border-radius: 8px;
  background-color: var(--card-bg) !important;
  margin: .7rem 0 1.1rem;
  font-size: .86rem;
  color: var(--muted) !important;
}}
.rail b {{ color: var(--ink) !important; font-weight: 600; }}

/* Alerts and Notices */
.note {{
  border-left: 3px solid;
  padding: .6rem .9rem;
  margin: .65rem 0;
  border-radius: 0 6px 6px 0;
  font-size: .94rem;
  line-height: 1.5;
  max-width: 78ch;
  color: var(--ink) !important;
}}
.note-warn  {{ border-color: var(--amber);  background: var(--amber-soft) !important; }}
.note-info  {{ border-color: var(--indigo); background: var(--indigo-soft) !important; }}
.note-error {{ border-color: var(--eosin);  background: var(--eosin-soft) !important; }}

.quote {{
  border-left: 3px solid var(--indigo);
  padding: .2rem 0 .2rem .95rem;
  margin: .65rem 0;
  font-family: var(--serif);
  font-size: 1rem;
  line-height: 1.62;
  max-width: 76ch;
  color: var(--ink);
}}
.quote .meta {{
  font-family: var(--sans);
  font-size: .78rem;
  color: var(--muted) !important;
  margin-top: .25rem;
}}
mark {{ background: rgba(251, 191, 36, 0.25); color: inherit; padding: 0 .15em; border-radius: 2px; }}

/* Cards */
.card {{
  border: 1px solid var(--rule);
  border-radius: 8px;
  background-color: var(--card-bg) !important;
  color: var(--ink) !important;
  padding: 1rem 1.2rem;
  margin: .8rem 0 .4rem;
}}
.card .title {{
  font-family: var(--serif);
  font-weight: 600;
  font-size: 1.12rem;
  line-height: 1.35;
  margin-bottom: .25rem;
  color: var(--ink) !important;
}}
.card .bib {{ font-size: .84rem; color: var(--muted) !important; margin-bottom: .45rem; }}
.card .bib a, .refs a {{ color: var(--indigo) !important; text-decoration: none; }}
.card .bib a:hover, .refs a:hover {{ text-decoration: underline; }}

/* Mechanism Steps */
ol.steps {{ list-style: none; counter-reset: s; padding: 0; margin: .4rem 0 1.1rem; max-width: 72ch; }}
ol.steps li {{
  counter-increment: s;
  display: grid;
  grid-template-columns: 1.8rem 1fr;
  gap: .75rem;
  padding: .6rem 0;
  border-bottom: 1px solid var(--rule);
  font-family: var(--serif);
  font-size: 1.02rem;
  line-height: 1.6;
  color: var(--ink);
}}
ol.steps li::before {{
  content: counter(s);
  width: 1.55rem;
  height: 1.55rem;
  border-radius: 50%;
  background: var(--indigo);
  color: #fff;
  font: 600 .78rem var(--sans);
  display: flex;
  align-items: center;
  justify-content: center;
  margin-top: .15rem;
}}

.term {{ padding: .5rem 0; border-bottom: 1px solid var(--rule); max-width: 72ch; font-size: .98rem; line-height: 1.55; color: var(--ink); }}
.term b {{ font-family: var(--serif); font-size: 1.02rem; margin-right: .35rem; color: var(--ink); }}
ul.refs {{ padding-left: 1.2rem; max-width: 80ch; color: var(--ink); }} ul.refs li {{ margin: .35rem 0; font-size: .9rem; line-height: 1.5; color: var(--ink); }}
.empty {{ border: 1px dashed var(--rule); border-radius: 8px; padding: 1.3rem 1.5rem; color: var(--muted) !important; max-width: 72ch; line-height: 1.55; }}

/* Form containers */
[data-testid="stForm"] {{
  border: 1px solid var(--rule) !important;
  border-radius: 10px !important;
  background-color: var(--card-bg) !important;
  padding: 1.2rem !important;
}}

/* BaseWeb input overrides */
div[data-baseweb="input"] > div,
div[data-baseweb="base-input"],
input[type="text"],
input[type="password"],
textarea {{
  background-color: var(--card-bg) !important;
  color: var(--ink) !important;
  border: 1px solid var(--rule) !important;
  border-radius: 6px !important;
  font-size: 0.95rem !important;
}}
div[data-baseweb="input"] > div:focus-within {{
  border-color: var(--indigo) !important;
  box-shadow: 0 0 0 1px var(--indigo) !important;
}}

div[data-baseweb="select"] > div {{
  background-color: var(--card-bg) !important;
  color: var(--ink) !important;
  border: 1px solid var(--rule) !important;
  border-radius: 6px !important;
}}
div[data-baseweb="select"] svg {{
  fill: var(--ink) !important;
}}

div[data-baseweb="popover"],
div[data-baseweb="popover"] > div,
ul[role="listbox"],
li[role="option"] {{
  background-color: var(--card-bg) !important;
  color: var(--ink) !important;
}}
li[role="option"]:hover, li[aria-selected="true"] {{
  background-color: var(--indigo-soft) !important;
  color: var(--indigo) !important;
}}

/* Expanders */
[data-testid="stExpander"] {{
  border: 1px solid var(--rule) !important;
  border-radius: 8px !important;
  background-color: var(--card-bg) !important;
  margin-bottom: 0.5rem;
}}
[data-testid="stExpander"] details {{
  background-color: var(--card-bg) !important;
  color: var(--ink) !important;
}}
[data-testid="stExpander"] summary {{
  background-color: var(--card-bg) !important;
  color: var(--ink) !important;
  border-radius: 7px !important;
  font-weight: 500;
  font-size: 0.92rem;
}}
[data-testid="stExpander"] summary:hover {{
  color: var(--indigo) !important;
}}
[data-testid="stExpander"] summary svg {{
  fill: var(--ink) !important;
}}

/* Tabs */
button[data-baseweb="tab"] {{
  font-weight: 500 !important;
  font-size: 0.94rem !important;
  color: var(--muted) !important;
  padding: 0.55rem 1.1rem !important;
  border-bottom: 2px solid transparent !important;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
  color: var(--indigo) !important;
  border-bottom: 2px solid var(--indigo) !important;
  font-weight: 600 !important;
}}
button[data-baseweb="tab"]:hover {{
  color: var(--ink) !important;
}}

/* Secondary Buttons */
button[data-testid="baseButton-secondary"],
.stButton > button,
button[kind="secondary"],
div[data-testid="stButton"] button {{
  border-radius: 6px !important;
  font-weight: 500 !important;
  font-size: 0.88rem !important;
  border: 1px solid var(--rule) !important;
  background-color: var(--card-bg) !important;
  color: var(--ink) !important;
}}
.stButton > button *,
button[data-testid="baseButton-secondary"] * {{
  color: var(--ink) !important;
}}
.stButton > button:hover,
button[data-testid="baseButton-secondary"]:hover {{
  border-color: var(--indigo) !important;
  color: var(--indigo) !important;
  background-color: var(--indigo-soft) !important;
}}
.stButton > button:hover *,
button[data-testid="baseButton-secondary"]:hover * {{
  color: var(--indigo) !important;
}}

/* Primary Buttons */
.stButton > button[kind="primary"],
.stFormSubmitButton > button,
.stDownloadButton > button[kind="primary"] {{
  background-color: #2563EB !important;
  background-image: none !important;
  color: #FFFFFF !important;
  border: 1px solid #1D4ED8 !important;
  font-weight: 500 !important;
  border-radius: 6px !important;
  padding: 0.5rem 1.25rem !important;
  box-shadow: none !important;
}}
.stButton > button[kind="primary"]:hover,
.stFormSubmitButton > button:hover {{
  background-color: #1D4ED8 !important;
}}

/* Clean Technical Header Tag */
.tech-tag {{
  display: inline-block;
  font-size: 0.72rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  padding: 0.2rem 0.6rem;
  border-radius: 4px;
  background: var(--indigo-soft);
  color: var(--indigo) !important;
  border: 1px solid var(--rule);
  margin-bottom: 0.4rem;
}}

:focus-visible {{ outline: 2px solid var(--indigo) !important; outline-offset: 2px; }}

@media (max-width: 700px) {{
  h1 {{ font-size: 1.8rem; }} .hero h1 {{ font-size: 2rem; }} .lead {{ font-size: 1.1rem; }}
  .block-container {{ padding-left: 1rem; padding-right: 1rem; }}
}}
</style>
"""


CSS = get_theme_css("auto")
