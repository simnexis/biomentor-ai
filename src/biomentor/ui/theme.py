"""Visual identity: a histology slide.

Hematoxylin indigo is the ink, eosin rose is used only for citations, amber only for warnings.
Reading text is set in a serif at a 68-character measure; the interface is a plain sans.
Fonts load from Google Fonts when online and fall back to Georgia / system sans offline.
Supports Auto, Light, and Dark modes smoothly across Streamlit and browser themes.
"""

from __future__ import annotations

LIGHT_VARS = """
  --ink: #1B1B33;
  --muted: #5B5B78;
  --paper: #FBFBFE;
  --card-bg: #FFFFFF;
  --sidebar-bg: #F3F3FA;
  --rule: #DEDEEE;
  --indigo: #3F3D8F;
  --indigo-soft: #ECECF8;
  --eosin: #C94F74;
  --eosin-soft: #FBE9EF;
  --amber: #B7791F;
  --amber-soft: #FBF1DD;
"""

DARK_VARS = """
  --ink: #F0F1F8;
  --muted: #9E9EB5;
  --paper: #0E1117;
  --card-bg: #1A1D27;
  --sidebar-bg: #131620;
  --rule: #2B2F42;
  --indigo: #9A98F0;
  --indigo-soft: #232242;
  --eosin: #F472B6;
  --eosin-soft: #381A26;
  --amber: #FBBF24;
  --amber-soft: #382A10;
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
  --sans: 'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
}}

{theme_rules}

.stApp {{
  background: var(--paper) !important;
  color: var(--ink) !important;
  font-family: var(--sans);
}}

.block-container {{
  max-width: 1080px;
  padding-top: 2.2rem;
  padding-bottom: 4rem;
}}

[data-testid="stSidebar"] {{
  background: var(--sidebar-bg) !important;
  border-right: 1px solid var(--rule) !important;
}}
[data-testid="stSidebar"] .block-container {{
  padding-top: 1.4rem;
}}
[data-testid="stSidebarNav"] {{
  display: none;
}}  /* custom navigation is in the sidebar */

h1, h2, h3, h4, h5, h6 {{
  font-family: var(--serif);
  color: var(--ink) !important;
  letter-spacing: -0.012em;
}}
h1 {{ font-weight: 600; font-size: 2.2rem; line-height: 1.15; }}
h2 {{ font-weight: 600; font-size: 1.45rem; margin-top: 1.6rem; }}
h3 {{ font-weight: 600; font-size: 1.15rem; }}

.hero {{ padding: 1.2rem 0 .6rem; }}
.hero h1 {{ font-size: 3rem; line-height: 1.08; max-width: 17ch; margin: 0 0 .9rem; }}
.hero p  {{ font-size: 1.12rem; line-height: 1.6; color: var(--muted) !important; max-width: 56ch; margin: 0; }}

.prose {{ font-family: var(--serif); font-size: 1.08rem; line-height: 1.72; max-width: 68ch; margin: .35rem 0 .9rem; color: var(--ink); }}
.lead  {{ font-family: var(--serif); font-size: 1.28rem; line-height: 1.55; max-width: 62ch;
          border-left: 3px solid var(--indigo); padding: .1rem 0 .1rem 1rem; margin: .4rem 0 1.1rem; color: var(--ink); }}

sup.cite {{ font-family: var(--sans); font-size: .68rem; font-weight: 600; color: var(--eosin) !important;
           background: var(--eosin-soft) !important; border-radius: 4px; padding: 0 .3em; margin-left: .22em; vertical-align: super; line-height: 1; }}
.basis {{ font-family: var(--sans); font-size: .72rem; color: var(--muted) !important; border: 1px solid var(--rule);
         border-radius: 999px; padding: 0 .5em; margin-left: .35em; white-space: nowrap; }}

.chip {{ display: inline-block; font-size: .78rem; border-radius: 999px; padding: .12rem .65rem;
        margin: 0 .3rem .2rem 0; border: 1px solid var(--rule); background: var(--card-bg) !important; color: var(--ink) !important; }}
.chip.src-pubmed {{ background: var(--indigo-soft) !important; border-color: var(--rule); color: var(--indigo) !important; }}
.chip.src-synth  {{ background: var(--amber-soft) !important; border-color: var(--rule); color: var(--amber) !important; }}
.chip.src-upload {{ background: var(--eosin-soft) !important; border-color: var(--rule); color: var(--eosin) !important; }}

.rail {{ display: flex; flex-wrap: wrap; align-items: center; gap: .2rem .6rem; padding: .55rem 0;
        border-top: 1px solid var(--rule); border-bottom: 1px solid var(--rule); margin: .5rem 0 1.2rem;
        font-size: .86rem; color: var(--muted) !important; }}
.rail b {{ color: var(--ink) !important; font-weight: 600; }}

.note {{ border-left: 3px solid; padding: .55rem .9rem; margin: .65rem 0; border-radius: 0 6px 6px 0; font-size: .95rem; line-height: 1.5; max-width: 78ch; color: var(--ink) !important; }}
.note-warn  {{ border-color: var(--amber);  background: var(--amber-soft) !important; }}
.note-info  {{ border-color: var(--indigo); background: var(--indigo-soft) !important; }}
.note-error {{ border-color: var(--eosin);  background: var(--eosin-soft) !important; }}

.quote {{ border-left: 3px solid var(--indigo); padding: .1rem 0 .1rem .9rem; margin: .55rem 0;
         font-family: var(--serif); font-size: 1rem; line-height: 1.62; max-width: 76ch; color: var(--ink); }}
.quote .meta {{ font-family: var(--sans); font-size: .78rem; color: var(--muted) !important; margin-top: .2rem; }}
mark {{ background: #FFE9A8; color: #1B1B33; padding: 0 .08em; border-radius: 2px; }}

.card {{ border: 1px solid var(--rule); border-radius: 10px; background: var(--card-bg) !important; color: var(--ink) !important; padding: .85rem 1.1rem .5rem; margin: 1rem 0 .4rem; }}
.card .title {{ font-family: var(--serif); font-weight: 600; font-size: 1.12rem; line-height: 1.35; margin-bottom: .25rem; color: var(--ink) !important; }}
.card .bib {{ font-size: .84rem; color: var(--muted) !important; margin-bottom: .45rem; }}
.card .bib a, .refs a {{ color: var(--indigo) !important; }}

ol.steps {{ list-style: none; counter-reset: s; padding: 0; margin: .3rem 0 1.1rem; max-width: 70ch; }}
ol.steps li {{ counter-increment: s; display: grid; grid-template-columns: 1.9rem 1fr; gap: .7rem; padding: .5rem 0;
              border-bottom: 1px solid var(--rule); font-family: var(--serif); font-size: 1.04rem; line-height: 1.6; color: var(--ink); }}
ol.steps li::before {{ content: counter(s); width: 1.55rem; height: 1.55rem; border-radius: 50%; background: var(--indigo);
                      color: #fff; font: 600 .78rem var(--sans); display: flex; align-items: center; justify-content: center; margin-top: .15rem; }}

.term {{ padding: .45rem 0; border-bottom: 1px solid var(--rule); max-width: 70ch; font-size: .98rem; line-height: 1.55; color: var(--ink); }}
.term b {{ font-family: var(--serif); font-size: 1.02rem; margin-right: .35rem; color: var(--ink); }}
ul.refs {{ padding-left: 1.1rem; max-width: 80ch; color: var(--ink); }} ul.refs li {{ margin: .35rem 0; font-size: .9rem; line-height: 1.5; color: var(--ink); }}
.empty {{ border: 1px dashed var(--rule); border-radius: 10px; padding: 1.2rem 1.3rem; color: var(--muted) !important; max-width: 70ch; line-height: 1.55; }}

.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {{ border-radius: 8px; font-weight: 500; }}
button[data-baseweb="tab"] {{ font-weight: 500; }}
[data-testid="stForm"] {{ border: 1px solid var(--rule); border-radius: 10px; background: var(--card-bg) !important; }}
[data-testid="stExpander"] {{ border-radius: 8px; background: var(--card-bg) !important; border: 1px solid var(--rule); }}
:focus-visible {{ outline: 2px solid var(--indigo) !important; outline-offset: 2px; }}

@media (max-width: 700px) {{
  h1 {{ font-size: 1.8rem; }} .hero h1 {{ font-size: 2.1rem; }} .lead {{ font-size: 1.12rem; }}
  .block-container {{ padding-left: 1rem; padding-right: 1rem; }}
}}
</style>
"""


CSS = get_theme_css("auto")
