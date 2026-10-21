"""Visual identity: modern biomedical clinical workstation.

Hematoxylin indigo is the ink, eosin rose is used for citations, amber for warnings.
Reading text is set in a serif at a 68-character measure; the interface is a plain sans.
Supports Auto, Light, and Dark modes smoothly across Streamlit and browser themes.
"""

from __future__ import annotations

LIGHT_VARS = """
  --ink: #111827;
  --muted: #4B5563;
  --paper: #F9FAFB;
  --card-bg: #FFFFFF;
  --sidebar-bg: #F3F4F6;
  --rule: #E5E7EB;
  --indigo: #4F46E5;
  --indigo-soft: #EEF2FF;
  --eosin: #DB2777;
  --eosin-soft: #FDF2F8;
  --amber: #D97706;
  --amber-soft: #FFFBEB;
"""

DARK_VARS = """
  --ink: #F3F4F6;
  --muted: #9CA3AF;
  --paper: #0F172A;
  --card-bg: #1E293B;
  --sidebar-bg: #0B1120;
  --rule: #334155;
  --indigo: #818CF8;
  --indigo-soft: #1E1B4B;
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
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&display=swap');

:root {{
  --serif: 'Source Serif 4', Georgia, 'Times New Roman', serif;
  --sans: 'Plus Jakarta Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
}}

{theme_rules}

/* Global app container */
.stApp {{
  background-color: var(--paper) !important;
  color: var(--ink) !important;
  font-family: var(--sans);
}}

/* Top navigation header: seamless transparent bar */
header[data-testid="stHeader"] {{
  background: transparent !important;
  color: var(--ink) !important;
  backdrop-filter: blur(8px);
  border-bottom: 1px solid transparent;
}}
[data-testid="stToolbar"] {{
  right: 1.5rem;
  top: 0.5rem;
}}
[data-testid="stToolbar"] * {{
  color: var(--muted) !important;
}}

.block-container {{
  max-width: 1120px;
  padding-top: 1.8rem;
  padding-bottom: 4rem;
}}

/* Sidebar styling */
[data-testid="stSidebar"] {{
  background-color: var(--sidebar-bg) !important;
  border-right: 1px solid var(--rule) !important;
}}
[data-testid="stSidebar"] .block-container {{
  padding-top: 1.2rem;
}}
[data-testid="stSidebarNav"] {{
  display: none;
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
  font-size: 0.94rem;
  padding: 0.45rem 0.75rem;
  border-radius: 8px;
  display: flex;
  align-items: center;
  transition: all 0.15s ease;
}}
[data-testid="stSidebar"] a:hover {{
  background-color: var(--indigo-soft) !important;
  color: var(--indigo) !important;
  transform: translateX(2px);
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
h1 {{ font-weight: 700; font-size: 2.3rem; line-height: 1.15; }}
h2 {{ font-weight: 600; font-size: 1.5rem; margin-top: 1.6rem; }}
h3 {{ font-weight: 600; font-size: 1.18rem; }}

.hero {{ padding: 0.8rem 0 1.2rem; }}
.hero h1 {{ font-size: 2.9rem; line-height: 1.1; max-width: 20ch; margin: 0 0 .8rem; }}
.hero p  {{ font-size: 1.15rem; line-height: 1.65; color: var(--muted) !important; max-width: 60ch; margin: 0; }}

.prose {{ font-family: var(--serif); font-size: 1.08rem; line-height: 1.75; max-width: 70ch; margin: .4rem 0 1rem; color: var(--ink); }}
.lead  {{ font-family: var(--serif); font-size: 1.28rem; line-height: 1.55; max-width: 64ch;
          border-left: 3px solid var(--indigo); padding: .2rem 0 .2rem 1.1rem; margin: .5rem 0 1.2rem; color: var(--ink); }}

/* Citations & badges */
sup.cite {{
  font-family: var(--sans);
  font-size: .70rem;
  font-weight: 600;
  color: var(--eosin) !important;
  background: var(--eosin-soft) !important;
  border-radius: 4px;
  padding: 0.1em .4em;
  margin-left: .25em;
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
  gap: 0.35rem;
  font-size: .78rem;
  font-weight: 500;
  border-radius: 999px;
  padding: .18rem .7rem;
  margin: 0 .3rem .3rem 0;
  border: 1px solid var(--rule);
  background: var(--card-bg) !important;
  color: var(--ink) !important;
}}
.chip.src-pubmed {{ background: var(--indigo-soft) !important; border-color: var(--rule); color: var(--indigo) !important; }}
.chip.src-synth  {{ background: var(--amber-soft) !important; border-color: var(--rule); color: var(--amber) !important; }}
.chip.src-upload {{ background: var(--eosin-soft) !important; border-color: var(--rule); color: var(--eosin) !important; }}

/* Telemetry & Provenance Ribbon */
.rail {{
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: .4rem .8rem;
  padding: .65rem 1rem;
  border: 1px solid var(--rule);
  border-radius: 10px;
  background-color: var(--card-bg) !important;
  margin: .8rem 0 1.3rem;
  font-size: .88rem;
  color: var(--muted) !important;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
}}
.rail b {{ color: var(--ink) !important; font-weight: 600; }}

/* Alerts and Notices */
.note {{
  border-left: 3px solid;
  padding: .65rem 1rem;
  margin: .75rem 0;
  border-radius: 0 8px 8px 0;
  font-size: .95rem;
  line-height: 1.55;
  max-width: 78ch;
  color: var(--ink) !important;
}}
.note-warn  {{ border-color: var(--amber);  background: var(--amber-soft) !important; }}
.note-info  {{ border-color: var(--indigo); background: var(--indigo-soft) !important; }}
.note-error {{ border-color: var(--eosin);  background: var(--eosin-soft) !important; }}

.quote {{
  border-left: 3px solid var(--indigo);
  padding: .2rem 0 .2rem 1rem;
  margin: .7rem 0;
  font-family: var(--serif);
  font-size: 1.02rem;
  line-height: 1.65;
  max-width: 76ch;
  color: var(--ink);
}}
.quote .meta {{
  font-family: var(--sans);
  font-size: .80rem;
  color: var(--muted) !important;
  margin-top: .3rem;
}}
mark {{ background: #FEF08A; color: #1E293B; padding: 0 .15em; border-radius: 3px; font-weight: 500; }}

/* Cards */
.card {{
  border: 1px solid var(--rule);
  border-radius: 12px;
  background-color: var(--card-bg) !important;
  color: var(--ink) !important;
  padding: 1.1rem 1.3rem;
  margin: 1rem 0 .5rem;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.04);
}}
.card .title {{
  font-family: var(--serif);
  font-weight: 600;
  font-size: 1.15rem;
  line-height: 1.35;
  margin-bottom: .3rem;
  color: var(--ink) !important;
}}
.card .bib {{ font-size: .86rem; color: var(--muted) !important; margin-bottom: .5rem; }}
.card .bib a, .refs a {{ color: var(--indigo) !important; text-decoration: none; font-weight: 500; }}
.card .bib a:hover, .refs a:hover {{ text-decoration: underline; }}

/* Mechanism Steps */
ol.steps {{ list-style: none; counter-reset: s; padding: 0; margin: .5rem 0 1.2rem; max-width: 72ch; }}
ol.steps li {{
  counter-increment: s;
  display: grid;
  grid-template-columns: 2rem 1fr;
  gap: .8rem;
  padding: .65rem 0;
  border-bottom: 1px solid var(--rule);
  font-family: var(--serif);
  font-size: 1.04rem;
  line-height: 1.65;
  color: var(--ink);
}}
ol.steps li::before {{
  content: counter(s);
  width: 1.65rem;
  height: 1.65rem;
  border-radius: 50%;
  background: var(--indigo);
  color: #fff;
  font: 600 .82rem var(--sans);
  display: flex;
  align-items: center;
  justify-content: center;
  margin-top: .15rem;
}}

.term {{ padding: .55rem 0; border-bottom: 1px solid var(--rule); max-width: 72ch; font-size: 1rem; line-height: 1.6; color: var(--ink); }}
.term b {{ font-family: var(--serif); font-size: 1.05rem; margin-right: .4rem; color: var(--ink); }}
ul.refs {{ padding-left: 1.2rem; max-width: 80ch; color: var(--ink); }} ul.refs li {{ margin: .4rem 0; font-size: .92rem; line-height: 1.55; color: var(--ink); }}
.empty {{ border: 1px dashed var(--rule); border-radius: 12px; padding: 1.4rem 1.6rem; color: var(--muted) !important; max-width: 72ch; line-height: 1.6; }}

/* Form inputs & BaseWeb overrides */
[data-testid="stForm"] {{
  border: 1px solid var(--rule) !important;
  border-radius: 12px !important;
  background-color: var(--card-bg) !important;
  padding: 1.4rem !important;
  box-shadow: 0 4px 18px rgba(0, 0, 0, 0.04);
}}

div[data-baseweb="input"] > div,
div[data-baseweb="base-input"],
input[type="text"],
input[type="password"],
textarea {{
  background-color: var(--card-bg) !important;
  color: var(--ink) !important;
  border: 1px solid var(--rule) !important;
  border-radius: 8px !important;
  font-size: 0.98rem !important;
}}
div[data-baseweb="input"] > div:focus-within {{
  border-color: var(--indigo) !important;
  box-shadow: 0 0 0 2px var(--indigo-soft) !important;
}}

div[data-baseweb="select"] > div {{
  background-color: var(--card-bg) !important;
  color: var(--ink) !important;
  border: 1px solid var(--rule) !important;
  border-radius: 8px !important;
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
  border-radius: 10px !important;
  background-color: var(--card-bg) !important;
  margin-bottom: 0.6rem;
}}
[data-testid="stExpander"] details {{
  background-color: var(--card-bg) !important;
  color: var(--ink) !important;
}}
[data-testid="stExpander"] summary {{
  background-color: var(--card-bg) !important;
  color: var(--ink) !important;
  border-radius: 9px !important;
  font-weight: 500;
}}
[data-testid="stExpander"] summary:hover {{
  color: var(--indigo) !important;
}}
[data-testid="stExpander"] summary svg {{
  fill: var(--ink) !important;
}}

/* Tabs */
button[data-baseweb="tab"] {{
  font-weight: 600 !important;
  font-size: 0.96rem !important;
  color: var(--muted) !important;
  padding: 0.65rem 1.3rem !important;
  border-bottom: 2px solid transparent !important;
  transition: all 0.15s ease;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
  color: var(--indigo) !important;
  border-bottom: 2px solid var(--indigo) !important;
}}
button[data-baseweb="tab"]:hover {{
  color: var(--ink) !important;
}}

/* Buttons */
.stButton > button,
.stDownloadButton > button {{
  border-radius: 8px !important;
  font-weight: 500 !important;
  transition: all 0.15s ease;
  border: 1px solid var(--rule) !important;
  background-color: var(--card-bg) !important;
  color: var(--ink) !important;
}}
.stButton > button:hover {{
  border-color: var(--indigo) !important;
  color: var(--indigo) !important;
  background-color: var(--indigo-soft) !important;
}}

.stButton > button[kind="primary"],
.stFormSubmitButton > button,
.stDownloadButton > button[kind="primary"] {{
  background: linear-gradient(135deg, #4F46E5 0%, #3730A3 100%) !important;
  color: #FFFFFF !important;
  border: 1px solid rgba(255, 255, 255, 0.15) !important;
  font-weight: 600 !important;
  letter-spacing: 0.01em;
  box-shadow: 0 4px 14px 0 rgba(79, 70, 229, 0.35);
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
  padding: 0.55rem 1.4rem !important;
}}
.stButton > button[kind="primary"]:hover,
.stFormSubmitButton > button:hover {{
  background: linear-gradient(135deg, #6366F1 0%, #4338CA 100%) !important;
  box-shadow: 0 6px 20px 0 rgba(79, 70, 229, 0.5);
  transform: translateY(-1px);
}}

/* Recruiter & Engineering Badges */
.recruiter-badge {{
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  font-size: 0.74rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  padding: 0.25rem 0.75rem;
  border-radius: 999px;
  background: var(--indigo-soft);
  color: var(--indigo) !important;
  border: 1px solid var(--rule);
  margin-bottom: 0.6rem;
}}
.recruiter-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
  gap: 0.75rem;
  margin: 1rem 0 1.5rem;
}}
.recruiter-card {{
  border: 1px solid var(--rule);
  border-radius: 10px;
  background-color: var(--card-bg);
  padding: 0.85rem 1rem;
  transition: transform 0.15s ease, border-color 0.15s ease;
}}
.recruiter-card:hover {{
  border-color: var(--indigo);
  transform: translateY(-2px);
}}
.recruiter-card .val {{
  font-size: 1.35rem;
  font-weight: 700;
  color: var(--indigo);
  line-height: 1.1;
}}
.recruiter-card .lbl {{
  font-size: 0.80rem;
  font-weight: 500;
  color: var(--muted);
  margin-top: 0.25rem;
}}

:focus-visible {{ outline: 2px solid var(--indigo) !important; outline-offset: 2px; }}

@media (max-width: 700px) {{
  h1 {{ font-size: 1.9rem; }} .hero h1 {{ font-size: 2.2rem; }} .lead {{ font-size: 1.12rem; }}
  .block-container {{ padding-left: 1rem; padding-right: 1rem; }}
}}
</style>
"""


CSS = get_theme_css("auto")
