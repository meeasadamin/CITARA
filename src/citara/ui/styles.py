"""Visual register of the interface (features 60, 61).

Institutional rather than consumer: navy and steel, restrained type, hairline rules, and no
decoration that does not carry meaning. Colour says one of four things - the brand bar, a
citation, the state of an answer, or a warning - and nothing else is coloured.

An answer is set like a briefing note. Small-caps labels over hairline rules separate the
question, the answer, the evidence and the sources, so the eye can find the part it needs
without reading the whole turn. Figures inside an answer are marked, because a figure is
usually what the reader came for and what they will check against the cited page.

Phone width is designed for, not patched: the demo happens on a phone in an interview room,
so chips stay legible without hover, the header shrinks, and the page never scrolls sideways.
"""

from __future__ import annotations

NAVY = "#10243D"
NAVY_DEEP = "#0A1728"
STEEL = "#2B6CA3"
STEEL_TINT = "#EAF1F8"
MIST = "#F4F6F9"
BORDER = "#D6DCE4"
INK = "#1B1F24"
MUTED = "#5A6472"
AMBER = "#B4791F"
AMBER_TINT = "#FDF3E2"
RED = "#9B2226"
RED_TINT = "#FBEDED"

CSS = f"""
<style>
[data-testid="stMainBlockContainer"] {{ padding-top: 2.4rem; padding-bottom: 5rem; }}
html, body, [data-testid="stAppViewContainer"] {{ color: {INK}; }}

/* Leftovers from the previous run fade out entirely rather than sitting at a third opacity
   under an answer being written, after the same half-second delay Streamlit uses. */
[data-testid="stMainBlockContainer"] [data-testid="stElementContainer"][data-stale="true"] {{
  opacity: 0 !important; transition: opacity 0.3s ease-in 0.5s;
}}

/* -- brand bar ---------------------------------------------------------------------- */
.citara-header {{
  background: linear-gradient(180deg, {NAVY} 0%, {NAVY_DEEP} 100%);
  color: #FFFFFF; padding: 15px 18px 14px; border-radius: 3px;
}}
.brand {{ display: flex; align-items: center; gap: 12px; }}
.brand-mark {{ display: flex; color: #FFFFFF; opacity: 0.95; }}
.brand-text {{ display: flex; flex-direction: column; }}
.brand-name {{ font-size: 1.32rem; font-weight: 700; letter-spacing: 0.18em; line-height: 1.1; }}
.brand-sub {{ font-size: 0.84rem; opacity: 0.88; margin-top: 3px; }}
.citara-disclaimer {{
  font-size: 0.76rem; color: {MUTED}; background: {MIST};
  border-left: 3px solid {STEEL}; padding: 7px 10px; margin: 8px 0 16px;
}}
.citara-intro {{ font-size: 0.94rem; color: {INK}; margin: 2px 0 12px; }}
.citara-scope {{ font-size: 0.8rem; color: {MUTED}; margin: 0 0 6px; }}
.citara-searching {{ font-size: 0.9rem; color: {MUTED}; font-style: italic; }}

/* -- section labels ------------------------------------------------------------------ */
.section-label {{
  display: block; font-size: 0.68rem; font-weight: 700; letter-spacing: 0.14em;
  text-transform: uppercase; color: {STEEL}; border-bottom: 1px solid {BORDER};
  padding-bottom: 4px; margin: 16px 0 8px;
}}
.section-label.first {{ margin-top: 2px; }}
.section-label .count {{ float: right; font-weight: 600; color: {MUTED}; letter-spacing: 0.04em; }}

/* -- citations ----------------------------------------------------------------------- */
.cite-chip {{
  display: inline-block; font-size: 0.71rem; font-weight: 700; line-height: 1.3;
  padding: 0 5px; margin: 0 2px; border: 1px solid {STEEL}; border-radius: 2px;
  color: {STEEL}; background: {STEEL_TINT}; white-space: nowrap; vertical-align: 1px;
  font-variant-numeric: tabular-nums;
}}
.cite-invalid {{ border-color: {RED}; color: {RED}; background: {RED_TINT}; }}
.cite-tail {{ white-space: nowrap; }}
.figure {{
  background: {STEEL_TINT}; border-bottom: 1px solid {STEEL}; padding: 0 2px;
  font-weight: 600; font-variant-numeric: tabular-nums;
}}

/* -- answer state -------------------------------------------------------------------- */
.citara-meta {{
  display: flex; flex-wrap: wrap; gap: 6px 12px; align-items: center;
  font-size: 0.76rem; color: {MUTED};
}}
.band {{
  font-weight: 700; font-size: 0.7rem; letter-spacing: 0.1em; padding: 2px 8px;
  border-radius: 2px; border: 1px solid {NAVY}; text-transform: uppercase;
}}
.band-High {{ background: {NAVY}; color: #FFFFFF; }}
.band-Moderate {{ background: {STEEL_TINT}; color: {NAVY}; border-color: {STEEL}; }}
.band-Low {{ background: {AMBER_TINT}; color: {AMBER}; border-color: {AMBER}; }}

.citara-notice {{
  border-left: 4px solid {NAVY}; background: {MIST}; padding: 9px 12px;
  margin-bottom: 8px; font-size: 0.9rem;
}}
.citara-notice strong {{ display: block; margin-bottom: 3px; color: {NAVY}; }}
.notice-degraded {{ border-color: {AMBER}; background: {AMBER_TINT}; }}
.notice-degraded strong {{ color: {AMBER}; }}
.notice-blocked {{ border-color: {RED}; background: {RED_TINT}; }}
.notice-blocked strong {{ color: {RED}; }}
.notice-scope, .notice-limit {{ border-color: {MUTED}; background: #F5F6F7; }}
.notice-scope strong, .notice-limit strong {{ color: {INK}; }}

/* -- question and sources ------------------------------------------------------------ */
.citara-question {{ font-size: 1.02rem; font-weight: 600; color: {NAVY}; margin: 0; }}
[data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"] {{
  display: none;
}}
[data-testid="stChatMessage"] {{ background: transparent; padding: 0; gap: 0; }}

.source-head {{ font-size: 0.82rem; margin: 10px 0 2px; }}
.source-head .score {{ color: {MUTED}; font-variant-numeric: tabular-nums; }}
.source-head .state {{ color: {MUTED}; font-size: 0.74rem; }}

/* -- sidebar ------------------------------------------------------------------------- */
[data-testid="stSidebar"] {{ background: {MIST}; border-right: 1px solid {BORDER}; }}
[data-testid="stMarkdownContainer"] ul.corpus-list {{
  font-size: 0.78rem; line-height: 1.35; padding-left: 0; margin-left: 0; list-style: none;
}}
[data-testid="stMarkdownContainer"] ul.corpus-list li {{ margin: 0 0 5px 0; padding-left: 0; }}
.corpus-list .pages {{ color: {MUTED}; }}
.corpus-list .gap {{ color: {AMBER}; }}

/* Starter questions read as a list, so they align left; sidebar buttons stay centred. */
[data-testid="stMainBlockContainer"] .stButton button {{
  justify-content: flex-start; text-align: left; white-space: normal;
}}
[data-testid="stMainBlockContainer"] .stButton button > div {{
  justify-content: flex-start; width: 100%;
}}
[data-testid="stMainBlockContainer"] .stButton button p {{ text-align: left; }}

@media (max-width: 640px) {{
  /* The top bar holding the sidebar toggle overlays the page on a phone; less clearance
     than this hid the brand bar and its title underneath it. */
  [data-testid="stMainBlockContainer"] {{
    padding-left: 0.8rem; padding-right: 0.8rem; padding-top: 3.8rem;
  }}
  .citara-header {{ padding: 12px 13px 11px; }}
  .brand-name {{ font-size: 1.1rem; letter-spacing: 0.14em; }}
  .brand-sub {{ font-size: 0.75rem; }}
  .cite-chip {{ font-size: 0.68rem; padding: 0 4px; }}
  .section-label {{ margin: 13px 0 7px; }}
  [data-testid="stMarkdownContainer"] table {{ display: block; overflow-x: auto; }}
}}
</style>
"""
