"""Visual register of the interface (features 60, 61).

Institutional rather than consumer: navy and steel, restrained type, hairline rules, and no
decoration that does not carry meaning. Colour says one of four things - the brand bar, a
citation, the state of an answer, or a warning - and nothing else is coloured.

The styling follows the document structure in ``markup`` rather than replacing it: headings
are styled to read as small-caps labels, ``<cite>`` loses its italics, ``<mark>`` loses the
highlighter yellow, and ``<details>`` becomes the source panel. Nothing here invents a
component that the markup does not already name.

Two accessibility rules are kept honestly: every text colour clears WCAG AA against the
surface it sits on (the amber gap note failed at 3.4:1 before), and motion is dropped
entirely for readers who ask for that.

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
# 5.5:1 on the sidebar mist; the previous #B4791F was 3.4:1 and failed AA.
AMBER = "#8A5A0B"
AMBER_TINT = "#FDF3E2"
RED = "#8F1D21"
RED_TINT = "#FBEDED"

CSS = f"""
<style>
[data-testid="stMainBlockContainer"] {{ padding-top: 2.4rem; padding-bottom: 4rem; }}
html, body, [data-testid="stAppViewContainer"] {{ color: {INK}; }}

/* Leftovers from the previous run fade out entirely rather than sitting at a third opacity
   under an answer being written, after the same half-second delay Streamlit uses. */
[data-testid="stMainBlockContainer"] [data-testid="stElementContainer"][data-stale="true"] {{
  opacity: 0 !important; transition: opacity 0.3s ease-in 0.5s;
}}

:focus-visible {{ outline: 2px solid {STEEL}; outline-offset: 2px; border-radius: 2px; }}
.skip-link {{
  position: absolute; left: -9999px; top: 0; z-index: 1000; background: {NAVY};
  color: #FFFFFF; padding: 8px 14px; border-radius: 0 0 3px 0; font-size: 0.85rem;
}}
.skip-link:focus {{ left: 0; }}
.sr-only {{
  position: absolute; width: 1px; height: 1px; margin: -1px; padding: 0;
  overflow: hidden; clip-path: inset(50%); white-space: nowrap; border: 0;
}}

/* -- brand bar ---------------------------------------------------------------------- */
.citara-header {{
  display: flex; align-items: center; gap: 12px;
  background: linear-gradient(180deg, {NAVY} 0%, {NAVY_DEEP} 100%);
  color: #FFFFFF; padding: 15px 18px 14px; border-radius: 3px;
}}
.brand-mark {{ display: flex; color: #FFFFFF; opacity: 0.95; }}
.brand-text {{ display: flex; flex-direction: column; }}
/* Streamlit styles h1-h6 inside its markdown container, so the headings that carry this
   page's structure have to out-specify it rather than fight it with !important. */
[data-testid="stMarkdownContainer"] .citara-header .brand-name,
.citara-header .brand-name {{
  font-size: 1.32rem; font-weight: 700; letter-spacing: 0.18em; line-height: 1.15;
  color: #FFFFFF; margin: 0; padding: 0; scroll-margin: 0;
}}
.brand-sub {{ font-size: 0.84rem; opacity: 0.9; margin: 3px 0 0; }}
.citara-disclaimer {{
  font-size: 0.76rem; color: {MUTED}; background: {MIST};
  border-left: 3px solid {STEEL}; padding: 7px 10px; margin: 8px 0 14px;
}}
.citara-intro {{ font-size: 0.94rem; color: {INK}; margin: 2px 0 12px; }}
.citara-scope {{ font-size: 0.8rem; color: {MUTED}; margin: 0 0 6px; }}
.citara-searching {{ font-size: 0.9rem; color: {MUTED}; font-style: italic; }}

/* -- section headings ---------------------------------------------------------------- */
[data-testid="stMarkdownContainer"] .section-label,
.section-label {{
  display: block; font-size: 0.68rem; font-weight: 700; letter-spacing: 0.14em;
  text-transform: uppercase; color: {STEEL}; border-bottom: 1px solid {BORDER};
  padding: 0 0 4px; margin: 16px 0 8px; line-height: 1.4; scroll-margin: 0;
}}
/* Streamlit hangs an anchor link off every heading; these headings are labels, not targets. */
.section-label a, .question a {{ display: none; }}
.section-label .count {{
  float: right; font-weight: 600; color: {MUTED}; letter-spacing: 0.04em;
  font-size: 0.66rem; text-transform: uppercase;
}}
[data-testid="stSidebar"] .section-label {{ margin-top: 20px; }}

/* -- one exchange --------------------------------------------------------------------- */
.turn {{ padding: 0 0 6px; }}
.turn + .turn, .stMarkdown + .stMarkdown .turn {{ border-top: 1px solid {BORDER}; }}
[data-testid="stMarkdownContainer"] .turn .question,
.turn .question {{
  font-size: 1.04rem; font-weight: 600; color: {NAVY}; margin: 0 0 2px; padding: 0;
  line-height: 1.35; letter-spacing: 0; scroll-margin: 0;
}}
.turn .section-label:first-of-type {{ margin-top: 4px; }}
.answer-body p {{ margin: 0 0 0.6rem; }}
.answer-body ul, .answer-body ol {{ margin: 0 0 0.6rem; padding-left: 1.2rem; }}
.answer-body li {{ margin-bottom: 0.25rem; }}

/* -- citations and figures ------------------------------------------------------------ */
.cite-chip {{
  display: inline-block; font-style: normal; font-size: 0.71rem; font-weight: 700;
  line-height: 1.3; padding: 0 5px; margin: 0 2px; border: 1px solid {STEEL};
  border-radius: 2px; color: {STEEL}; background: {STEEL_TINT}; white-space: nowrap;
  vertical-align: 1px; font-variant-numeric: tabular-nums;
}}
.cite-invalid {{ border-color: {RED}; color: {RED}; background: {RED_TINT}; }}
.cite-tail {{ white-space: nowrap; }}
mark.figure {{
  background: {STEEL_TINT}; color: {INK}; border-bottom: 1px solid {STEEL};
  padding: 0 2px; font-weight: 600; font-variant-numeric: tabular-nums;
}}

/* -- answer state ---------------------------------------------------------------------- */
.citara-meta {{
  display: flex; flex-wrap: wrap; gap: 6px 12px; align-items: center;
  font-size: 0.76rem; color: {MUTED}; margin: 0;
}}
.band {{
  font-weight: 700; font-size: 0.7rem; letter-spacing: 0.1em; padding: 2px 8px;
  border-radius: 2px; border: 1px solid {NAVY}; text-transform: uppercase;
}}
.band-High {{ background: {NAVY}; color: #FFFFFF; }}
.band-Moderate {{ background: {STEEL_TINT}; color: {NAVY}; border-color: {STEEL}; }}
.band-Low {{ background: {AMBER_TINT}; color: {AMBER}; border-color: {AMBER}; }}

.citara-notice {{
  border-left: 4px solid {NAVY}; background: {MIST}; padding: 9px 12px; font-size: 0.9rem;
}}
.citara-notice strong {{ display: block; margin-bottom: 3px; color: {NAVY}; }}
.citara-notice p {{ margin: 0; }}
.notice-degraded {{ border-color: {AMBER}; background: {AMBER_TINT}; }}
.notice-degraded strong {{ color: {AMBER}; }}
.notice-blocked {{ border-color: {RED}; background: {RED_TINT}; }}
.notice-blocked strong {{ color: {RED}; }}
.notice-scope, .notice-limit {{ border-color: {MUTED}; background: #F5F6F7; }}
.notice-scope strong, .notice-limit strong {{ color: {INK}; }}

/* -- sources --------------------------------------------------------------------------- */
.sources-panel summary {{
  cursor: pointer; font-size: 0.86rem; color: {STEEL}; font-weight: 600;
  padding: 6px 8px; border: 1px solid {BORDER}; border-radius: 3px; background: {MIST};
  list-style-position: inside;
}}
.sources-panel[open] summary {{ margin-bottom: 8px; }}
.source-list {{ list-style: none; margin: 0; padding: 0; }}
.source-list > li {{ margin-bottom: 14px; }}
.source-head {{ font-size: 0.82rem; margin: 0 0 2px; }}
.source-head cite {{ font-style: normal; font-weight: 700; }}
.source-head .score, .source-head .state {{ color: {MUTED}; font-size: 0.76rem; }}
/* Streamlit renders blockquotes at 0.6 opacity, which washed the quoted passage out and
   dropped it under AA: the evidence is the point of this panel, not a faded aside. */
[data-testid="stMarkdownContainer"] .passage,
.passage {{
  margin: 0; padding: 6px 0 6px 12px; border-left: 2px solid {BORDER};
  font-size: 0.88rem; color: {INK}; opacity: 1;
}}
.passage p {{ margin: 0 0 0.4rem; }}
.passage table {{ border-collapse: collapse; font-size: 0.8rem; width: 100%; }}
/* Streamlit colours table cells #76797C, which is 4.37:1 on white and fails AA at this size. */
[data-testid="stMarkdownContainer"] .passage th,
[data-testid="stMarkdownContainer"] .passage td {{
  border: 1px solid {BORDER}; padding: 3px 6px; text-align: left; color: {INK};
}}
[data-testid="stMarkdownContainer"] .passage th {{ background: {MIST}; font-weight: 700; }}

/* -- footer ----------------------------------------------------------------------------- */
.citara-footer {{
  margin-top: 28px; padding-top: 12px; border-top: 1px solid {BORDER};
  font-size: 0.76rem; color: {MUTED};
}}
.citara-footer p {{ margin: 0 0 4px; }}
.citara-footer a {{ color: {STEEL}; }}

/* -- sidebar ----------------------------------------------------------------------------- */
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
  .citara-header .brand-name {{ font-size: 1.1rem; letter-spacing: 0.14em; }}
  .brand-sub {{ font-size: 0.75rem; }}
  .cite-chip {{ font-size: 0.68rem; padding: 0 4px; }}
  .section-label {{ margin: 13px 0 7px; }}
  .passage {{ overflow-x: auto; }}
}}

@media (prefers-reduced-motion: reduce) {{
  *, *::before, *::after {{
    transition-duration: 0.01ms !important; animation-duration: 0.01ms !important;
  }}
}}
</style>
"""
