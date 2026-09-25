"""Visual register of the interface (features 60, 61).

Institutional rather than consumer: deep teal with a copper accent, restrained type, hairline
rules, and no decoration that does not carry meaning. Colour says one of four things - the
brand, a citation, a figure worth checking, or the state of an answer - and nothing else is
coloured.

The styling follows the document structure in ``markup`` rather than replacing it: headings
are styled to read as section labels, ``<cite>`` loses its italics, ``<mark>`` loses the
highlighter yellow, and ``<details>`` becomes the source panel. Nothing here invents a
component that the markup does not already name.

Two accessibility rules are kept honestly. Every text colour clears WCAG AA against the
surface it sits on, which is why copper is #8F4E13 rather than the brighter #B4651E that
measured 4.36:1 on white. And motion is dropped entirely for readers who ask for that.

Phone width is designed for, not patched: the demo happens on a phone in an interview room,
so chips stay legible without hover, the header shrinks, and the page never scrolls sideways.
"""

from __future__ import annotations

TEAL = "#0E3B3E"
TEAL_DEEP = "#092A2C"
TEAL_MID = "#14666B"
TEAL_TINT = "#E6F0F0"
COPPER = "#8F4E13"
COPPER_TINT = "#FBF0E5"
PALE = "#F5F7F7"
BORDER = "#D3DBDB"
INK = "#1A1F1F"
MUTED = "#55605F"
RED = "#8F1D21"
RED_TINT = "#FBEDED"

# The page fills the window rather than floating in the middle of it. Only the measure of
# running text is capped, at the width where a line stops being comfortable to read, and the
# panels around it - brand bar, rules, source list - span the full column.
CONTENT_WIDTH = "1680px"
READING_WIDTH = "98ch"
SIDEBAR_WIDTH = "312px"

CSS = f"""
<style>
[data-testid="stMainBlockContainer"] {{
  padding: 0.9rem 2.5rem 1.2rem; max-width: {CONTENT_WIDTH}; margin: 0 auto;
}}
/* Streamlit reserves a 60px strip for its toolbar even when the toolbar is empty, and
   holds it there with a min-height that a height alone does not beat. Shortened, so the
   brand bar starts at the top of the page rather than a thumb's width below it. */
[data-testid="stHeader"] {{
  height: 2.2rem; min-height: 2.2rem; background: transparent;
}}
/* The composer lives in Streamlit's own bottom container, which indents itself 80px on
   each side and leaves 56px of nothing underneath: the box floats narrower than the column
   it belongs to, above a band of empty page. Padded to match the column instead, so the
   two line up and the opening view fits a 768px-tall laptop without scrolling - which is
   what keeps the brand bar on screen, since Streamlit scrolls a taller page to its end. */
[data-testid="stBottomBlockContainer"] {{
  padding: 0.55rem 2.5rem 1rem; max-width: {CONTENT_WIDTH}; margin: 0 auto;
}}
/* Streamlit spaces every element 1rem apart; over a dozen stacked rows that is a screenful. */
[data-testid="stMainBlockContainer"] [data-testid="stVerticalBlock"] {{ gap: 0.75rem; }}
.answer-body, .citara-intro, .citara-notice p {{ max-width: {READING_WIDTH}; }}
html, body, [data-testid="stAppViewContainer"] {{ color: {INK}; }}

/* Leftovers from the previous run fade out entirely rather than sitting at a third opacity
   under an answer being written, after the same half-second delay Streamlit uses. */
[data-testid="stMainBlockContainer"] [data-testid="stElementContainer"][data-stale="true"] {{
  opacity: 0 !important; transition: opacity 0.3s ease-in 0.5s;
}}

:focus-visible {{ outline: 2px solid {TEAL_MID}; outline-offset: 2px; border-radius: 2px; }}
.sr-only {{
  position: absolute; width: 1px; height: 1px; margin: -1px; padding: 0;
  overflow: hidden; clip-path: inset(50%); white-space: nowrap; border: 0;
}}
.skip-link {{
  position: absolute; left: -9999px; top: 0; z-index: 1000; background: {TEAL};
  color: #FFFFFF; padding: 8px 14px; border-radius: 0 0 3px 0; font-size: 0.85rem;
}}
.skip-link:focus {{ left: 0; }}

/* -- brand bar ---------------------------------------------------------------------- */
.citara-header {{
  display: flex; align-items: center; gap: 14px;
  background: linear-gradient(135deg, {TEAL} 0%, {TEAL_DEEP} 100%);
  color: #FFFFFF; padding: 15px 22px 14px; border-radius: 4px;
}}
.brand-mark {{ display: flex; }}
.brand-text {{ display: flex; flex-direction: column; }}
[data-testid="stMarkdownContainer"] .citara-header .brand-name,
.citara-header .brand-name {{
  font-size: 1.5rem; font-weight: 700; letter-spacing: 0.2em; line-height: 1.1;
  color: #FFFFFF; margin: 0; padding: 0; scroll-margin: 0;
}}
.brand-sub {{ font-size: 0.88rem; opacity: 0.92; margin: 4px 0 0; }}
.citara-disclaimer {{
  font-size: 0.78rem; color: {MUTED}; background: {PALE};
  border-left: 3px solid {COPPER}; padding: 7px 12px; margin: 8px 0 12px;
}}
.citara-intro {{ font-size: 0.98rem; color: {INK}; margin: 2px 0 8px; }}
.citara-scope {{ font-size: 0.82rem; color: {MUTED}; margin: 0 0 8px; }}
.citara-searching {{ font-size: 0.95rem; color: {MUTED}; font-style: italic; }}

/* -- section headings ---------------------------------------------------------------- */
[data-testid="stMarkdownContainer"] .section-label,
.section-label {{
  display: block; font-size: 0.82rem; font-weight: 700; letter-spacing: 0.12em;
  text-transform: uppercase; color: {TEAL_MID}; border-bottom: 2px solid {BORDER};
  padding: 0 0 5px; margin: 22px 0 10px; line-height: 1.4; scroll-margin: 0;
}}
.section-label .count {{
  float: right; font-weight: 600; color: {MUTED}; letter-spacing: 0.04em;
  font-size: 0.72rem; text-transform: uppercase; padding-top: 2px;
}}
/* Each part of an answer gets its own colour, so the eye can find the one it wants. */
.label-answer {{ color: {TEAL}; border-bottom-color: {TEAL}; }}
.label-evidence {{ color: {TEAL_MID}; }}
.label-sources {{ color: {COPPER}; border-bottom-color: {COPPER_TINT}; }}
.label-notice {{ color: {COPPER}; border-bottom-color: {COPPER}; }}
/* Streamlit hangs an anchor link off every heading; these headings are labels, not targets. */
.section-label a, .question a {{ display: none; }}

/* -- one exchange --------------------------------------------------------------------- */
.turn {{ padding: 0 0 10px; }}
.turn + .turn, .stMarkdown + .stMarkdown .turn {{ border-top: 1px solid {BORDER}; }}
[data-testid="stMarkdownContainer"] .turn .question,
.turn .question {{
  font-size: 1.38rem; font-weight: 700; color: {TEAL}; margin: 0 0 4px; padding: 0;
  line-height: 1.3; letter-spacing: -0.01em; scroll-margin: 0;
}}
.turn .section-label:first-of-type {{ margin-top: 6px; }}
.answer-body {{ font-size: 1.02rem; line-height: 1.6; }}
.answer-body p {{ margin: 0 0 0.7rem; }}
.answer-body ul, .answer-body ol {{ margin: 0 0 0.7rem; padding-left: 1.3rem; }}
.answer-body li {{ margin-bottom: 0.35rem; }}

/* -- citations and figures ------------------------------------------------------------ */
.cite-chip {{
  display: inline-block; font-style: normal; font-size: 0.74rem; font-weight: 700;
  line-height: 1.35; padding: 1px 6px; margin: 0 2px; border: 1px solid {TEAL_MID};
  border-radius: 3px; color: {TEAL_MID}; background: {TEAL_TINT}; white-space: nowrap;
  vertical-align: 1px; font-variant-numeric: tabular-nums;
}}
.cite-invalid {{ border-color: {RED}; color: {RED}; background: {RED_TINT}; }}
.cite-tail {{ white-space: nowrap; }}
mark.figure {{
  background: {COPPER_TINT}; color: {INK}; border-bottom: 2px solid {COPPER};
  padding: 0 3px; font-weight: 600; font-variant-numeric: tabular-nums;
}}

/* -- answer state ---------------------------------------------------------------------- */
.citara-meta {{
  display: flex; flex-wrap: wrap; gap: 8px 14px; align-items: center;
  font-size: 0.82rem; color: {MUTED}; margin: 0;
}}
.band {{
  font-weight: 700; font-size: 0.74rem; letter-spacing: 0.1em; padding: 3px 10px;
  border-radius: 3px; border: 1px solid {TEAL}; text-transform: uppercase;
}}
.band-High {{ background: {TEAL}; color: #FFFFFF; }}
.band-Moderate {{ background: {TEAL_TINT}; color: {TEAL}; border-color: {TEAL_MID}; }}
.band-Low {{ background: {COPPER_TINT}; color: {COPPER}; border-color: {COPPER}; }}

.citara-notice {{
  border-left: 4px solid {TEAL}; background: {PALE}; padding: 12px 14px; font-size: 0.98rem;
}}
.citara-notice strong {{
  display: block; margin-bottom: 4px; color: {TEAL}; font-size: 1.04rem;
}}
.citara-notice p {{ margin: 0; }}
.notice-degraded {{ border-color: {COPPER}; background: {COPPER_TINT}; }}
.notice-degraded strong {{ color: {COPPER}; }}
.notice-blocked {{ border-color: {RED}; background: {RED_TINT}; }}
.notice-blocked strong {{ color: {RED}; }}
.notice-scope, .notice-limit {{ border-color: {MUTED}; background: #F4F6F6; }}
.notice-scope strong, .notice-limit strong {{ color: {INK}; }}

/* -- sources --------------------------------------------------------------------------- */
.sources-panel summary {{
  cursor: pointer; font-size: 0.92rem; color: {TEAL_MID}; font-weight: 600;
  padding: 8px 12px; border: 1px solid {BORDER}; border-radius: 4px; background: {PALE};
  list-style-position: inside;
}}
.sources-panel summary:hover {{ background: {TEAL_TINT}; }}
.sources-panel[open] summary {{ margin-bottom: 10px; }}
.source-list {{ list-style: none; margin: 0; padding: 0; }}
.source-list > li {{ margin-bottom: 16px; }}
.source-head {{ font-size: 0.88rem; margin: 0 0 3px; }}
.source-head cite {{ font-style: normal; font-weight: 700; color: {TEAL}; }}
.source-head .score, .source-head .state {{ color: {MUTED}; font-size: 0.78rem; }}
/* Streamlit renders blockquotes at 0.6 opacity, which washed the quoted passage out and
   dropped it under AA: the evidence is the point of this panel, not a faded aside. */
[data-testid="stMarkdownContainer"] .passage,
.passage {{
  margin: 0; padding: 8px 0 8px 14px; border-left: 3px solid {TEAL_TINT};
  font-size: 0.92rem; color: {INK}; opacity: 1;
}}
.passage p {{ margin: 0 0 0.4rem; }}
.passage table {{ border-collapse: collapse; font-size: 0.84rem; width: 100%; }}
/* Streamlit colours table cells #76797C, which is 4.37:1 on white and fails AA at this size. */
[data-testid="stMarkdownContainer"] .passage th,
[data-testid="stMarkdownContainer"] .passage td {{
  border: 1px solid {BORDER}; padding: 4px 7px; text-align: left; color: {INK};
}}
[data-testid="stMarkdownContainer"] .passage th {{ background: {PALE}; font-weight: 700; }}

/* -- footer ----------------------------------------------------------------------------- */
.citara-footer {{
  margin-top: 32px; padding-top: 14px; border-top: 2px solid {BORDER};
  font-size: 0.8rem; color: {MUTED};
}}
.citara-footer p {{ margin: 0 0 5px; }}
.citara-footer a {{ color: {TEAL_MID}; }}

/* -- sidebar ----------------------------------------------------------------------------- */
/* Held at one width rather than Streamlit's resizable default, so the mark and the
   headings below it have a column to sit in. */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div {{ width: {SIDEBAR_WIDTH} !important; }}
[data-testid="stSidebar"] {{ background: {PALE}; border-right: 1px solid {BORDER}; }}
.sidebar-brand {{
  display: flex; flex-direction: column; align-items: center; gap: 10px;
  padding: 6px 0 16px; border-bottom: 2px solid {TEAL}; margin-bottom: 6px; text-align: center;
}}
.sidebar-brand-name {{
  font-size: 1.6rem; font-weight: 700; letter-spacing: 0.26em; color: {TEAL};
  text-indent: 0.26em;
}}
[data-testid="stSidebar"] .section-label {{
  font-size: 1.02rem; letter-spacing: 0.1em; color: {TEAL}; border-bottom-color: {TEAL_MID};
  margin: 26px 0 12px; padding-bottom: 6px;
}}
[data-testid="stSidebar"] label {{ font-size: 0.92rem; }}
/* Streamlit dims captions with opacity as well as colour, which composites to #99A09F on
   white - 2.66:1, which axe-core reports and AA does not allow. The palette's muted ink
   says the same thing at full strength and 6.5:1. */
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {{ color: {MUTED}; opacity: 1; }}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{ font-size: 0.86rem; }}
.corpus-summary {{
  cursor: pointer; font-size: 0.84rem; font-weight: 600; color: {TEAL_MID};
  padding: 7px 10px; border: 1px solid {BORDER}; border-radius: 4px; background: #FFFFFF;
  list-style-position: inside;
}}
.corpus-summary:hover {{ background: {TEAL_TINT}; }}
[data-testid="stMarkdownContainer"] ul.corpus-list {{
  font-size: 0.8rem; line-height: 1.4; padding: 10px 0 0; margin: 0; list-style: none;
}}
[data-testid="stMarkdownContainer"] ul.corpus-list li {{
  margin: 0 0 8px 0; padding-left: 0;
}}
.corpus-list .pages {{ color: {MUTED}; }}
.corpus-list .gap {{ color: {COPPER}; }}

/* Starter questions read as a list, so they align left; sidebar buttons stay centred. */
[data-testid="stMainBlockContainer"] .stButton button {{
  justify-content: flex-start; text-align: left; white-space: normal;
  min-height: 0; padding-top: 0.42rem; padding-bottom: 0.42rem;
}}
[data-testid="stMainBlockContainer"] .stButton button > div {{
  justify-content: flex-start; width: 100%;
}}
[data-testid="stMainBlockContainer"] .stButton button p {{ text-align: left; }}

@media (max-width: 640px) {{
  /* The top bar holding the sidebar toggle overlays the page on a phone, so the brand bar
     keeps clear of it - but only just, now that the bar itself is 2.2rem rather than 60px. */
  [data-testid="stMainBlockContainer"] {{
    padding-left: 0.8rem; padding-right: 0.8rem; padding-top: 1.4rem;
  }}
  [data-testid="stBottomBlockContainer"] {{
    padding-left: 0.8rem; padding-right: 0.8rem;
  }}
  .sidebar-brand-name {{ font-size: 1.35rem; }}
  .citara-header {{ padding: 13px 14px 12px; gap: 10px; }}
  [data-testid="stMarkdownContainer"] .citara-header .brand-name,
  .citara-header .brand-name {{ font-size: 1.16rem; letter-spacing: 0.15em; }}
  .brand-sub {{ font-size: 0.78rem; }}
  [data-testid="stMarkdownContainer"] .turn .question,
  .turn .question {{ font-size: 1.16rem; }}
  .answer-body {{ font-size: 0.98rem; }}
  .cite-chip {{ font-size: 0.7rem; padding: 0 4px; }}
  .section-label {{ margin: 16px 0 8px; font-size: 0.76rem; }}
  .passage {{ overflow-x: auto; }}
}}

@media (prefers-reduced-motion: reduce) {{
  *, *::before, *::after {{
    transition-duration: 0.01ms !important; animation-duration: 0.01ms !important;
  }}
}}
</style>
"""
