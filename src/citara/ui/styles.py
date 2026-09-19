"""Visual register of the interface (features 60, 61).

Institutional rather than consumer: NDMA green, restrained type, dense information, and no
decoration that does not carry meaning. Colour is used for three things only - the brand bar,
citation chips, and the state of an answer (evidence band, refusal, degraded).

Phone width is designed for, not patched: the demo happens on a phone in an interview room,
so the chips stay legible without hover, the header shrinks, and the page never scrolls
sideways.
"""

from __future__ import annotations

GREEN = "#01411C"
GREEN_DARK = "#012A12"
GREEN_TINT = "#EEF3EF"
BORDER = "#CFDCD2"
MUTED = "#55645A"
AMBER = "#7A4B00"
AMBER_TINT = "#FDF1DC"
RED = "#9B1C1C"
RED_TINT = "#FDECEC"

CSS = f"""
<style>
[data-testid="stMainBlockContainer"] {{ padding-top: 2.4rem; padding-bottom: 5rem; }}

/* Streamlit fades elements left over from the previous run until the new run finishes. The
   starter buttons would linger at a third opacity under an answer being written, so leftovers
   in the main column fade out entirely instead - after the same half-second delay Streamlit
   uses, so anything re-drawn quickly never flickers. */
[data-testid="stMainBlockContainer"] [data-testid="stElementContainer"][data-stale="true"] {{
  opacity: 0 !important; transition: opacity 0.3s ease-in 0.5s;
}}

.citara-header {{
  background: {GREEN}; color: #FFFFFF; padding: 14px 18px 12px; border-radius: 4px;
  border-bottom: 3px solid {GREEN_DARK};
}}
.citara-title {{ font-size: 1.35rem; font-weight: 700; letter-spacing: 0.08em; }}
.citara-subtitle {{ font-size: 0.88rem; opacity: 0.92; margin-top: 2px; }}
.citara-disclaimer {{
  font-size: 0.78rem; color: {MUTED}; background: {GREEN_TINT};
  border-left: 3px solid {GREEN}; padding: 6px 10px; margin: 8px 0 14px;
}}
.citara-intro {{ font-size: 0.92rem; color: #1A1A1A; margin: 4px 0 10px; }}
.citara-scope {{ font-size: 0.8rem; color: {MUTED}; margin: 0 0 6px; }}
.citara-searching {{ font-size: 0.88rem; color: {MUTED}; font-style: italic; }}

.cite-chip {{
  display: inline-block; font-size: 0.72rem; font-weight: 600; line-height: 1.25;
  padding: 0 6px; margin: 0 2px; border: 1px solid {GREEN}; border-radius: 3px;
  color: {GREEN}; background: {GREEN_TINT}; white-space: nowrap; vertical-align: 1px;
  font-variant-numeric: tabular-nums;
}}
.cite-invalid {{ border-color: {RED}; color: {RED}; background: {RED_TINT}; }}
.cite-tail {{ white-space: nowrap; }}

.citara-meta {{
  display: flex; flex-wrap: wrap; gap: 6px 10px; align-items: center;
  font-size: 0.76rem; color: {MUTED}; margin-top: 6px;
}}
.band {{
  font-weight: 700; font-size: 0.72rem; letter-spacing: 0.02em; padding: 1px 7px;
  border-radius: 3px; border: 1px solid {GREEN};
}}
.band-High {{ background: {GREEN}; color: #FFFFFF; }}
.band-Moderate {{ background: {GREEN_TINT}; color: {GREEN}; }}
.band-Low {{ background: {AMBER_TINT}; color: {AMBER}; border-color: {AMBER}; }}

.citara-notice {{
  border-left: 4px solid {GREEN}; background: {GREEN_TINT}; padding: 8px 12px;
  margin-bottom: 8px; font-size: 0.9rem;
}}
.citara-notice strong {{ display: block; margin-bottom: 2px; }}
.notice-degraded {{ border-color: {AMBER}; background: {AMBER_TINT}; }}
.notice-blocked {{ border-color: {RED}; background: {RED_TINT}; }}
.notice-scope, .notice-limit {{ border-color: {MUTED}; background: #F4F5F4; }}

.source-head {{ font-size: 0.82rem; margin: 8px 0 2px; }}
.source-head .score {{ color: {MUTED}; font-variant-numeric: tabular-nums; }}
.source-head .state {{ color: {MUTED}; font-size: 0.74rem; }}

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
  .citara-header {{ padding: 10px 12px 9px; }}
  .citara-title {{ font-size: 1.12rem; }}
  .citara-subtitle {{ font-size: 0.78rem; }}
  .cite-chip {{ font-size: 0.68rem; padding: 0 4px; }}
  [data-testid="stChatMessage"] {{ padding-left: 0.2rem; padding-right: 0.2rem; }}
  [data-testid="stMarkdownContainer"] table {{ display: block; overflow-x: auto; }}
}}
</style>
"""
