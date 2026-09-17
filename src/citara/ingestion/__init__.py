"""[1] Ingestion: PDF extraction, provenance, normalisation, tables, manifest (features 1-11)."""

from citara.ingestion.loader import ingest_corpus, load_document
from citara.ingestion.models import CorpusManifest, DocumentReport, PageRecord
from citara.ingestion.normalise import has_font_corruption, normalise_text
from citara.ingestion.store import read_manifest, read_records, write_manifest, write_records

__all__ = [
    "CorpusManifest",
    "DocumentReport",
    "PageRecord",
    "has_font_corruption",
    "ingest_corpus",
    "load_document",
    "normalise_text",
    "read_manifest",
    "read_records",
    "write_manifest",
    "write_records",
]
