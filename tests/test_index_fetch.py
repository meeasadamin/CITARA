"""Fetching the published index on a deployment that starts without one.

The archive is untrusted input arriving over the network, so the tests that matter are the
ones where it is wrong: a corrupted download, a substituted asset, an archive that tries to
write outside the data directory.
"""

from __future__ import annotations

import hashlib
import io
import tarfile
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from citara.config import Paths, Settings
from citara.indexing import fetch


def build_archive(members: dict[str, bytes]) -> bytes:
    """A tar.gz holding *members*, keyed by the path each one takes inside the archive."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


INDEX = {
    "chunks.jsonl": b'{"chunk_id": "c1"}\n',
    "index_manifest.json": b'{"chunk_count": 1, "documents": [{"doc_id": "ndrp"}]}',
    "corpus_manifest.json": b'{"documents": []}',
    "bm25/index.json": b"{}",
    "chroma/chroma.sqlite3": b"not really sqlite",
}


@pytest.fixture
def published() -> Iterator[tuple[str, bytes, list[str]]]:
    """Serve one archive over HTTP, recording every request that arrives."""
    body = build_archive(INDEX)
    hits: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            hits.append(self.path)
            payload = SERVED[0]
            self.send_response(200)
            self.send_header("content-type", "application/gzip")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args: object) -> None: ...

    SERVED = [body]
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}/citara-index.tar.gz"
    yield url, body, hits
    server.shutdown()
    server.server_close()


def settings_for(tmp_path: Path, url: str, payload: bytes | None = None) -> Settings:
    checksum = hashlib.sha256(payload).hexdigest() if payload is not None else None
    return Settings(  # type: ignore[call-arg]
        _env_file=None,
        paths=Paths(data_dir=tmp_path / "data"),
        index_url=url,
        index_sha256=checksum,
    )


def test_a_deployment_without_an_index_fetches_one(
    tmp_path: Path, published: tuple[str, bytes, list[str]]
) -> None:
    url, body, hits = published
    settings = settings_for(tmp_path, url, body)

    assert fetch.index_present(settings) is False
    assert fetch.ensure_index(settings) is True

    data = tmp_path / "data"
    assert fetch.index_present(settings) is True
    assert (data / "chunks.jsonl").read_bytes() == INDEX["chunks.jsonl"]
    assert (data / "chroma" / "chroma.sqlite3").exists()
    assert len(hits) == 1


def test_progress_is_reported_while_downloading(
    tmp_path: Path, published: tuple[str, bytes, list[str]]
) -> None:
    url, body, _ = published
    seen: list[tuple[int, int]] = []
    fetch.ensure_index(
        settings_for(tmp_path, url, body), lambda received, total: seen.append((received, total))
    )
    assert seen, "the interface needs progress to show during a 26 MB download"
    assert seen[-1][0] == len(body)


def test_an_index_already_there_is_not_downloaded_again(
    tmp_path: Path, published: tuple[str, bytes, list[str]]
) -> None:
    url, body, hits = published
    settings = settings_for(tmp_path, url, body)
    fetch.ensure_index(settings)

    assert fetch.ensure_index(settings) is False
    assert len(hits) == 1  # the second call never asked for it


def test_a_checksum_mismatch_is_refused_rather_than_searched(
    tmp_path: Path, published: tuple[str, bytes, list[str]]
) -> None:
    """A substituted or corrupted asset must not become the corpus answers are drawn from."""
    url, _, _ = published
    settings = settings_for(tmp_path, url, b"a different archive entirely")

    with pytest.raises(fetch.IndexFetchError, match="checksum"):
        fetch.ensure_index(settings)
    assert fetch.index_present(settings) is False
    assert not (tmp_path / "data" / "chunks.jsonl").exists()


def test_an_archive_cannot_write_outside_the_data_directory(tmp_path: Path) -> None:
    """A path-traversal member would otherwise land anywhere the process can reach."""
    escape = build_archive({**INDEX, "../../escaped.txt": b"owned"})
    hits: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            hits.append(self.path)
            self.send_response(200)
            self.send_header("content-length", str(len(escape)))
            self.end_headers()
            self.wfile.write(escape)

        def log_message(self, *args: object) -> None: ...

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/evil.tar.gz"
    try:
        with pytest.raises(fetch.IndexFetchError, match="unpacked"):
            fetch.ensure_index(settings_for(tmp_path, url, escape))
    finally:
        server.shutdown()
        server.server_close()

    assert not (tmp_path / "escaped.txt").exists()
    assert not (tmp_path.parent / "escaped.txt").exists()


def test_an_unreachable_asset_says_so(tmp_path: Path) -> None:
    settings = settings_for(tmp_path, "http://127.0.0.1:9/nothing.tar.gz", b"")
    with pytest.raises(fetch.IndexFetchError, match="did not complete"):
        fetch.ensure_index(settings)


def test_an_incomplete_archive_leaves_nothing_half_unpacked(tmp_path: Path) -> None:
    partial = build_archive({"chunks.jsonl": b"{}", "bm25/index.json": b"{}"})
    hits: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            hits.append(self.path)
            self.send_response(200)
            self.send_header("content-length", str(len(partial)))
            self.end_headers()
            self.wfile.write(partial)

        def log_message(self, *args: object) -> None: ...

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/partial.tar.gz"
    try:
        with pytest.raises(fetch.IndexFetchError, match="missing"):
            fetch.ensure_index(settings_for(tmp_path, url, partial))
    finally:
        server.shutdown()
        server.server_close()

    assert not (tmp_path / "data" / "chunks.jsonl").exists()


def test_no_configured_source_is_not_an_error(tmp_path: Path) -> None:
    """Locally there is no URL to fetch from; the health check reports a missing index."""
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None, paths=Paths(data_dir=tmp_path / "data"), index_url=""
    )
    assert settings.index_url is None
    assert fetch.ensure_index(settings) is False
