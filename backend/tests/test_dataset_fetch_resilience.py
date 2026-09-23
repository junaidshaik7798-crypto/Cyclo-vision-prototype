"""Regression tests for the dataset-fetch failures seen in the dashboard.

Three independent problems used to surface as "could not fetch the datasets":

* ``/api/reference-dataset/summary`` answered 404 for older (cached) client
  bundles,
* a single failed IBTrACS refresh threw away a perfectly readable on-disk
  cache and silently degraded the results to the tiny bundled table,
* a dropped NOAA connection was never retried and a partial ``*.part`` file
  was left behind, ready to be mistaken for a cache.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import types

import pytest
from fastapi.testclient import TestClient

import app.services.ibtracs as ib

HEADER = (
    "SID,SEASON,NUMBER,BASIN,SUBBASIN,NAME,ISO_TIME,NATURE,LAT,LON,"
    "USA_WIND,USA_PRES\n"
)


class _FakeResponse:
    """Minimal stand-in for ``requests.Response`` (context-manager API)."""

    def __init__(self, payload: bytes = b"", fail: bool = False) -> None:
        self._payload = payload
        self._fail = fail
        self.headers: dict[str, str] = {}

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False

    def raise_for_status(self) -> None:
        if self._fail:
            raise ConnectionError("simulated NOAA connection reset")

    def iter_content(self, chunk_size: int = 1 << 20):
        yield self._payload


def _offline(force: bool = False):
    """Stand-in for ``download_dataset`` when the network is unreachable."""
    raise ConnectionError("offline")


def test_stale_ca_bundle_env_is_ignored(monkeypatch, tmp_path):
    """A CURL_CA_BUNDLE pointing at a missing file must not break HTTPS."""
    from app.core import net

    monkeypatch.setenv("CURL_CA_BUNDLE", str(tmp_path / "gone" / "ca-bundle.crt"))
    monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)
    net.ca_bundle.cache_clear()
    try:
        resolved = net.ca_bundle()
    finally:
        net.ca_bundle.cache_clear()

    assert resolved is not str(tmp_path / "gone" / "ca-bundle.crt")
    assert resolved is True or os.path.isfile(str(resolved))


def test_existing_ca_bundle_env_is_honoured(monkeypatch, tmp_path):
    """An explicitly configured bundle that exists must be used as-is."""
    from app.core import net

    bundle = tmp_path / "corp-ca.pem"
    bundle.write_text("-----BEGIN CERTIFICATE-----\n", encoding="utf-8")
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(bundle))
    net.ca_bundle.cache_clear()
    try:
        resolved = net.ca_bundle()
    finally:
        net.ca_bundle.cache_clear()

    assert resolved == str(bundle)


@pytest.mark.parametrize(
    "origin",
    [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:4173",
        "http://127.0.0.1:5500",
        "null",  # standalone pages opened straight from disk
    ],
)
def test_dataset_fetch_is_cors_allowed_for_local_frontends(
    client: TestClient, origin: str
):
    """Every documented way of opening the UI must be able to fetch datasets.

    VS Code Live Server (5500) and the on-disk standalone pages send an origin
    the API used to reject, so the browser blocked the request and the UI
    reported it as a failed dataset fetch.
    """
    resp = client.get("/api/ibtracs/status", headers={"Origin": origin})
    assert resp.status_code == 200, resp.text
    allowed = resp.headers.get("access-control-allow-origin")
    assert allowed in (origin, "*"), f"{origin} was not echoed back: {resp.headers}"


def test_reference_summary_alias_matches_full_payload(client: TestClient):
    """The compatibility alias must serve exactly the same summary."""
    full = client.get("/api/reference-dataset")
    summary = client.get("/api/reference-dataset/summary")
    assert full.status_code == 200, full.text
    assert summary.status_code == 200, summary.text

    expected = dict(full.json()["summary"])
    actual = dict(summary.json())
    # ``updated`` is regenerated per call, so it cannot be compared verbatim.
    assert expected.pop("updated")
    actual.pop("updated", None)
    assert actual == expected
    assert actual["total_events"] > 0


def test_download_retries_then_writes_cache_atomically(monkeypatch, tmp_path):
    """Two dropped connections must be retried, and no ``.part`` file survive."""
    csv = HEADER + "SID-1,2020,1,NI,BB,FAKE,2020-05-01 00:00:00,TS,10.0,85.0,50,990\n"
    calls = {"n": 0}

    def fake_get(url, **kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            return _FakeResponse(fail=True)
        return _FakeResponse(payload=csv.encode())

    monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace(get=fake_get))
    monkeypatch.setattr(ib, "time", types.SimpleNamespace(sleep=lambda _s: None))
    monkeypatch.setattr(ib, "MIN_VALID_BYTES", 10)
    cache = tmp_path / "ibtracs.NI.list.v04r01.csv"
    monkeypatch.setattr(ib, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(ib, "CACHE_FILE", cache)
    monkeypatch.setattr(ib, "_source", "cold")
    monkeypatch.setattr(ib, "_last_error", None)

    assert ib.download_dataset(force=True) == cache
    assert calls["n"] == 3, "the download was not retried"
    assert cache.read_bytes() == csv.encode()
    assert ib._source == "live"
    assert not list(tmp_path.glob("*.part")), "a partial file was left behind"


def test_html_error_page_is_never_promoted_to_cache(monkeypatch, tmp_path):
    """A captive-portal/HTML body must be rejected, not cached as the dataset."""
    html = b"<html><body>403 Forbidden</body></html>" * 20
    monkeypatch.setitem(
        sys.modules,
        "requests",
        types.SimpleNamespace(get=lambda *a, **k: _FakeResponse(payload=html)),
    )
    monkeypatch.setattr(ib, "time", types.SimpleNamespace(sleep=lambda _s: None))
    monkeypatch.setattr(ib, "DOWNLOAD_ATTEMPTS", 2)
    cache = tmp_path / "ibtracs.NI.list.v04r01.csv"
    monkeypatch.setattr(ib, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(ib, "CACHE_FILE", cache)

    with pytest.raises(IOError):
        ib.download_dataset(force=True)
    assert not cache.exists(), "an HTML error page was cached as the dataset"
    assert not list(tmp_path.glob("*.part"))


def test_stale_cache_is_preferred_over_bundled_table(monkeypatch, tmp_path):
    """A failed refresh must keep serving the full cached record (accuracy)."""
    cache = tmp_path / "ibtracs.NI.list.v04r01.csv"
    cache.write_bytes(b"cached")
    storms = [object()]

    monkeypatch.setattr(ib, "CACHE_FILE", cache)
    monkeypatch.setattr(ib, "download_dataset", _offline)
    monkeypatch.setattr(ib, "_parse", lambda path: list(storms))
    monkeypatch.setattr(ib, "_storms", None)
    monkeypatch.setattr(ib, "_source", "cold")
    monkeypatch.setattr(ib, "_last_error", None)

    loaded = ib.load_storms(force=True)

    assert loaded == storms
    assert ib._source == "cache"
    assert "ConnectionError" in (ib._last_error or "")


def test_missing_cache_falls_back_to_bundled_table(monkeypatch, tmp_path):
    """Only a missing/unusable cache may degrade to the bundled table."""
    monkeypatch.setattr(ib, "CACHE_FILE", tmp_path / "does-not-exist.csv")
    monkeypatch.setattr(ib, "download_dataset", _offline)
    monkeypatch.setattr(ib, "_storms", None)
    monkeypatch.setattr(ib, "_source", "cold")
    monkeypatch.setattr(ib, "_last_error", None)

    loaded = ib.load_storms(force=True)

    assert loaded, "bundled fallback returned nothing"
    assert ib._source == "bundled"


# ---------------------------------------------------------------------------
# Regression: the dashboard must never wait for the NOAA download
# ---------------------------------------------------------------------------

def test_status_is_instant_while_a_download_is_running(monkeypatch, tmp_path):
    """The dataset panels used to time out (and ask for a retry) because
    /api/ibtracs/status waited for the whole NOAA download + parse.

    It must answer immediately -- with the bundled anchors and an explicit
    ``state`` -- while the refresh continues in the background.
    """
    release = threading.Event()
    started = threading.Event()

    def slow_download(force: bool = False):
        started.set()
        release.wait(15)
        raise ConnectionError("simulated slow NOAA response")

    monkeypatch.setattr(ib, "CACHE_FILE", tmp_path / "missing.csv")
    monkeypatch.setattr(ib, "download_dataset", slow_download)
    monkeypatch.setattr(ib, "_storms", None)
    monkeypatch.setattr(ib, "_source", "cold")
    monkeypatch.setattr(ib, "_last_error", None)
    monkeypatch.setattr(ib, "_loading", False)

    ib.warm_async()
    assert started.wait(5), "the background refresh never started"

    began = time.perf_counter()
    status = ib.get_status()
    elapsed = time.perf_counter() - began

    assert elapsed < 1.0, f"status blocked for {elapsed:.2f}s on the download"
    assert status["records"] > 0, "no provisional dataset was served"
    assert status["state"] in {"warming", "degraded"}
    assert status["loading"] is True

    release.set()


def test_local_cache_is_served_without_waiting_for_the_download(monkeypatch, tmp_path):
    """A stale cache must be parsed and served immediately (accuracy + speed).

    A day-old best-track file is still the real observed record, so the UI
    shows it right away while the refresh runs in the background.
    """
    csv = tmp_path / "ibtracs.NI.list.v04r01.csv"
    csv.write_text(
        HEADER
        + "SID,SEASON,NUMBER,BASIN,SUBBASIN,NAME,ISO_TIME,NATURE,LAT,LON,USA_WIND,USA_PRES\n"
        + "SID-CACHED,2019,1,NI,BB,FANI,2019-05-01 00:00:00,TS,10.0,85.0,130,915\n",
        encoding="utf-8",
    )
    started = threading.Event()
    release = threading.Event()

    def slow_download(force: bool = False):
        started.set()
        release.wait(15)
        return csv

    monkeypatch.setattr(ib, "CACHE_FILE", csv)
    monkeypatch.setattr(ib, "download_dataset", slow_download)
    monkeypatch.setattr(ib, "_storms", None)
    monkeypatch.setattr(ib, "_source", "cold")
    monkeypatch.setattr(ib, "_loading", False)

    began = time.perf_counter()
    snapshot = ib.dataset_snapshot()
    elapsed = time.perf_counter() - began

    assert elapsed < 10.0, "reading the local cache must be a local operation"
    assert [s.name for s in snapshot.storms] == ["FANI"]
    assert snapshot.source == "cache"
    assert snapshot.ready is True, "a readable cache must not be reported as warming"

    release.set()


def _storm(name: str) -> "ib.IBTrACSStorm":
    """Minimal IBTrACSStorm used by the sidecar tests."""
    return ib.IBTrACSStorm(
        sid=f"SID-{name}",
        name=name,
        year=2020,
        basin="NI",
        category="Severe Cyclonic Storm",
        max_wind_knots=70.0,
        min_pressure_hpa=980.0,
        genesis_lat=10.0,
        genesis_lon=85.0,
        landfall_lat=15.0,
        landfall_lon=83.0,
        track_direction_deg=315.0,
        forward_speed_knots=8.0,
        intensity_index=4,
        notes="test",
    )


def test_sidecar_is_preferred_over_reparsing_the_csv(monkeypatch, tmp_path):
    """The JSON sidecar makes a restart instant; it must be used when valid."""
    csv = tmp_path / "ibtracs.NI.list.v04r01.csv"
    csv.write_text(
        HEADER
        + "SID,SEASON,NUMBER,BASIN,SUBBASIN,NAME,ISO_TIME,NATURE,LAT,LON,USA_WIND,USA_PRES\n"
        + "SID-SIDE,2021,1,NI,BB,SIDE-CAR,2021-05-01 00:00:00,TS,10.0,85.0,70,980\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(ib, "CACHE_FILE", csv)
    monkeypatch.setattr(ib, "_storms", None)
    monkeypatch.setattr(ib, "_source", "cold")
    monkeypatch.setattr(ib, "_loading", True)  # keep the refresh thread out of the way

    ib.write_sidecar(ib._parse(csv))
    assert ib._sidecar_file().exists()

    def boom(_path):
        raise AssertionError("the CSV was re-parsed although the sidecar was valid")

    monkeypatch.setattr(ib, "_parse", boom)
    snapshot = ib.dataset_snapshot()

    assert snapshot.ready is True
    assert [s.name for s in snapshot.storms] == ["SIDE-CAR"]


def test_replaced_csv_invalidates_the_sidecar(monkeypatch, tmp_path):
    """A re-downloaded CSV must never be shadowed by a stale sidecar."""
    csv = tmp_path / "ibtracs.NI.list.v04r01.csv"
    csv.write_text("first", encoding="utf-8")
    monkeypatch.setattr(ib, "CACHE_FILE", csv)

    ib.write_sidecar([_storm("OLD")])          # sidecar describing "first"
    csv.write_text("second-and-different", encoding="utf-8")

    monkeypatch.setattr(ib, "_storms", None)
    monkeypatch.setattr(ib, "_source", "cold")
    monkeypatch.setattr(ib, "_loading", True)
    monkeypatch.setattr(ib, "_parse", lambda _p: [_storm("NEW")])

    snapshot = ib.dataset_snapshot()
    assert [s.name for s in snapshot.storms] == ["NEW"]



# ---------------------------------------------------------------------------
# Regression: every dataset must be visible, with every attribute
# ---------------------------------------------------------------------------

def test_full_dataset_endpoint_exposes_every_attribute(client: TestClient):
    """The dataset view needs the whole record, not a 15-row teaser."""
    resp = client.get("/api/ibtracs/dataset")
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["total"] == body["summary"]["records"] > 0
    assert body["returned"] == body["total"], "the default must return every storm"
    assert len(body["attributes"]) >= 15
    assert body["scale"], "the IMD scale table is part of the payload"

    fields = {attr["field"] for attr in body["attributes"]}
    assert fields <= set(body["storms"][0]), "a documented field is missing from the rows"
    for key in ("records_by_basin", "records_by_category"):
        assert key in body["summary"]


def test_full_dataset_endpoint_filters_and_sorts(client: TestClient):
    """Filters/sorting must work on the whole record and report match counts."""
    resp = client.get("/api/ibtracs/dataset", params={"min_wind": 100})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["matched"] == body["returned"]
    assert all(s["max_wind_knots"] >= 100 for s in body["storms"])

    page = client.get(
        "/api/ibtracs/dataset", params={"limit": 3, "offset": 0, "sort": "intense"}
    ).json()
    assert page["returned"] == 3
    winds = [s["max_wind_knots"] for s in page["storms"]]
    assert winds == sorted(winds, reverse=True)

    all_storms = client.get("/api/ibtracs/dataset").json()["storms"]
    needle = all_storms[0]["name"]
    found = client.get("/api/ibtracs/dataset", params={"search": needle}).json()
    assert any(s["name"] == needle for s in found["storms"])


def test_overview_endpoint_returns_every_dataset_in_one_call(client: TestClient):
    """One request must hydrate the dashboard, and never fail as a whole."""
    resp = client.get("/api/datasets/overview")
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["samples"], "demo samples missing"
    assert body["sources"], "data sources missing"
    assert body["reference"]["dataset"], "reference dataset missing"
    assert body["ibtracs"]["status"]["records"] > 0
    assert body["ibtracs"]["dataset"]["storms"], "live record missing"
    assert isinstance(body["errors"], dict)
    assert body["generated_at"]

