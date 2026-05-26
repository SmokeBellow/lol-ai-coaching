"""
Тесты для benchmarks_client.py.

Запуск:
  python test_benchmarks_client.py               # unit-тесты (без API)
  python test_benchmarks_client.py --probe       # живой запрос к Lolalytics + U.GG
  python test_benchmarks_client.py --probe Yasuo mid gold   # свой чемп/роль/тир
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
from unittest.mock import AsyncMock, MagicMock, patch

from benchmarks_client import (
    BenchmarkData,
    BenchmarkProvider,
    BenchmarksClient,
    LolalyticsProvider,
    OPGGProvider,
    Percentiles,
    StaticBenchmarkProvider,
    UggProvider,
    _approx_percentiles,
    _cache_get,
    _cache_put,
    _deserialize,
    _init_cache_db,
    _pick,
    _serialize,
    ROLE_TO_LOLA_LANE,
    TIER_TO_LOLA_TIER,
)

PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"
_failures: list[str] = []


def ok(cond: bool, label: str) -> None:
    if cond:
        print(f"{PASS} {label}")
    else:
        print(f"{FAIL} {label}")
        _failures.append(label)


def eq(a, b, label: str) -> None:
    ok(a == b, f"{label}  (got {a!r}, expected {b!r})" if a != b else label)


# ---------------------------------------------------------------------------
# _pick helper
# ---------------------------------------------------------------------------

def test_pick() -> None:
    d = {"wr": 52.1, "n": 1000}
    eq(_pick(d, "win_rate", "wr"), 52.1, "_pick: finds second key")
    eq(_pick(d, "missing"),        None,  "_pick: returns None for missing keys")
    eq(_pick(d, "n"),              1000.0, "_pick: returns numeric as float")


# ---------------------------------------------------------------------------
# Percentile approximation
# ---------------------------------------------------------------------------

def test_approx_percentiles_ordering() -> None:
    p = _approx_percentiles(7.0, "cs_per_min")
    ok(p.p25 < p.p50 < p.p75, "CS percentiles ordered p25 < p50 < p75")
    eq(p.p50, 7.0, "p50 == mean")


def test_approx_percentiles_vision() -> None:
    p = _approx_percentiles(1.2, "vision_score_per_min")
    ok(p.p25 < 1.2 < p.p75, "Vision percentiles bracket mean")


def test_approx_percentiles_unknown_metric() -> None:
    p = _approx_percentiles(5.0, "some_unknown_metric")
    ok(p.p25 < p.p50 < p.p75, "Unknown metric: falls back to default IQR")


# ---------------------------------------------------------------------------
# Role / tier mapping
# ---------------------------------------------------------------------------

def test_role_mapping() -> None:
    cases = [
        ("BOTTOM", "adc"), ("ADC", "adc"),
        ("UTILITY", "support"), ("SUPPORT", "support"),
        ("MIDDLE", "mid"), ("MID", "mid"),
        ("TOP", "top"), ("JUNGLE", "jungle"),
    ]
    for role, expected in cases:
        eq(ROLE_TO_LOLA_LANE.get(role.upper()), expected, f"Role map: {role} -> {expected}")


def test_tier_mapping() -> None:
    cases = [
        ("GOLD", "gold"), ("MASTER", "master_plus"),
        ("GRANDMASTER", "master_plus"), ("CHALLENGER", "master_plus"),
        ("PLATINUM", "platinum"),
    ]
    for tier, expected in cases:
        eq(TIER_TO_LOLA_TIER.get(tier.upper()), expected, f"Tier map: {tier} -> {expected}")


# ---------------------------------------------------------------------------
# Serialization roundtrip
# ---------------------------------------------------------------------------

def _make_benchmark(champion: str = "Jinx", stale: bool = False) -> BenchmarkData:
    return BenchmarkData(
        champion=champion,
        role="BOTTOM",
        tier="GOLD",
        patch="15.10",
        source="lolalytics",
        sample_size=50_000,
        winrate=51.5,
        cs_per_min=Percentiles(5.5, 7.0, 8.3),
        vision_score_per_min=Percentiles(0.8, 1.1, 1.4),
        deaths_per_game=Percentiles(2.5, 4.0, 5.8),
        kill_participation=Percentiles(45.0, 58.0, 68.0),
        scraped_at=time.time(),
        stale=stale,
    )


def test_serialize_roundtrip() -> None:
    bd = _make_benchmark()
    d = _serialize(bd)
    bd2 = _deserialize(d)

    eq(bd2.champion, bd.champion, "Roundtrip: champion")
    eq(bd2.winrate, bd.winrate,   "Roundtrip: winrate")
    eq(bd2.cs_per_min.p50, 7.0,  "Roundtrip: cs_per_min.p50")
    eq(bd2.source, "lolalytics",  "Roundtrip: source")


# ---------------------------------------------------------------------------
# SQLite cache
# ---------------------------------------------------------------------------

def test_cache_miss() -> None:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        conn = _init_cache_db(db_path)
        result = _cache_get(conn, "Jinx", "BOTTOM", "GOLD", "15.10")
        eq(result, None, "Cache miss returns None")
    finally:
        conn.close()
        os.unlink(db_path)


def test_cache_put_and_get_fresh() -> None:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        conn = _init_cache_db(db_path)
        bd = _make_benchmark()
        _cache_put(conn, bd)

        result = _cache_get(conn, "Jinx", "BOTTOM", "GOLD", "15.10")
        ok(result is not None, "Cache hit after put")
        if result:
            bd2, is_stale = result
            eq(bd2.winrate, 51.5, "Cached winrate matches")
            eq(is_stale, False,   "Fresh entry: is_stale=False")
    finally:
        conn.close()
        os.unlink(db_path)


def test_cache_stale_detection() -> None:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        conn = _init_cache_db(db_path)
        bd = _make_benchmark()
        bd.scraped_at = time.time() - (49 * 3600)  # 49 hours ago → stale

        _cache_put(conn, bd)
        result = _cache_get(conn, "Jinx", "BOTTOM", "GOLD", "15.10")
        ok(result is not None, "Stale entry: still returned")
        if result:
            _, is_stale = result
            eq(is_stale, True, "Stale entry: is_stale=True")
    finally:
        conn.close()
        os.unlink(db_path)


def test_cache_key_case_insensitive() -> None:
    """champion stored lowercase, role/tier uppercase — lookups should match."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        conn = _init_cache_db(db_path)
        bd = _make_benchmark("Jinx")
        _cache_put(conn, bd)

        result = _cache_get(conn, "jinx", "bottom", "gold", "15.10")
        ok(result is not None, "Cache: case-insensitive lookup hits correctly")
    finally:
        conn.close()
        os.unlink(db_path)


def test_cache_upsert() -> None:
    """Second put for same key overwrites first."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        conn = _init_cache_db(db_path)
        bd1 = _make_benchmark()
        bd1.winrate = 50.0
        _cache_put(conn, bd1)

        bd2 = _make_benchmark()
        bd2.winrate = 55.0
        _cache_put(conn, bd2)

        result = _cache_get(conn, "Jinx", "BOTTOM", "GOLD", "15.10")
        if result:
            bd_out, _ = result
            eq(bd_out.winrate, 55.0, "Cache upsert: second write overwrites first")
    finally:
        conn.close()
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# LolalyticsProvider._parse (unit test with mock response)
# ---------------------------------------------------------------------------

def test_lolalytics_parse_known_keys() -> None:
    """Tests parser against a plausible Lolalytics response structure."""
    mock_response = {
        "n":      45000,
        "wr":     51.8,
        "cs":     7.2,
        "vs":     1.05,
        "deaths": 4.1,
        "kp":     57.3,
    }
    provider = LolalyticsProvider()
    bd = provider._parse(mock_response, "Jinx", "BOTTOM", "GOLD", "15.10")

    eq(bd.sample_size, 45000,            "Lola parse: sample_size from 'n'")
    eq(bd.winrate, 51.8,                 "Lola parse: winrate from 'wr'")
    eq(bd.cs_per_min.p50, 7.2,           "Lola parse: cs_per_min median from 'cs'")
    eq(bd.vision_score_per_min.p50, 1.05,"Lola parse: vision_score median from 'vs'")
    eq(bd.deaths_per_game.p50, 4.1,      "Lola parse: deaths median from 'deaths'")
    eq(bd.kill_participation.p50, 57.3,  "Lola parse: kp median from 'kp'")
    eq(bd.source, "lolalytics",          "Lola parse: source tag")


def test_lolalytics_parse_missing_keys() -> None:
    """Parser should survive completely empty response without crashing."""
    provider = LolalyticsProvider()
    bd = provider._parse({}, "Jinx", "BOTTOM", "GOLD", "15.10")

    eq(bd.sample_size, 0,  "Lola parse: missing sample_size -> 0")
    eq(bd.winrate, 0.0,    "Lola parse: missing winrate -> 0.0")
    ok(bd.cs_per_min.p50 == 0.0, "Lola parse: missing cs -> p50=0.0")


def test_lolalytics_parse_alt_keys() -> None:
    """Parser should also handle alternative key names."""
    mock_response = {
        "games":         30000,
        "win_rate":      49.5,
        "cspm":          6.8,
        "vspm":          0.95,
        "avg_deaths":    3.8,
        "kill_participation": 55.0,
    }
    provider = LolalyticsProvider()
    bd = provider._parse(mock_response, "Yasuo", "MIDDLE", "GOLD", "15.10")

    eq(bd.sample_size, 30000, "Lola alt-keys: sample_size from 'games'")
    eq(bd.winrate, 49.5,       "Lola alt-keys: winrate from 'win_rate'")
    eq(bd.cs_per_min.p50, 6.8, "Lola alt-keys: cs from 'cspm'")


# ---------------------------------------------------------------------------
# UggProvider._parse
# ---------------------------------------------------------------------------

def test_ugg_parse() -> None:
    mock_response = {
        "total_matches":         22000,
        "win_rate":              52.1,
        "cs_per_min":            7.5,
        "vision_score_per_min":  1.1,
        "deaths":                3.9,
        "kill_participation":    61.0,
    }
    provider = UggProvider()
    bd = provider._parse(mock_response, "Caitlyn", "BOTTOM", "PLATINUM", "15.10")

    eq(bd.sample_size, 22000, "UGG parse: total_matches")
    eq(bd.winrate, 52.1,      "UGG parse: win_rate")
    eq(bd.cs_per_min.p50, 7.5,"UGG parse: cs_per_min")


# ---------------------------------------------------------------------------
# BenchmarksClient integration (mocked providers)
# ---------------------------------------------------------------------------

async def test_client_uses_cache() -> None:
    """Second call for same key must not hit providers."""

    class MockProvider(BenchmarkProvider):
        name = "mock"
        calls = 0

        async def fetch(self, champion, role, tier, patch, http):
            MockProvider.calls += 1
            return _make_benchmark(champion)

    db_fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(db_fd)
    client = BenchmarksClient(db_path=db_path, providers=[MockProvider()])

    import benchmarks_client as bc
    original = bc.get_current_patch
    async def fake_patch(_): return "15.10"
    bc.get_current_patch = fake_patch

    try:
        bd1 = await client.get("Jinx", "BOTTOM", "GOLD", "15.10")
        bd2 = await client.get("Jinx", "BOTTOM", "GOLD", "15.10")
        eq(MockProvider.calls, 1, "Client: provider called once, second call served from cache")
        eq(bd1.winrate, bd2.winrate, "Client: cached result matches original")
    finally:
        bc.get_current_patch = original
        await client.close()
        try:
            os.unlink(db_path)
        except PermissionError:
            pass  # Windows: файл иногда удерживается после close, не критично


async def test_client_fallback_to_second_provider() -> None:
    """If first provider raises, second should be used."""

    class FailProvider(BenchmarkProvider):
        name = "fail"
        async def fetch(self, *args, **kwargs):
            raise ConnectionError("simulated failure")

    class OkProvider(BenchmarkProvider):
        name = "ok"
        async def fetch(self, champion, role, tier, patch, http):
            bd = _make_benchmark(champion)
            bd.source = "ok"
            return bd

    db_fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(db_fd)
    client = BenchmarksClient(db_path=db_path, providers=[FailProvider(), OkProvider()])

    import benchmarks_client as bc
    original = bc.get_current_patch
    async def fake_patch(_): return "15.10"
    bc.get_current_patch = fake_patch

    try:
        bd = await client.get("Jinx", "BOTTOM", "GOLD", "15.10")
        eq(bd.source, "ok", "Fallback: second provider used after first fails")
    finally:
        bc.get_current_patch = original
        await client.close()
        try:
            os.unlink(db_path)
        except PermissionError:
            pass


async def test_client_stale_fallback() -> None:
    """If all providers fail and stale cache exists, return stale data."""

    class AlwaysFailProvider(BenchmarkProvider):
        name = "fail"
        async def fetch(self, *args, **kwargs):
            raise RuntimeError("always fails")

    db_fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(db_fd)

    # Pre-populate stale cache
    conn = _init_cache_db(db_path)
    bd = _make_benchmark()
    bd.scraped_at = time.time() - (50 * 3600)  # 50 hours old = stale
    _cache_put(conn, bd)
    conn.close()

    client = BenchmarksClient(db_path=db_path, providers=[AlwaysFailProvider()])

    import benchmarks_client as bc
    original = bc.get_current_patch
    async def fake_patch(_): return "15.10"
    bc.get_current_patch = fake_patch

    try:
        result = await client.get("Jinx", "BOTTOM", "GOLD", "15.10")
        eq(result.stale, True,   "Stale fallback: stale flag is True")
        eq(result.winrate, 51.5, "Stale fallback: original data preserved")
    finally:
        bc.get_current_patch = original
        await client.close()
        try:
            os.unlink(db_path)
        except PermissionError:
            pass


# ---------------------------------------------------------------------------
# StaticBenchmarkProvider
# ---------------------------------------------------------------------------

def test_static_gold_bottom_baselines() -> None:
    """Gold BOTTOM baselines must match the hardcoded table exactly."""
    provider = StaticBenchmarkProvider()
    bd = provider.fetch_static("Jinx", "BOTTOM", "GOLD", "15.10")

    eq(bd.source, "static",  "Static: source tag")
    eq(bd.winrate, 50.2,     "Static: Gold win rate")
    eq(bd.cs_per_min.p25, 5.5,  "Static: Gold BOTTOM cs p25")
    eq(bd.cs_per_min.p50, 7.0,  "Static: Gold BOTTOM cs p50")
    eq(bd.cs_per_min.p75, 8.5,  "Static: Gold BOTTOM cs p75")
    eq(bd.kill_participation.p50, 54.0, "Static: Gold BOTTOM kp p50")
    eq(bd.sample_size, 0,    "Static: sample_size is 0 (no real data)")


def test_static_tier_scaling() -> None:
    """Diamond stats should be meaningfully different from Gold stats."""
    provider = StaticBenchmarkProvider()
    bd_gold = provider.fetch_static("Jinx", "BOTTOM", "GOLD", "15.10")
    bd_dia  = provider.fetch_static("Jinx", "BOTTOM", "DIAMOND", "15.10")

    ok(bd_dia.cs_per_min.p50 > bd_gold.cs_per_min.p50,
       "Static: Diamond CS > Gold CS")
    ok(bd_dia.deaths_per_game.p50 < bd_gold.deaths_per_game.p50,
       "Static: Diamond deaths < Gold deaths")
    ok(bd_dia.winrate > bd_gold.winrate,
       "Static: Diamond winrate > Gold winrate")


def test_static_iron_scaling() -> None:
    """Iron stats should be lower CS / higher deaths than Gold."""
    provider = StaticBenchmarkProvider()
    bd_gold = provider.fetch_static("Jinx", "BOTTOM", "GOLD", "15.10")
    bd_iron = provider.fetch_static("Jinx", "BOTTOM", "IRON", "15.10")

    ok(bd_iron.cs_per_min.p50 < bd_gold.cs_per_min.p50,
       "Static: Iron CS < Gold CS")
    ok(bd_iron.deaths_per_game.p50 > bd_gold.deaths_per_game.p50,
       "Static: Iron deaths > Gold deaths")


def test_static_role_alias_adc() -> None:
    """ADC and BOTTOM aliases should produce identical baselines."""
    provider = StaticBenchmarkProvider()
    bd_bot = provider.fetch_static("Jinx", "BOTTOM", "GOLD", "15.10")
    bd_adc = provider.fetch_static("Jinx", "ADC",    "GOLD", "15.10")

    eq(bd_bot.cs_per_min.p50, bd_adc.cs_per_min.p50,
       "Static: ADC alias == BOTTOM cs/min")
    eq(bd_bot.kill_participation.p50, bd_adc.kill_participation.p50,
       "Static: ADC alias == BOTTOM kp")


def test_static_role_alias_support() -> None:
    """SUPPORT alias should map to UTILITY baselines."""
    provider = StaticBenchmarkProvider()
    bd_util = provider.fetch_static("Lulu", "UTILITY", "GOLD", "15.10")
    bd_sup  = provider.fetch_static("Lulu", "SUPPORT", "GOLD", "15.10")

    eq(bd_util.vision_score_per_min.p50, bd_sup.vision_score_per_min.p50,
       "Static: SUPPORT alias == UTILITY vision p50")


def test_static_jungle_baselines() -> None:
    """Jungle has lower CS but higher vision and KP than ADC."""
    provider = StaticBenchmarkProvider()
    bd_jg  = provider.fetch_static("Vi", "JUNGLE", "GOLD", "15.10")
    bd_adc = provider.fetch_static("Jinx", "BOTTOM", "GOLD", "15.10")

    ok(bd_jg.cs_per_min.p50 < bd_adc.cs_per_min.p50,
       "Static: Jungle CS < ADC CS")
    ok(bd_jg.vision_score_per_min.p50 > bd_adc.vision_score_per_min.p50,
       "Static: Jungle vision > ADC vision")
    ok(bd_jg.kill_participation.p50 > bd_adc.kill_participation.p50,
       "Static: Jungle KP > ADC KP")


async def test_static_fetch_async() -> None:
    """fetch() (async ABC method) must delegate to fetch_static correctly."""
    provider = StaticBenchmarkProvider()
    # fetch() accepts an http client but doesn't use it (no network)
    bd = await provider.fetch("Jinx", "BOTTOM", "GOLD", "15.10", http=None)  # type: ignore[arg-type]

    eq(bd.source, "static", "Static async fetch: source tag")
    eq(bd.cs_per_min.p50, 7.0, "Static async fetch: cs p50")


# ---------------------------------------------------------------------------
# OPGGProvider
# ---------------------------------------------------------------------------

async def test_opgg_extracts_winrate_and_games() -> None:
    """OPGGProvider should extract win rate and sample size from HTML."""
    html = (
        "<html><body>"
        "Win rate | 52.14 | %"
        "238,400 Games"
        "</body></html>"
    )
    mock_resp = MagicMock()
    mock_resp.text = html
    mock_resp.raise_for_status = MagicMock()

    mock_http = MagicMock()
    mock_http.get = AsyncMock(return_value=mock_resp)

    provider = OPGGProvider()
    bd = await provider.fetch("Jinx", "BOTTOM", "GOLD", "15.10", mock_http)

    eq(bd.source, "opgg",    "OPGG: source tag")
    eq(bd.winrate, 52.14,    "OPGG: win rate extracted from HTML")
    eq(bd.sample_size, 238400, "OPGG: sample size parsed (commas stripped)")
    ok(bd.cs_per_min.p50 > 0,  "OPGG: cs_per_min filled from static provider")
    ok(bd.vision_score_per_min.p50 > 0, "OPGG: vision filled from static provider")


async def test_opgg_fallback_winrate_when_missing() -> None:
    """If HTML has no recognisable win rate, fall back to static winrate."""
    html = "<html><body>No stats here at all.</body></html>"
    mock_resp = MagicMock()
    mock_resp.text = html
    mock_resp.raise_for_status = MagicMock()

    mock_http = MagicMock()
    mock_http.get = AsyncMock(return_value=mock_resp)

    provider = OPGGProvider()
    bd = await provider.fetch("Jinx", "BOTTOM", "GOLD", "15.10", mock_http)

    # winrate should come from StaticBenchmarkProvider (50.2 for Gold)
    eq(bd.winrate, 50.2,  "OPGG fallback: winrate from static when HTML has none")
    eq(bd.sample_size, 0, "OPGG fallback: sample_size=0 when HTML has no Games count")


async def test_opgg_games_without_comma() -> None:
    """Sample size parsing should work even without comma separators."""
    html = "Win rate | 49.88 | %  12345 Games"
    mock_resp = MagicMock()
    mock_resp.text = html
    mock_resp.raise_for_status = MagicMock()

    mock_http = MagicMock()
    mock_http.get = AsyncMock(return_value=mock_resp)

    provider = OPGGProvider()
    bd = await provider.fetch("Jinx", "BOTTOM", "GOLD", "15.10", mock_http)

    eq(bd.sample_size, 12345, "OPGG: sample size without commas")
    eq(bd.winrate, 49.88,     "OPGG: win rate without commas in games count")


async def test_opgg_http_error_propagates() -> None:
    """HTTP errors should propagate so BenchmarksClient can try next provider."""
    import httpx as _httpx

    mock_http = MagicMock()
    mock_http.get = AsyncMock(side_effect=_httpx.HTTPStatusError(
        "403", request=MagicMock(), response=MagicMock()
    ))

    provider = OPGGProvider()
    raised = False
    try:
        await provider.fetch("Jinx", "BOTTOM", "GOLD", "15.10", mock_http)
    except Exception:
        raised = True
    ok(raised, "OPGG: HTTP error propagates as exception")


# ---------------------------------------------------------------------------
# Live probe  (--probe flag)
# ---------------------------------------------------------------------------

async def run_probe(champion: str = "Jinx", role: str = "adc", tier: str = "gold") -> None:
    """
    Fires real HTTP requests to Lolalytics and U.GG.
    Prints raw JSON so the parser can be verified/tuned.
    """
    import httpx as _httpx

    role_map = {
        "adc": "BOTTOM", "bot": "BOTTOM", "bottom": "BOTTOM",
        "mid": "MIDDLE", "middle": "MIDDLE",
        "top": "TOP", "jg": "JUNGLE", "jungle": "JUNGLE",
        "sup": "UTILITY", "support": "UTILITY", "utility": "UTILITY",
    }
    std_role = role_map.get(role.lower(), role.upper())

    print(f"\nProbe: {champion} | role={std_role} | tier={tier}\n")

    http = _httpx.AsyncClient(timeout=20.0)
    from benchmarks_client import (
        get_current_patch, resolve_champion_id,
        ROLE_TO_LOLA_LANE, TIER_TO_LOLA_TIER,
        _LOLA_BROWSER_HEADERS, _UGG_BROWSER_HEADERS,
        _fetch_ddragon_version,
    )

    patch = await get_current_patch(http)
    cid = await resolve_champion_id(champion, http)
    print(f"Patch: {patch}  |  Champion ID: {cid}")

    # --- Lolalytics ---
    lane = ROLE_TO_LOLA_LANE.get(std_role, "adc")
    lola_tier = TIER_TO_LOLA_TIER.get(tier.upper(), tier.lower())
    lola_url = "https://a3.lolalytics.com/mega/"
    lola_params = {
        "ep": "champion", "p": "d", "v": "1",
        "patch": patch, "cid": str(cid),
        "lane": lane, "tier": lola_tier,
        "queue": "420", "region": "all",
    }
    print(f"\n--- Lolalytics ---")
    print(f"URL: {lola_url}?{'&'.join(f'{k}={v}' for k,v in lola_params.items())}")
    try:
        resp = await http.get(lola_url, params=lola_params, headers=_LOLA_BROWSER_HEADERS)
        print(f"Status: {resp.status_code}")
        if resp.status_code == 200:
            raw = resp.json()
            print("Keys:", list(raw.keys()) if isinstance(raw, dict) else type(raw).__name__)
            print("Raw (first 2000 chars):")
            print(json.dumps(raw, indent=2)[:2000])
        else:
            print("Body:", resp.text[:500])
    except Exception as e:
        print(f"Error: {e}")

    # --- U.GG ---
    from benchmarks_client import ROLE_TO_UGG_ID, TIER_TO_UGG_KEY
    role_id = ROLE_TO_UGG_ID.get(std_role, 3)
    tier_key = TIER_TO_UGG_KEY.get(tier.upper(), "10")
    ddragon_ver = await _fetch_ddragon_version(http)
    ugg_patch = ddragon_ver.replace(".", "_")
    ugg_url = (
        f"https://stats2.u.gg/lol/1.5/champion_ranking"
        f"/{ugg_patch}/{tier_key}/12/{cid}/{role_id}/world/1_1.json"
    )
    print(f"\n--- U.GG ---")
    print(f"URL: {ugg_url}")
    try:
        resp = await http.get(ugg_url, headers=_UGG_BROWSER_HEADERS)
        print(f"Status: {resp.status_code}")
        if resp.status_code == 200:
            raw = resp.json()
            t = type(raw).__name__
            print(f"Type: {t}")
            if isinstance(raw, dict):
                print("Keys:", list(raw.keys()))
            elif isinstance(raw, list):
                print(f"List len: {len(raw)}, first item type: {type(raw[0]).__name__ if raw else 'empty'}")
            print("Raw (first 2000 chars):")
            print(json.dumps(raw, indent=2)[:2000])
        else:
            print("Body:", resp.text[:500])
    except Exception as e:
        print(f"Error: {e}")

    await http.aclose()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def main() -> None:
    if "--probe" in sys.argv:
        args = [a for a in sys.argv[1:] if not a.startswith("--")]
        champion = args[0] if len(args) > 0 else "Jinx"
        role     = args[1] if len(args) > 1 else "adc"
        tier     = args[2] if len(args) > 2 else "gold"
        await run_probe(champion, role, tier)
        return

    print("\n=== benchmarks_client.py - unit tests ===\n")

    # Sync tests
    test_pick()
    test_approx_percentiles_ordering()
    test_approx_percentiles_vision()
    test_approx_percentiles_unknown_metric()
    test_role_mapping()
    test_tier_mapping()
    test_serialize_roundtrip()
    test_cache_miss()
    test_cache_put_and_get_fresh()
    test_cache_stale_detection()
    test_cache_key_case_insensitive()
    test_cache_upsert()
    test_lolalytics_parse_known_keys()
    test_lolalytics_parse_missing_keys()
    test_lolalytics_parse_alt_keys()
    test_ugg_parse()

    # StaticBenchmarkProvider
    test_static_gold_bottom_baselines()
    test_static_tier_scaling()
    test_static_iron_scaling()
    test_static_role_alias_adc()
    test_static_role_alias_support()
    test_static_jungle_baselines()

    # Async tests
    await test_static_fetch_async()
    await test_opgg_extracts_winrate_and_games()
    await test_opgg_fallback_winrate_when_missing()
    await test_opgg_games_without_comma()
    await test_opgg_http_error_propagates()
    await test_client_uses_cache()
    await test_client_fallback_to_second_provider()
    await test_client_stale_fallback()

    print()
    if _failures:
        print(f"\033[91mFAILED: {len(_failures)} test(s)\033[0m")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("\033[92mAll tests passed.\033[0m")


if __name__ == "__main__":
    asyncio.run(main())
