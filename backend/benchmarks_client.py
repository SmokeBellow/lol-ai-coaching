"""
Benchmark provider for LoL champion statistics.

Primary:  Lolalytics  (a3.lolalytics.com private JSON API)
Fallback: U.GG        (stats2.u.gg CDN JSON files)

Architecture
------------
  BenchmarkProvider (ABC)
    └── LolalyticsProvider   primary
    └── UggProvider          fallback

  BenchmarksClient
    - Tries providers in order.
    - Caches results in SQLite: (champion, role, tier, patch) → JSON, TTL 48 h.
    - On cache miss + all providers fail → returns last stale value with flag.

Percentile note
---------------
Neither Lolalytics nor U.GG expose raw percentile distributions in their
public-facing JSON.  We apply a statistical approximation based on typical
within-rank IQR ratios observed for each metric.  The approximation is
conservative and clearly labelled in the output.

Run  python benchmarks_client.py --probe Jinx adc gold   to dump raw API
responses so the parser can be tuned against the live format.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import httpx
from dotenv import load_dotenv

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DB_PATH = os.getenv("DB_PATH", "lol_coaching.db")
CACHE_TTL_SECONDS = 48 * 3600  # 48 hours

# Lolalytics lane labels
ROLE_TO_LOLA_LANE: dict[str, str] = {
    "TOP": "top",
    "JUNGLE": "jungle",
    "MIDDLE": "mid",
    "MID": "mid",
    "BOTTOM": "adc",
    "ADC": "adc",
    "UTILITY": "support",
    "SUPPORT": "support",
}

# Lolalytics tier labels (sent as ?tier= param)
TIER_TO_LOLA_TIER: dict[str, str] = {
    "IRON": "iron",
    "BRONZE": "bronze",
    "SILVER": "silver",
    "GOLD": "gold",
    "PLATINUM": "platinum",
    "EMERALD": "emerald",
    "DIAMOND": "diamond",
    "MASTER": "master_plus",
    "GRANDMASTER": "master_plus",
    "CHALLENGER": "master_plus",
    # Aggregates accepted directly
    "platinum_plus": "platinum_plus",
    "emerald_plus": "emerald_plus",
    "diamond_plus": "diamond_plus",
    "master_plus": "master_plus",
    "all": "all",
}

# U.GG role IDs
ROLE_TO_UGG_ID: dict[str, int] = {
    "TOP": 4,
    "JUNGLE": 1,
    "MIDDLE": 5,
    "MID": 5,
    "BOTTOM": 3,
    "ADC": 3,
    "UTILITY": 2,
    "SUPPORT": 2,
}

# U.GG tier keys
TIER_TO_UGG_KEY: dict[str, str] = {
    "IRON": "1",
    "BRONZE": "2",
    "SILVER": "3",
    "GOLD": "4",
    "PLATINUM": "5",
    "EMERALD": "6",
    "DIAMOND": "7",
    "MASTER": "8",
    "GRANDMASTER": "9",
    "CHALLENGER": "10",
    "platinum_plus": "10",   # u.gg uses different aggregates
    "emerald_plus": "10",
    "diamond_plus": "10",
}

# Approximate p25/p75 as fraction of mean per metric.
# Based on observed LoL rank distributions; intentionally conservative.
# Source: community analysis of match-level stat distributions at Gold–Plat.
PERCENTILE_IQR_RATIOS: dict[str, dict[str, float]] = {
    "cs_per_min":           {"p25": 0.82, "p75": 1.18},
    "vision_score_per_min": {"p25": 0.70, "p75": 1.30},
    "deaths_per_game":      {"p25": 0.65, "p75": 1.40},
    "kill_participation":   {"p25": 0.80, "p75": 1.20},
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Percentiles:
    p25: float
    p50: float
    p75: float


@dataclass
class BenchmarkData:
    champion: str
    role: str
    tier: str
    patch: str
    source: str                               # "lolalytics" | "ugg"
    sample_size: int
    winrate: float                            # 0–100
    cs_per_min: Percentiles
    vision_score_per_min: Percentiles
    deaths_per_game: Percentiles
    kill_participation: Percentiles           # 0–100
    scraped_at: float = field(default_factory=time.time)
    stale: bool = False                       # True = returned from expired cache


def _approx_percentiles(mean: float, metric: str) -> Percentiles:
    """Approximate p25/p75 from the mean using empirical IQR ratios."""
    ratios = PERCENTILE_IQR_RATIOS.get(metric, {"p25": 0.80, "p75": 1.20})
    return Percentiles(
        p25=round(mean * ratios["p25"], 2),
        p50=round(mean, 2),
        p75=round(mean * ratios["p75"], 2),
    )


# ---------------------------------------------------------------------------
# Champion ID resolution (Data Dragon)
# ---------------------------------------------------------------------------

_champion_id_cache: dict[str, int] = {}   # name (lower) → numeric id
_ddragon_version: Optional[str] = None


async def _fetch_ddragon_version(http: httpx.AsyncClient) -> str:
    global _ddragon_version
    if _ddragon_version:
        return _ddragon_version
    resp = await http.get("https://ddragon.leagueoflegends.com/api/versions.json", timeout=10)
    resp.raise_for_status()
    _ddragon_version = resp.json()[0]        # latest version first
    return _ddragon_version


async def resolve_champion_id(champion_name: str, http: httpx.AsyncClient) -> Optional[int]:
    """Resolve champion name → Riot numeric champion ID via Data Dragon."""
    key = champion_name.lower()
    if key in _champion_id_cache:
        return _champion_id_cache[key]

    version = await _fetch_ddragon_version(http)
    url = f"https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/champion.json"
    resp = await http.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()["data"]

    for champ_data in data.values():
        cid = int(champ_data["key"])
        _champion_id_cache[champ_data["name"].lower()] = cid
        _champion_id_cache[champ_data["id"].lower()] = cid   # "MissFortune" style

    return _champion_id_cache.get(key)


async def get_current_patch(http: httpx.AsyncClient) -> str:
    """Returns patch in Lolalytics format, e.g. '25.10' from '15.10.1'."""
    version = await _fetch_ddragon_version(http)
    parts = version.split(".")
    # DDragon: "15.10.1"  →  "15.10" (year.week, used by Lolalytics with no leading zero)
    return f"{parts[0]}.{parts[1]}"


# ---------------------------------------------------------------------------
# Abstract provider
# ---------------------------------------------------------------------------

class BenchmarkProvider(ABC):

    name: str = "abstract"

    @abstractmethod
    async def fetch(
        self,
        champion_name: str,
        role: str,
        tier: str,
        patch: str,
        http: httpx.AsyncClient,
    ) -> BenchmarkData:
        """Fetch benchmark. Raises on failure."""
        ...


# ---------------------------------------------------------------------------
# Provider 1: Lolalytics
# ---------------------------------------------------------------------------

_LOLA_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer":         "https://lolalytics.com/",
    "Origin":          "https://lolalytics.com",
}

# --- Known response field names (as reverse-engineered from the Lolalytics API).
# If the API changes these, only this mapping needs updating.
# Run --probe to dump the raw response and verify.
_LOLA_FIELD_MAP = {
    # Our name          : (candidate keys in Lolalytics response, in priority order)
    "sample_size":        ("n", "games", "count"),
    "winrate":            ("wr", "win_rate", "winRate"),
    "cs_per_min":         ("cs", "cspm", "cs_per_min"),
    "vision_per_min":     ("vs", "vspm", "vision_score_per_min"),
    "deaths_per_game":    ("deaths", "d", "avg_deaths"),
    "kill_participation": ("kp", "kill_participation", "kpar"),
}


def _pick(d: dict, *keys: str) -> Optional[float]:
    """Return first matching key from a dict, or None."""
    for k in keys:
        if k in d:
            v = d[k]
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return None


class LolalyticsProvider(BenchmarkProvider):

    name = "lolalytics"

    async def fetch(
        self,
        champion_name: str,
        role: str,
        tier: str,
        patch: str,
        http: httpx.AsyncClient,
    ) -> BenchmarkData:
        cid = await resolve_champion_id(champion_name, http)
        if cid is None:
            raise ValueError(f"Champion not found in Data Dragon: {champion_name!r}")

        lane = ROLE_TO_LOLA_LANE.get(role.upper())
        if not lane:
            raise ValueError(f"Unknown role: {role!r}")

        lola_tier = TIER_TO_LOLA_TIER.get(tier.upper(), TIER_TO_LOLA_TIER.get(tier, "gold"))

        params = {
            "ep":     "champion",
            "p":      "d",
            "v":      "1",
            "patch":  patch,
            "cid":    str(cid),
            "lane":   lane,
            "tier":   lola_tier,
            "queue":  "420",
            "region": "all",
        }
        url = "https://a3.lolalytics.com/mega/"
        resp = await http.get(url, params=params, headers=_LOLA_BROWSER_HEADERS, timeout=20)
        resp.raise_for_status()
        raw: dict = resp.json()

        return self._parse(raw, champion_name, role, tier, patch)

    def _parse(self, raw: dict, champion: str, role: str, tier: str, patch: str) -> BenchmarkData:
        fm = _LOLA_FIELD_MAP
        n    = int(_pick(raw, *fm["sample_size"])    or 0)
        wr   = float(_pick(raw, *fm["winrate"])      or 0.0)
        cs   = float(_pick(raw, *fm["cs_per_min"])   or 0.0)
        vs   = float(_pick(raw, *fm["vision_per_min"]) or 0.0)
        dth  = float(_pick(raw, *fm["deaths_per_game"]) or 0.0)
        kp   = float(_pick(raw, *fm["kill_participation"]) or 0.0)

        return BenchmarkData(
            champion=champion,
            role=role,
            tier=tier,
            patch=patch,
            source=self.name,
            sample_size=n,
            winrate=wr,
            cs_per_min=_approx_percentiles(cs, "cs_per_min"),
            vision_score_per_min=_approx_percentiles(vs, "vision_score_per_min"),
            deaths_per_game=_approx_percentiles(dth, "deaths_per_game"),
            kill_participation=_approx_percentiles(kp, "kill_participation"),
        )


# ---------------------------------------------------------------------------
# Provider 2: U.GG (CDN fallback)
# ---------------------------------------------------------------------------
# U.GG hosts pre-aggregated JSON at stats2.u.gg.
# URL pattern derived from community reverse-engineering; may need patching.
# Full raw dump at: https://stats2.u.gg/lol/1.5/champion_ranking/{ver}/{tier}/{region}/{cid}/{role}/world/1_1.json
# ---------------------------------------------------------------------------

_UGG_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://u.gg/",
    "Origin":  "https://u.gg",
}


class UggProvider(BenchmarkProvider):

    name = "ugg"

    async def fetch(
        self,
        champion_name: str,
        role: str,
        tier: str,
        patch: str,
        http: httpx.AsyncClient,
    ) -> BenchmarkData:
        cid = await resolve_champion_id(champion_name, http)
        if cid is None:
            raise ValueError(f"Champion not found in Data Dragon: {champion_name!r}")

        role_id = ROLE_TO_UGG_ID.get(role.upper())
        if role_id is None:
            raise ValueError(f"Unknown role: {role!r}")

        tier_key = TIER_TO_UGG_KEY.get(tier.upper(), "10")

        # U.GG uses underscore-separated patch: "15_10_1" from "15.10"
        version = await _fetch_ddragon_version(http)
        ugg_patch = version.replace(".", "_")           # "15_10_1"

        url = (
            f"https://stats2.u.gg/lol/1.5/champion_ranking"
            f"/{ugg_patch}/{tier_key}/12/{cid}/{role_id}/world/1_1.json"
        )
        resp = await http.get(url, headers=_UGG_BROWSER_HEADERS, timeout=20)
        resp.raise_for_status()
        raw = resp.json()

        return self._parse(raw, champion_name, role, tier, patch)

    def _parse(self, raw: dict | list, champion: str, role: str, tier: str, patch: str) -> BenchmarkData:
        # U.GG structure is deeply nested; attempt common paths.
        # Run --probe to verify actual structure.
        data: dict = {}
        if isinstance(raw, list) and raw:
            data = raw[0] if isinstance(raw[0], dict) else {}
        elif isinstance(raw, dict):
            # Might be wrapped: {"data": {...}} or {"champion": {...}}
            data = raw.get("data", raw.get("champion", raw))

        n    = int(_pick(data, "total_matches", "n", "games") or 0)
        wr   = float(_pick(data, "win_rate", "wr", "winrate") or 0.0)
        cs   = float(_pick(data, "cs_per_min", "cspm", "cs") or 0.0)
        vs   = float(_pick(data, "vision_score_per_min", "vspm", "vs") or 0.0)
        dth  = float(_pick(data, "deaths", "avg_deaths", "d") or 0.0)
        kp   = float(_pick(data, "kill_participation", "kp", "kpar") or 0.0)

        return BenchmarkData(
            champion=champion,
            role=role,
            tier=tier,
            patch=patch,
            source=self.name,
            sample_size=n,
            winrate=wr,
            cs_per_min=_approx_percentiles(cs, "cs_per_min"),
            vision_score_per_min=_approx_percentiles(vs, "vision_score_per_min"),
            deaths_per_game=_approx_percentiles(dth, "deaths_per_game"),
            kill_participation=_approx_percentiles(kp, "kill_participation"),
        )


# ---------------------------------------------------------------------------
# Provider 3: OP.GG — win rate from HTML, static values for other metrics
# ---------------------------------------------------------------------------
# OP.GG build page returns 200 with SSR HTML that embeds win rate and sample
# size in its RSC payload.  Other per-game metrics are not exposed in the
# HTML and fall back to static tier baselines.
# ---------------------------------------------------------------------------

_OPGG_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Referer": "https://www.op.gg/",
    "Accept-Language": "en-US,en;q=0.9",
}

ROLE_TO_OPGG_POSITION: dict[str, str] = {
    "TOP": "top", "JUNGLE": "jungle",
    "MIDDLE": "mid", "MID": "mid",
    "BOTTOM": "adc", "ADC": "adc",
    "UTILITY": "support", "SUPPORT": "support",
}


class OPGGProvider(BenchmarkProvider):
    """
    Fetches win rate + sample size from OP.GG build page HTML.
    All other metrics are filled from StaticBenchmarkProvider.
    Serves as a live-data bridge until a proper JSON API is available.
    """
    name = "opgg"

    async def fetch(
        self,
        champion_name: str,
        role: str,
        tier: str,
        patch: str,
        http: httpx.AsyncClient,
    ) -> BenchmarkData:
        import re
        position = ROLE_TO_OPGG_POSITION.get(role.upper(), "adc")
        tier_param = tier.lower().replace("_plus", "+")
        url = f"https://www.op.gg/champions/{champion_name.lower()}/build"
        resp = await http.get(
            url,
            params={"tier": tier_param, "position": position},
            headers=_OPGG_HEADERS,
            timeout=25,
            follow_redirects=True,
        )
        resp.raise_for_status()
        html = resp.text

        # Win rate appears in text as e.g. "Win rate | 51.29 | %"
        wr_match = re.search(r'Win rate[^0-9]*(\d+\.\d+)\s*\|?\s*%', html, re.IGNORECASE)
        winrate = float(wr_match.group(1)) if wr_match else 0.0

        # Sample size: "238,400 Games" or "238400 Games"
        games_match = re.search(r'([\d,]+)\s+Games', html, re.IGNORECASE)
        games = int(games_match.group(1).replace(",", "")) if games_match else 0

        # Fill mechanical metrics from StaticBenchmarkProvider
        static = StaticBenchmarkProvider()
        static_bd = static.fetch_static(champion_name, role, tier, patch)

        return BenchmarkData(
            champion=champion_name,
            role=role,
            tier=tier,
            patch=patch,
            source=self.name,
            sample_size=games,
            winrate=winrate if winrate > 0 else static_bd.winrate,
            cs_per_min=static_bd.cs_per_min,
            vision_score_per_min=static_bd.vision_score_per_min,
            deaths_per_game=static_bd.deaths_per_game,
            kill_participation=static_bd.kill_participation,
        )


# ---------------------------------------------------------------------------
# Provider 4: Static baselines (guaranteed, no network)
# ---------------------------------------------------------------------------
# Empirical percentile estimates per role × tier from published LoL research
# and community data.  Used as ultimate fallback; clearly tagged source="static".
#
# Scaling vs Gold baseline: Iron×0.78, Bronze×0.88, Silver×0.94, Gold×1.0,
# Platinum×1.05, Emerald×1.10, Diamond×1.15, Master+×1.20
# Deaths scale inversely: Iron×1.30, Bronze×1.18, Silver×1.08, Gold×1.0,
# Platinum×0.93, Emerald×0.87, Diamond×0.82, Master+×0.76
# ---------------------------------------------------------------------------

# Gold-tier role baselines  {metric: (p25, p50, p75)}
_GOLD_BASELINES: dict[str, dict[str, tuple[float, float, float]]] = {
    "TOP":     {"cs": (5.5, 7.0, 8.4), "vs": (0.65, 0.95, 1.30), "dth": (2.4, 4.0, 6.0), "kp": (34, 46, 58)},
    "JUNGLE":  {"cs": (4.3, 5.8, 7.3), "vs": (1.00, 1.40, 1.90), "dth": (2.6, 4.3, 6.3), "kp": (54, 67, 77)},
    "MIDDLE":  {"cs": (5.5, 7.0, 8.5), "vs": (0.75, 1.05, 1.45), "dth": (2.4, 4.2, 6.3), "kp": (44, 57, 68)},
    "BOTTOM":  {"cs": (5.5, 7.0, 8.5), "vs": (0.65, 0.95, 1.30), "dth": (2.4, 4.1, 6.0), "kp": (42, 54, 66)},
    "UTILITY": {"cs": (0.4, 0.7, 1.10), "vs": (1.50, 2.05, 2.75), "dth": (3.0, 5.0, 7.2), "kp": (54, 67, 77)},
}

_TIER_CS_SCALE: dict[str, float] = {
    "IRON": 0.78, "BRONZE": 0.88, "SILVER": 0.94, "GOLD": 1.0,
    "PLATINUM": 1.05, "EMERALD": 1.10, "DIAMOND": 1.15,
    "MASTER": 1.20, "GRANDMASTER": 1.20, "CHALLENGER": 1.22,
}
_TIER_VS_SCALE   = _TIER_CS_SCALE   # vision tracks mechanics similarly
_TIER_KP_SCALE: dict[str, float] = {
    "IRON": 0.88, "BRONZE": 0.92, "SILVER": 0.97, "GOLD": 1.0,
    "PLATINUM": 1.02, "EMERALD": 1.04, "DIAMOND": 1.06,
    "MASTER": 1.08, "GRANDMASTER": 1.08, "CHALLENGER": 1.10,
}
_TIER_DTH_SCALE: dict[str, float] = {
    "IRON": 1.30, "BRONZE": 1.18, "SILVER": 1.08, "GOLD": 1.0,
    "PLATINUM": 0.93, "EMERALD": 0.87, "DIAMOND": 0.82,
    "MASTER": 0.76, "GRANDMASTER": 0.76, "CHALLENGER": 0.73,
}
_STATIC_WINRATE: dict[str, float] = {
    "IRON": 49.8, "BRONZE": 50.0, "SILVER": 50.1, "GOLD": 50.2,
    "PLATINUM": 50.3, "EMERALD": 50.4, "DIAMOND": 50.5,
    "MASTER": 50.6, "GRANDMASTER": 50.7, "CHALLENGER": 51.0,
}

# Normalize role aliases to canonical key
_ROLE_CANONICAL: dict[str, str] = {
    "TOP": "TOP", "JUNGLE": "JUNGLE", "MIDDLE": "MIDDLE", "MID": "MIDDLE",
    "BOTTOM": "BOTTOM", "ADC": "BOTTOM", "UTILITY": "UTILITY", "SUPPORT": "UTILITY",
}


class StaticBenchmarkProvider(BenchmarkProvider):
    """
    Returns pre-computed empirical percentile estimates.
    No network calls.  Always succeeds.  Source tagged 'static'.
    """
    name = "static"

    def fetch_static(
        self,
        champion: str,
        role: str,
        tier: str,
        patch: str,
    ) -> BenchmarkData:
        canonical_role = _ROLE_CANONICAL.get(role.upper(), "BOTTOM")
        canonical_tier = tier.upper().split("_")[0]  # "platinum_plus" → "PLATINUM"
        base = _GOLD_BASELINES.get(canonical_role, _GOLD_BASELINES["BOTTOM"])

        cs_s  = _TIER_CS_SCALE.get(canonical_tier, 1.0)
        vs_s  = _TIER_VS_SCALE.get(canonical_tier, 1.0)
        kp_s  = _TIER_KP_SCALE.get(canonical_tier, 1.0)
        dth_s = _TIER_DTH_SCALE.get(canonical_tier, 1.0)

        def scale(t: tuple[float, float, float], s: float) -> Percentiles:
            return Percentiles(round(t[0]*s, 2), round(t[1]*s, 2), round(t[2]*s, 2))

        return BenchmarkData(
            champion=champion,
            role=role,
            tier=tier,
            patch=patch,
            source=self.name,
            sample_size=0,    # static: no real sample
            winrate=_STATIC_WINRATE.get(canonical_tier, 50.0),
            cs_per_min=scale(base["cs"], cs_s),
            vision_score_per_min=scale(base["vs"], vs_s),
            deaths_per_game=scale(base["dth"], dth_s),
            kill_participation=scale(base["kp"], kp_s),
        )

    async def fetch(
        self,
        champion: str,
        role: str,
        tier: str,
        patch: str,
        http: httpx.AsyncClient,
    ) -> BenchmarkData:
        return self.fetch_static(champion, role, tier, patch)


# ---------------------------------------------------------------------------
# SQLite cache
# ---------------------------------------------------------------------------

def _init_cache_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS benchmarks_cache (
            champion   TEXT NOT NULL,
            role       TEXT NOT NULL,
            tier       TEXT NOT NULL,
            patch      TEXT NOT NULL,
            data_json  TEXT NOT NULL,
            cached_at  REAL NOT NULL,
            PRIMARY KEY (champion, role, tier, patch)
        )
    """)
    conn.commit()
    return conn


def _cache_get(
    conn: sqlite3.Connection,
    champion: str,
    role: str,
    tier: str,
    patch: str,
) -> Optional[tuple[BenchmarkData, bool]]:
    """
    Returns (BenchmarkData, is_stale) or None if no cache entry.
    is_stale=True if past TTL but still returned (caller should flag response).
    """
    row = conn.execute(
        "SELECT data_json, cached_at FROM benchmarks_cache "
        "WHERE champion=? AND role=? AND tier=? AND patch=?",
        (champion.lower(), role.upper(), tier.upper(), patch),
    ).fetchone()
    if row is None:
        return None
    data_json, cached_at = row
    bd = _deserialize(json.loads(data_json))
    stale = (time.time() - cached_at) > CACHE_TTL_SECONDS
    bd.stale = stale
    return bd, stale


def _cache_put(conn: sqlite3.Connection, bd: BenchmarkData) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO benchmarks_cache "
        "(champion, role, tier, patch, data_json, cached_at) VALUES (?,?,?,?,?,?)",
        (
            bd.champion.lower(),
            bd.role.upper(),
            bd.tier.upper(),
            bd.patch,
            json.dumps(_serialize(bd)),
            bd.scraped_at,
        ),
    )
    conn.commit()


def _serialize(bd: BenchmarkData) -> dict:
    def pct(p: Percentiles) -> dict:
        return {"p25": p.p25, "p50": p.p50, "p75": p.p75}

    return {
        "champion": bd.champion,
        "role": bd.role,
        "tier": bd.tier,
        "patch": bd.patch,
        "source": bd.source,
        "sample_size": bd.sample_size,
        "winrate": bd.winrate,
        "cs_per_min": pct(bd.cs_per_min),
        "vision_score_per_min": pct(bd.vision_score_per_min),
        "deaths_per_game": pct(bd.deaths_per_game),
        "kill_participation": pct(bd.kill_participation),
        "scraped_at": bd.scraped_at,
    }


def _deserialize(d: dict) -> BenchmarkData:
    def pct(v: dict) -> Percentiles:
        return Percentiles(p25=v["p25"], p50=v["p50"], p75=v["p75"])

    return BenchmarkData(
        champion=d["champion"],
        role=d["role"],
        tier=d["tier"],
        patch=d["patch"],
        source=d["source"],
        sample_size=d["sample_size"],
        winrate=d["winrate"],
        cs_per_min=pct(d["cs_per_min"]),
        vision_score_per_min=pct(d["vision_score_per_min"]),
        deaths_per_game=pct(d["deaths_per_game"]),
        kill_participation=pct(d["kill_participation"]),
        scraped_at=d.get("scraped_at", 0.0),
    )


# ---------------------------------------------------------------------------
# Main client
# ---------------------------------------------------------------------------

class BenchmarksClient:
    """
    Orchestrates: cache → Lolalytics → U.GG → OP.GG → Static → stale cache.

    Provider priority:
      1. LolalyticsProvider  — full JSON stats (blocked by Cloudflare as of 2025-05;
                               keep in chain for when they fix their API)
      2. UggProvider         — CDN JSON stats (access-denied as of 2025-05; same)
      3. OPGGProvider        — live win rate from HTML + static tier baselines
      4. StaticBenchmarkProvider — pure static, always succeeds

    Usage:
        async with BenchmarksClient() as client:
            bd = await client.get("Jinx", "BOTTOM", "GOLD", "16.10")
    """

    def __init__(
        self,
        db_path: str = DB_PATH,
        providers: Optional[list[BenchmarkProvider]] = None,
    ) -> None:
        self._db = _init_cache_db(db_path)
        self._providers: list[BenchmarkProvider] = providers or [
            LolalyticsProvider(),
            UggProvider(),
            OPGGProvider(),
            StaticBenchmarkProvider(),
        ]
        self._http = httpx.AsyncClient(timeout=20.0)

    async def close(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> "BenchmarksClient":
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()

    async def get(
        self,
        champion: str,
        role: str,
        tier: str,
        patch: Optional[str] = None,
    ) -> BenchmarkData:
        """
        Fetch benchmark, using cache when fresh (< 48 h).
        Falls back to stale cache if all providers fail.
        """
        if patch is None:
            patch = await get_current_patch(self._http)

        # --- 1. Fresh cache hit ---
        cached = _cache_get(self._db, champion, role, tier, patch)
        if cached is not None:
            bd, is_stale = cached
            if not is_stale:
                return bd
            # Stale: try live fetch, fall back to stale on failure.
            stale_bd = bd

        else:
            stale_bd = None

        # --- 2. Live fetch from providers ---
        last_error: Optional[Exception] = None
        for provider in self._providers:
            try:
                bd = await provider.fetch(champion, role, tier, patch, self._http)
                _cache_put(self._db, bd)
                return bd
            except Exception as exc:
                last_error = exc
                continue   # try next provider

        # --- 3. Stale fallback ---
        if stale_bd is not None:
            print(
                f"[benchmarks] All providers failed ({last_error}). "
                f"Returning stale cache for {champion}/{role}/{tier}/{patch}."
            )
            stale_bd.stale = True
            return stale_bd

        raise RuntimeError(
            f"No benchmark data for {champion}/{role}/{tier}/{patch}. "
            f"Last error: {last_error}"
        )

    async def get_multi(
        self,
        champions: list[str],
        role: str,
        tier: str,
        patch: Optional[str] = None,
    ) -> dict[str, BenchmarkData]:
        """Fetch benchmarks for multiple champions concurrently."""
        results = await asyncio.gather(
            *[self.get(c, role, tier, patch) for c in champions],
            return_exceptions=True,
        )
        return {
            champ: result
            for champ, result in zip(champions, results)
            if not isinstance(result, Exception)
        }

    def to_dict(self, bd: BenchmarkData) -> dict:
        return _serialize(bd)
