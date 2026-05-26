"""
Tests for riot_client.py.

Unit tests run with no API key needed (pure logic + mocks).
Integration test requires RIOT_API_KEY in .env and --integration flag.

Usage:
  python test_riot_client.py                   # unit tests only
  python test_riot_client.py --integration     # + live API call (interactive)
"""

import asyncio
import sys
import time
from unittest.mock import AsyncMock, MagicMock, patch

from riot_client import (
    PLATFORM_TO_REGION,
    REGION_ALIASES,
    PlayerData,
    RiotClient,
    SummonerRank,
    TokenBucket,
)

PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"

_failures: list[str] = []


def assert_eq(a, b, label: str) -> None:
    if a == b:
        print(f"{PASS} {label}")
    else:
        print(f"{FAIL} {label}  (got {a!r}, expected {b!r})")
        _failures.append(label)


def assert_true(cond: bool, label: str) -> None:
    assert_eq(cond, True, label)


# ---------------------------------------------------------------------------
# Token bucket tests
# ---------------------------------------------------------------------------

async def test_token_bucket_full_bucket_immediate() -> None:
    bucket = TokenBucket(rate=100.0, capacity=10.0)
    t0 = time.monotonic()
    await bucket.acquire()
    elapsed = time.monotonic() - t0
    assert_true(elapsed < 0.05, "Full bucket: acquire is instant (<50 ms)")


async def test_token_bucket_empty_bucket_waits() -> None:
    bucket = TokenBucket(rate=2.0, capacity=1.0)
    await bucket.acquire()  # drain the single token
    t0 = time.monotonic()
    await bucket.acquire()  # refill rate is 2/s → ~0.5 s wait
    elapsed = time.monotonic() - t0
    assert_true(0.35 < elapsed < 0.75, f"Empty bucket: waited ~0.5 s (got {elapsed:.3f}s)")


async def test_token_bucket_respects_capacity() -> None:
    """Refill should never exceed capacity."""
    bucket = TokenBucket(rate=100.0, capacity=5.0)
    await asyncio.sleep(0.2)  # would add 20 tokens if uncapped
    # Drain 5 — should all be instant
    t0 = time.monotonic()
    await asyncio.gather(*[bucket.acquire() for _ in range(5)])
    elapsed = time.monotonic() - t0
    assert_true(elapsed < 0.05, "Capacity respected: 5 tokens after sleep is instant")


# ---------------------------------------------------------------------------
# Region resolution tests
# ---------------------------------------------------------------------------

def test_resolve_platform_aliases() -> None:
    cases = [
        ("na", "na1"),
        ("NA", "na1"),
        ("euw", "euw1"),
        ("EUW", "euw1"),
        ("kr", "kr"),
        ("KR", "kr"),
        ("na1", "na1"),
        ("eun1", "eun1"),
    ]
    for inp, expected in cases:
        assert_eq(RiotClient.resolve_platform(inp), expected, f"resolve_platform({inp!r}) == {expected!r}")


def test_platform_to_region_coverage() -> None:
    assert_eq(PLATFORM_TO_REGION["na1"], "americas", "na1 -> americas")
    assert_eq(PLATFORM_TO_REGION["euw1"], "europe", "euw1 -> europe")
    assert_eq(PLATFORM_TO_REGION["kr"], "asia", "kr -> asia")
    assert_eq(PLATFORM_TO_REGION["oc1"], "sea", "oc1 -> sea")


# ---------------------------------------------------------------------------
# SummonerRank numeric_rank ordering
# ---------------------------------------------------------------------------

def test_rank_ordering() -> None:
    iron4 = SummonerRank("IRON", "IV", 0, "RANKED_SOLO_5x5")
    bronze1 = SummonerRank("BRONZE", "I", 99, "RANKED_SOLO_5x5")
    gold2 = SummonerRank("GOLD", "II", 50, "RANKED_SOLO_5x5")
    assert_true(iron4.numeric_rank < bronze1.numeric_rank, "IRON IV < BRONZE I")
    assert_true(bronze1.numeric_rank < gold2.numeric_rank, "BRONZE I < GOLD II")


def test_rank_label() -> None:
    r = SummonerRank("PLATINUM", "III", 75, "RANKED_SOLO_5x5")
    assert_eq(r.label, "PLATINUM III 75 LP", "SummonerRank.label format")


# ---------------------------------------------------------------------------
# Solo death extraction
# ---------------------------------------------------------------------------

def _make_client_stub() -> RiotClient:
    """Create RiotClient without requiring an API key (for unit testing helpers)."""
    obj = RiotClient.__new__(RiotClient)
    return obj


def test_solo_deaths_before_10_basic() -> None:
    puuid = "test-puuid-abc"
    participants = [
        {"participantId": 1, "puuid": puuid},
        {"participantId": 2, "puuid": "enemy-puuid"},
    ]
    timeline = {
        "info": {
            "frames": [
                {
                    "timestamp": 300_000,  # 5 min — inside window
                    "events": [
                        # solo death: no assistingParticipantIds
                        {"type": "CHAMPION_KILL", "victimId": 1, "assistingParticipantIds": []},
                        # assisted death: should NOT count
                        {"type": "CHAMPION_KILL", "victimId": 1, "assistingParticipantIds": [2]},
                    ],
                },
                {
                    "timestamp": 700_000,  # 11.7 min — outside window
                    "events": [
                        # solo death after 10 min: should NOT count
                        {"type": "CHAMPION_KILL", "victimId": 1, "assistingParticipantIds": []},
                    ],
                },
            ]
        }
    }
    client = _make_client_stub()
    result = client._extract_solo_deaths_before_10(timeline, puuid, participants)
    assert_eq(result, 1, "Counts only solo deaths before 10 min")


def test_solo_deaths_before_10_no_deaths() -> None:
    puuid = "test-puuid"
    participants = [{"participantId": 1, "puuid": puuid}]
    timeline = {"info": {"frames": [{"timestamp": 100_000, "events": []}]}}
    client = _make_client_stub()
    result = client._extract_solo_deaths_before_10(timeline, puuid, participants)
    assert_eq(result, 0, "Zero deaths returns 0")


def test_solo_deaths_before_10_missing_puuid() -> None:
    """If player not in participant list, return 0 safely."""
    timeline = {
        "info": {
            "frames": [
                {
                    "timestamp": 60_000,
                    "events": [
                        {"type": "CHAMPION_KILL", "victimId": 99, "assistingParticipantIds": []},
                    ],
                }
            ]
        }
    }
    client = _make_client_stub()
    result = client._extract_solo_deaths_before_10(timeline, "ghost-puuid", [])
    assert_eq(result, 0, "Missing puuid returns 0 safely")


def test_solo_deaths_frame_boundary() -> None:
    """Death at exactly 600_000 ms should NOT count (>= comparison)."""
    puuid = "test-puuid"
    participants = [{"participantId": 3, "puuid": puuid}]
    timeline = {
        "info": {
            "frames": [
                {
                    "timestamp": 600_000,
                    "events": [
                        {"type": "CHAMPION_KILL", "victimId": 3, "assistingParticipantIds": []},
                    ],
                }
            ]
        }
    }
    client = _make_client_stub()
    result = client._extract_solo_deaths_before_10(timeline, puuid, participants)
    assert_eq(result, 0, "Death at frame timestamp 600000 is excluded (boundary)")


# ---------------------------------------------------------------------------
# fetch_player_data integration (mocked HTTP)
# ---------------------------------------------------------------------------

MOCK_SUMMONER = {
    "id": "summoner-id-123",
    "accountId": "account-id-456",
    "puuid": "puuid-789",
    "name": "TestPlayer",
    "profileIconId": 1,
    "revisionDate": 0,
    "summonerLevel": 100,
}

MOCK_RANK = [
    {
        "queueType": "RANKED_SOLO_5x5",
        "tier": "GOLD",
        "rank": "II",
        "leaguePoints": 55,
        "wins": 80,
        "losses": 70,
    }
]

MOCK_MATCH_IDS = ["NA1_1111", "NA1_2222"]

def _make_match(match_id: str, puuid: str, deaths: int = 2, win: bool = True) -> dict:
    return {
        "metadata": {"matchId": match_id, "participants": [puuid]},
        "info": {
            "gameDuration": 1800,
            "participants": [
                {
                    "participantId": 1,
                    "puuid": puuid,
                    "championName": "Jinx",
                    "teamPosition": "BOTTOM",
                    "kills": 5,
                    "deaths": deaths,
                    "assists": 10,
                    "totalMinionsKilled": 180,
                    "neutralMinionsKilled": 10,
                    "visionScore": 30,
                    "totalDamageDealtToChampions": 25000,
                    "win": win,
                }
            ],
        },
    }

MOCK_TIMELINE = {
    "info": {
        "frames": [
            {
                "timestamp": 450_000,
                "events": [
                    {
                        "type": "CHAMPION_KILL",
                        "victimId": 1,
                        "assistingParticipantIds": [],
                    }
                ],
            }
        ]
    }
}


async def test_fetch_player_data_mocked() -> None:
    """Full pipeline test with mocked HTTP — validates control flow and data shaping."""

    async def mock_get(url: str) -> dict | list:
        if "summoners/by-name" in url:
            return MOCK_SUMMONER
        if "entries/by-summoner" in url:
            return MOCK_RANK
        if "matches/by-puuid" in url and "/ids" in url:
            return MOCK_MATCH_IDS
        if url.endswith("/timeline"):
            return MOCK_TIMELINE
        for mid in MOCK_MATCH_IDS:
            if mid in url:
                deaths = 2 if mid == "NA1_1111" else 0
                return _make_match(mid, MOCK_SUMMONER["puuid"], deaths=deaths)
        return {}

    client = RiotClient.__new__(RiotClient)
    client.api_key = "MOCK"
    client._limiter = MagicMock()
    client._limiter.acquire = AsyncMock()
    client._http = MagicMock()
    client._get = mock_get

    data = await client.fetch_player_data("TestPlayer", "na")

    assert_eq(data.summoner_name, "TestPlayer", "Mock: summoner_name set")
    assert_eq(data.puuid, "puuid-789", "Mock: puuid set")
    assert_eq(data.platform, "na1", "Mock: platform resolved")
    assert_eq(data.region, "americas", "Mock: region resolved")
    assert_true(data.rank is not None, "Mock: rank parsed")
    assert_eq(data.rank.tier, "GOLD", "Mock: rank tier")
    assert_eq(data.rank.division, "II", "Mock: rank division")
    assert_eq(data.rank.lp, 55, "Mock: rank LP")
    assert_eq(len(data.matches), 2, "Mock: 2 matches fetched")

    # Only NA1_1111 has deaths, so only its timeline should be fetched
    assert_true("NA1_1111" in data.timelines, "Mock: timeline fetched for game with deaths")
    assert_true("NA1_2222" not in data.timelines, "Mock: no timeline for game with 0 deaths")

    # Check solo_deaths_before_10 annotation
    p1 = next(p for p in data.matches[0]["info"]["participants"] if p["puuid"] == data.puuid)
    assert_eq(p1["solo_deaths_before_10"], 1, "Mock: solo_deaths_before_10 annotated from timeline")

    p2 = next(p for p in data.matches[1]["info"]["participants"] if p["puuid"] == data.puuid)
    assert_eq(p2["solo_deaths_before_10"], 0, "Mock: 0 deaths -> solo_deaths_before_10 = 0")


# ---------------------------------------------------------------------------
# Integration test (live API)
# ---------------------------------------------------------------------------

async def run_integration() -> None:
    from dotenv import load_dotenv
    load_dotenv()

    summoner = input("Summoner (Name or Name#TAG): ").strip()
    region = input("Region (na, euw, kr, ...): ").strip()
    count = int(input("Match count [default 5]: ").strip() or "5")

    print(f"\nFetching {count} matches for {summoner!r} on {region!r}...\n")

    async with RiotClient() as client:
        data = await client.fetch_player_data(summoner, region, count=count)

    print("=" * 60)
    print(f"Summoner : {data.summoner_name}")
    print(f"PUUID    : {data.puuid[:24]}...")
    print(f"Platform : {data.platform}  →  Region: {data.region}")
    print(f"Rank     : {data.rank.label if data.rank else 'Unranked'}")
    print(f"Matches  : {len(data.matches)} fetched")
    print(f"Timelines: {len(data.timelines)} fetched (only for games with deaths)")
    print()

    for match in data.matches[:count]:
        mid = match["metadata"]["matchId"]
        info = match["info"]
        duration_min = info["gameDuration"] / 60
        player = next(p for p in info["participants"] if p["puuid"] == data.puuid)
        cs = player["totalMinionsKilled"] + player["neutralMinionsKilled"]
        print(f"  {mid}")
        print(f"    Champion : {player['championName']} | Role: {player.get('teamPosition') or 'N/A'}")
        print(f"    KDA      : {player['kills']}/{player['deaths']}/{player['assists']}")
        print(f"    CS/min   : {cs / duration_min:.2f}  |  Duration: {duration_min:.1f} min")
        print(f"    Vision   : {player['visionScore']}  |  Win: {player['win']}")
        print(f"    Solo deaths <10m: {player.get('solo_deaths_before_10', 'N/A')}")
        print()

    print("Integration test PASSED.")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def main() -> None:
    print("\n=== riot_client.py — unit tests ===\n")

    # Token bucket (async)
    await test_token_bucket_full_bucket_immediate()
    await test_token_bucket_empty_bucket_waits()
    await test_token_bucket_respects_capacity()

    # Region helpers (sync)
    test_resolve_platform_aliases()
    test_platform_to_region_coverage()

    # Rank helpers (sync)
    test_rank_ordering()
    test_rank_label()

    # Solo death extraction (sync)
    test_solo_deaths_before_10_basic()
    test_solo_deaths_before_10_no_deaths()
    test_solo_deaths_before_10_missing_puuid()
    test_solo_deaths_frame_boundary()

    # Mocked integration
    await test_fetch_player_data_mocked()

    print()
    if _failures:
        print(f"\033[91mFAILED: {len(_failures)} test(s)\033[0m")  # noqa: T201
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("\033[92mAll tests passed.\033[0m")

    if "--integration" in sys.argv:
        print("\n=== Live API integration test ===\n")
        await run_integration()


if __name__ == "__main__":
    asyncio.run(main())
