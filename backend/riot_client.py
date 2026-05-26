"""
Riot API client with async token-bucket rate limiting.
Free dev key limits: 20 req/s, 100 req/2 min.

Supports both legacy summoner names and modern Riot IDs (Name#TAG).
Timelines are fetched only for games where the player registered deaths,
preserving API quota while still enabling solo-death-before-10 analysis.
"""

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

# platform routing (summoner-v4, league-v4) → regional routing (match-v5, account-v1)
PLATFORM_TO_REGION: dict[str, str] = {
    "na1": "americas",
    "br1": "americas",
    "la1": "americas",
    "la2": "americas",
    "euw1": "europe",
    "eun1": "europe",
    "tr1": "europe",
    "ru": "europe",
    "kr": "asia",
    "jp1": "asia",
    "oc1": "sea",
    "ph2": "sea",
    "sg2": "sea",
    "th2": "sea",
    "tw2": "sea",
    "vn2": "sea",
}

REGION_ALIASES: dict[str, str] = {
    "na": "na1",
    "euw": "euw1",
    "eune": "eun1",
    "oce": "oc1",
    "br": "br1",
    "jp": "jp1",
    "lan": "la1",
    "las": "la2",
    "tr": "tr1",
    "kr": "kr",
    "ru": "ru",
}


@dataclass
class SummonerRank:
    tier: str
    division: str
    lp: int
    queue_type: str

    @property
    def label(self) -> str:
        return f"{self.tier} {self.division} {self.lp} LP"

    @property
    def numeric_rank(self) -> int:
        """Lower = lower rank. Used for rank-change detection."""
        tier_order = ["IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND", "MASTER", "GRANDMASTER", "CHALLENGER"]
        div_order = {"IV": 0, "III": 1, "II": 2, "I": 3}
        t = tier_order.index(self.tier) if self.tier in tier_order else 0
        d = div_order.get(self.division, 0)
        return t * 400 + d * 100 + self.lp


@dataclass
class PlayerData:
    summoner_name: str
    summoner_id: str
    puuid: str
    account_id: str
    platform: str
    region: str
    rank: Optional[SummonerRank]
    matches: list[dict]
    timelines: dict[str, dict] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class TokenBucket:
    """
    Async token bucket. Sleeps *outside* the lock so waiters don't block
    each other — important when many coroutines are queued concurrently.
    """

    def __init__(self, rate: float, capacity: float) -> None:
        self.rate = rate          # tokens added per second
        self.capacity = capacity
        self.tokens = capacity
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self._last) * self.rate)
        self._last = now

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                self._refill()
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                wait = (1.0 - self.tokens) / self.rate
            # Sleep outside the lock so other coroutines can check state.
            await asyncio.sleep(wait)


class RateLimiter:
    """
    Enforces both Riot free-key limits simultaneously:
      - 18/s  (leaving 2 headroom below the 20/s limit)
      - 90/2min (leaving 10 headroom below the 100/2min limit)
    """

    def __init__(self) -> None:
        self._per_second = TokenBucket(rate=18.0, capacity=18.0)
        self._per_2min = TokenBucket(rate=90 / 120, capacity=90.0)

    async def acquire(self) -> None:
        # Both buckets must grant a token before the request fires.
        await asyncio.gather(
            self._per_second.acquire(),
            self._per_2min.acquire(),
        )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class RiotClient:

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("RIOT_API_KEY", "")
        if not self.api_key:
            raise ValueError("RIOT_API_KEY missing — set it in .env or pass explicitly")
        self._limiter = RateLimiter()
        self._http = httpx.AsyncClient(timeout=30.0)

    async def close(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> "RiotClient":
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Internal HTTP
    # ------------------------------------------------------------------

    async def _get(self, url: str) -> dict | list:
        await self._limiter.acquire()
        resp = await self._http.get(url, headers={"X-Riot-Token": self.api_key})
        if resp.status_code == 429:
            # Honour the server-side rate-limit header rather than a fixed sleep.
            retry_after = float(resp.headers.get("Retry-After", "5"))
            await asyncio.sleep(retry_after)
            return await self._get(url)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Region resolution
    # ------------------------------------------------------------------

    @staticmethod
    def resolve_platform(region_input: str) -> str:
        """'na' → 'na1', 'EUW' → 'euw1', 'na1' → 'na1'."""
        r = region_input.lower().strip()
        return REGION_ALIASES.get(r, r)

    @staticmethod
    def platform_to_region(platform: str) -> str:
        return PLATFORM_TO_REGION[platform]

    # ------------------------------------------------------------------
    # API wrappers
    # ------------------------------------------------------------------

    async def get_account_by_riot_id(self, game_name: str, tag_line: str, region: str) -> dict:
        """account-v1 — modern Riot ID lookup (returns puuid)."""
        url = (
            f"https://{region}.api.riotgames.com"
            f"/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}"
        )
        return await self._get(url)

    async def get_summoner_by_name(self, name: str, platform: str) -> dict:
        """summoner-v4 — legacy name lookup."""
        encoded = name.replace(" ", "%20")
        url = f"https://{platform}.api.riotgames.com/lol/summoner/v4/summoners/by-name/{encoded}"
        return await self._get(url)

    async def get_summoner_by_puuid(self, puuid: str, platform: str) -> dict:
        """summoner-v4 — resolve puuid → summoner data."""
        url = f"https://{platform}.api.riotgames.com/lol/summoner/v4/summoners/by-puuid/{puuid}"
        return await self._get(url)

    async def get_rank(self, puuid: str, platform: str) -> list[dict]:
        """league-v4 — all ranked queue entries for a summoner (PUUID-based, post-2024 API)."""
        url = f"https://{platform}.api.riotgames.com/lol/league/v4/entries/by-puuid/{puuid}"
        return await self._get(url)

    async def get_match_ids(
        self,
        puuid: str,
        region: str,
        count: int = 40,
        queue: Optional[int] = 420,  # 420 = ranked solo/duo
    ) -> list[str]:
        q = f"&queue={queue}" if queue is not None else ""
        url = (
            f"https://{region}.api.riotgames.com"
            f"/lol/match/v5/matches/by-puuid/{puuid}/ids?count={count}{q}"
        )
        return await self._get(url)

    async def get_match(self, match_id: str, region: str) -> dict:
        url = f"https://{region}.api.riotgames.com/lol/match/v5/matches/{match_id}"
        return await self._get(url)

    async def get_timeline(self, match_id: str, region: str) -> dict:
        url = f"https://{region}.api.riotgames.com/lol/match/v5/matches/{match_id}/timeline"
        return await self._get(url)

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def _extract_solo_deaths_before_10(
        self,
        timeline: dict,
        puuid: str,
        participants: list[dict],
    ) -> int:
        """
        Counts deaths before 10 min where assistingParticipantIds is empty.
        Empty assists list = no teammates helped secure the kill → likely a
        positioning or decision error the player made alone.
        """
        pid_map: dict[int, str] = {p["participantId"]: p["puuid"] for p in participants}
        player_pid = next((pid for pid, uid in pid_map.items() if uid == puuid), None)
        if player_pid is None:
            return 0

        count = 0
        for frame in timeline.get("info", {}).get("frames", []):
            if frame["timestamp"] >= 600_000:  # 10 minutes in ms
                break
            for event in frame.get("events", []):
                if (
                    event.get("type") == "CHAMPION_KILL"
                    and event.get("victimId") == player_pid
                    and not event.get("assistingParticipantIds")
                ):
                    count += 1
        return count

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def fetch_player_data(
        self,
        summoner_input: str,
        region_input: str,
        count: int = 40,
    ) -> PlayerData:
        """
        Full pipeline: resolve summoner → fetch rank → fetch matches (parallel)
        → fetch timelines (only for games with deaths, parallel).

        summoner_input accepts:
          - "PlayerName"         (legacy summoner name)
          - "PlayerName#TAG"     (Riot ID, preferred)
        """
        platform = self.resolve_platform(region_input)
        if platform not in PLATFORM_TO_REGION:
            raise ValueError(
                f"Unknown region {region_input!r}. "
                f"Valid platforms: {sorted(PLATFORM_TO_REGION)}"
            )
        routing_region = PLATFORM_TO_REGION[platform]

        # --- 1. Resolve summoner → puuid ---
        # NOTE: As of 2024, summoner-v4 no longer returns 'id' or 'name'.
        # We use account-v1 gameName/tagLine as the display name,
        # and league-v4 by-puuid for rank (no summoner ID needed).
        summoner_name: str
        if "#" in summoner_input:
            game_name, tag_line = summoner_input.split("#", 1)
            account = await self.get_account_by_riot_id(game_name, tag_line, routing_region)
            puuid = account["puuid"]
            summoner_name = f"{account['gameName']}#{account['tagLine']}"
            # Fetch summoner for profileIconId / level (optional, but keeps the structure)
            summoner = await self.get_summoner_by_puuid(puuid, platform)
        else:
            summoner = await self.get_summoner_by_name(summoner_input, platform)
            puuid = summoner["puuid"]
            summoner_name = summoner.get("name", summoner_input)

        # --- 2. Rank (now uses PUUID directly) ---
        rank_entries = await self.get_rank(puuid, platform)
        solo_raw = next(
            (r for r in rank_entries if r.get("queueType") == "RANKED_SOLO_5x5"),
            None,
        )
        rank = (
            SummonerRank(
                tier=solo_raw["tier"],
                division=solo_raw["rank"],
                lp=solo_raw["leaguePoints"],
                queue_type="RANKED_SOLO_5x5",
            )
            if solo_raw
            else None
        )

        # --- 3. Match IDs → match details (parallel, rate-limited) ---
        match_ids = await self.get_match_ids(puuid, routing_region, count)
        matches: list[dict] = list(
            await asyncio.gather(*[self.get_match(mid, routing_region) for mid in match_ids])
        )

        # --- 4. Timelines — only for games where player has deaths ---
        needs_timeline: list[str] = []
        for match in matches:
            player = next(
                (p for p in match["info"]["participants"] if p["puuid"] == puuid),
                None,
            )
            if player and player.get("deaths", 0) > 0:
                needs_timeline.append(match["metadata"]["matchId"])

        timeline_results = await asyncio.gather(
            *[self.get_timeline(mid, routing_region) for mid in needs_timeline],
            return_exceptions=True,
        )
        timelines: dict[str, dict] = {
            mid: result
            for mid, result in zip(needs_timeline, timeline_results)
            if not isinstance(result, Exception)
        }

        # --- 5. Annotate each match with solo_deaths_before_10 ---
        for match in matches:
            mid = match["metadata"]["matchId"]
            participants = match["info"]["participants"]
            player = next((p for p in participants if p["puuid"] == puuid), None)
            if player is not None:
                if mid in timelines:
                    player["solo_deaths_before_10"] = self._extract_solo_deaths_before_10(
                        timelines[mid], puuid, participants
                    )
                else:
                    player["solo_deaths_before_10"] = 0

        return PlayerData(
            summoner_name=summoner_name,
            summoner_id=puuid,           # summoner_id deprecated; store puuid here
            puuid=puuid,
            account_id=summoner.get("accountId", ""),
            platform=platform,
            region=routing_region,
            rank=rank,
            matches=matches,
            timelines=timelines,
        )
