from __future__ import annotations

import time
from collections import defaultdict
from typing import DefaultDict

from redis.asyncio import Redis

from app.config import settings

USER_LIMIT_PER_MINUTE = settings.USER_RATE_LIMIT_PER_MINUTE
USER_BURST = settings.USER_RATE_BURST
GLOBAL_LIMIT_PER_MINUTE = settings.GLOBAL_RATE_LIMIT_PER_MINUTE


class MemoryRateLimiter:
    """Fallback in-memory limiter used when Redis is unavailable during local or test execution."""

    def __init__(self) -> None:
        self._buckets: DefaultDict[str, list[float]] = defaultdict(list)

    async def allow(self, user_id: str) -> bool:
        now = time.time()
        user_key = f"user:{user_id}"
        global_key = "global:requests"

        self._buckets[user_key] = [ts for ts in self._buckets.get(user_key, []) if now - ts < 60]
        self._buckets[global_key] = [ts for ts in self._buckets.get(global_key, []) if now - ts < 60]

        if len(self._buckets[user_key]) >= USER_LIMIT_PER_MINUTE + USER_BURST:
            return False
        if len(self._buckets[global_key]) >= GLOBAL_LIMIT_PER_MINUTE + USER_BURST:
            return False

        self._buckets[user_key].append(now)
        self._buckets[global_key].append(now)
        return True


class RedisRateLimiter:
    """Redis-backed limiter that enforces a per-user window and a global application window."""

    def __init__(self, redis_url: str) -> None:
        self.client = Redis.from_url(redis_url, decode_responses=True)

    async def allow(self, user_id: str) -> bool:
        now = time.time()
        user_key = f"rate:user:{user_id}"
        global_key = "rate:global"

        user_count = await self.client.zcount(user_key, now - 60, now)
        if user_count >= USER_LIMIT_PER_MINUTE + USER_BURST:
            return False

        global_count = await self.client.zcount(global_key, now - 60, now)
        if global_count >= GLOBAL_LIMIT_PER_MINUTE + USER_BURST:
            return False

        await self.client.zadd(user_key, {f"{now}:{user_id}": now})
        await self.client.zadd(global_key, {f"{now}:global": now})
        await self.client.expire(user_key, 60)
        await self.client.expire(global_key, 60)
        return True


class RateLimiter:
    """Thin wrapper that prefers Redis and falls back to the in-memory bucket when needed."""

    def __init__(self, redis_url: str | None = None) -> None:
        self._redis = RedisRateLimiter(redis_url) if redis_url else None
        self._memory = MemoryRateLimiter()

    async def allow(self, user_id: str) -> bool:
        user_id = user_id or "anonymous"
        if self._redis is not None:
            try:
                return await self._redis.allow(user_id)
            except Exception:
                pass
        return await self._memory.allow(user_id)


rate_limiter = RateLimiter(settings.REDIS_URL)


def extract_user_id(authorization: str | None, user_header: str | None) -> str:
    """Extract the user identifier from standard API-key headers and keep the client contract simple."""
    if authorization:
        candidate = authorization.strip()
        if candidate.lower().startswith("apikey "):
            value = candidate.split(" ", 1)[1].strip()
            if value:
                return value
    if user_header:
        return user_header.strip()
    return "anonymous"
