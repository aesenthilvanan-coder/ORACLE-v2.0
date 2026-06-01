import json
import hashlib
import pickle
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar
import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")

DEFAULT_CACHE_DIR = Path.home() / ".oracle_cache"


class DiskCache:
    """Simple disk-backed key-value cache with optional TTL."""

    def __init__(self, cache_dir: Optional[Path] = None, namespace: str = "default"):
        self.cache_dir = (cache_dir or DEFAULT_CACHE_DIR) / namespace
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key_to_path(self, key: str) -> Path:
        h = hashlib.sha256(key.encode()).hexdigest()[:16]
        return self.cache_dir / f"{h}.pkl"

    def get(self, key: str) -> Optional[Any]:
        path = self._key_to_path(key)
        if path.exists():
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.debug(f"Cache read error for {key}: {e}")
                path.unlink(missing_ok=True)
        return None

    def set(self, key: str, value: Any) -> None:
        path = self._key_to_path(key)
        try:
            with open(path, "wb") as f:
                pickle.dump(value, f)
        except Exception as e:
            logger.debug(f"Cache write error for {key}: {e}")

    def has(self, key: str) -> bool:
        return self._key_to_path(key).exists()

    def delete(self, key: str) -> None:
        self._key_to_path(key).unlink(missing_ok=True)

    def clear(self) -> None:
        for p in self.cache_dir.glob("*.pkl"):
            p.unlink(missing_ok=True)
        logger.info(f"Cleared cache at {self.cache_dir}")


class JSONCache:
    """JSON-serializable disk cache for simple types."""

    def __init__(self, cache_dir: Optional[Path] = None, namespace: str = "json"):
        self.cache_dir = (cache_dir or DEFAULT_CACHE_DIR) / namespace
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> Optional[Any]:
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            try:
                with open(path) as f:
                    return json.load(f)
            except Exception:
                pass
        return None

    def set(self, key: str, value: Any) -> None:
        path = self.cache_dir / f"{key}.json"
        try:
            with open(path, "w") as f:
                json.dump(value, f)
        except Exception as e:
            logger.debug(f"JSON cache write error for {key}: {e}")


def cached(cache: DiskCache, key_fn: Optional[Callable] = None):
    """Decorator for caching function results to disk."""
    def decorator(fn: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            if key_fn is not None:
                key = key_fn(*args, **kwargs)
            else:
                key = f"{fn.__name__}:{str(args)}:{str(kwargs)}"
            result = cache.get(key)
            if result is not None:
                return result
            result = fn(*args, **kwargs)
            cache.set(key, result)
            return result
        return wrapper
    return decorator
