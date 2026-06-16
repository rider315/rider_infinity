"""
news_ingestion.py - Multi-source forex news aggregator.

Fetches financial news from multiple providers — Finnhub, NewsAPI,
and configurable RSS feeds — deduplicates by headline fingerprint,
and surfaces them for downstream AI analysis.

Architecture
------------
* ``NewsArticle`` – Immutable dataclass representing a single article.
* ``BaseNewsSource`` – Abstract interface every provider implements.
* ``FinnhubNewsSource`` – REST client for Finnhub general + forex news.
* ``NewsAPISource`` – REST client for NewsAPI.org business + forex news.
* ``RSSNewsSource`` – Generic RSS / Atom feed parser (no API key required).
* ``NewsIngestionEngine`` – Orchestrator that polls all sources on a
  background daemon thread, deduplicates, and exposes recent articles
  in a thread-safe manner.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import ssl
import threading
import time
import urllib.error
import urllib.request
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =====================================================================
# Constants
# =====================================================================

# Known major forex currencies for symbol extraction.
_KNOWN_CURRENCIES: List[str] = [
    "USD", "EUR", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD",
]

# Map single-currency mentions / nicknames → likely affected pairs.
_CURRENCY_PAIR_MAP: Dict[str, List[str]] = {
    "USD":    ["EURUSD", "USDJPY", "GBPUSD", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"],
    "EUR":    ["EURUSD", "EURGBP", "EURJPY", "EURAUD", "EURCHF", "EURCAD"],
    "GBP":    ["GBPUSD", "EURGBP", "GBPJPY", "GBPAUD", "GBPCAD", "GBPCHF"],
    "JPY":    ["USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY", "CADJPY"],
    "CHF":    ["USDCHF", "EURCHF", "GBPCHF"],
    "AUD":    ["AUDUSD", "EURAUD", "GBPAUD", "AUDJPY", "AUDCAD", "AUDNZD"],
    "NZD":    ["NZDUSD", "AUDNZD", "NZDJPY", "EURNZD"],
    "CAD":    ["USDCAD", "EURCAD", "GBPCAD", "AUDCAD", "CADJPY"],
}

# Informal names → ISO currency code.
_CURRENCY_NICKNAMES: Dict[str, str] = {
    "dollar":       "USD",
    "greenback":    "USD",
    "buck":         "USD",
    "euro":         "EUR",
    "pound":        "GBP",
    "sterling":     "GBP",
    "cable":        "GBP",
    "yen":          "JPY",
    "franc":        "CHF",
    "swissy":       "CHF",
    "aussie":       "AUD",
    "kiwi":         "NZD",
    "loonie":       "CAD",
}

# Regex: matches explicit pairs like EURUSD, EUR/USD, EUR-USD.
_PAIR_PATTERN = re.compile(
    r"\b([A-Z]{3})[/\-]?([A-Z]{3})\b",
)

# Maximum article age before automatic cleanup (hours).
_MAX_ARTICLE_AGE_HOURS: int = 48

# Default retry settings for HTTP sources.
_MAX_RETRIES: int = 3
_BASE_BACKOFF: float = 1.0

# Create an SSL context that works even when system certs are incomplete.
_SSL_CTX: ssl.SSLContext = ssl.create_default_context()


# =====================================================================
# NewsArticle dataclass
# =====================================================================

@dataclass
class NewsArticle:
    """Immutable representation of a single news article.

    Attributes
    ----------
    id : str
        Unique identifier (UUID-4).
    headline : str
        Article headline / title.
    summary : str
        Body text or short description.
    source : str
        Provider identifier (e.g. ``"finnhub"``, ``"newsapi"``).
    source_name : str
        Human-readable source name (e.g. ``"Reuters"``).
    url : str
        Canonical URL to the full article.
    published_at : datetime
        Publication timestamp (UTC-aware).
    symbols : list[str]
        Forex pairs mentioned or implied by the article.
    category : str
        Topic category (e.g. ``"forex"``, ``"general"``).
    raw_data : dict
        Original payload from the upstream provider.
    fingerprint : str
        MD5 hex-digest of the normalised headline — used for dedup.
    """

    headline: str
    summary: str = ""
    source: str = ""
    source_name: str = ""
    url: str = ""
    published_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    symbols: List[str] = field(default_factory=list)
    category: str = ""
    raw_data: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    fingerprint: str = ""

    def __post_init__(self) -> None:
        """Compute fingerprint automatically if not provided."""
        if not self.fingerprint and self.headline:
            self.fingerprint = _compute_fingerprint(self.headline)


# =====================================================================
# Abstract source
# =====================================================================

class BaseNewsSource(ABC):
    """Abstract interface for a news data provider."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable source name used in logs and status dicts."""
        ...

    @abstractmethod
    def fetch(self) -> List[NewsArticle]:
        """Fetch the latest articles from this provider.

        Implementations must handle their own error cases and return
        an empty list on failure — they must *never* raise.

        Returns
        -------
        list[NewsArticle]
        """
        ...


# =====================================================================
# Finnhub
# =====================================================================

class FinnhubNewsSource(BaseNewsSource):
    """Fetches general market and forex news from Finnhub REST API.

    Parameters
    ----------
    api_key : str
        Finnhub API key (free tier is sufficient).
    """

    _BASE_URL = "https://finnhub.io/api/v1/news"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    @property
    def name(self) -> str:  # noqa: D102
        return "Finnhub"

    def fetch(self) -> List[NewsArticle]:
        """Fetch general + forex news from Finnhub."""
        articles: List[NewsArticle] = []

        for category in ("general", "forex"):
            url = f"{self._BASE_URL}?category={category}&token={self._api_key}"
            data = self._request_json(url)
            if data is None:
                continue

            for item in data:
                try:
                    # Finnhub uses unix timestamps.
                    published = datetime.fromtimestamp(
                        item.get("datetime", 0), tz=timezone.utc,
                    )
                    headline = item.get("headline", "").strip()
                    if not headline:
                        continue

                    symbols = _extract_forex_symbols(
                        f"{headline} {item.get('summary', '')}",
                    )

                    articles.append(
                        NewsArticle(
                            headline=headline,
                            summary=item.get("summary", ""),
                            source="finnhub",
                            source_name=item.get("source", "Finnhub"),
                            url=item.get("url", ""),
                            published_at=published,
                            symbols=symbols,
                            category=category,
                            raw_data=item,
                        ),
                    )
                except Exception:  # noqa: BLE001
                    logger.debug(
                        "Skipping malformed Finnhub item: %s",
                        item, exc_info=True,
                    )

        logger.debug("Finnhub returned %d articles.", len(articles))
        return articles

    # ── HTTP helper with retries ─────────────────────────────────────

    @staticmethod
    def _request_json(
        url: str,
        retries: int = _MAX_RETRIES,
        backoff: float = _BASE_BACKOFF,
    ) -> Optional[list]:
        """GET *url* and parse JSON with exponential-backoff retry.

        Returns ``None`` on total failure instead of raising.
        """
        last_error: Optional[Exception] = None
        for attempt in range(1, retries + 1):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "RiderInfinity/1.0"})
                with urllib.request.urlopen(req, timeout=15, context=_SSL_CTX) as resp:
                    body = resp.read().decode("utf-8")
                    return json.loads(body)
            except urllib.error.HTTPError as exc:
                last_error = exc
                logger.warning(
                    "Finnhub HTTP %d on attempt %d/%d: %s",
                    exc.code, attempt, retries, exc.reason,
                )
                if exc.code in (401, 403):
                    # Auth errors won't resolve with retries.
                    break
            except (urllib.error.URLError, OSError) as exc:
                last_error = exc
                logger.warning(
                    "Finnhub network error (attempt %d/%d): %s",
                    attempt, retries, exc,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(
                    "Finnhub unexpected error (attempt %d/%d): %s",
                    attempt, retries, exc,
                )

            if attempt < retries:
                wait = backoff * (2 ** (attempt - 1))
                time.sleep(wait)

        logger.error("Finnhub: all %d attempts failed. Last error: %s", retries, last_error)
        return None


# =====================================================================
# NewsAPI.org
# =====================================================================

class NewsAPISource(BaseNewsSource):
    """Fetches business and forex news from NewsAPI.org.

    Parameters
    ----------
    api_key : str
        NewsAPI.org API key.  If empty, the source is effectively
        disabled — :meth:`fetch` returns an empty list.
    """

    _TOP_HEADLINES = (
        "https://newsapi.org/v2/top-headlines"
        "?category=business&language=en&apiKey={key}"
    )
    _EVERYTHING = (
        "https://newsapi.org/v2/everything"
        "?q=forex+OR+currency+OR+%22central+bank%22"
        "&sortBy=publishedAt&language=en&apiKey={key}&pageSize=20"
    )

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    @property
    def name(self) -> str:  # noqa: D102
        return "NewsAPI"

    def fetch(self) -> List[NewsArticle]:
        """Fetch top headlines and forex-related articles from NewsAPI."""
        if not self._api_key:
            logger.warning("NewsAPI key not configured — skipping source.")
            return []

        articles: List[NewsArticle] = []
        urls = [
            self._TOP_HEADLINES.format(key=self._api_key),
            self._EVERYTHING.format(key=self._api_key),
        ]

        for url in urls:
            data = self._request_json(url)
            if data is None:
                continue

            for item in data.get("articles", []):
                try:
                    headline = (item.get("title") or "").strip()
                    if not headline:
                        continue

                    # NewsAPI provides ISO-8601 timestamps.
                    published = self._parse_iso(item.get("publishedAt"))

                    source_obj = item.get("source", {})
                    source_name = source_obj.get("name", "NewsAPI") if isinstance(source_obj, dict) else "NewsAPI"

                    symbols = _extract_forex_symbols(
                        f"{headline} {item.get('description', '')}",
                    )

                    articles.append(
                        NewsArticle(
                            headline=headline,
                            summary=item.get("description", "") or "",
                            source="newsapi",
                            source_name=source_name,
                            url=item.get("url", ""),
                            published_at=published,
                            symbols=symbols,
                            category="business",
                            raw_data=item,
                        ),
                    )
                except Exception:  # noqa: BLE001
                    logger.debug(
                        "Skipping malformed NewsAPI item: %s",
                        item, exc_info=True,
                    )

        logger.debug("NewsAPI returned %d articles.", len(articles))
        return articles

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _parse_iso(value: Optional[str]) -> datetime:
        """Parse an ISO-8601 datetime string; fall back to *now* on error."""
        if not value:
            return datetime.now(timezone.utc)
        try:
            # Remove trailing 'Z' and attach UTC.
            cleaned = value.replace("Z", "+00:00")
            return datetime.fromisoformat(cleaned)
        except (ValueError, TypeError):
            return datetime.now(timezone.utc)

    @staticmethod
    def _request_json(url: str) -> Optional[dict]:
        """GET *url* and parse JSON; returns ``None`` on failure."""
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "RiderInfinity/1.0"})
            with urllib.request.urlopen(req, timeout=15, context=_SSL_CTX) as resp:
                body = resp.read().decode("utf-8")
                data = json.loads(body)
                if data.get("status") != "ok":
                    logger.warning(
                        "NewsAPI responded with status=%s: %s",
                        data.get("status"), data.get("message", ""),
                    )
                    return None
                return data
        except urllib.error.HTTPError as exc:
            logger.warning("NewsAPI HTTP %d: %s", exc.code, exc.reason)
        except (urllib.error.URLError, OSError) as exc:
            logger.warning("NewsAPI network error: %s", exc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("NewsAPI unexpected error: %s", exc)
        return None


# =====================================================================
# RSS feeds
# =====================================================================

class RSSNewsSource(BaseNewsSource):
    """Parses configurable RSS / Atom feeds for financial news.

    Parameters
    ----------
    feeds : list[tuple[str, str]] | None
        Optional list of ``(url, display_name)`` tuples.  Falls back to
        a curated set of well-known forex / business feeds.
    """

    _DEFAULT_FEEDS: List[tuple] = [
        ("https://www.forexlive.com/feed/news", "ForexLive"),
        ("https://www.fxstreet.com/rss/news", "FXStreet"),
        ("https://feeds.reuters.com/reuters/businessNews", "Reuters Business"),
        ("https://feeds.bbci.co.uk/news/business/rss.xml", "BBC Business"),
    ]

    def __init__(self, feeds: Optional[List[tuple]] = None) -> None:
        self._feeds = feeds or self._DEFAULT_FEEDS

    @property
    def name(self) -> str:  # noqa: D102
        return "RSS"

    def fetch(self) -> List[NewsArticle]:
        """Parse all configured feeds and return combined articles."""
        try:
            import feedparser  # type: ignore[import-untyped]
        except ImportError:
            logger.warning(
                "feedparser is not installed — RSS source disabled.  "
                "Install with: pip install feedparser",
            )
            return []

        articles: List[NewsArticle] = []

        for feed_url, feed_name in self._feeds:
            try:
                parsed = feedparser.parse(feed_url)

                if parsed.bozo and not parsed.entries:
                    logger.warning(
                        "RSS feed '%s' returned an error: %s",
                        feed_name, parsed.bozo_exception,
                    )
                    continue

                for entry in parsed.entries:
                    try:
                        headline = (
                            getattr(entry, "title", "") or ""
                        ).strip()
                        if not headline:
                            continue

                        summary = (
                            getattr(entry, "summary", "")
                            or getattr(entry, "description", "")
                            or ""
                        )

                        published = self._parse_feed_date(entry)
                        link = getattr(entry, "link", "") or ""

                        symbols = _extract_forex_symbols(
                            f"{headline} {summary}",
                        )

                        articles.append(
                            NewsArticle(
                                headline=headline,
                                summary=summary,
                                source="rss",
                                source_name=feed_name,
                                url=link,
                                published_at=published,
                                symbols=symbols,
                                category="news",
                                raw_data=dict(entry) if hasattr(entry, "keys") else {},
                            ),
                        )
                    except Exception:  # noqa: BLE001
                        logger.debug(
                            "Skipping malformed RSS entry in %s",
                            feed_name, exc_info=True,
                        )

            except Exception:  # noqa: BLE001
                logger.warning(
                    "Failed to fetch RSS feed '%s' (%s).",
                    feed_name, feed_url, exc_info=True,
                )

        logger.debug("RSS returned %d articles.", len(articles))
        return articles

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _parse_feed_date(entry: Any) -> datetime:
        """Extract a UTC datetime from a feedparser entry.

        Uses ``published_parsed`` or ``updated_parsed`` struct_time
        if available; falls back to the current time.
        """
        import calendar

        for attr in ("published_parsed", "updated_parsed"):
            struct = getattr(entry, attr, None)
            if struct is not None:
                try:
                    ts = calendar.timegm(struct)
                    return datetime.fromtimestamp(ts, tz=timezone.utc)
                except (TypeError, ValueError, OverflowError):
                    pass
        return datetime.now(timezone.utc)


# =====================================================================
# Ingestion engine
# =====================================================================

class NewsIngestionEngine:
    """Orchestrates polling of all configured news sources.

    Runs a daemon thread that periodically fetches articles from every
    registered source, deduplicates them by headline fingerprint, and
    stores them in an in-memory list protected by a threading lock.

    Parameters
    ----------
    finnhub_api_key : str
        API key for Finnhub.  If empty, the Finnhub source is skipped.
    news_api_key : str
        API key for NewsAPI.org.  If empty, the NewsAPI source is skipped.
    rss_feeds : list[tuple[str, str]] | None
        Optional RSS feed overrides — ``(url, name)`` tuples.
    polling_interval : int
        Seconds between poll cycles (default 60).
    logger : logging.Logger | None
        Optional logger override; defaults to module-level logger.
    """

    def __init__(
        self,
        finnhub_api_key: str,
        news_api_key: str = "",
        rss_feeds: Optional[List[tuple]] = None,
        polling_interval: int = 60,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._log = logger or logging.getLogger(__name__)
        self._polling_interval = max(polling_interval, 10)  # Floor at 10 s

        # Thread-safety primitives.
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Article storage.
        self._articles: List[NewsArticle] = []
        self._seen_fingerprints: set = set()

        # Per-source tracking.
        self._source_stats: Dict[str, Dict[str, Any]] = {}

        # ── Register sources ─────────────────────────────────────────
        self._sources: List[BaseNewsSource] = []

        if finnhub_api_key:
            src = FinnhubNewsSource(api_key=finnhub_api_key)
            self._sources.append(src)
            self._source_stats[src.name] = {
                "enabled": True,
                "last_fetch": None,
                "article_count": 0,
            }
            self._log.info("News source registered: %s", src.name)
        else:
            self._log.warning(
                "Finnhub API key not provided — Finnhub source disabled.",
            )

        if news_api_key:
            src = NewsAPISource(api_key=news_api_key)
            self._sources.append(src)
            self._source_stats[src.name] = {
                "enabled": True,
                "last_fetch": None,
                "article_count": 0,
            }
            self._log.info("News source registered: %s", src.name)
        else:
            self._log.info(
                "NewsAPI key not provided — NewsAPI source disabled.",
            )

        # RSS is always available (no key required).
        rss = RSSNewsSource(feeds=rss_feeds or None)
        self._sources.append(rss)
        self._source_stats[rss.name] = {
            "enabled": True,
            "last_fetch": None,
            "article_count": 0,
        }
        self._log.info("News source registered: %s (always-on)", rss.name)

    # ----------------------------------------------------------------
    # Lifecycle
    # ----------------------------------------------------------------

    def start(self) -> None:
        """Start the background polling daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            self._log.warning("Ingestion engine is already running.")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="NewsIngestion",
            daemon=True,
        )
        self._thread.start()
        self._log.info(
            "NewsIngestionEngine started — polling every %ds, %d source(s).",
            self._polling_interval,
            len(self._sources),
        )

    def stop(self) -> None:
        """Signal the polling thread to stop and wait for it to finish."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._polling_interval + 5)
            if self._thread.is_alive():
                self._log.warning(
                    "Ingestion thread did not exit within timeout.",
                )
            else:
                self._log.info("NewsIngestionEngine stopped gracefully.")
            self._thread = None

    # ----------------------------------------------------------------
    # Properties
    # ----------------------------------------------------------------

    @property
    def article_count(self) -> int:
        """Total number of articles currently held in memory."""
        with self._lock:
            return len(self._articles)

    @property
    def is_running(self) -> bool:
        """Whether the background polling thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    @property
    def source_status(self) -> Dict[str, Dict[str, Any]]:
        """Per-source health snapshot.

        Returns
        -------
        dict
            ``{source_name: {enabled: bool, last_fetch: datetime | None,
            article_count: int}}``
        """
        with self._lock:
            # Return a deep-enough copy so callers can't mutate internals.
            return {
                name: dict(stats)
                for name, stats in self._source_stats.items()
            }

    # ----------------------------------------------------------------
    # Query
    # ----------------------------------------------------------------

    def get_latest(self, max_age_hours: int = 4) -> List[NewsArticle]:
        """Return articles published within the last *max_age_hours*.

        Parameters
        ----------
        max_age_hours : int
            Look-back window in hours (default 4).

        Returns
        -------
        list[NewsArticle]
            Articles ordered newest-first.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        with self._lock:
            recent = [
                a for a in self._articles if a.published_at >= cutoff
            ]
        # Newest first.
        recent.sort(key=lambda a: a.published_at, reverse=True)
        return recent

    # ----------------------------------------------------------------
    # Internal loop
    # ----------------------------------------------------------------

    def _run_loop(self) -> None:
        """Main polling loop executed on the background thread."""
        self._log.debug("Ingestion loop entered.")

        # Do an immediate first poll before sleeping.
        self._poll_sources()

        while not self._stop_event.is_set():
            # Wait interruptibly so stop() takes effect promptly.
            if self._stop_event.wait(timeout=self._polling_interval):
                break
            self._poll_sources()
            self._cleanup_old_articles()

        self._log.debug("Ingestion loop exited.")

    def _poll_sources(self) -> None:
        """Iterate every source, fetch articles, and deduplicate."""
        for source in self._sources:
            try:
                self._log.debug("Polling source: %s …", source.name)
                raw_articles = source.fetch()
                new_count = 0

                with self._lock:
                    for article in raw_articles:
                        fp = article.fingerprint
                        if fp and fp not in self._seen_fingerprints:
                            self._seen_fingerprints.add(fp)
                            self._articles.append(article)
                            new_count += 1

                    # Update per-source stats.
                    if source.name in self._source_stats:
                        self._source_stats[source.name]["last_fetch"] = (
                            datetime.now(timezone.utc)
                        )
                        self._source_stats[source.name]["article_count"] += new_count

                if new_count:
                    self._log.info(
                        "Source %s: %d new article(s) ingested (%d fetched).",
                        source.name, new_count, len(raw_articles),
                    )
                else:
                    self._log.debug(
                        "Source %s: 0 new articles (%d fetched, all duplicates).",
                        source.name, len(raw_articles),
                    )

            except Exception:  # noqa: BLE001
                # One source failing must never bring down the others.
                self._log.exception(
                    "Unhandled error polling source '%s' — skipping.",
                    source.name,
                )

    def _cleanup_old_articles(self) -> None:
        """Remove in-memory articles older than ``_MAX_ARTICLE_AGE_HOURS``."""
        cutoff = datetime.now(timezone.utc) - timedelta(
            hours=_MAX_ARTICLE_AGE_HOURS,
        )
        with self._lock:
            before = len(self._articles)
            self._articles = [
                a for a in self._articles if a.published_at >= cutoff
            ]
            removed = before - len(self._articles)

        if removed:
            self._log.info(
                "Cleanup: purged %d article(s) older than %dh.",
                removed, _MAX_ARTICLE_AGE_HOURS,
            )

    # ----------------------------------------------------------------
    # Static helpers
    # ----------------------------------------------------------------

    @staticmethod
    def _extract_forex_symbols(text: str) -> List[str]:
        """Extract forex pair symbols from *text*.

        Kept as a static method on the engine for backward compatibility;
        delegates to the module-level helper.
        """
        return _extract_forex_symbols(text)


# =====================================================================
# Module-level helpers
# =====================================================================

def _compute_fingerprint(headline: str) -> str:
    """Return an MD5 hex-digest of the normalised headline."""
    normalised = headline.lower().strip().encode("utf-8")
    return hashlib.md5(normalised).hexdigest()


def _extract_forex_symbols(text: str) -> List[str]:
    """Extract forex pair symbols from free-form *text*.

    Detection strategy
    ------------------
    1. Explicit pair patterns: ``EURUSD``, ``EUR/USD``, ``EUR-USD``.
    2. Single ISO currency codes (e.g. ``USD`` mentioned alone) →
       mapped to a set of commonly-affected pairs.
    3. Informal currency nicknames (``dollar``, ``euro``, ``yen``, …)
       → mapped via ``_CURRENCY_NICKNAMES`` → pairs.

    Returns a de-duplicated, sorted list.

    Parameters
    ----------
    text : str
        The input text to scan.

    Returns
    -------
    list[str]
        Sorted unique forex pair symbols.
    """
    found: set = set()
    upper_text = text.upper()

    # 1. Explicit pairs  (e.g. EURUSD, EUR/USD, EUR-USD).
    for match in _PAIR_PATTERN.finditer(upper_text):
        base, quote = match.group(1), match.group(2)
        if base in _KNOWN_CURRENCIES and quote in _KNOWN_CURRENCIES:
            found.add(f"{base}{quote}")

    # 2. Standalone ISO currency codes.
    for ccy in _KNOWN_CURRENCIES:
        # Use word boundary to avoid matching inside longer words.
        if re.search(rf"\b{ccy}\b", upper_text):
            found.update(_CURRENCY_PAIR_MAP.get(ccy, []))

    # 3. Informal nicknames (case-insensitive).
    lower_text = text.lower()
    for nickname, ccy in _CURRENCY_NICKNAMES.items():
        if re.search(rf"\b{nickname}\b", lower_text):
            found.update(_CURRENCY_PAIR_MAP.get(ccy, []))

    return sorted(found)
