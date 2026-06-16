"""
News Intelligence Engine — Rider Infinity
==========================================

Real-time news analysis and trading signal generation powered by
AI sentiment analysis.  Aggregates financial news from multiple
sources, analyses them via Anthropic Claude, and produces
actionable trading signals.
"""

from .news_ingestion import NewsArticle, NewsIngestionEngine
from .news_cache import NewsCache
from .news_analyzer import ClaudeNewsAnalyzer
from .signal_composer import NewsSignalComposer

__all__ = [
    "NewsArticle",
    "NewsIngestionEngine",
    "NewsCache",
    "ClaudeNewsAnalyzer",
    "NewsSignalComposer",
]
