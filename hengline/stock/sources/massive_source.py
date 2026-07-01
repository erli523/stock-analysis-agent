#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Massive REST stock data source.
"""

import os
from datetime import date, timedelta
from typing import Any, Dict, List

import pandas as pd
import requests

from hengline.logger import debug, error, info, warning
from hengline.tools.cache_tool import get_price_cache, set_price_cache, get_stock_info_cache, set_stock_info_cache


class MassiveSource:
    """Massive/Polygon-compatible REST stock data source."""

    def __init__(self, api_keys: Dict[str, Any] = None):
        api_keys = api_keys or {}
        self.api_key = api_keys.get("massive") or os.environ.get("MASSIVE_API_KEY", "")
        self.base_url = os.environ.get("MASSIVE_BASE_URL", "https://api.massive.com").rstrip("/")
        self.timeout = int(os.environ.get("MASSIVE_TIMEOUT", "15"))
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "HengLine/1.0"
        })

        if self.api_key:
            info("Massive data source initialized")
        else:
            warning("Massive API key is not configured")

    def _request(self, path: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        if not self.api_key:
            raise ValueError("Massive API key is not configured")

        request_params = dict(params or {})
        request_params["apiKey"] = self.api_key
        url = f"{self.base_url}{path}"

        response = self.session.get(url, params=request_params, timeout=self.timeout)
        if response.status_code == 401:
            raise ValueError("Massive API authentication failed; check MASSIVE_API_KEY")
        if response.status_code == 429:
            raise ValueError("Massive API rate limit reached")

        response.raise_for_status()
        data = response.json()
        status = str(data.get("status", "")).upper()
        if status == "ERROR":
            raise ValueError(data.get("error") or data.get("message") or "Massive API returned an error")
        return data

    @staticmethod
    def _normalize_symbol(stock_code: str) -> str:
        return str(stock_code).strip().upper()

    @staticmethod
    def _period_to_dates(period: str) -> tuple[str, str]:
        today = date.today()
        days_map = {
            "1d": 7,
            "1w": 14,
            "1m": 40,
            "3m": 100,
            "6m": 200,
            "1y": 370,
            "2y": 740,
            "5y": 1850,
            "10y": 3700,
            "ytd": (today - date(today.year, 1, 1)).days + 7,
            "max": 3700,
        }
        start = today - timedelta(days=days_map.get(period, 370))
        return start.isoformat(), today.isoformat()

    @staticmethod
    def _interval_to_range(interval: str) -> tuple[int, str]:
        interval_map = {
            "1m": (1, "minute"),
            "5m": (5, "minute"),
            "15m": (15, "minute"),
            "30m": (30, "minute"),
            "60m": (1, "hour"),
            "1h": (1, "hour"),
            "1d": (1, "day"),
            "1wk": (1, "week"),
            "1w": (1, "week"),
            "1mo": (1, "month"),
        }
        return interval_map.get(interval, (1, "day"))

    def get_stock_price_data(self, stock_code: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        symbol = self._normalize_symbol(stock_code)
        cached_data = get_price_cache(symbol, period, interval)
        if cached_data is not None:
            debug(f"Massive: returning cached price data for {symbol}")
            return cached_data

        multiplier, timespan = self._interval_to_range(interval)
        start_date, end_date = self._period_to_dates(period)
        path = f"/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start_date}/{end_date}"

        debug(f"Massive: fetching price data for {symbol}, period={period}, interval={interval}")
        data = self._request(path, {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
        })

        rows = data.get("results") or []
        if not rows:
            warning(f"Massive: no price data returned for {symbol}")
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["Date"] = pd.to_datetime(df["t"], unit="ms")
        df = df.rename(columns={
            "o": "Open",
            "h": "High",
            "l": "Low",
            "c": "Close",
            "v": "Volume",
        })
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
        df = df.astype({
            "Open": float,
            "High": float,
            "Low": float,
            "Close": float,
            "Volume": float,
        })

        set_price_cache(symbol, period, interval, df)
        info(f"Massive: fetched {len(df)} price rows for {symbol}")
        return df

    def get_stock_info(self, stock_code: str) -> Dict[str, Any]:
        symbol = self._normalize_symbol(stock_code)
        cached_info = get_stock_info_cache(symbol)
        if cached_info:
            return cached_info

        debug(f"Massive: fetching ticker info for {symbol}")
        data = self._request(f"/v3/reference/tickers/{symbol}")
        result = data.get("results") or {}
        info_data = {
            "symbol": result.get("ticker", symbol),
            "name": result.get("name"),
            "company_name": result.get("name"),
            "market": result.get("market"),
            "locale": result.get("locale"),
            "primary_exchange": result.get("primary_exchange"),
            "type": result.get("type"),
            "currency_name": result.get("currency_name"),
            "market_cap": result.get("market_cap"),
            "description": result.get("description"),
            "homepage_url": result.get("homepage_url"),
            "list_date": result.get("list_date"),
        }
        info_data = {key: value for key, value in info_data.items() if value not in (None, "")}
        set_stock_info_cache(symbol, info_data)
        return info_data

    def get_stock_realtime_data(self, stock_code: str) -> Dict[str, Any]:
        symbol = self._normalize_symbol(stock_code)
        data = self._request(f"/v2/aggs/ticker/{symbol}/prev", {"adjusted": "true"})
        rows = data.get("results") or []
        if not rows:
            return {}

        row = rows[0]
        return {
            "symbol": symbol,
            "open": row.get("o"),
            "high": row.get("h"),
            "low": row.get("l"),
            "close": row.get("c"),
            "volume": row.get("v"),
            "timestamp": row.get("t"),
        }

    def get_stock_news(self, stock_code: str, limit: int = 5) -> List[Dict[str, Any]]:
        symbol = self._normalize_symbol(stock_code)
        data = self._request("/v2/reference/news", {
            "ticker": symbol,
            "limit": limit,
            "order": "desc",
            "sort": "published_utc",
        })
        news_items = []
        for item in data.get("results") or []:
            news_items.append({
                "title": item.get("title"),
                "publisher": (item.get("publisher") or {}).get("name"),
                "link": item.get("article_url"),
                "providerPublishTime": item.get("published_utc"),
                "summary": item.get("description"),
            })
        return news_items
