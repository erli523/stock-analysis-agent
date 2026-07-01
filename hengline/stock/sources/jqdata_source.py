#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JQData source for A-share market data.
"""

import os
from datetime import date, timedelta
from typing import Any, Dict, List

import pandas as pd

from hengline.logger import debug, info, warning
from hengline.tools.cache_tool import get_price_cache, set_price_cache, get_stock_info_cache, set_stock_info_cache


class JQDataSource:
    """JoinQuant JQData source based on jqdatasdk."""

    def __init__(self, api_keys: Dict[str, Any] = None):
        try:
            import jqdatasdk as jq
        except ImportError as exc:
            raise ImportError("jqdatasdk is not installed. Run: pip install jqdatasdk") from exc

        self.jq = jq
        self.username = os.environ.get("JQDATA_USERNAME", "")
        self.password = os.environ.get("JQDATA_PASSWORD", "")
        if not self.username or not self.password:
            raise ValueError("JQDATA_USERNAME and JQDATA_PASSWORD are required")

        self.jq.auth(self.username, self.password)
        info("Initialized jqdata source")

    @staticmethod
    def _normalize_code(stock_code: str) -> str:
        code = str(stock_code).strip().upper()
        if code.endswith((".XSHE", ".XSHG")):
            return code
        if code.startswith("SZ") and len(code) >= 8:
            return f"{code[-6:]}.XSHE"
        if code.startswith("SH") and len(code) >= 8:
            return f"{code[-6:]}.XSHG"
        if code.endswith(".SZ"):
            return f"{code[:6]}.XSHE"
        if code.endswith((".SH", ".SS")):
            return f"{code[:6]}.XSHG"
        if code.isdigit() and len(code) == 6:
            if code.startswith(("5", "6", "9")):
                return f"{code}.XSHG"
            return f"{code}.XSHE"
        return code

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
    def _frequency(interval: str) -> str:
        mapping = {
            "1d": "daily",
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "60m": "60m",
            "1h": "60m",
        }
        return mapping.get(interval, "daily")

    def get_stock_price_data(self, stock_code: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        code = self._normalize_code(stock_code)
        cached_data = get_price_cache(f"jqdata_{code}", period, interval)
        if cached_data is not None:
            return cached_data

        start_date, end_date = self._period_to_dates(period)
        frequency = self._frequency(interval)
        debug(f"JQData: fetching price data for {code}, period={period}, interval={interval}")

        df = self.jq.get_price(
            code,
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
            fields=["open", "high", "low", "close", "volume"],
            skip_paused=True,
            fq="pre",
        )
        if df is None or df.empty:
            warning(f"JQData: no price data returned for {code}")
            return pd.DataFrame()

        df = df.reset_index().rename(columns={
            "index": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        })
        if "time" in df.columns:
            df = df.rename(columns={"time": "Date"})
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
        set_price_cache(f"jqdata_{code}", period, interval, df)
        info(f"JQData: fetched {len(df)} price rows for {code}")
        return df

    def get_stock_info(self, stock_code: str) -> Dict[str, Any]:
        code = self._normalize_code(stock_code)
        cached_info = get_stock_info_cache(f"jqdata_{code}")
        if cached_info:
            return cached_info

        securities = self.jq.get_all_securities(types=["stock"])
        if code not in securities.index:
            return {}

        row = securities.loc[code]
        info_data = {
            "symbol": code,
            "name": row.get("display_name"),
            "company_name": row.get("name"),
            "start_date": str(row.get("start_date")),
            "end_date": str(row.get("end_date")),
            "type": row.get("type"),
        }
        info_data = {key: value for key, value in info_data.items() if value not in (None, "")}
        set_stock_info_cache(f"jqdata_{code}", info_data)
        return info_data

    def get_stock_news(self, stock_code: str, limit: int = 5) -> List[Dict[str, Any]]:
        return []

    def get_stock_realtime_data(self, stock_code: str) -> Dict[str, Any]:
        return {}
