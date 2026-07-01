#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified stock data manager.
"""

import os
import time
from typing import Any, Dict, List

import pandas as pd

from config.config import get_api_keys_config
from hengline.logger import debug, error, info, warning
from hengline.stock.simulated.stock_data_manager import stock_data_manager
from hengline.stock.sources.akshare_source import AKShareSource
from hengline.stock.sources.alltick_source import AlltickSource
from hengline.stock.sources.alpha_vantage_source import AlphaVantageSource
from hengline.stock.sources.baostock_source import BaoStockSource
from hengline.stock.sources.iex_cloud_source import IEXCloudSource
from hengline.stock.sources.jqdata_source import JQDataSource
from hengline.stock.sources.massive_source import MassiveSource
from hengline.stock.sources.yahoo_direct_source import YahooDirectSource
from hengline.stock.sources.yfinance_source import YFinanceSource


class StockDataManager:
    """Fetch stock data from multiple sources with fallback."""

    def __init__(self):
        self.api_keys: Dict[str, Any] = get_api_keys_config()
        self._data_sources: Dict[str, Any] = {}
        self._last_request_time: Dict[str, float] = {}
        self._consecutive_failures: Dict[str, int] = {}
        self._failed_sources: Dict[str, tuple[float, int]] = {}
        self._cache: Dict[str, tuple[Any, float]] = {}
        self._cache_ttl = 300
        self._init_data_sources()

    def _valid_key(self, value: Any) -> bool:
        return isinstance(value, str) and bool(value.strip()) and not value.strip().startswith("$")

    def _enabled(self, name: str, default: bool = True) -> bool:
        value = os.environ.get(f"{name.upper()}_ENABLED")
        if value is None:
            return default
        return value.lower() in {"1", "true", "yes", "on"}

    def _init_data_sources(self):
        info("Initializing stock data sources...")

        if self._enabled("akshare", False):
            try:
                self._data_sources["akshare"] = AKShareSource()
                debug("Initialized akshare source")
            except Exception as exc:
                error(f"Failed to initialize akshare source: {exc}")
        else:
            info("AKShare disabled by AKSHARE_ENABLED=false")

        try:
            key = self.api_keys.get("alltick", "")
            if self._enabled("alltick", True) and self._valid_key(key):
                self._data_sources["alltick"] = AlltickSource(self.api_keys)
                info("Initialized alltick source")
            else:
                info("Alltick source is disabled or API key is not configured")
        except Exception as exc:
            error(f"Failed to initialize alltick source: {exc}")

        if self._enabled("baostock", True):
            try:
                self._data_sources["baostock"] = BaoStockSource(self.api_keys)
                info("Initialized baostock source")
            except Exception as exc:
                error(f"Failed to initialize baostock source: {exc}")
        else:
            info("BaoStock disabled by BAOSTOCK_ENABLED=false")

        if self._enabled("jqdata", False):
            try:
                self._data_sources["jqdata"] = JQDataSource(self.api_keys)
                info("Initialized jqdata source")
            except Exception as exc:
                error(f"Failed to initialize jqdata source: {exc}")
        else:
            info("JQData disabled by JQDATA_ENABLED=false")

        try:
            key = self.api_keys.get("massive", "")
            if self._enabled("massive", True) and self._valid_key(key):
                self._data_sources["massive"] = MassiveSource(self.api_keys)
                info("Initialized massive source")
            else:
                info("Massive source is disabled or API key is not configured")
        except Exception as exc:
            error(f"Failed to initialize massive source: {exc}")

        if self._enabled("yfinance", False):
            try:
                self._data_sources["yfinance"] = YFinanceSource(self.api_keys)
                info("Initialized yfinance source")
            except Exception as exc:
                error(f"Failed to initialize yfinance source: {exc}")
        else:
            info("YFinance disabled by YFINANCE_ENABLED=false")

        if self._enabled("yahoo_direct", False):
            try:
                self._data_sources["yahoo_direct"] = YahooDirectSource(self.api_keys)
                info("Initialized yahoo_direct source")
            except Exception as exc:
                error(f"Failed to initialize yahoo_direct source: {exc}")
        else:
            info("Yahoo Direct disabled by YAHOO_DIRECT_ENABLED=false")

        try:
            key = self.api_keys.get("alpha_vantage", "")
            if self._valid_key(key):
                self._data_sources["alpha_vantage"] = AlphaVantageSource(key)
                info("Initialized alpha_vantage source")
            else:
                info("Alpha Vantage API key is not configured")
        except Exception as exc:
            error(f"Failed to initialize alpha_vantage source: {exc}")

        try:
            key = self.api_keys.get("iex_cloud", "")
            if self._valid_key(key):
                self._data_sources["iex_cloud"] = IEXCloudSource({"iex_cloud": key})
                info("Initialized iex_cloud source")
            else:
                info("IEX Cloud API key is not configured")
        except Exception as exc:
            error(f"Failed to initialize iex_cloud source: {exc}")

        for source_name in self._data_sources:
            self._last_request_time[source_name] = 0
            self._consecutive_failures[source_name] = 0

        info(f"Loaded {len(self._data_sources)} stock data sources")

    def _get_source_instance(self, source_name: str):
        return self._data_sources.get(source_name)

    def _normalize_code_for_source(self, source_name: str, stock_code: str) -> str:
        if source_name not in {"yfinance", "yahoo_direct"}:
            return stock_code
        if not isinstance(stock_code, str):
            return stock_code

        code = stock_code.strip()
        lower_code = code.lower()
        if lower_code.endswith((".sz", ".ss", ".hk")):
            return code
        if lower_code.startswith("sz") and len(code) >= 8:
            return f"{code[-6:]}.SZ"
        if lower_code.startswith("sh") and len(code) >= 8:
            return f"{code[-6:]}.SS"
        if code.isdigit() and len(code) == 6:
            if code.startswith(("0", "2", "3")):
                return f"{code}.SZ"
            if code.startswith(("5", "6", "9")):
                return f"{code}.SS"
        return code

    def _rate_limit_delay(self, source_name: str):
        intervals = {
            "akshare": 3.0,
            "alltick": 1.0,
            "baostock": 0.5,
            "jqdata": 0.5,
            "massive": 0.25,
            "alpha_vantage": 12.0,
            "iex_cloud": 0.2,
            "yfinance": 0.5,
            "yahoo_direct": 0.5,
        }
        interval = intervals.get(source_name, 1.0)
        now = time.time()
        elapsed = now - self._last_request_time.get(source_name, 0)
        if elapsed < interval:
            time.sleep(interval - elapsed)
        self._last_request_time[source_name] = time.time()

    def _try_data_source(self, source_name: str, method_name: str, *args, **kwargs):
        source = self._get_source_instance(source_name)
        if not source:
            warning(f"Data source unavailable: {source_name}")
            return None

        method = getattr(source, method_name, None)
        if not method:
            warning(f"Data source {source_name} does not support {method_name}")
            return None

        try:
            self._rate_limit_delay(source_name)
            call_args = list(args)
            if call_args and method_name in {
                "get_stock_price_data",
                "get_stock_info",
                "get_stock_news",
                "get_financial_data",
                "get_stock_realtime_data",
            }:
                call_args[0] = self._normalize_code_for_source(source_name, call_args[0])

            debug(f"Using {source_name}.{method_name}")
            result = method(*call_args, **kwargs)
            if self._is_valid_result(result, method_name):
                self._consecutive_failures[source_name] = 0
                return result

            warning(f"{source_name} returned empty data for {method_name}")
            return None
        except Exception as exc:
            self._consecutive_failures[source_name] = self._consecutive_failures.get(source_name, 0) + 1
            error(f"Data source {source_name} failed on {method_name}: {exc}")
            return None

    def _is_valid_result(self, result: Any, method_name: str) -> bool:
        if result is None:
            return False
        if method_name == "get_stock_price_data":
            return isinstance(result, pd.DataFrame) and not result.empty
        if method_name in {"get_stock_info", "get_financial_data", "get_stock_realtime_data"}:
            return isinstance(result, dict) and bool(result)
        if method_name == "get_stock_news":
            return isinstance(result, list)
        return True

    def _cache_key(self, method_name: str, *args, **kwargs) -> str:
        key_parts = [method_name]
        key_parts.extend(str(arg) for arg in args)
        key_parts.extend(f"{key}={value}" for key, value in sorted(kwargs.items()))
        return ":".join(key_parts)

    def _is_a_stock_code(self, stock_code: str) -> bool:
        if not isinstance(stock_code, str):
            return False
        code = stock_code.strip().lower()
        return code.startswith(("sh", "sz")) or (code.isdigit() and len(code) == 6)

    def _source_priority(self, stock_code: str) -> List[str]:
        if self._is_a_stock_code(stock_code):
            priority = ["alltick", "baostock", "jqdata", "akshare", "yfinance", "yahoo_direct", "alpha_vantage"]
            return priority

        return ["massive", "alltick", "alpha_vantage", "iex_cloud", "yfinance", "yahoo_direct"]

    def _available_sources(self, source_priority: List[str]) -> List[str]:
        now = time.time()
        available = []
        for source_name in source_priority:
            if source_name not in self._data_sources:
                continue
            failed = self._failed_sources.get(source_name)
            if failed:
                fail_time, cool_down = failed
                if now - fail_time < cool_down:
                    debug(f"Skipping cooling-down source: {source_name}")
                    continue
                del self._failed_sources[source_name]
            available.append(source_name)
        return available

    def _mark_source_failed(self, source_name: str):
        failures = self._consecutive_failures.get(source_name, 0)
        cool_down = min(300, 30 * max(1, failures))
        self._failed_sources[source_name] = (time.time(), cool_down)

    def _load_mock_data(self, method_name: str, *args, **kwargs):
        stock_code = args[0] if args else ""
        debug(f"Loading mock data for {stock_code}: {method_name}")
        try:
            if method_name == "get_stock_price_data":
                period = args[1] if len(args) > 1 else kwargs.get("period", "1y")
                interval = args[2] if len(args) > 2 else kwargs.get("interval", "1d")
                return stock_data_manager.get_stock_price_data(stock_code, period, interval)
            if method_name == "get_stock_info":
                return stock_data_manager.get_stock_info(stock_code)
            if method_name == "get_stock_news":
                return stock_data_manager.get_stock_news(stock_code)
            if method_name == "get_financial_data":
                return stock_data_manager.get_financial_data(stock_code)
            if method_name == "get_stock_realtime_data":
                return stock_data_manager.get_stock_realtime_data(stock_code)
        except Exception as exc:
            error(f"Mock data failed for {method_name}: {exc}")
        return None

    def _get_data_with_fallback(self, method_name: str, *args, **kwargs):
        cache_key = self._cache_key(method_name, *args, **kwargs)
        cached = self._cache.get(cache_key)
        if cached:
            data, timestamp = cached
            if time.time() - timestamp < self._cache_ttl:
                return data
            del self._cache[cache_key]

        stock_code = args[0] if args else ""
        source_priority = self._source_priority(stock_code)
        info(f"Source priority for {stock_code}: {source_priority}")
        available_sources = self._available_sources(source_priority)

        max_attempts = 2
        attempts = 0
        for source_name in available_sources:
            if attempts >= max_attempts:
                break
            attempts += 1
            result = self._try_data_source(source_name, method_name, *args, **kwargs)
            if result is not None:
                self._cache[cache_key] = (result, time.time())
                return result
            self._mark_source_failed(source_name)

        info("Online data sources failed; returning mock data")
        mock_result = self._load_mock_data(method_name, *args, **kwargs)
        if mock_result is not None:
            self._cache[cache_key] = (mock_result, time.time())
            return mock_result

        defaults = {
            "get_stock_price_data": pd.DataFrame(),
            "get_stock_info": {},
            "get_stock_news": [],
            "get_financial_data": {},
            "get_stock_realtime_data": {},
        }
        return defaults.get(method_name)

    def get_stock_price_data(self, stock_code: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        return self._get_data_with_fallback("get_stock_price_data", stock_code, period, interval)

    def get_stock_info(self, stock_code: str) -> Dict[str, Any]:
        return self._get_data_with_fallback("get_stock_info", stock_code)

    def get_stock_news(self, stock_code: str, limit: int = 5) -> List[Dict[str, Any]]:
        return self._get_data_with_fallback("get_stock_news", stock_code, limit)

    def get_financial_data(self, stock_code: str) -> Dict[str, pd.DataFrame]:
        financial_data = self._get_data_with_fallback("get_financial_data", stock_code)
        if not isinstance(financial_data, dict):
            return {}

        valid_financial_data = {}
        for key, df in financial_data.items():
            if isinstance(df, pd.DataFrame) and not df.empty and len(df.columns) > 0:
                valid_financial_data[key] = df
        return valid_financial_data

    def get_stock_realtime_data(self, stock_code: str) -> Dict[str, Any]:
        return self._get_data_with_fallback("get_stock_realtime_data", stock_code)


def get_stock_price_data(stock_code: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    return manager.get_stock_price_data(stock_code, period, interval)


def get_stock_info(stock_code: str) -> Dict[str, Any]:
    return manager.get_stock_info(stock_code)


def get_stock_news(stock_code: str, limit: int = 5) -> List[Dict[str, Any]]:
    return manager.get_stock_news(stock_code, limit)


def get_financial_data(stock_code: str) -> Dict[str, pd.DataFrame]:
    return manager.get_financial_data(stock_code)


manager = StockDataManager()
