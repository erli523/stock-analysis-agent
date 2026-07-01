#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BaoStock source for A-share historical market data.
"""

from datetime import date, timedelta
from typing import Any, Dict, List

import pandas as pd
import requests
from bs4 import BeautifulSoup

from hengline.logger import debug, error, info, warning
from hengline.tools.cache_tool import get_price_cache, set_price_cache, get_stock_info_cache, set_stock_info_cache


class BaoStockSource:
    """Free A-share historical data source based on baostock."""

    def __init__(self, api_keys: Dict[str, Any] = None):
        try:
            import baostock as bs
        except ImportError as exc:
            raise ImportError("baostock is not installed. Run: pip install baostock") from exc

        self.bs = bs
        self._logged_in = False
        info("Initialized baostock source")

    def _ensure_login(self):
        if self._logged_in:
            return
        login_result = self.bs.login()
        if getattr(login_result, "error_code", "0") != "0":
            raise RuntimeError(f"BaoStock login failed: {getattr(login_result, 'error_msg', '')}")
        self._logged_in = True

    @staticmethod
    def _normalize_code(stock_code: str) -> str:
        code = str(stock_code).strip().lower()
        if code.startswith(("sh.", "sz.")):
            return code
        if code.startswith("sh") and len(code) >= 8:
            return f"sh.{code[-6:]}"
        if code.startswith("sz") and len(code) >= 8:
            return f"sz.{code[-6:]}"
        if code.endswith((".sh", ".ss")):
            return f"sh.{code[:6]}"
        if code.endswith(".sz"):
            return f"sz.{code[:6]}"
        if code.isdigit() and len(code) == 6:
            if code.startswith(("5", "6", "9")):
                return f"sh.{code}"
            return f"sz.{code}"
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
            "1d": "d",
            "1w": "w",
            "1wk": "w",
            "1m": "5",
            "5m": "5",
            "15m": "15",
            "30m": "30",
            "60m": "60",
            "1h": "60",
            "1mo": "m",
        }
        return mapping.get(interval, "d")

    def get_stock_price_data(self, stock_code: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        code = self._normalize_code(stock_code)
        cached_data = get_price_cache(f"baostock_{code}", period, interval)
        if cached_data is not None:
            return cached_data

        self._ensure_login()
        start_date, end_date = self._period_to_dates(period)
        fields = "date,code,open,high,low,close,volume,amount"
        frequency = self._frequency(interval)

        debug(f"BaoStock: fetching price data for {code}, period={period}, interval={interval}")
        result = self.bs.query_history_k_data_plus(
            code,
            fields,
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
            adjustflag="2",
        )
        if result.error_code != "0":
            raise RuntimeError(f"BaoStock query failed: {result.error_msg}")

        rows = []
        while result.next():
            rows.append(result.get_row_data())
        if not rows:
            warning(f"BaoStock: no price data returned for {code}")
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=result.fields)
        df = df.rename(columns={
            "date": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
            "amount": "Amount",
        })
        df["Date"] = pd.to_datetime(df["Date"])
        for column in ["Open", "High", "Low", "Close", "Volume"]:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]

        set_price_cache(f"baostock_{code}", period, interval, df)
        info(f"BaoStock: fetched {len(df)} price rows for {code}")
        return df

    # 每次新增/修改返回字段时递增此版本号，强制旧缓存失效
    _CACHE_VERSION = "v2"

    def get_stock_info(self, stock_code: str) -> Dict[str, Any]:
        code = self._normalize_code(stock_code)
        cache_key = f"baostock_{self._CACHE_VERSION}_{code}"
        cached_info = get_stock_info_cache(cache_key)
        if cached_info:
            return cached_info

        self._ensure_login()
        result = self.bs.query_stock_basic(code=code)
        if result.error_code != "0":
            raise RuntimeError(f"BaoStock basic query failed: {result.error_msg}")

        rows = []
        while result.next():
            rows.append(result.get_row_data())
        if not rows:
            return {}

        data = dict(zip(result.fields, rows[0]))
        info_data = {
            "symbol": code,
            "code": data.get("code"),
            "name": data.get("code_name"),
            "company_name": data.get("code_name"),
            "ipo_date": data.get("ipoDate"),
            "out_date": data.get("outDate"),
            "type": data.get("type"),
            "status": data.get("status"),
        }
        industry = self._get_industry_info(code)
        if industry:
            info_data.update(industry)

        # 合并为一次 API 调用同时获取市值估算和估值指标
        market_metrics = self._get_market_metrics(code)
        if market_metrics:
            info_data.update(market_metrics)

        info_data = {key: value for key, value in info_data.items() if value not in (None, "")}
        set_stock_info_cache(cache_key, info_data)
        return info_data

    def _get_market_metrics(self, code: str) -> Dict[str, Any]:
        """一次 BaoStock 调用同时获取市值估算（turn×volume×close）和估值比率（PE/PB/PS）。"""
        from datetime import date, timedelta
        today = date.today().isoformat()
        start = (date.today() - timedelta(days=10)).isoformat()
        try:
            result = self.bs.query_history_k_data_plus(
                code,
                "date,close,volume,turn,peTTM,pbMRQ,psTTM",
                start_date=start,
                end_date=today,
                frequency="d",
                adjustflag="2",
            )
            if result.error_code != "0":
                warning(f"BaoStock market metrics query error for {code}: [{result.error_code}] {result.error_msg}")
                return {}
            rows = []
            while result.next():
                rows.append(result.get_row_data())
            if not rows:
                warning(f"BaoStock market metrics: no rows returned for {code}")
                return {}
            df = pd.DataFrame(rows, columns=result.fields)
            num_cols = ("close", "volume", "turn", "peTTM", "pbMRQ", "psTTM")
            for col in num_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df[df["turn"] > 0].dropna(subset=["close", "volume", "turn"])
            if df.empty:
                warning(f"BaoStock market metrics: df empty after turn>0 filter for {code}")
                return {}
            latest = df.iloc[-1]
            metrics: Dict[str, Any] = {}
            # 总市值估算
            total_shares = latest["volume"] / (latest["turn"] / 100)
            metrics["market_cap"] = f"{total_shares * latest['close'] / 1e8:.2f}亿元"
            # 估值比率
            for src, dst in (("peTTM", "pe_ratio"), ("pbMRQ", "pb_ratio"), ("psTTM", "ps_ratio")):
                val = latest.get(src)
                if pd.notna(val):
                    metrics[dst] = round(float(val), 2)
            return metrics
        except Exception as exc:
            warning(f"BaoStock market metrics failed for {code}: {exc}")
            return {}

    def _get_industry_info(self, code: str) -> Dict[str, Any]:
        try:
            result = self.bs.query_stock_industry(code=code)
            if result.error_code != "0":
                return {}
            rows = []
            while result.next():
                rows.append(result.get_row_data())
            if not rows:
                return {}
            data = dict(zip(result.fields, rows[0]))
            industry = data.get("industry") or data.get("industryClassification")
            return {
                "sector": industry,
                "industry": industry,
                "exchange": code.split(".")[0].upper(),
            }
        except Exception as exc:
            warning(f"BaoStock industry query failed for {code}: {exc}")
            return {}

    def get_stock_news(self, stock_code: str, limit: int = 5) -> List[Dict[str, Any]]:
        code = self._normalize_code(stock_code)
        sina_symbol = code.replace(".", "")
        url = f"https://vip.stock.finance.sina.com.cn/corp/go.php/vCB_AllNewsStock/symbol/{sina_symbol}.phtml"
        try:
            response = requests.get(
                url,
                timeout=10,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
            )
            response.raise_for_status()
            response.encoding = "gbk"
            soup = BeautifulSoup(response.text, "html.parser")
            news = []
            seen = set()
            for link in soup.select("a"):
                title = link.get_text(strip=True)
                href = link.get("href", "")
                if not title or not href:
                    continue
                if len(title) < 8:
                    continue
                if "sina.com.cn" not in href:
                    continue
                if href in seen:
                    continue
                seen.add(href)
                news.append({
                    "title": title,
                    "source": "Sina Finance",
                    "summary": title,
                    "link": href,
                })
                if len(news) >= limit:
                    break
            return news
        except Exception as exc:
            warning(f"BaoStock/Sina news query failed for {code}: {exc}")
            return []

    def get_stock_realtime_data(self, stock_code: str) -> Dict[str, Any]:
        return {}

    def get_financial_data(self, stock_code: str) -> Dict[str, pd.DataFrame]:
        code = self._normalize_code(stock_code)
        self._ensure_login()

        tables = {
            "income_statement": self.bs.query_profit_data,
            "balance_sheet": self.bs.query_balance_data,
            "cash_flow": self.bs.query_cash_flow_data,
            "financial_ratios": self.bs.query_dupont_data,
            "growth": self.bs.query_growth_data,
            "operation": self.bs.query_operation_data,
        }

        result: Dict[str, pd.DataFrame] = {}
        today = date.today()
        periods = []
        for offset in range(0, 4):
            month_index = today.month - 1 - offset * 3
            year = today.year + month_index // 12
            quarter = month_index % 12 // 3 + 1
            periods.append((year, quarter))

        for table_name, query_func in tables.items():
            rows = []
            fields = None
            for year, quarter in periods:
                try:
                    query_result = query_func(code=code, year=year, quarter=quarter)
                    if query_result.error_code != "0":
                        continue
                    fields = query_result.fields
                    while query_result.next():
                        rows.append(query_result.get_row_data())
                except Exception as exc:
                    debug(f"BaoStock financial query failed for {code} {table_name} {year}Q{quarter}: {exc}")
            if rows and fields:
                df = pd.DataFrame(rows, columns=fields).drop_duplicates()
                for column in df.columns:
                    if column not in {"code", "pubDate", "statDate"}:
                        df[column] = pd.to_numeric(df[column], errors="coerce")
                result[table_name] = df

        if result:
            info(f"BaoStock: fetched {len(result)} financial tables for {code}")
        else:
            warning(f"BaoStock: no financial data returned for {code}")
        return result
