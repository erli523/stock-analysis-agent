#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for Streamlit product feature helpers."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from hengline.streamlit import st_product_features as features


class ProductFeatureHelperTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self.original_user_dir = features.USER_DATA_DIR
        self.original_favorites_file = features.FAVORITES_FILE
        self.original_output_dir = features.OUTPUT_DIR
        features.USER_DATA_DIR = self.tmp_path / "user"
        features.FAVORITES_FILE = features.USER_DATA_DIR / "favorites.json"
        features.OUTPUT_DIR = self.tmp_path / "output"

    def tearDown(self):
        features.USER_DATA_DIR = self.original_user_dir
        features.FAVORITES_FILE = self.original_favorites_file
        features.OUTPUT_DIR = self.original_output_dir
        self._tmp.cleanup()

    def test_favorites_are_deduplicated_and_uppercase(self):
        features.save_favorites(["300502", "300502", "nvda", ""])
        self.assertEqual(features.load_favorites(), ["300502", "NVDA"])
        features.add_favorite("aapl")
        self.assertEqual(features.load_favorites(), ["300502", "NVDA", "AAPL"])
        features.remove_favorite("nvda")
        self.assertEqual(features.load_favorites(), ["300502", "AAPL"])

    def test_technical_indicator_columns_are_calculated(self):
        dates = pd.date_range("2026-01-01", periods=40, freq="B")
        price_data = pd.DataFrame(
            {
                "Date": dates,
                "Open": range(100, 140),
                "High": range(101, 141),
                "Low": range(99, 139),
                "Close": range(100, 140),
                "Volume": [1000000 + i * 1000 for i in range(40)],
            }
        )
        result = features.calculate_technical_indicators(price_data)
        for column in ["RSI14", "MACD", "MACD_SIGNAL", "MACD_HIST", "BOLL_UPPER", "BOLL_LOWER"]:
            self.assertIn(column, result.columns)
        self.assertTrue(result["MACD"].notna().any())
        self.assertTrue(result["BOLL_UPPER"].notna().any())

    def test_analysis_history_save_and_load(self):
        report = {
            "success": True,
            "stock_code": "300502",
            "analysis_time": "2026-07-01T12:00:00",
            "final_recommendation": {
                "investment_recommendation": "持有",
                "comprehensive_metrics": {"risk_level": "medium"},
            },
        }
        saved = features.save_analysis_result(report, "300502")
        self.assertIsNotNone(saved)
        self.assertTrue(saved.exists())
        loaded = features.load_analysis_history()
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["stock_code"], "300502")
        self.assertEqual(loaded[0]["recommendation"], "持有")

    def test_json_list_helpers_and_agent_selection(self):
        path = self.tmp_path / "user" / "portfolio.json"
        rows = [{"symbol": "300502", "quantity": 100}]
        features.save_json_list(path, rows)
        self.assertEqual(features.load_json_list(path), rows)
        self.assertEqual(
            features.normalize_agent_selection(["技术面", "资金流", "不存在"]),
            ["TechnicalAgent", "FundFlowAgent"],
        )

    def test_markdown_report_contains_recommendation_and_findings(self):
        result = {
            "success": True,
            "stock_code": "NVDA",
            "analysis_time": "2026-07-01T12:00:00",
            "final_recommendation": {
                "investment_recommendation": "买入",
                "analysis_summary": "盈利和趋势表现较强。",
                "key_findings": ["AI 需求强劲"],
                "comprehensive_metrics": {"composite_score": 82, "risk_level": "medium"},
            },
            "detailed_results": {
                "TechnicalAgent": {"key_findings": ["趋势向上"]},
            },
        }
        markdown = features.build_markdown_report(result)
        self.assertIn("股票 AI 分析报告 - NVDA", markdown)
        self.assertIn("最终建议：买入", markdown)
        self.assertIn("AI 需求强劲", markdown)
        self.assertIn("趋势向上", markdown)


if __name__ == "__main__":
    unittest.main()
