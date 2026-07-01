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

ST_MAIN = Path(ROOT) / "hengline" / "streamlit" / "st_main.py"


def read_st_main() -> str:
    return ST_MAIN.read_text(encoding="utf-8-sig")


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

    def test_check_alerts_returns_trigger_status(self):
        def fake_price_data(symbol, period="1m"):
            return pd.DataFrame({"Close": [10.0, 12.5]})

        rows = features.check_alerts(
            fake_price_data,
            [{"symbol": "300502", "above": 12.0, "below": 0, "enabled": True}],
        )
        self.assertEqual(rows[0]["状态"], "触发")
        self.assertEqual(rows[0]["最新价"], 12.5)

    def test_moving_average_backtest_includes_costs_and_risk_metrics(self):
        dates = pd.date_range("2026-01-01", periods=90, freq="B")
        closes = [100 + i * 0.4 for i in range(90)]
        price_data = pd.DataFrame({"Date": dates, "Close": closes})

        result = features.compute_moving_average_backtest(
            price_data,
            fast=5,
            slow=20,
            fee_bps=5,
            slippage_bps=5,
        )

        self.assertTrue(result["success"])
        self.assertIn("max_drawdown_pct", result["metrics"])
        self.assertIn("sharpe", result["metrics"])
        self.assertEqual(result["assumptions"]["fee_bps"], 5)
        self.assertEqual(result["assumptions"]["limit_up_down"], "not_modeled")

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
        html = features.build_html_report(result)
        self.assertIn("<!doctype html>", html)
        self.assertIn("股票 AI 分析报告 - NVDA", html)

    def test_workspace_ui_helpers_are_present(self):
        src = read_st_main()
        for name in [
            "apply_theme_override",
            "quality_badge_html",
            "render_workspace_right_rail",
            "render_agent_workflow_strip",
            "build_overview_cards",
            "render_ai_overview_panel",
        ]:
            self.assertIn(f"def {name}", src)
        self.assertIn("right-rail", src)
        self.assertIn("workflow-strip", src)
        self.assertIn("terminal-card", src)

    def test_main_layout_uses_content_and_right_rail_columns(self):
        src = read_st_main()
        self.assertIn("content_col, rail_col = st.columns([4.2, 1.15]", src)
        self.assertIn("render_workspace_right_rail(", src)
        self.assertIn("st.session_state[\"last_analysis_result\"] = result", src)
        self.assertIn("界面主题", src)
        self.assertIn("深色终端", src)
        self.assertIn("Fallback", src)
        self.assertIn("st.status(\"运行 AI 分析\"", src)


if __name__ == "__main__":
    unittest.main()
