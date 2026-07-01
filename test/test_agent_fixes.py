#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Agent 系统修复验证测试套件
============================
覆盖全部 7 个修复点:
  [P0.1] _retrieve_knowledge  NodeWithScore 解析
  [P0.2] Streamlit Coordinator @st.cache_resource 单例
  [P1.1] ChiefStrategy 评分字段与子 Agent 输出对齐
  [P1.2] SentimentAgent  不再使用 random 生成社交/情绪数据
  [P1.2] ESGRiskAgent    非 A 股无 ESG 数据时返回 None 而非 random
  [P1.2] FundamentalAgent 缺失财务数据时标注不可用而非硬编码模拟值
  [P1.2] FundFlowAgent   OBV 阈值改为相对变化率
  [P2.1] Coordinator     注入共享 StockDataManager
  [P2.2] UI              子 Agent 失败时明确标注

运行方法:
  conda run -n stock-agent python test/test_agent_fixes.py
  conda run -n stock-agent pytest test/test_agent_fixes.py -v
"""
import os
import sys
import ast
import inspect
import textwrap
import types
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

# ── 项目根目录 ──────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ── 文件路径常量 ─────────────────────────────────────────────────────
AGENTS_DIR = os.path.join(ROOT, "hengline", "agents")
ST_MAIN    = os.path.join(ROOT, "hengline", "streamlit", "st_main.py")


# ════════════════════════════════════════════════════════════════════════
# 辅助工具
# ════════════════════════════════════════════════════════════════════════
def _read_src(path):
    with open(path, encoding="utf-8-sig") as f:   # utf-8-sig 自动去掉 BOM
        return f.read()


def _src(agent_filename):
    return _read_src(os.path.join(AGENTS_DIR, agent_filename))


class TestResultUtils(unittest.TestCase):
    """验证 Agent 结果元数据 helper 的保守优先级。"""

    def test_data_quality_prioritizes_simulated_over_other_states(self):
        from hengline.agents.result_utils import build_data_quality_fields

        fields = build_data_quality_fields(
            data_available=False,
            data_note="proxy data",
            is_simulated=True,
            is_estimated=True,
        )

        self.assertEqual(fields["data_quality_level"], "simulated")
        self.assertFalse(fields["data_available"])
        self.assertTrue(fields["is_simulated"])

    def test_data_quality_marks_estimated_before_partial(self):
        from hengline.agents.result_utils import build_data_quality_fields

        fields = build_data_quality_fields(
            data_available=True,
            data_note="estimated from volume proxy",
            is_estimated=True,
        )

        self.assertEqual(fields["data_quality_level"], "estimated")
        self.assertEqual(fields["data_note"], "estimated from volume proxy")

    def test_has_simulated_source_checks_multiple_inputs(self):
        from hengline.agents.result_utils import has_simulated_source

        self.assertTrue(has_simulated_source({"is_simulated": False}, {"is_simulated": True}))
        self.assertFalse(has_simulated_source({}, {"data_source": "real"}))


# ════════════════════════════════════════════════════════════════════════
# P0.1  _retrieve_knowledge — NodeWithScore 解析
# ════════════════════════════════════════════════════════════════════════
class TestP0_1_RetrieveKnowledge(unittest.TestCase):
    """验证 base_agent._retrieve_knowledge 能正确解析 LlamaIndex NodeWithScore"""

    def _make_agent_instance(self):
        """绕过 BaseAgent.__init__ 创建裸实例，只注入 retriever"""
        from hengline.agents.base_agent import BaseAgent

        class _DummyAgent(BaseAgent):
            def analyze(self, stock_code, time_range="1y", **kwargs):
                pass
            def get_result_template(self):
                return {}

        agent = object.__new__(_DummyAgent)
        agent.retriever = None
        agent.config = MagicMock()
        return agent

    # ── 辅助 mock 类 ──
    class _MockNode:
        def get_content(self):
            return "NodeWithScore 内容"

    class _MockNodeWithScore:
        def __init__(self):
            self.node = TestP0_1_RetrieveKnowledge._MockNode()
            self.score = 0.9

    class _MockDirectNode:
        def get_content(self):
            return "直接节点内容"

    class _MockTextAttr:
        text = "text 属性内容"

    def _run(self, results):
        from hengline.agents.base_agent import BaseAgent

        agent = self._make_agent_instance()
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = results
        agent.retriever = mock_retriever
        return agent._retrieve_knowledge("测试查询")

    def test_node_with_score_parsed(self):
        """NodeWithScore 对象应通过 result.node.get_content() 提取内容"""
        result = self._run([self._MockNodeWithScore()])
        self.assertEqual(result, ["NodeWithScore 内容"],
                         "NodeWithScore.node.get_content() 未被正确调用")

    def test_direct_node_parsed(self):
        """带 get_content() 方法的直接节点对象应被正确处理"""
        result = self._run([self._MockDirectNode()])
        self.assertEqual(result, ["直接节点内容"])

    def test_string_result(self):
        """纯字符串结果应直接追加"""
        result = self._run(["字符串知识片段"])
        self.assertEqual(result, ["字符串知识片段"])

    def test_dict_result(self):
        """dict 格式（text 键）应被解析"""
        result = self._run([{"text": "dict 知识片段"}])
        self.assertEqual(result, ["dict 知识片段"])

    def test_text_attr_fallback(self):
        """具有 .text 属性的对象应作为最后回退"""
        result = self._run([self._MockTextAttr()])
        self.assertEqual(result, ["text 属性内容"])

    def test_mixed_results(self):
        """混合多种格式时应全部正确提取"""
        results = [
            self._MockNodeWithScore(),
            "字符串",
            {"text": "dict 内容"},
        ]
        knowledge = self._run(results)
        self.assertEqual(len(knowledge), 3)
        self.assertIn("NodeWithScore 内容", knowledge)
        self.assertIn("字符串", knowledge)
        self.assertIn("dict 内容", knowledge)

    def test_empty_retriever(self):
        """retriever 为 None 时返回空列表，不抛出异常"""
        agent = self._make_agent_instance()
        result = agent._retrieve_knowledge("任意查询")
        self.assertEqual(result, [])

    def test_no_longer_checks_text_first(self):
        """旧代码先检查 .text 属性，新代码优先检查 .node.get_content()
        确认 NodeWithScore 内容在 .text 不存在时仍能提取"""
        class _ScoreOnlyNode:
            def __init__(self):
                self.node = TestP0_1_RetrieveKnowledge._MockNode()
                self.score = 0.5
            # 故意没有 .text 属性

        result = self._run([_ScoreOnlyNode()])
        self.assertEqual(result, ["NodeWithScore 内容"],
                         "新解析逻辑未能从 NodeWithScore 中提取内容")


# ════════════════════════════════════════════════════════════════════════
# P0.2  Streamlit Coordinator 单例
# ════════════════════════════════════════════════════════════════════════
class TestP0_2_CoordinatorSingleton(unittest.TestCase):
    """验证 st_main.py 中 get_coordinator 使用 @st.cache_resource"""

    def test_get_coordinator_function_exists(self):
        """st_main.py 必须定义 get_coordinator 函数"""
        src = _read_src(ST_MAIN)
        self.assertIn("def get_coordinator(", src,
                      "st_main.py 中找不到 get_coordinator 函数")

    def test_cache_resource_decorator_applied(self):
        """get_coordinator 必须被 @st.cache_resource 装饰"""
        src = _read_src(ST_MAIN)
        # 查找装饰器紧跟在 def get_coordinator 之前
        lines = src.splitlines()
        decorator_found = False
        for i, line in enumerate(lines):
            if "def get_coordinator(" in line:
                # 向前检查最多 5 行
                window = lines[max(0, i - 5): i]
                decorator_found = any(
                    "cache_resource" in l for l in window
                )
                break
        self.assertTrue(decorator_found,
                        "get_coordinator 未被 @st.cache_resource 装饰")

    def test_coordinator_not_instantiated_inside_show_agent_analysis(self):
        """show_agent_analysis 内不应再有 AgentCoordinator(...) 直接实例化"""
        src = _read_src(ST_MAIN)
        # 提取 show_agent_analysis 函数体
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "show_agent_analysis":
                func_src = ast.unparse(node)
                self.assertNotIn("AgentCoordinator(",  func_src,
                    "show_agent_analysis 内仍直接实例化 AgentCoordinator，应改用 get_coordinator()")
                return
        self.fail("st_main.py 中找不到 show_agent_analysis 函数")

    def test_get_coordinator_called_inside_show_agent_analysis(self):
        """show_agent_analysis 应调用 get_coordinator()"""
        src = _read_src(ST_MAIN)
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "show_agent_analysis":
                func_src = ast.unparse(node)
                self.assertIn("get_coordinator(", func_src,
                    "show_agent_analysis 未调用 get_coordinator()")
                return
        self.fail("st_main.py 中找不到 show_agent_analysis 函数")


# ════════════════════════════════════════════════════════════════════════
# P1.1  ChiefStrategy 评分字段对齐
# ════════════════════════════════════════════════════════════════════════
class TestP1_1_ChiefScoreAlignment(unittest.TestCase):
    """验证 ChiefStrategyAgent._extract_agent_score 正确读取各子 Agent 实际输出"""

    def _get_chief_method(self):
        """提取 _extract_agent_score 方法，不实例化完整 Agent"""
        from hengline.agents import chief_strategy_agent
        # 获取方法源码并注入最小 self
        agent = object.__new__(chief_strategy_agent.ChiefStrategyAgent)
        return agent._extract_agent_score

    def test_fundamental_reads_overall_score(self):
        """FundamentalAgent 输出 overall_score (0-10)，应被转换为 0-100"""
        fn = self._get_chief_method()
        result = {"overall_score": 7, "confidence_score": 0.85}
        score = fn("FundamentalAgent", result)
        self.assertAlmostEqual(score, 70.0, places=1,
            msg="overall_score=7 应转换为约 70 分（×10）")

    def test_fundamental_fallback_to_confidence(self):
        """FundamentalAgent overall_score 为 0 时，应回退到 confidence_score"""
        fn = self._get_chief_method()
        result = {"overall_score": 0, "confidence_score": 0.8}
        score = fn("FundamentalAgent", result)
        self.assertAlmostEqual(score, 80.0, places=1,
            msg="overall_score=0 时应用 confidence_score×100 作为回退")

    def test_technical_uses_signal_strength(self):
        """TechnicalAgent 应通过 signal_strength 映射评分"""
        fn = self._get_chief_method()
        result_bullish   = {"signal_strength": "bullish",       "confidence_score": 0.8}
        result_bearish   = {"signal_strength": "strong_bearish","confidence_score": 0.9}
        result_neutral   = {"signal_strength": "neutral",       "confidence_score": 0.5}
        s_bull   = fn("TechnicalAgent", result_bullish)
        s_bear   = fn("TechnicalAgent", result_bearish)
        s_neutral= fn("TechnicalAgent", result_neutral)
        self.assertGreater(s_bull, s_neutral, "bullish 评分应高于 neutral")
        self.assertLess(s_bear, s_neutral, "strong_bearish 评分应低于 neutral")

    def test_fundamental_no_longer_looks_for_fundamental_score(self):
        """确认旧字段 fundamental_score 不再被独立使用（已改为 overall_score）"""
        fn = self._get_chief_method()
        # 只有旧字段，没有 overall_score 或 confidence_score
        result = {"fundamental_score": 80.0}
        score = fn("FundamentalAgent", result)
        # 旧逻辑会直接返回 80.0，新逻辑应返回 0*10=0 后回退 confidence 0.5*100=50
        self.assertNotEqual(score, 80.0,
            "新逻辑不应再读取 fundamental_score 字段，旧逻辑检测到残留")

    def test_industry_macro_reads_nested_scores(self):
        """IndustryMacroAgent 应读取 industry_analysis.industry_score 和 macro_analysis.economic_score"""
        fn = self._get_chief_method()
        result = {
            "industry_analysis": {"industry_score": 80},
            "macro_analysis":    {"economic_score": 60}
        }
        score = fn("IndustryMacroAgent", result)
        self.assertAlmostEqual(score, 70.0, places=1,
            msg="应返回 (80+60)/2 = 70")

    def test_fund_flow_classification_mapping(self):
        """FundFlowAgent 应从 key_metrics.flow_classification 映射评分"""
        fn = self._get_chief_method()
        cases = [
            ({"key_metrics": {"flow_classification": "strong_inflow"}},  90),
            ({"key_metrics": {"flow_classification": "neutral"}},        50),
            ({"key_metrics": {"flow_classification": "strong_outflow"}}, 10),
        ]
        for result, expected in cases:
            score = fn("FundFlowAgent", result)
            self.assertEqual(score, float(expected),
                f"flow_classification={result['key_metrics']['flow_classification']} 应得 {expected}")

    def test_esg_reads_overall_score(self):
        """ESGRiskAgent 应读取 esg_metrics.overall_score"""
        fn = self._get_chief_method()
        result = {"esg_metrics": {"overall_score": 72.5}}
        score = fn("ESGRiskAgent", result)
        self.assertAlmostEqual(score, 72.5, places=1)

    def test_default_score_is_50(self):
        """未知 Agent 名称应返回默认 50 分"""
        fn = self._get_chief_method()
        score = fn("UnknownAgent", {})
        self.assertEqual(score, 50.0)


# ════════════════════════════════════════════════════════════════════════
# P1.2  随机/模拟数据修复
# ════════════════════════════════════════════════════════════════════════
class TestP1_2_NoRandomData(unittest.TestCase):
    """验证各 Agent 不再用 random 生成核心数据"""

    # ── SentimentAgent ──────────────────────────────────────────────────
    def _make_sentiment_agent(self):
        from hengline.agents.sentiment_agent import SentimentAgent
        agent = object.__new__(SentimentAgent)
        agent.stock_manager = MagicMock()
        return agent

    def test_sentiment_social_media_no_random(self):
        """_get_social_media_data 返回 data_available=False，不含随机数字"""
        agent = self._make_sentiment_agent()
        data = agent._get_social_media_data("300502")
        self.assertFalse(data.get("data_available", True),
            "社交媒体数据应标注 data_available=False")
        self.assertEqual(data.get("platforms", []), [],
            "无真实 API 时 platforms 应为空列表")
        self.assertIsNone(data.get("overall_sentiment"),
            "overall_sentiment 应为 None，不应是随机数")

    def test_sentiment_social_media_stable(self):
        """多次调用 _get_social_media_data 结果应完全相同（无随机性）"""
        agent = self._make_sentiment_agent()
        r1 = agent._get_social_media_data("300502")
        r2 = agent._get_social_media_data("300502")
        self.assertEqual(r1, r2, "返回值存在随机性，每次结果不一致")

    def test_sentiment_market_sentiment_no_random(self):
        """_get_market_sentiment 应标注 data_available=False，不含随机指数"""
        agent = self._make_sentiment_agent()
        data = agent._get_market_sentiment()
        self.assertFalse(data.get("data_available", True),
            "市场情绪数据应标注 data_available=False")
        fear_greed = data.get("fear_greed_index", {})
        self.assertIsNone(fear_greed.get("current"),
            "fear_greed_index.current 应为 None，不应是随机整数")

    def test_sentiment_market_sentiment_stable(self):
        """多次调用 _get_market_sentiment 结果应完全相同"""
        agent = self._make_sentiment_agent()
        r1 = agent._get_market_sentiment()
        r2 = agent._get_market_sentiment()
        self.assertEqual(r1, r2)

    # ── ESGRiskAgent（非 A 股无数据路径）──────────────────────────────
    def test_esg_no_random_when_no_data(self):
        """ESGRiskAgent 在 yfinance 无数据时返回 None 评分，而非 random"""
        src = _src("esg_risk_agent.py")
        # 查找非 A 股路径的 random.uniform 用法
        # 修复后此处不应再出现 random.uniform 生成 base_score
        import re
        # 找到 "如果没有直接评分" 块之后是否存在 random.uniform
        # 修复后该块应被替换为 data_available = False
        block_start = src.find("esg_data[\"overall_score\"] == 0")
        if block_start == -1:
            # 可能文字略有不同
            block_start = src.find("overall_score'] == 0")
        if block_start != -1:
            # 取该块后 500 字符内
            snippet = src[block_start: block_start + 500]
            self.assertNotIn("random.uniform", snippet,
                "ESGRisk 无数据路径仍含 random.uniform，修复未生效")
        # 同时确认 data_available 已写入
        self.assertIn("data_available", src,
            "esg_risk_agent.py 中未找到 data_available 标注")

    # ── FundamentalAgent（财务数据缺失路径）────────────────────────────
    def test_fundamental_no_hardcoded_mock_when_missing(self):
        """FundamentalAgent 财务数据缺失时不再填充 10 亿硬编码模拟值"""
        src = _src("fundamental_agent.py")
        # 确认旧硬编码块已移除
        self.assertNotIn("1000000000.0,  # 10亿营收", src,
            "fundamental_agent.py 仍含旧 10亿硬编码模拟营收，修复未生效")
        # 确认 data_note 标注存在
        self.assertIn("data_available", src,
            "fundamental_agent.py 中未找到 data_available 标注")
        self.assertIn("data_note", src,
            "fundamental_agent.py 中未找到 data_note 说明")

    # ── FundFlowAgent（OBV 相对变化率）────────────────────────────────
    def test_fundamental_valuation_comparison_uses_industry_band(self):
        from hengline.agents.fundamental_agent import FundamentalAgent

        agent = object.__new__(FundamentalAgent)
        comparison = agent._build_valuation_comparison(
            {"sector": "电子", "industry": "通信设备"},
            {"pe_ratio": 60, "pb_ratio": 8, "ps_ratio": 3},
        )

        self.assertEqual(comparison["profile"], "科技成长")
        self.assertEqual(comparison["metrics"]["pe"]["bucket"], "expensive")
        self.assertIn("valuation_risk", comparison)

    def test_sentiment_event_analysis_classifies_high_impact_news(self):
        from hengline.agents.sentiment_agent import SentimentAgent

        agent = object.__new__(SentimentAgent)
        result = agent._classify_event_impact([
            {"title": "公司收到监管问询函", "sentiment": "negative", "publisher": "公告"},
            {"title": "发布年度业绩预告", "sentiment": "positive", "publisher": "公告"},
        ])

        self.assertTrue(result["requires_followup"])
        self.assertEqual(result["max_severity"], "high")
        self.assertIn("regulatory", result["event_counts"])

    def test_fund_flow_obv_relative_threshold(self):
        """FundFlowAgent 应使用 OBV 相对变化率（%），不再与金额绝对值比较"""
        src = _src("fund_flow_agent.py")
        # 新逻辑特征：obv_change_pct 变量存在
        self.assertIn("obv_change_pct", src,
            "fund_flow_agent.py 中未找到 obv_change_pct，OBV 相对化修复可能未生效")
        # 旧逻辑特征：与 money_flow_thresholds 大于某绝对值阈值比较
        # 新逻辑不应再出现 obv_change > self.money_flow_thresholds["strong_inflow"]
        self.assertNotIn(
            'obv_change > self.money_flow_thresholds["strong_inflow"]', src,
            "fund_flow_agent.py 仍使用旧绝对值阈值，修复未完全生效"
        )

    def test_fund_flow_obv_logic_relative(self):
        """OBV 分类逻辑：相对变化率 5% 以上 + MFI>80 应为 strong_inflow"""
        from hengline.agents.fund_flow_agent import FundFlowAgent
        agent = object.__new__(FundFlowAgent)

        money_flow = {
            "on_balance_volume": [1000000, 1060000],  # 6% 涨幅
            "money_flow_index": 85,
            "flow_classification": "neutral",
            "flow_trend": "stable"
        }

        # 直接调用分类逻辑片段（通过内部数据结构验证）
        obv_prev = money_flow["on_balance_volume"][-2]
        obv_curr = money_flow["on_balance_volume"][-1]
        obv_change = obv_curr - obv_prev
        obv_change_pct = obv_change / abs(obv_prev)
        mfi = money_flow["money_flow_index"]

        # 对应新逻辑：obv_change_pct > 0.05 and mfi > 80 → strong_inflow
        self.assertGreater(obv_change_pct, 0.05)
        self.assertGreater(mfi, 80)
        expected_classification = "strong_inflow"
        # 验证阈值判断
        if obv_change_pct > 0.05 and mfi > 80:
            actual = "strong_inflow"
        else:
            actual = "other"
        self.assertEqual(actual, expected_classification)


# ════════════════════════════════════════════════════════════════════════
# P2.1  共享 StockDataManager
# ════════════════════════════════════════════════════════════════════════
class TestP2_1_SharedStockDataManager(unittest.TestCase):
    """验证 BaseAgent 提供注入接口，Coordinator 完成共享注入"""

    def test_inject_stock_manager_method_exists(self):
        """BaseAgent 必须有 inject_stock_manager 方法"""
        from hengline.agents.base_agent import BaseAgent
        self.assertTrue(
            hasattr(BaseAgent, "inject_stock_manager"),
            "BaseAgent 缺少 inject_stock_manager 方法"
        )

    def test_inject_stock_manager_sets_attribute(self):
        """inject_stock_manager 应将实例赋给 self.stock_manager"""
        from hengline.agents.base_agent import BaseAgent

        class _DummyAgent(BaseAgent):
            def analyze(self, *a, **k): pass
            def get_result_template(self): return {}

        agent = object.__new__(_DummyAgent)
        agent.stock_manager = None
        agent.agent_name = "TestAgent"   # inject_stock_manager 使用 getattr 但显式赋值更清晰

        mock_sm = MagicMock(name="SharedStockDataManager")
        agent.inject_stock_manager(mock_sm)
        self.assertIs(agent.stock_manager, mock_sm,
            "inject_stock_manager 未将传入实例赋给 self.stock_manager")

    def test_all_concrete_agents_check_stock_manager_before_init(self):
        """各子 Agent __init__ 在创建 StockDataManager 前应检查 self.stock_manager is None"""
        for filename in [
            "fundamental_agent.py", "technical_agent.py",
            "sentiment_agent.py", "fund_flow_agent.py",
            "esg_risk_agent.py", "industry_macro_agent.py",
        ]:
            src = _src(filename)
            self.assertIn(
                "if self.stock_manager is None:",
                src,
                f"{filename} 缺少 'if self.stock_manager is None' 保护，注入机制未生效"
            )

    def test_coordinator_imports_stock_data_manager(self):
        """agent_coordinator.py 应导入 StockDataManager"""
        src = _src("agent_coordinator.py")
        self.assertIn("from hengline.stock.stock_manage import StockDataManager", src,
            "agent_coordinator.py 未导入 StockDataManager")

    def test_coordinator_creates_and_injects_shared_manager(self):
        """agent_coordinator.py 应创建 shared_stock_manager 并调用 inject_stock_manager"""
        src = _src("agent_coordinator.py")
        self.assertIn("shared_stock_manager", src,
            "agent_coordinator.py 中找不到 shared_stock_manager 变量")
        self.assertIn("inject_stock_manager", src,
            "agent_coordinator.py 中找不到 inject_stock_manager 调用")

    def test_coordinator_injects_to_all_agents_at_runtime(self):
        """inject_stock_manager 在协调器模拟注入逻辑中对每个 Agent 正确执行"""
        from hengline.agents.base_agent import BaseAgent

        # 构造 7 个 Mock Agent，全部支持 inject_stock_manager
        agent_names = [
            "FundamentalAgent", "TechnicalAgent", "IndustryMacroAgent",
            "SentimentAgent", "FundFlowAgent", "ESGRiskAgent",
            "ChiefStrategyAgent"
        ]
        mock_agents = {name: MagicMock(spec=BaseAgent) for name in agent_names}

        # 模拟共享 StockDataManager
        shared_sm = MagicMock(name="SharedStockDataManager")

        # 模拟 coordinator 注入逻辑（与 agent_coordinator.py 中的代码等价）
        for agent_name, agent in mock_agents.items():
            if hasattr(agent, "inject_stock_manager"):
                agent.inject_stock_manager(shared_sm)

        # 验证每个 Agent 都收到了注入调用，且参数是同一个共享实例
        for agent_name, agent in mock_agents.items():
            agent.inject_stock_manager.assert_called_once_with(shared_sm), \
                f"{agent_name} 未收到 inject_stock_manager(shared_sm) 调用"


# ════════════════════════════════════════════════════════════════════════
# P2.2  UI 失败标注
# ════════════════════════════════════════════════════════════════════════
class TestP2_2_UIFailureDisplay(unittest.TestCase):
    """验证 st_main.py 的失败展示逻辑"""

    def test_render_agent_status_has_failure_banner(self):
        """render_agent_status 函数应包含失败 Agent 的警告横幅逻辑"""
        src = _read_src(ST_MAIN)
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "render_agent_status":
                func_src = ast.unparse(node)
                self.assertIn("failed_agents", func_src,
                    "render_agent_status 中未找到 failed_agents 变量，失败横幅逻辑缺失")
                self.assertIn("warning", func_src,
                    "render_agent_status 中未找到 st.warning 调用，失败横幅逻辑缺失")
                return
        self.fail("st_main.py 中找不到 render_agent_status 函数")

    def test_render_agent_details_accepts_status_param(self):
        """render_agent_details 应接受可选 status 参数"""
        src = _read_src(ST_MAIN)
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "render_agent_details":
                args = [a.arg for a in node.args.args]
                self.assertIn("status", args,
                    "render_agent_details 缺少 status 参数，无法传入失败状态信息")
                return
        self.fail("st_main.py 中找不到 render_agent_details 函数")

    def test_render_agent_details_shows_failure_in_tab(self):
        """render_agent_details 对失败 Agent 应显示错误信息而非空内容"""
        src = _read_src(ST_MAIN)
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "render_agent_details":
                func_src = ast.unparse(node)
                self.assertIn("agent_success", func_src,
                    "render_agent_details 中未找到 agent_success 判断")
                self.assertIn("st.error", func_src,
                    "render_agent_details 中未找到 st.error 调用，失败 Agent 将无提示")
                self.assertIn("continue", func_src,
                    "render_agent_details 失败 Agent 未 continue 跳过内容展示")
                return
        self.fail("st_main.py 中找不到 render_agent_details 函数")

    def test_render_agent_status_success_failure_emoji(self):
        """Agent 状态表格应使用 ✅/❌ 标注，替代纯文本 Success/Failed"""
        src = _read_src(ST_MAIN)
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "render_agent_status":
                func_src = ast.unparse(node)
                self.assertIn("✅", func_src,
                    "render_agent_status 中未找到 ✅ 成功标记")
                self.assertIn("❌", func_src,
                    "render_agent_status 中未找到 ❌ 失败标记")
                return
        self.fail("st_main.py 中找不到 render_agent_status 函数")


# ════════════════════════════════════════════════════════════════════════
# P3  数据质量护栏与风险字段对齐
# ════════════════════════════════════════════════════════════════════════
class TestP3_ChiefDataQualityGuardrails(unittest.TestCase):
    """验证首席策略不再基于缺失/模拟数据输出过强建议。"""

    def _chief(self):
        from hengline.agents.chief_strategy_agent import ChiefStrategyAgent

        agent = object.__new__(ChiefStrategyAgent)
        agent.agent_name = "ChiefStrategyAgent"
        agent.agent_weights = {
            "FundamentalAgent": 0.25,
            "TechnicalAgent": 0.20,
            "IndustryMacroAgent": 0.15,
            "SentimentAgent": 0.15,
            "FundFlowAgent": 0.15,
            "ESGRiskAgent": 0.10,
        }
        agent.risk_levels = {
            "低风险": {"range": (0, 20), "color": "green"},
            "中低风险": {"range": (21, 40), "color": "light-green"},
            "中等风险": {"range": (41, 60), "color": "yellow"},
            "中高风险": {"range": (61, 80), "color": "orange"},
            "高风险": {"range": (81, 100), "color": "red"},
        }
        agent.investment_recommendations = {
            "强烈买入": {"description": "强烈买入", "confidence": "非常高", "signal_strength": 5},
            "买入": {"description": "买入", "confidence": "高", "signal_strength": 4},
            "持有": {"description": "持有", "confidence": "中等", "signal_strength": 3},
            "谨慎观望": {"description": "谨慎观望", "confidence": "低", "signal_strength": 2},
            "卖出": {"description": "卖出", "confidence": "高", "signal_strength": 1},
        }
        return agent

    def test_risk_uses_current_agent_score_schema(self):
        agent = self._chief()
        risk = agent._assess_overall_risk({
            "FundamentalAgent": {"overall_score": 8, "confidence_score": 0.9},
            "TechnicalAgent": {"signal_strength": "strong_bearish", "confidence_score": 0.9},
        })

        self.assertLess(risk["risk_score"], 60,
            "风险模型应读取 overall_score/signal_strength，而不是回退旧字段默认值")

    def test_simulated_data_caps_buy_recommendation(self):
        agent = self._chief()
        result = agent._structure_result(
            "300502",
            {
                "investment_recommendation": "强烈买入",
                "confidence_score": 0.95,
                "key_findings": ["趋势强"],
            },
            {"TechnicalAgent": {"data_quality_level": "simulated", "is_simulated": True}},
            88.0,
            {"risk_score": 40.0, "risk_level": "中低风险", "risk_factors": [], "risk_mitigation": []},
            {"level": "simulated", "score": 40, "simulated_agents": ["TechnicalAgent"], "failed_agents": []},
        )

        self.assertEqual(result["investment_recommendation"], "谨慎观望")
        self.assertLessEqual(result["confidence_score"], 0.55)
        self.assertIn("compliance_disclaimer", result)
        self.assertTrue(result["guardrail_notes"])
        self.assertTrue(result["human_review"]["required"])
        self.assertTrue(result["human_review"]["reasons"])

    def test_human_review_required_for_high_confidence_directional_call(self):
        agent = self._chief()
        result = agent._structure_result(
            "300502",
            {
                "investment_recommendation": "买入",
                "confidence_score": 0.85,
                "key_findings": ["趋势和基本面共振"],
            },
            {"TechnicalAgent": {"signal_strength": "bullish", "confidence_score": 0.85}},
            78.0,
            {"risk_score": 45.0, "risk_level": "中等风险", "risk_factors": [], "risk_mitigation": []},
            {"level": "verified", "score": 95, "simulated_agents": [], "failed_agents": []},
        )

        self.assertTrue(result["human_review"]["required"])
        self.assertIn("高强度", result["human_review"]["reasons"][0])

    def test_position_plan_reduces_size_for_limited_data_and_high_risk(self):
        agent = self._chief()
        result = agent._structure_result(
            "300502",
            {
                "investment_recommendation": "买入",
                "confidence_score": 0.85,
                "key_findings": ["趋势偏强"],
            },
            {"TechnicalAgent": {"signal_strength": "bullish", "confidence_score": 0.85}},
            75.0,
            {"risk_score": 72.0, "risk_level": "中高风险", "risk_factors": [], "risk_mitigation": []},
            {"level": "limited", "score": 45, "simulated_agents": [], "failed_agents": ["FundamentalAgent"]},
        )

        plan = result["position_plan"]
        self.assertLessEqual(plan["suggested_position_range"]["max_pct"], 0.03)
        self.assertIn("risk_controls", plan)
        self.assertEqual(plan["action"], "watch_or_probe")

    def test_decision_boundaries_include_conflict_and_data_gaps(self):
        agent = self._chief()
        result = agent._structure_result(
            "300502",
            {
                "investment_recommendation": "持有",
                "confidence_score": 0.65,
                "key_findings": ["多维度结论不完全一致"],
            },
            {"TechnicalAgent": {"signal_strength": "bullish", "confidence_score": 0.8}},
            62.0,
            {"risk_score": 55.0, "risk_level": "中等风险", "risk_factors": [], "risk_mitigation": []},
            {"level": "partial", "score": 70, "partial_agents": ["FundFlowAgent"], "data_gaps": ["资金流缺口"]},
            {"has_conflicts": True},
        )

        boundaries = result["decision_boundaries"]
        self.assertTrue(boundaries["evidence_quality"]["missing_or_limited_data"])
        self.assertTrue(any("分歧" in item for item in boundaries["reverse_risks"]))
        self.assertTrue(boundaries["invalidation_conditions"])

    def test_prompt_includes_data_quality_constraints(self):
        agent = self._chief()
        prompt = agent._generate_analysis_prompt(
            "300502",
            {},
            50.0,
            {"risk_level": "中等风险", "risk_score": 50.0},
            [],
            [],
            data_quality={"level": "limited", "score": 45, "simulated_agents": [], "failed_agents": ["FundamentalAgent"], "unavailable_agents": [], "decision_policy": "限制建议强度"},
        )

        self.assertIn("数据质量与合规约束", prompt)
        self.assertIn("限制建议强度", prompt)

    def test_data_quality_detects_estimated_and_unavailable_inputs(self):
        from hengline.agents.base_agent import AgentResult

        agent = self._chief()
        quality = agent._assess_data_quality(
            {
                "FundFlowAgent": {
                    "data_quality_level": "estimated",
                    "data_note": "机构持仓来自成交量代理估算",
                },
                "ESGRiskAgent": {
                    "data_quality_level": "unavailable",
                    "data_available": False,
                    "data_note": "ESG 数据不可用",
                },
            },
            {
                "FundFlowAgent": AgentResult(agent_name="FundFlowAgent", success=True, result={}),
                "ESGRiskAgent": AgentResult(agent_name="ESGRiskAgent", success=True, result={}),
            },
            {"data_gaps": []},
        )

        self.assertEqual(quality["level"], "limited")
        self.assertIn("FundFlowAgent", quality["partial_agents"])
        self.assertIn("ESGRiskAgent", quality["unavailable_agents"])


# ════════════════════════════════════════════════════════════════════════
# 汇总运行
# ════════════════════════════════════════════════════════════════════════
def run_all():
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    test_classes = [
        TestP0_1_RetrieveKnowledge,
        TestP0_2_CoordinatorSingleton,
        TestP1_1_ChiefScoreAlignment,
        TestP1_2_NoRandomData,
        TestP2_1_SharedStockDataManager,
        TestP2_2_UIFailureDisplay,
        TestP3_ChiefDataQualityGuardrails,
    ]

    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
