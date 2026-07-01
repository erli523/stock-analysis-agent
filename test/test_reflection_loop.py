#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reflection Loop 改造验证测试
=================================
覆盖范围：
  [R1] base_agent: set/clear/get reflection hint
  [R2] base_agent: _validate_output 结构验证逻辑
  [R3] base_agent: _generate_analysis 的 Prompt 中注入 Reflection 文本
  [R4] agent_coordinator: AgentState 含 conflict_analysis 字段
  [R5] agent_coordinator: REFLECTION_MAX_RETRIES / AGENT_TIMEOUTS 常量
  [R6] agent_coordinator: _compute_agent_score 纯逻辑
  [R7] agent_coordinator: ConflictAnalyzer 节点逻辑
  [R8] agent_coordinator: build_workflow 含 conflict_analyzer 节点
  [R9] agent_coordinator: 修复 functools.partial（run_in_executor 不再接受 **kwargs）
  [R10] chief_strategy_agent: analyze() 接受 conflict_analysis 参数
  [R11] chief_strategy_agent: _generate_analysis_prompt 注入冲突文本

运行：
  conda run -n stock-agent python test/test_reflection_loop.py
  conda run -n stock-agent pytest test/test_reflection_loop.py -v
"""

import os
import sys
import ast
import inspect
import unittest
from unittest.mock import MagicMock, patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

AGENTS_DIR = os.path.join(ROOT, "hengline", "agents")
COORD_FILE = os.path.join(AGENTS_DIR, "agent_coordinator.py")


def _src(filename):
    return open(os.path.join(AGENTS_DIR, filename), encoding='utf-8-sig').read()


# ════════════════════════════════════════════════════════════════════════
# R1-R3  BaseAgent Reflection Hint 机制
# ════════════════════════════════════════════════════════════════════════
class TestR1_R3_BaseAgentReflection(unittest.TestCase):

    def _make_bare_agent(self):
        from hengline.agents.base_agent import BaseAgent

        class _Dummy(BaseAgent):
            def analyze(self, *a, **k): pass
            def get_result_template(self): return {}

        agent = object.__new__(_Dummy)
        agent._reflection_hint = None
        agent.stock_manager = None
        agent.agent_name = "TestAgent"
        return agent

    # R1: set / clear
    def test_set_reflection_hint(self):
        agent = self._make_bare_agent()
        agent.set_reflection_hint("confidence_score 为 0")
        self.assertEqual(agent._reflection_hint, "confidence_score 为 0")

    def test_clear_reflection_hint(self):
        agent = self._make_bare_agent()
        agent.set_reflection_hint("some error")
        agent.clear_reflection_hint()
        self.assertIsNone(agent._reflection_hint)

    def test_hint_text_empty_when_none(self):
        agent = self._make_bare_agent()
        self.assertEqual(agent._reflection_hint_text(), "")

    def test_hint_text_nonempty_when_set(self):
        agent = self._make_bare_agent()
        agent.set_reflection_hint("missing key_findings")
        text = agent._reflection_hint_text()
        self.assertIn("missing key_findings", text)
        self.assertIn("Reflection", text)

    # R2: _validate_output
    def test_validate_success_valid(self):
        """成功且有 key_findings 的结果应通过验证"""
        from hengline.agents.base_agent import AgentResult
        agent = self._make_bare_agent()
        result = AgentResult(
            agent_name="Test", success=True, confidence_score=0.85,
            result={"key_findings": ["业绩增长"], "confidence_score": 0.85}
        )
        self.assertIsNone(agent._validate_output(result))

    def test_validate_failed_agent(self):
        """success=False 的结果应返回错误描述"""
        from hengline.agents.base_agent import AgentResult
        agent = self._make_bare_agent()
        result = AgentResult(
            agent_name="Test", success=False, confidence_score=0.0,
            result={}, error_message="Network error"
        )
        err = agent._validate_output(result)
        self.assertIsNotNone(err)
        self.assertIn("Network error", err)

    def test_validate_empty_result(self):
        """result 为空 dict 应返回错误"""
        from hengline.agents.base_agent import AgentResult
        agent = self._make_bare_agent()
        result = AgentResult(
            agent_name="Test", success=True, confidence_score=0.5, result={}
        )
        err = agent._validate_output(result)
        self.assertIsNotNone(err)

    def test_validate_low_confidence(self):
        """confidence_score <= 0.1 应触发验证失败"""
        from hengline.agents.base_agent import AgentResult
        agent = self._make_bare_agent()
        result = AgentResult(
            agent_name="Test", success=True, confidence_score=0.05,
            result={"key_findings": ["x"], "confidence_score": 0.05}
        )
        err = agent._validate_output(result)
        self.assertIsNotNone(err)
        self.assertIn("0.1", err)

    def test_validate_missing_key_findings(self):
        """缺少 key_findings 字段应返回错误"""
        from hengline.agents.base_agent import AgentResult
        agent = self._make_bare_agent()
        result = AgentResult(
            agent_name="Test", success=True, confidence_score=0.8,
            result={"confidence_score": 0.8}  # no key_findings
        )
        err = agent._validate_output(result)
        self.assertIsNotNone(err)
        self.assertIn("key_findings", err)

    # R3: _generate_analysis 中包含 reflection_section
    def test_generate_analysis_template_has_reflection_section(self):
        """_generate_analysis 源码中应包含 {reflection_section} 占位符"""
        src = _src("base_agent.py")
        self.assertIn("{reflection_section}", src,
            "base_agent.py _generate_analysis 中未找到 {reflection_section} 占位符")

    def test_reflection_hint_text_injected_into_chain(self):
        """_generate_analysis 应将 _reflection_hint_text() 传给 chain assign"""
        src = _src("base_agent.py")
        self.assertIn("reflection_section=lambda _: reflection_text", src,
            "_generate_analysis 中未将 reflection_text 注入 chain")


# ════════════════════════════════════════════════════════════════════════
# R4-R9  AgentCoordinator 改造
# ════════════════════════════════════════════════════════════════════════
class TestR4_R9_CoordinatorReflection(unittest.TestCase):

    def _coord_src(self):
        return open(COORD_FILE, encoding='utf-8-sig').read()

    # R4: AgentState 含 conflict_analysis
    def test_state_has_conflict_analysis(self):
        src = self._coord_src()
        self.assertIn("conflict_analysis", src,
            "AgentState 中缺少 conflict_analysis 字段")

    # R5: 常量存在
    def test_reflection_max_retries_constant(self):
        src = self._coord_src()
        self.assertIn("REFLECTION_MAX_RETRIES", src,
            "coordinator 中缺少 REFLECTION_MAX_RETRIES 常量")

    def test_agent_timeouts_dict(self):
        src = self._coord_src()
        self.assertIn("AGENT_TIMEOUTS", src,
            "coordinator 中缺少 AGENT_TIMEOUTS 字典")
        self.assertIn("FundamentalAgent", src)
        self.assertIn("ChiefStrategyAgent", src)

    # R6: _compute_agent_score 纯逻辑测试
    def test_compute_fundamental_score(self):
        from hengline.agents.agent_coordinator import AgentCoordinator
        score = AgentCoordinator._compute_agent_score(
            "FundamentalAgent",
            {"success": True, "result": {"overall_score": 7, "confidence_score": 0.8}}
        )
        self.assertAlmostEqual(score, 70.0, places=1)

    def test_compute_technical_bullish(self):
        from hengline.agents.agent_coordinator import AgentCoordinator
        score = AgentCoordinator._compute_agent_score(
            "TechnicalAgent",
            {"success": True, "result": {"signal_strength": "bullish", "confidence_score": 0.9}}
        )
        self.assertGreater(score, 60)

    def test_compute_fundflow_strong_inflow(self):
        from hengline.agents.agent_coordinator import AgentCoordinator
        score = AgentCoordinator._compute_agent_score(
            "FundFlowAgent",
            {"success": True, "result": {"key_metrics": {"flow_classification": "strong_inflow"}}}
        )
        self.assertEqual(score, 90.0)

    def test_compute_failed_agent_returns_none(self):
        from hengline.agents.agent_coordinator import AgentCoordinator
        score = AgentCoordinator._compute_agent_score(
            "FundamentalAgent",
            {"success": False, "result": {}}
        )
        self.assertIsNone(score)

    # R7: ConflictAnalyzer 节点逻辑
    def test_conflict_analyzer_detects_divergence(self):
        """当两个 Agent 评分差距超过 30 时，应报告 score_divergences"""
        from hengline.agents.base_agent import AgentResult
        from hengline.agents.agent_coordinator import AgentCoordinator

        coord = object.__new__(AgentCoordinator)
        coord.agents = {}
        node_fn = coord._create_conflict_analyzer_node()

        # 构造高分歧的假 State
        state = {
            "agent_results": {
                "FundamentalAgent": AgentResult(
                    agent_name="FundamentalAgent", success=True,
                    confidence_score=0.9,
                    result={"overall_score": 8, "key_findings": ["好"]}
                ),
                "TechnicalAgent": AgentResult(
                    agent_name="TechnicalAgent", success=True,
                    confidence_score=0.8,
                    result={"signal_strength": "strong_bearish",
                            "confidence_score": 0.8, "key_findings": ["弱"]}
                ),
            }
        }
        result = node_fn(state)
        ca = result["conflict_analysis"]
        self.assertTrue(ca["has_conflicts"],
            "FundamentalAgent(80分) vs TechnicalAgent(<30分) 应检测到冲突")
        self.assertGreater(len(ca["score_divergences"]), 0)

    def test_conflict_analyzer_no_divergence(self):
        """评分一致时 has_conflicts 应为 False"""
        from hengline.agents.base_agent import AgentResult
        from hengline.agents.agent_coordinator import AgentCoordinator

        coord = object.__new__(AgentCoordinator)
        coord.agents = {}
        node_fn = coord._create_conflict_analyzer_node()

        state = {
            "agent_results": {
                "FundamentalAgent": AgentResult(
                    agent_name="FundamentalAgent", success=True,
                    confidence_score=0.9,
                    result={"overall_score": 7, "key_findings": ["好"]}
                ),
                "TechnicalAgent": AgentResult(
                    agent_name="TechnicalAgent", success=True,
                    confidence_score=0.8,
                    result={"signal_strength": "bullish",
                            "confidence_score": 0.8, "key_findings": ["涨"]}
                ),
            }
        }
        result = node_fn(state)
        ca = result["conflict_analysis"]
        # FundamentalAgent≈70, TechnicalAgent≈73 → 差距<30
        self.assertFalse(ca["has_conflicts"])
        self.assertEqual(ca["score_divergences"], [])

    def test_conflict_analyzer_failed_agents_counted(self):
        """失败的 Agent 应出现在 failed_agents 列表中"""
        from hengline.agents.base_agent import AgentResult
        from hengline.agents.agent_coordinator import AgentCoordinator

        coord = object.__new__(AgentCoordinator)
        coord.agents = {}
        node_fn = coord._create_conflict_analyzer_node()

        state = {
            "agent_results": {
                "ESGRiskAgent": AgentResult(
                    agent_name="ESGRiskAgent", success=False,
                    confidence_score=0.0, result={}, error_message="超时"
                ),
            }
        }
        result = node_fn(state)
        ca = result["conflict_analysis"]
        self.assertIn("ESGRiskAgent", ca["failed_agents"])

    def test_conflict_analyzer_consensus_direction(self):
        """大多数 Agent 看多时，consensus_direction 应为 '偏多'"""
        from hengline.agents.base_agent import AgentResult
        from hengline.agents.agent_coordinator import AgentCoordinator

        coord = object.__new__(AgentCoordinator)
        coord.agents = {}
        node_fn = coord._create_conflict_analyzer_node()

        state = {
            "agent_results": {
                "FundamentalAgent": AgentResult(
                    agent_name="FundamentalAgent", success=True,
                    confidence_score=0.9,
                    result={"overall_score": 8, "key_findings": ["x"]}
                ),
                "TechnicalAgent": AgentResult(
                    agent_name="TechnicalAgent", success=True,
                    confidence_score=0.9,
                    result={"signal_strength": "strong_bullish",
                            "confidence_score": 0.9, "key_findings": ["x"]}
                ),
                "FundFlowAgent": AgentResult(
                    agent_name="FundFlowAgent", success=True,
                    confidence_score=0.8,
                    result={"key_metrics": {"flow_classification": "strong_inflow"},
                            "key_findings": ["x"]}
                ),
            }
        }
        result = node_fn(state)
        ca = result["conflict_analysis"]
        self.assertEqual(ca["consensus_direction"], "偏多",
            f"Expected '偏多', got '{ca['consensus_direction']}'")

    # R8: build_workflow 包含 conflict_analyzer 节点
    def test_build_workflow_has_conflict_analyzer_node(self):
        src = self._coord_src()
        self.assertIn('conflict_analyzer', src,
            "agent_coordinator.py 中未找到 conflict_analyzer 节点")
        self.assertIn('_create_conflict_analyzer_node', src,
            "agent_coordinator.py 中未找到 _create_conflict_analyzer_node 方法")

    def test_build_workflow_graph_edge_to_conflict_analyzer(self):
        """build_workflow 应在所有 Agent 到 conflict_analyzer 之间添加边"""
        src = self._coord_src()
        self.assertIn('"conflict_analyzer"', src,
            "build_workflow 中未找到到 conflict_analyzer 的边")
        self.assertIn('add_edge(name, "conflict_analyzer")', src,
            "build_workflow 中未找到各 Agent 到 conflict_analyzer 的 add_edge 调用")

    # R9: functools.partial 修复
    def test_functools_partial_used_in_agent_node(self):
        src = self._coord_src()
        self.assertIn("functools.partial(agent.analyze", src,
            "agent_node 中未使用 functools.partial 修复 run_in_executor kwargs 传递")

    def test_import_functools(self):
        src = self._coord_src()
        self.assertIn("import functools", src,
            "agent_coordinator.py 未导入 functools")


# ════════════════════════════════════════════════════════════════════════
# R9B  LangGraph 拓扑与执行轨迹
# ════════════════════════════════════════════════════════════════════════
class TestR9B_LangGraphDiagnostics(unittest.TestCase):

    def test_merge_lists_keeps_parallel_trace_events(self):
        from hengline.agents.agent_coordinator import merge_lists

        merged = merge_lists([{"node": "A"}], [{"node": "B"}])
        self.assertEqual([item["node"] for item in merged], ["A", "B"])

    def test_workflow_topology_describes_fan_out_fan_in_graph(self):
        from hengline.agents.agent_coordinator import AgentCoordinator

        coord = object.__new__(AgentCoordinator)
        coord.specialist_agent_names = ["TechnicalAgent", "FundamentalAgent"]

        topology = coord.get_workflow_topology()
        self.assertIn({"from": "START", "to": "TechnicalAgent"}, topology["edges"])
        self.assertIn({"from": "TechnicalAgent", "to": "conflict_analyzer"}, topology["edges"])
        self.assertIn({"from": "conflict_analyzer", "to": "ChiefStrategyAgent"}, topology["edges"])
        self.assertEqual(topology["reducers"]["agent_results"], "merge_dicts")
        self.assertEqual(topology["reducers"]["workflow_trace"], "merge_lists")

    def test_conflict_analyzer_returns_workflow_trace(self):
        from hengline.agents.base_agent import AgentResult
        from hengline.agents.agent_coordinator import AgentCoordinator

        coord = object.__new__(AgentCoordinator)
        coord.agents = {}
        node_fn = coord._create_conflict_analyzer_node()

        result = node_fn({
            "agent_results": {
                "TechnicalAgent": AgentResult(
                    agent_name="TechnicalAgent",
                    success=True,
                    confidence_score=0.8,
                    result={
                        "signal_strength": "bullish",
                        "confidence_score": 0.8,
                        "key_findings": ["x"],
                    },
                )
            }
        })

        trace = result["workflow_trace"]
        self.assertEqual(trace[0]["node"], "conflict_analyzer")
        self.assertEqual(trace[0]["event"], "completed")
        self.assertIn("consensus_direction", trace[0])

    def test_report_includes_workflow_trace_and_topology(self):
        from hengline.agents.base_agent import AgentResult
        from hengline.agents.agent_coordinator import AgentCoordinator

        coord = object.__new__(AgentCoordinator)
        coord.specialist_agent_names = ["TechnicalAgent"]

        final_state = {
            "stock_code": "300502",
            "analysis_start_time": "2026-07-01T00:00:00",
            "agent_results": {
                "TechnicalAgent": AgentResult(
                    agent_name="TechnicalAgent",
                    success=True,
                    confidence_score=0.8,
                    result={"key_findings": ["x"]},
                )
            },
            "conflict_analysis": {"consensus_direction": "偏多"},
            "workflow_trace": [{"node": "TechnicalAgent", "event": "completed"}],
            "final_result": AgentResult(
                agent_name="ChiefStrategyAgent",
                success=True,
                confidence_score=0.8,
                result={"investment_recommendation": "持有", "key_findings": ["x"]},
            ),
        }

        report = coord._generate_analysis_report(final_state)
        self.assertEqual(report["workflow_trace"], final_state["workflow_trace"])
        self.assertIn("topology", report["workflow_metadata"])
        self.assertIn({"from": "START", "to": "TechnicalAgent"},
                      report["workflow_metadata"]["topology"]["edges"])


# ════════════════════════════════════════════════════════════════════════
# R10-R11  ChiefStrategyAgent 接受冲突分析
# ════════════════════════════════════════════════════════════════════════
class TestR10_R11_ChiefConflictAnalysis(unittest.TestCase):

    def _chief_src(self):
        return open(
            os.path.join(AGENTS_DIR, "chief_strategy_agent.py"),
            encoding='utf-8-sig'
        ).read()

    def test_analyze_accepts_conflict_analysis(self):
        """analyze() 签名应包含 conflict_analysis 参数"""
        src = self._chief_src()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "analyze":
                args = [a.arg for a in node.args.args]
                self.assertIn("conflict_analysis", args,
                    "ChiefStrategyAgent.analyze() 缺少 conflict_analysis 参数")
                return
        self.fail("ChiefStrategyAgent 中找不到 analyze 方法")

    def test_generate_analysis_prompt_accepts_conflict_analysis(self):
        """_generate_analysis_prompt 签名应包含 conflict_analysis 参数"""
        src = self._chief_src()
        self.assertIn("conflict_analysis: Dict = None", src,
            "_generate_analysis_prompt 缺少 conflict_analysis 参数")

    def test_conflict_text_injected_into_prompt(self):
        """_generate_analysis_prompt 中应构建并追加 conflict_text 到 prompt"""
        src = self._chief_src()
        self.assertIn("conflict_text", src,
            "chief_strategy_agent.py 中未找到 conflict_text 变量")
        self.assertIn("conflict_analysis.get(", src,
            "chief_strategy_agent.py 中未读取 conflict_analysis 内容")

    def test_conflict_analysis_passed_to_prompt_call(self):
        """analyze() 中调用 _generate_analysis_prompt 时应传入 conflict_analysis"""
        src = self._chief_src()
        self.assertIn("conflict_analysis=conflict_analysis", src,
            "analyze() 未将 conflict_analysis 传给 _generate_analysis_prompt")

    def test_chief_analyze_runtime_with_conflict(self):
        """运行时 analyze() 在收到 conflict_analysis 时能正常调用 _generate_analysis_prompt"""
        from hengline.agents.chief_strategy_agent import ChiefStrategyAgent

        agent = object.__new__(ChiefStrategyAgent)
        agent._reflection_hint = None
        agent.agent_name = "首席策略官"
        agent.description = "资深投资策略专家"
        agent.retriever = None
        agent.memory = None
        agent.investment_recommendations = {"持有": {"description": "持有", "confidence": "中"}}
        agent.agent_weights = {"FundamentalAgent": 0.5, "TechnicalAgent": 0.5}
        agent.risk_levels = {"中等风险": {"range": (41, 60), "color": "yellow"}}

        # 构造 prompt，验证 conflict_text 被注入
        from hengline.agents.base_agent import AgentResult
        prompt = agent._generate_analysis_prompt(
            stock_code="300502",
            filtered_results={"FundamentalAgent": {"overall_score": 7}},
            composite_score=65.0,
            overall_risk={"risk_level": "中等风险", "risk_score": 50.0},
            key_strengths=["营收增长"],
            key_risks=["估值偏高"],
            conflict_analysis={
                "conflict_summary": "存在评分分歧",
                "score_divergences": ["FundamentalAgent(70) vs TechnicalAgent(30)：差距40分"],
                "data_gaps": [],
                "failed_agents": []
            }
        )
        self.assertIn("冲突分析", prompt,
            "conflict_analysis 未被注入到最终 Prompt 中")
        self.assertIn("FundamentalAgent", prompt)


# ════════════════════════════════════════════════════════════════════════
# 汇总运行
# ════════════════════════════════════════════════════════════════════════
def run_all():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [
        TestR1_R3_BaseAgentReflection,
        TestR4_R9_CoordinatorReflection,
        TestR9B_LangGraphDiagnostics,
        TestR10_R11_ChiefConflictAnalysis,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
