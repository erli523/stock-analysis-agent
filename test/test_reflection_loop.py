#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reflection Loop 鏀归€犻獙璇佹祴璇?=================================
瑕嗙洊鑼冨洿锛?  [R1] base_agent: set/clear/get reflection hint
  [R2] base_agent: _validate_output 缁撴瀯楠岃瘉閫昏緫
  [R3] base_agent: _generate_analysis 鐨?Prompt 涓敞鍏?Reflection 鏂囨湰
  [R4] agent_coordinator: AgentState 鍚?conflict_analysis 瀛楁
  [R5] agent_coordinator: REFLECTION_MAX_RETRIES / AGENT_TIMEOUTS 甯搁噺
  [R6] agent_coordinator: _compute_agent_score 绾€昏緫
  [R7] agent_coordinator: ConflictAnalyzer 鑺傜偣閫昏緫
  [R8] agent_coordinator: build_workflow 鍚?conflict_analyzer 鑺傜偣
  [R9] agent_coordinator: 淇 functools.partial锛坮un_in_executor 涓嶅啀鎺ュ彈 **kwargs锛?  [R10] chief_strategy_agent: analyze() 鎺ュ彈 conflict_analysis 鍙傛暟
  [R11] chief_strategy_agent: _generate_analysis_prompt 娉ㄥ叆鍐茬獊鏂囨湰

杩愯锛?  conda run -n stock-agent python test/test_reflection_loop.py
  conda run -n stock-agent pytest test/test_reflection_loop.py -v
"""

import os
import sys
import ast
import inspect
import asyncio
import unittest
from unittest.mock import MagicMock, patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

AGENTS_DIR = os.path.join(ROOT, "hengline", "agents")
COORD_FILE = os.path.join(AGENTS_DIR, "agent_coordinator.py")


def _src(filename):
    return open(os.path.join(AGENTS_DIR, filename), encoding='utf-8-sig').read()


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
# R1-R3  BaseAgent Reflection Hint 鏈哄埗
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
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
        agent.set_reflection_hint("confidence_score 涓?0")
        self.assertEqual(agent._reflection_hint, "confidence_score 涓?0")

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
        """Test helper."""
        from hengline.agents.base_agent import AgentResult
        agent = self._make_bare_agent()
        result = AgentResult(
            agent_name="Test", success=True, confidence_score=0.85,
            result={"key_findings": ["涓氱哗澧為暱"], "confidence_score": 0.85}
        )
        self.assertIsNone(agent._validate_output(result))

    def test_validate_failed_agent(self):
        """Test helper."""
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
        """Test helper."""
        from hengline.agents.base_agent import AgentResult
        agent = self._make_bare_agent()
        result = AgentResult(
            agent_name="Test", success=True, confidence_score=0.5, result={}
        )
        err = agent._validate_output(result)
        self.assertIsNotNone(err)

    def test_validate_low_confidence(self):
        """Test helper."""
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
        """Test helper."""
        from hengline.agents.base_agent import AgentResult
        agent = self._make_bare_agent()
        result = AgentResult(
            agent_name="Test", success=True, confidence_score=0.8,
            result={"confidence_score": 0.8}  # no key_findings
        )
        err = agent._validate_output(result)
        self.assertIsNotNone(err)
        self.assertIn("key_findings", err)

    # R3: _generate_analysis contains reflection_section
    def test_generate_analysis_template_has_reflection_section(self):
        """_generate_analysis source should contain the reflection_section placeholder."""
        src = _src("base_agent.py")
        self.assertIn("{reflection_section}", src,
            "base_agent.py _generate_analysis is missing {reflection_section}")

    def test_reflection_hint_text_injected_into_chain(self):
        """Test helper."""
        src = _src("base_agent.py")
        self.assertIn("reflection_section=lambda _: reflection_text", src,
            "_generate_analysis 涓湭灏?reflection_text 娉ㄥ叆 chain")


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
# R4-R9  AgentCoordinator 鏀归€?# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
class TestR4_R9_CoordinatorReflection(unittest.TestCase):

    def _coord_src(self):
        return open(COORD_FILE, encoding='utf-8-sig').read()

    # R4: AgentState 鍚?conflict_analysis
    def test_state_has_conflict_analysis(self):
        src = self._coord_src()
        self.assertIn("conflict_analysis", src,
            "AgentState 涓己灏?conflict_analysis 瀛楁")

    # R5: 甯搁噺瀛樺湪
    def test_reflection_max_retries_constant(self):
        src = self._coord_src()
        self.assertIn("REFLECTION_MAX_RETRIES", src,
            "coordinator 涓己灏?REFLECTION_MAX_RETRIES 甯搁噺")

    def test_agent_timeouts_dict(self):
        src = self._coord_src()
        self.assertIn("AGENT_TIMEOUTS", src,
            "coordinator 涓己灏?AGENT_TIMEOUTS 瀛楀吀")
        self.assertIn("FundamentalAgent", src)
        self.assertIn("ChiefStrategyAgent", src)

    # R6: _compute_agent_score 绾€昏緫娴嬭瘯
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

    # R7: ConflictAnalyzer 鑺傜偣閫昏緫
    def test_conflict_analyzer_detects_divergence(self):
        """Test helper."""
        from hengline.agents.base_agent import AgentResult
        from hengline.agents.agent_coordinator import AgentCoordinator

        coord = object.__new__(AgentCoordinator)
        coord.agents = {}
        node_fn = coord._create_conflict_analyzer_node()

        # 鏋勯€犻珮鍒嗘鐨勫亣 State
        state = {
            "agent_results": {
                "FundamentalAgent": AgentResult(
                    agent_name="FundamentalAgent", success=True,
                    confidence_score=0.9,
                    result={"overall_score": 8, "key_findings": ["good"]}
                ),
                "TechnicalAgent": AgentResult(
                    agent_name="TechnicalAgent", success=True,
                    confidence_score=0.8,
                    result={"signal_strength": "strong_bearish",
                            "confidence_score": 0.8, "key_findings": ["weak"]}
                ),
            }
        }
        result = node_fn(state)
        ca = result["conflict_analysis"]
        self.assertTrue(ca["has_conflicts"],
            "FundamentalAgent(80鍒? vs TechnicalAgent(<30鍒? 搴旀娴嬪埌鍐茬獊")
        self.assertGreater(len(ca["score_divergences"]), 0)

    def test_conflict_analyzer_no_divergence(self):
        """has_conflicts should be false when specialist scores agree."""
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
                    result={"overall_score": 7, "key_findings": ["good"]}
                ),
                "TechnicalAgent": AgentResult(
                    agent_name="TechnicalAgent", success=True,
                    confidence_score=0.8,
                    result={"signal_strength": "bullish",
                            "confidence_score": 0.8, "key_findings": ["neutral"]}
                ),
            }
        }
        result = node_fn(state)
        ca = result["conflict_analysis"]
        # FundamentalAgent鈮?0, TechnicalAgent鈮?3 鈫?宸窛<30
        self.assertFalse(ca["has_conflicts"])
        self.assertEqual(ca["score_divergences"], [])

    def test_conflict_analyzer_failed_agents_counted(self):
        """Test helper."""
        from hengline.agents.base_agent import AgentResult
        from hengline.agents.agent_coordinator import AgentCoordinator

        coord = object.__new__(AgentCoordinator)
        coord.agents = {}
        node_fn = coord._create_conflict_analyzer_node()

        state = {
            "agent_results": {
                "ESGRiskAgent": AgentResult(
                    agent_name="ESGRiskAgent", success=False,
                    confidence_score=0.0, result={}, error_message="瓒呮椂"
                ),
            }
        }
        result = node_fn(state)
        ca = result["conflict_analysis"]
        self.assertIn("ESGRiskAgent", ca["failed_agents"])

    def test_conflict_analyzer_flags_simulated_data_quality(self):
        """Test helper."""
        from hengline.agents.base_agent import AgentResult
        from hengline.agents.agent_coordinator import AgentCoordinator

        coord = object.__new__(AgentCoordinator)
        coord.agents = {}
        node_fn = coord._create_conflict_analyzer_node()

        state = {
            "agent_results": {
                "TechnicalAgent": AgentResult(
                    agent_name="TechnicalAgent",
                    success=True,
                    confidence_score=0.8,
                    result={
                        "signal_strength": "bullish",
                        "confidence_score": 0.8,
                        "key_findings": ["x"],
                        "data_quality_level": "simulated",
                        "is_simulated": True,
                    },
                ),
            }
        }
        result = node_fn(state)
        ca = result["conflict_analysis"]
        self.assertTrue(any("simulated" in item for item in ca["data_gaps"]))

    def test_conflict_analyzer_consensus_direction(self):
        """Test helper."""
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
        self.assertEqual(ca["consensus_direction"], "bullish",
            f"Expected 'bullish', got '{ca['consensus_direction']}'")

    # R8: build_workflow 鍖呭惈 conflict_analyzer 鑺傜偣
    def test_build_workflow_has_conflict_analyzer_node(self):
        src = self._coord_src()
        self.assertIn('conflict_analyzer', src,
            "agent_coordinator.py 涓湭鎵惧埌 conflict_analyzer 鑺傜偣")
        self.assertIn('_create_conflict_analyzer_node', src,
            "agent_coordinator.py 涓湭鎵惧埌 _create_conflict_analyzer_node 鏂规硶")

    def test_build_workflow_graph_edge_to_conflict_analyzer(self):
        """Test helper."""
        src = self._coord_src()
        self.assertIn('"conflict_analyzer"', src,
            "build_workflow should include conflict_analyzer")
        self.assertIn('add_edge(name, "agent_quality_gate")', src,
            "build_workflow should route agents to agent_quality_gate")

    # R9: functools.partial 淇
    def test_functools_partial_used_in_agent_node(self):
        src = self._coord_src()
        self.assertIn("functools.partial(agent.analyze", src,
            "agent_node should use functools.partial for run_in_executor kwargs")

    def test_import_functools(self):
        src = self._coord_src()
        self.assertIn("import functools", src,
            "agent_coordinator.py should import functools")


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
# R9B  LangGraph 鎷撴墤涓庢墽琛岃建杩?# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
class TestR9B_LangGraphDiagnostics(unittest.TestCase):

    def test_merge_lists_keeps_parallel_trace_events(self):
        from hengline.agents.agent_coordinator import merge_lists

        merged = merge_lists([{"node": "A"}], [{"node": "B"}])
        self.assertEqual([item["node"] for item in merged], ["A", "B"])

    def test_workflow_topology_describes_fan_out_fan_in_graph(self):
        from hengline.agents.agent_coordinator import AgentCoordinator

        coord = object.__new__(AgentCoordinator)
        coord.specialist_agent_names = ["TechnicalAgent", "FundamentalAgent"]
        coord.checkpoint_enabled = False
        coord.checkpointer = None

        topology = coord.get_workflow_topology()
        self.assertIn({"from": "START", "to": "agent_router"}, topology["edges"])
        self.assertIn({"from": "agent_router", "to": "TechnicalAgent", "condition": "active_agent_names"}, topology["edges"])
        self.assertIn({"from": "TechnicalAgent", "to": "agent_quality_gate"}, topology["edges"])
        self.assertIn({"from": "agent_quality_gate", "to": "agent_router", "condition": "retry"}, topology["edges"])
        self.assertIn({"from": "agent_quality_gate", "to": "conflict_analyzer", "condition": "continue"}, topology["edges"])
        self.assertIn({"from": "conflict_analyzer", "to": "ChiefStrategyAgent"}, topology["edges"])
        self.assertEqual(topology["reducers"]["agent_results"], "merge_dicts")
        self.assertEqual(topology["reducers"]["workflow_trace"], "merge_lists")
        self.assertFalse(topology["checkpointing"]["enabled"])

    def test_router_selects_question_relevant_agents(self):
        from hengline.agents.agent_coordinator import AgentCoordinator

        coord = object.__new__(AgentCoordinator)
        coord.config = {}
        coord.specialist_agent_names = [
            "FundamentalAgent",
            "TechnicalAgent",
            "IndustryMacroAgent",
            "SentimentAgent",
            "FundFlowAgent",
            "ESGRiskAgent",
        ]

        decision = coord._select_active_agents({
            "time_range": "1m",
            "agent_params": {"question": "technical volume trend"},
        })

        self.assertIn("TechnicalAgent", decision["selected_agents"])
        self.assertIn("FundFlowAgent", decision["selected_agents"])
        self.assertNotIn("ESGRiskAgent", decision["selected_agents"])
        self.assertEqual(decision["routing_reason"], "question_keyword_routing")

    def test_router_respects_explicit_enabled_agents(self):
        from hengline.agents.agent_coordinator import AgentCoordinator

        coord = object.__new__(AgentCoordinator)
        coord.config = {}
        coord.specialist_agent_names = ["FundamentalAgent", "TechnicalAgent", "SentimentAgent"]

        decision = coord._select_active_agents({
            "time_range": "1y",
            "agent_params": {"enabled_agents": ["SentimentAgent"]},
        })

        self.assertEqual(decision["selected_agents"], ["SentimentAgent"])
        self.assertEqual(decision["routing_reason"], "explicit_enabled_agents")

    def test_agent_router_node_returns_trace_and_active_agents(self):
        from hengline.agents.agent_coordinator import AgentCoordinator

        coord = object.__new__(AgentCoordinator)
        coord.config = {}
        coord.specialist_agent_names = ["FundamentalAgent", "TechnicalAgent"]

        node = coord._create_agent_router_node()
        result = node({"time_range": "1y", "agent_params": {"analysis_focus": "valuation"}})

        self.assertIn("FundamentalAgent", result["active_agent_names"])
        self.assertEqual(result["workflow_trace"][0]["node"], "agent_router")
        self.assertEqual(result["routing_decision"]["routing_reason"], "question_keyword_routing")

    def test_quality_gate_flags_simulated_data(self):
        from hengline.agents.agent_coordinator import AgentCoordinator
        from hengline.agents.base_agent import AgentResult

        coord = object.__new__(AgentCoordinator)
        node = coord._create_quality_gate_node()
        result = node({
            "active_agent_names": ["TechnicalAgent"],
            "agent_results": {
                "TechnicalAgent": AgentResult(
                    agent_name="TechnicalAgent",
                    success=True,
                    result={
                        "key_findings": ["x"],
                        "data_quality_level": "simulated",
                        "is_simulated": True,
                    },
                    confidence_score=0.8,
                )
            },
        })

        self.assertFalse(result["quality_gate"]["passed"])
        self.assertEqual(result["workflow_trace"][0]["node"], "agent_quality_gate")
        self.assertEqual(result["quality_gate"]["high_severity_count"], 1)

    def test_quality_gate_requests_graph_retry_for_failed_agent(self):
        from hengline.agents.agent_coordinator import AgentCoordinator
        from hengline.agents.base_agent import AgentResult

        coord = object.__new__(AgentCoordinator)
        node = coord._create_quality_gate_node()
        result = node({
            "active_agent_names": ["TechnicalAgent"],
            "quality_retry_count": 0,
            "agent_results": {
                "TechnicalAgent": AgentResult(
                    agent_name="TechnicalAgent",
                    success=False,
                    result={},
                    error_message="temporary failure",
                    confidence_score=0.0,
                )
            },
        })

        self.assertEqual(result["retry_agent_names"], ["TechnicalAgent"])
        self.assertEqual(result["quality_retry_count"], 1)
        self.assertEqual(AgentCoordinator._route_after_quality_gate(result), "retry")

    def test_compiled_graph_routes_only_selected_specialists(self):
        from hengline.agents.agent_coordinator import AgentCoordinator
        from hengline.agents.base_agent import AgentResult

        class FakeSpecialist:
            def __init__(self, name, calls):
                self.name = name
                self.calls = calls

            def analyze(self, stock_code, time_range, **kwargs):
                self.calls.append(self.name)
                return AgentResult(
                    agent_name=self.name,
                    success=True,
                    result={
                        "key_findings": [f"{self.name} ok"],
                        "signal_strength": "neutral",
                        "confidence_score": 0.8,
                    },
                    confidence_score=0.8,
                )

            def _validate_output(self, result):
                return None

            def clear_reflection_hint(self):
                return None

            def set_reflection_hint(self, hint):
                return None

        class FakeChief(FakeSpecialist):
            investment_recommendations = {"鎸佹湁": "hold"}

            def analyze(self, stock_code, agent_results, conflict_analysis=None, **kwargs):
                self.calls.append(self.name)
                return AgentResult(
                    agent_name="ChiefStrategyAgent",
                    success=True,
                    result={
                        "investment_recommendation": "鎸佹湁",
                        "key_findings": ["chief ok"],
                        "confidence_score": 0.8,
                    },
                    confidence_score=0.8,
                )

        calls = []
        coord = object.__new__(AgentCoordinator)
        coord.config = {}
        coord.specialist_agent_names = ["FundamentalAgent", "TechnicalAgent"]
        coord.checkpoint_enabled = False
        coord.checkpointer = None
        coord._llm_semaphore = asyncio.Semaphore(2)
        coord.agents = {
            "FundamentalAgent": FakeSpecialist("FundamentalAgent", calls),
            "TechnicalAgent": FakeSpecialist("TechnicalAgent", calls),
            "ChiefStrategyAgent": FakeChief("ChiefStrategyAgent", calls),
        }
        coord.build_workflow()

        async def run_once():
            return await coord.graph.ainvoke(coord._create_initial_state(
                "300502",
                "1m",
                "2026-07-02T00:00:00",
                {"agent_params": {"enabled_agents": ["TechnicalAgent"]}},
            ))

        final_state = asyncio.run(run_once())
        self.assertIn("TechnicalAgent", calls)
        self.assertNotIn("FundamentalAgent", calls)
        self.assertEqual(final_state["active_agent_names"], ["TechnicalAgent"])
        self.assertIn("TechnicalAgent", final_state["agent_results"])
        self.assertNotIn("FundamentalAgent", final_state["agent_results"])

    def test_checkpoint_switch_reads_config_and_env(self):
        import os
        from hengline.agents.agent_coordinator import AgentCoordinator

        old_value = os.environ.get("LANGGRAPH_CHECKPOINTS_ENABLED")
        try:
            os.environ.pop("LANGGRAPH_CHECKPOINTS_ENABLED", None)
            coord = object.__new__(AgentCoordinator)
            coord.config = {"enable_checkpointing": True}
            self.assertTrue(coord._get_checkpoint_enabled())

            os.environ["LANGGRAPH_CHECKPOINTS_ENABLED"] = "false"
            self.assertFalse(coord._get_checkpoint_enabled())

            os.environ["LANGGRAPH_CHECKPOINTS_ENABLED"] = "1"
            self.assertTrue(coord._get_checkpoint_enabled())
        finally:
            if old_value is None:
                os.environ.pop("LANGGRAPH_CHECKPOINTS_ENABLED", None)
            else:
                os.environ["LANGGRAPH_CHECKPOINTS_ENABLED"] = old_value

    def test_checkpoint_code_uses_memory_saver_and_thread_id(self):
        src = open(os.path.join(AGENTS_DIR, "agent_coordinator.py"), encoding='utf-8-sig').read()
        self.assertIn("MemorySaver", src)
        self.assertIn("LANGGRAPH_CHECKPOINTS_ENABLED", src)
        self.assertIn('"thread_id"', src)

    def test_stream_event_normalizer_serializes_agent_result(self):
        from hengline.agents.base_agent import AgentResult
        from hengline.agents.agent_coordinator import AgentCoordinator

        coord = object.__new__(AgentCoordinator)
        raw = {
            "event": "on_chain_end",
            "name": "TechnicalAgent",
            "run_id": "run-1",
            "metadata": {"langgraph_node": "TechnicalAgent"},
            "data": {
                "output": {
                    "agent_results": {
                        "TechnicalAgent": AgentResult(
                            agent_name="TechnicalAgent",
                            success=True,
                            result={"key_findings": ["x"]},
                            confidence_score=0.8,
                        )
                    }
                }
            },
        }

        event = coord._normalize_stream_event(raw)
        result = event["data"]["output"]["agent_results"]["TechnicalAgent"]
        self.assertEqual(event["node"], "TechnicalAgent")
        self.assertTrue(result["success"])
        self.assertEqual(result["confidence_score"], 0.8)

    def test_stream_analysis_events_async_yields_lifecycle_events(self):
        from hengline.agents.agent_coordinator import AgentCoordinator

        class FakeGraph:
            async def astream_events(self, state, **kwargs):
                yield {
                    "event": "on_chain_start",
                    "name": "TechnicalAgent",
                    "run_id": "run-1",
                    "metadata": {"langgraph_node": "TechnicalAgent"},
                    "data": {"input": {"stock_code": state["stock_code"]}},
                }

        coord = object.__new__(AgentCoordinator)
        coord.graph = FakeGraph()
        coord.checkpoint_enabled = False

        async def collect():
            events = []
            async for event in coord.stream_analysis_events_async("300502", "1m"):
                events.append(event)
            return events

        events = asyncio.run(collect())
        self.assertEqual(events[0]["event"], "workflow_started")
        self.assertEqual(events[1]["node"], "TechnicalAgent")
        self.assertEqual(events[-1]["event"], "workflow_finished")

    def test_fastapi_stream_endpoint_is_registered(self):
        src = open(os.path.join(ROOT, "api", "stock_agent_api.py"), encoding='utf-8-sig').read()
        self.assertIn('"/analyze/stream"', src)
        self.assertIn("StreamingResponse", src)
        self.assertIn("stream_analysis_events_async", src)
        self.assertIn("text/event-stream", src)

    def test_sse_encoder_outputs_event_and_data_lines(self):
        from api.stock_agent_api import encode_sse_event

        text = encode_sse_event({"event": "workflow_started", "stock_code": "300502"})
        self.assertTrue(text.startswith("event: workflow_started\n"))
        self.assertIn('"stock_code": "300502"', text)
        self.assertTrue(text.endswith("\n\n"))

    def test_api_request_builds_agent_routing_params(self):
        from api.stock_agent_api import StockAnalysisRequest, build_agent_params

        request = StockAnalysisRequest(
            stock_code="300502",
            enabled_agents=["TechnicalAgent"],
            analysis_focus="technical",
            question="is volume abnormal",
            agent_params={"custom": "value"},
        )

        params = build_agent_params(request)
        self.assertEqual(params["enabled_agents"], ["TechnicalAgent"])
        self.assertEqual(params["analysis_focus"], "technical")
        self.assertEqual(params["question"], "is volume abnormal")
        self.assertEqual(params["custom"], "value")

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
            "active_agent_names": ["TechnicalAgent"],
            "routing_decision": {"routing_reason": "explicit_enabled_agents"},
            "agent_results": {
                "TechnicalAgent": AgentResult(
                    agent_name="TechnicalAgent",
                    success=True,
                    confidence_score=0.8,
                    result={"key_findings": ["x"]},
                )
            },
            "conflict_analysis": {"consensus_direction": "bullish"},
            "workflow_trace": [{"node": "TechnicalAgent", "event": "completed"}],
            "final_result": AgentResult(
                agent_name="ChiefStrategyAgent",
                success=True,
                confidence_score=0.8,
                result={"investment_recommendation": "鎸佹湁", "key_findings": ["x"]},
            ),
        }

        report = coord._generate_analysis_report(final_state)
        self.assertEqual(report["workflow_trace"], final_state["workflow_trace"])
        self.assertIn("topology", report["workflow_metadata"])
        self.assertEqual(report["workflow_metadata"]["active_agent_names"], ["TechnicalAgent"])
        self.assertIn({"from": "agent_router", "to": "TechnicalAgent", "condition": "active_agent_names"},
                      report["workflow_metadata"]["topology"]["edges"])


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
# R10-R11  ChiefStrategyAgent 鎺ュ彈鍐茬獊鍒嗘瀽
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
class TestR10_R11_ChiefConflictAnalysis(unittest.TestCase):

    def _chief_src(self):
        return open(
            os.path.join(AGENTS_DIR, "chief_strategy_agent.py"),
            encoding='utf-8-sig'
        ).read()

    def test_analyze_accepts_conflict_analysis(self):
        """Test helper."""
        src = self._chief_src()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "analyze":
                args = [a.arg for a in node.args.args]
                self.assertIn("conflict_analysis", args,
                    "ChiefStrategyAgent.analyze() 缂哄皯 conflict_analysis 鍙傛暟")
                return
        self.fail("ChiefStrategyAgent 涓壘涓嶅埌 analyze 鏂规硶")

    def test_generate_analysis_prompt_accepts_conflict_analysis(self):
        """Test helper."""
        src = self._chief_src()
        self.assertIn("conflict_analysis: Dict = None", src,
            "_generate_analysis_prompt 缂哄皯 conflict_analysis 鍙傛暟")

    def test_conflict_text_injected_into_prompt(self):
        """Test helper."""
        src = self._chief_src()
        self.assertIn("conflict_text", src,
            "chief_strategy_agent.py 涓湭鎵惧埌 conflict_text 鍙橀噺")
        self.assertIn("conflict_analysis.get(", src,
            "chief_strategy_agent.py 涓湭璇诲彇 conflict_analysis 鍐呭")

    def test_conflict_analysis_passed_to_prompt_call(self):
        """Test helper."""
        src = self._chief_src()
        self.assertIn("conflict_analysis=conflict_analysis", src,
            "analyze() 鏈皢 conflict_analysis 浼犵粰 _generate_analysis_prompt")

    def test_chief_analyze_runtime_with_conflict(self):
        """Test helper."""
        from hengline.agents.chief_strategy_agent import ChiefStrategyAgent

        agent = object.__new__(ChiefStrategyAgent)
        agent._reflection_hint = None
        agent.agent_name = "Chief Strategy Agent"
        agent.description = "Investment strategy expert"
        agent.retriever = None
        agent.memory = None
        agent.investment_recommendations = {"hold": {"description": "hold", "confidence": "medium"}}
        agent.agent_weights = {"FundamentalAgent": 0.5, "TechnicalAgent": 0.5}
        agent.risk_levels = {"medium_risk": {"range": (41, 60), "color": "yellow"}}

        # Build prompt and verify conflict text is injected.
        prompt = agent._generate_analysis_prompt(
            stock_code="300502",
            filtered_results={"FundamentalAgent": {"overall_score": 7}},
            composite_score=65.0,
            overall_risk={"risk_level": "medium_risk", "risk_score": 50.0},
            key_strengths=["revenue growth"],
            key_risks=["valuation pressure"],
            conflict_analysis={
                "conflict_summary": "score divergence exists",
                "score_divergences": ["FundamentalAgent(70) vs TechnicalAgent(30): gap 40"],
                "data_gaps": [],
                "failed_agents": []
            }
        )
        self.assertIn("score divergence exists", prompt)
        self.assertIn("FundamentalAgent", prompt)


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
# 姹囨€昏繍琛?# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
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
