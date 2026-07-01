#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LangGraph coordinator for the stock analysis agents."""

import asyncio
import functools
import os
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, AsyncIterator, Dict, List, Optional, TypedDict

import nest_asyncio
from langgraph.graph import END, START, StateGraph

from config.config import get_ai_config, get_embedding_config
from hengline.agents.base_agent import AgentConfig, AgentResult
from hengline.agents.chief_strategy_agent import ChiefStrategyAgent
from hengline.agents.esg_risk_agent import ESGRiskAgent
from hengline.agents.fund_flow_agent import FundFlowAgent
from hengline.agents.fundamental_agent import FundamentalAgent
from hengline.agents.industry_macro_agent import IndustryMacroAgent
from hengline.agents.sentiment_agent import SentimentAgent
from hengline.agents.technical_agent import TechnicalAgent
from hengline.logger import debug, error, info, performance_logger, warning
from hengline.stock.stock_manage import StockDataManager
from utils.log_utils import print_log_exception

nest_asyncio.apply()


def merge_dicts(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """Merge parallel LangGraph updates for agent_results."""
    merged = left.copy() if left else {}
    merged.update(right or {})
    return merged


def merge_lists(left: Optional[List[Dict[str, Any]]], right: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Append parallel LangGraph trace updates without losing sibling node events."""
    merged = list(left or [])
    merged.extend(right or [])
    return merged


REFLECTION_MAX_RETRIES = 2
QUALITY_GATE_MAX_RETRIES = int(os.getenv("QUALITY_GATE_MAX_RETRIES", "1"))

AGENT_TIMEOUTS: Dict[str, int] = {
    "FundamentalAgent": int(os.getenv("FUNDAMENTAL_TIMEOUT", "120")),
    "TechnicalAgent": int(os.getenv("TECHNICAL_TIMEOUT", "90")),
    "IndustryMacroAgent": int(os.getenv("INDUSTRY_TIMEOUT", "90")),
    "SentimentAgent": int(os.getenv("SENTIMENT_TIMEOUT", "60")),
    "FundFlowAgent": int(os.getenv("FUNDFLOW_TIMEOUT", "60")),
    "ESGRiskAgent": int(os.getenv("ESG_TIMEOUT", "45")),
    "ChiefStrategyAgent": int(os.getenv("CHIEF_TIMEOUT", "120")),
}


class AgentState(TypedDict, total=False):
    stock_code: str
    time_range: str
    agent_params: Dict[str, Any]
    chief_params: Dict[str, Any]
    active_agent_names: List[str]
    retry_agent_names: List[str]
    quality_retry_count: int
    routing_decision: Dict[str, Any]
    quality_gate: Dict[str, Any]
    agent_results: Annotated[Dict[str, AgentResult], merge_dicts]
    workflow_trace: Annotated[List[Dict[str, Any]], merge_lists]
    conflict_analysis: Optional[Dict[str, Any]]
    final_result: Optional[AgentResult]
    analysis_start_time: str


class AgentCoordinator:
    """Coordinates specialist agents, conflict checks, and the chief agent."""

    specialist_agent_names = [
        "FundamentalAgent",
        "TechnicalAgent",
        "IndustryMacroAgent",
        "SentimentAgent",
        "FundFlowAgent",
        "ESGRiskAgent",
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.agents: Dict[str, Any] = {}
        self.graph = None
        self.checkpointer = None
        self.checkpoint_backend_active = None
        self.checkpoint_path_active = None
        self.checkpoint_enabled = self._get_checkpoint_enabled()
        configured_agents = self.config.get("enabled_agents") or self.specialist_agent_names
        self.specialist_agent_names = [
            name for name in configured_agents
            if name in type(self).specialist_agent_names
        ] or list(type(self).specialist_agent_names)
        self._llm_semaphore = asyncio.Semaphore(self._get_max_concurrency())
        self.initialize_agents()
        self.build_workflow()

    def _get_max_concurrency(self) -> int:
        """Return the maximum number of agent LLM calls allowed in parallel."""
        raw_value = os.getenv("AGENT_LLM_CONCURRENCY", self.config.get("max_concurrency", 3))
        try:
            return max(1, int(raw_value))
        except (TypeError, ValueError):
            warning(f"Invalid AGENT_LLM_CONCURRENCY={raw_value!r}; falling back to 3")
            return 3

    def _get_checkpoint_enabled(self) -> bool:
        """Return whether LangGraph should compile with a checkpointer."""
        raw_value = os.getenv(
            "LANGGRAPH_CHECKPOINTS_ENABLED",
            str(getattr(self, "config", {}).get("enable_checkpointing", False)),
        )
        return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}

    def _get_checkpoint_backend(self) -> str:
        """Return requested checkpoint backend: sqlite or memory."""
        raw_value = os.getenv(
            "LANGGRAPH_CHECKPOINT_BACKEND",
            str(getattr(self, "config", {}).get("checkpoint_backend", "memory")),
        )
        return str(raw_value).strip().lower() or "memory"

    def _get_checkpoint_path(self) -> str:
        """Return SQLite checkpoint path when persistent checkpointing is available."""
        raw_value = os.getenv(
            "LANGGRAPH_CHECKPOINT_PATH",
            str(getattr(self, "config", {}).get("checkpoint_path", "data/checkpoints/langgraph.sqlite")),
        )
        return raw_value

    def initialize_agents(self):
        """Initialize all specialist agents and the chief strategy agent."""
        try:
            debug("Initializing stock analysis agents")

            llm_config = get_ai_config()
            embedding_config = get_embedding_config()
            model_name = llm_config.get("model_name", "gpt-4")
            model_type = llm_config.get("provider", "openai").lower()
            enable_memory = embedding_config.get("enable_memory", True)

            # 数据驱动的 Agent 规格表：(类, 显示名, 描述, 额外配置)
            agent_specs = {
                "FundamentalAgent": (
                    FundamentalAgent, "Fundamental Analysis Agent",
                    "Analyzes financial statements, valuation, profitability, and company fundamentals.", {},
                ),
                "TechnicalAgent": (
                    TechnicalAgent, "Technical Analysis Agent",
                    "Analyzes price action, volume, trend, and technical indicators.", {},
                ),
                "IndustryMacroAgent": (
                    IndustryMacroAgent, "Industry Macro Agent",
                    "Analyzes industry trends, macro conditions, and market context.", {},
                ),
                "SentimentAgent": (
                    SentimentAgent, "Sentiment Agent",
                    "Analyzes news, market sentiment, and investor behavior.", {},
                ),
                "FundFlowAgent": (
                    FundFlowAgent, "Fund Flow Agent",
                    "Analyzes capital flow, institutional activity, and volume behavior.", {},
                ),
                "ESGRiskAgent": (
                    ESGRiskAgent, "ESG Risk Agent",
                    "Analyzes ESG, governance, controversy, and sustainability risk.", {},
                ),
                "ChiefStrategyAgent": (
                    ChiefStrategyAgent, "Chief Strategy Agent",
                    "Synthesizes specialist outputs into a final investment recommendation.",
                    {"memory_top_k": 5},
                ),
            }

            overrides = self.config.get("agents", {})
            self.agents = {}
            for name, (agent_cls, display_name, description, extra) in agent_specs.items():
                agent_config = AgentConfig(
                    agent_name=display_name,
                    description=description,
                    model_name=model_name,
                    model_type=model_type,
                    enable_memory=enable_memory,
                    llm_config=llm_config,
                    embedding_config=embedding_config,
                    **extra,
                )
                if name in overrides:
                    agent_config = agent_config.model_copy(update=overrides[name])
                self.agents[name] = agent_cls(agent_config)

            shared_stock_manager = StockDataManager()
            for agent_name, agent in self.agents.items():
                if hasattr(agent, "inject_stock_manager"):
                    agent.inject_stock_manager(shared_stock_manager)
                    debug(f"Injected shared StockDataManager into {agent_name}")

            info(f"Initialized {len(self.agents)} agents")
            info(f"Agent memory enabled: {enable_memory}")

        except Exception as exc:
            error(f"Failed to initialize agents: {exc}")
            raise

    def build_workflow(self):
        """Build the LangGraph workflow."""
        try:
            debug("Building LangGraph workflow")
            graph = StateGraph(AgentState)

            graph.add_node("agent_router", self._create_agent_router_node())
            graph.add_edge(START, "agent_router")

            for name in self.specialist_agent_names:
                graph.add_node(name, self._create_agent_node(name))
                graph.add_edge(name, "agent_quality_gate")

            graph.add_conditional_edges(
                "agent_router",
                self._route_specialist_agents,
                {name: name for name in self.specialist_agent_names},
            )
            graph.add_node("agent_quality_gate", self._create_quality_gate_node())
            graph.add_conditional_edges(
                "agent_quality_gate",
                self._route_after_quality_gate,
                {"retry": "agent_router", "continue": "conflict_analyzer"},
            )
            graph.add_node("conflict_analyzer", self._create_conflict_analyzer_node())
            graph.add_node("ChiefStrategyAgent", self._create_chief_node())
            graph.add_edge("conflict_analyzer", "ChiefStrategyAgent")
            graph.add_edge("ChiefStrategyAgent", END)

            compile_kwargs = {}
            self.checkpointer = None
            self.checkpoint_backend_active = None
            self.checkpoint_path_active = None
            if self.checkpoint_enabled:
                backend = self._get_checkpoint_backend()
                if backend == "sqlite":
                    try:
                        from langgraph.checkpoint.sqlite import SqliteSaver

                        checkpoint_path = Path(self._get_checkpoint_path())
                        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                        self.checkpointer = SqliteSaver.from_conn_string(str(checkpoint_path))
                        self.checkpoint_backend_active = "sqlite"
                        self.checkpoint_path_active = str(checkpoint_path)
                        compile_kwargs["checkpointer"] = self.checkpointer
                        debug(f"LangGraph SQLite checkpointing enabled: {checkpoint_path}")
                    except Exception as exc:
                        warning(
                            "LangGraph SQLite checkpointing unavailable; "
                            f"falling back to MemorySaver: {exc}"
                        )
                        backend = "memory"
                if backend != "sqlite":
                    try:
                        from langgraph.checkpoint.memory import MemorySaver

                        self.checkpointer = MemorySaver()
                        self.checkpoint_backend_active = "memory"
                        self.checkpoint_path_active = None
                        compile_kwargs["checkpointer"] = self.checkpointer
                        debug("LangGraph MemorySaver checkpointing enabled")
                    except Exception as exc:
                        self.checkpointer = None
                        self.checkpoint_enabled = False
                        warning(f"LangGraph checkpointing disabled; MemorySaver unavailable: {exc}")
            else:
                self.checkpointer = None

            self.graph = graph.compile(**compile_kwargs)
            debug("LangGraph workflow built")

        except Exception as exc:
            error(f"Failed to build LangGraph workflow: {exc}")
            raise

    @staticmethod
    def _trace_event(node: str, event: str, **payload) -> Dict[str, Any]:
        """Create a compact workflow event for UI/API diagnostics."""
        clean_payload = {key: value for key, value in payload.items() if value is not None}
        return {
            "node": node,
            "event": event,
            "timestamp": datetime.now().isoformat(),
            **clean_payload,
        }

    def get_workflow_topology(self) -> Dict[str, Any]:
        """Return the concrete LangGraph topology built for this coordinator."""
        edges = [{"from": "START", "to": "agent_router"}]
        for name in self.specialist_agent_names:
            edges.append({"from": "agent_router", "to": name, "condition": "active_agent_names"})
            edges.append({"from": name, "to": "agent_quality_gate"})
        edges.extend([
            {"from": "agent_quality_gate", "to": "agent_router", "condition": "retry"},
            {"from": "agent_quality_gate", "to": "conflict_analyzer", "condition": "continue"},
            {"from": "conflict_analyzer", "to": "ChiefStrategyAgent"},
            {"from": "ChiefStrategyAgent", "to": "END"},
        ])

        return {
            "nodes": [
                "START",
                "agent_router",
                *self.specialist_agent_names,
                "agent_quality_gate",
                "conflict_analyzer",
                "ChiefStrategyAgent",
                "END",
            ],
            "edges": edges,
            "reducers": {
                "agent_results": "merge_dicts",
                "workflow_trace": "merge_lists",
            },
            "enabled_specialists": list(self.specialist_agent_names),
            "langgraph_features": [
                "StateGraph",
                "conditional routing from agent_router",
                "explicit agent quality gate node",
                "conditional graph retry from quality gate to router",
                "reducer-based parallel state merge",
                "fan-in conflict analysis node",
                "final synthesis node",
            ],
            "current_limitations": [
                "node-local reflection retry still handles transient call failures before graph-level retry",
                "checkpointing is in-memory only when LANGGRAPH_CHECKPOINTS_ENABLED is enabled",
                "human approval is represented as guardrails/disclaimers rather than an interrupt node",
            ],
            "checkpointing": {
                "enabled": bool(getattr(self, "checkpoint_enabled", False)),
                "type": type(getattr(self, "checkpointer", None)).__name__ if getattr(self, "checkpointer", None) else None,
                "requested_backend": self._get_checkpoint_backend() if getattr(self, "checkpoint_enabled", False) else None,
                "backend": getattr(self, "checkpoint_backend_active", None),
                "path": getattr(self, "checkpoint_path_active", None),
                "persistence": getattr(self, "checkpoint_backend_active", None),
            },
        }

    @staticmethod
    def _compute_agent_score(agent_name: str, result_dict: dict) -> Optional[float]:
        """Extract a 0-100 score from each specialist result where possible."""
        if not result_dict or not result_dict.get("success", False):
            return None

        data = result_dict.get("result", {}) or {}
        try:
            if agent_name == "FundamentalAgent":
                raw = data.get("overall_score")
                if raw is None:
                    return None
                raw = float(raw)
                return raw * 10 if raw <= 10 else raw

            if agent_name == "TechnicalAgent":
                signal = data.get("signal_strength", "neutral")
                mapping = {
                    "strong_bullish": 85,
                    "bullish": 70,
                    "weak_bullish": 60,
                    "neutral": 50,
                    "weak_bearish": 40,
                    "bearish": 30,
                    "strong_bearish": 15,
                }
                confidence = float(data.get("confidence_score", 0.5) or 0.5)
                return mapping.get(signal, 50) * 0.7 + confidence * 100 * 0.3

            if agent_name == "IndustryMacroAgent":
                industry_score = data.get("industry_analysis", {}).get("industry_score", 50)
                macro_score = data.get("macro_analysis", {}).get("economic_score", 50)
                return (float(industry_score) + float(macro_score)) / 2

            if agent_name == "SentimentAgent":
                news_sentiment = data.get("sentiment_metrics", {}).get("news_sentiment", {})
                positive = float(news_sentiment.get("positive", 0) or 0)
                negative = float(news_sentiment.get("negative", 0) or 0)
                return positive / (positive + negative) * 100 if positive + negative > 0 else 50.0

            if agent_name == "FundFlowAgent":
                flow = data.get("key_metrics", {}).get("flow_classification", "neutral")
                return {
                    "strong_inflow": 90,
                    "moderate_inflow": 75,
                    "weak_inflow": 60,
                    "neutral": 50,
                    "weak_outflow": 40,
                    "moderate_outflow": 25,
                    "strong_outflow": 10,
                }.get(flow, 50.0)

            if agent_name == "ESGRiskAgent":
                return float(data.get("esg_metrics", {}).get("overall_score", 50) or 50)
        except (TypeError, ValueError):
            return None

        return 50.0

    def _select_active_agents(self, state: AgentState) -> Dict[str, Any]:
        """Choose which specialist agents should run for this request."""
        agent_params = state.get("agent_params", {}) or {}
        retry_agents = [
            name for name in (state.get("retry_agent_names") or [])
            if name in self.specialist_agent_names
        ]
        if retry_agents:
            return {
                "selected_agents": retry_agents,
                "available_agents": list(self.specialist_agent_names),
                "routing_reason": "quality_gate_retry",
                "analysis_focus": agent_params.get("analysis_focus") or agent_params.get("analysis_type"),
                "question": agent_params.get("question") or agent_params.get("user_question"),
            }

        requested = (
            agent_params.get("enabled_agents")
            or agent_params.get("agents")
            or self.config.get("enabled_agents")
            or []
        )
        focus_text = " ".join(
            str(item or "")
            for item in (
                agent_params.get("analysis_focus"),
                agent_params.get("analysis_type"),
                agent_params.get("question"),
                agent_params.get("user_question"),
                state.get("time_range"),
            )
        ).lower()

        explicit_agents = [
            name for name in requested
            if name in self.specialist_agent_names
        ] if isinstance(requested, (list, tuple, set)) else []
        if explicit_agents:
            selected = explicit_agents
            reason = "explicit_enabled_agents"
        else:
            selected_set = set()
            keyword_map = {
                "FundamentalAgent": [
                    "fundamental", "financial", "finance", "valuation", "pe", "pb",
                    "eps", "income", "cash flow", "profit", "基本面", "财务", "估值",
                    "盈利", "利润", "现金流", "资产负债",
                ],
                "TechnicalAgent": [
                    "technical", "trend", "kline", "candlestick", "volume", "rsi",
                    "macd", "ma", "技术", "趋势", "k线", "k 线", "成交量", "量价",
                    "均线",
                ],
                "IndustryMacroAgent": [
                    "industry", "macro", "sector", "policy", "benchmark", "compare",
                    "行业", "宏观", "政策", "同业", "可比", "指数", "基准",
                ],
                "SentimentAgent": [
                    "sentiment", "news", "public opinion", "announcement",
                    "情绪", "新闻", "舆情", "公告", "研报", "消息",
                ],
                "FundFlowAgent": [
                    "fund flow", "money flow", "capital", "institution", "northbound", "volume",
                    "资金", "主力", "北向", "机构", "成交", "融资融券",
                ],
                "ESGRiskAgent": [
                    "esg", "governance", "risk", "controversy", "sustainability",
                    "治理", "风险", "争议", "合规", "监管", "可持续",
                ],
            }
            for agent_name, keywords in keyword_map.items():
                if agent_name in self.specialist_agent_names and any(keyword in focus_text for keyword in keywords):
                    selected_set.add(agent_name)

            if not selected_set:
                selected = list(self.specialist_agent_names)
                reason = "default_full_research"
            else:
                selected_set.add("TechnicalAgent")
                selected = [name for name in self.specialist_agent_names if name in selected_set]
                reason = "question_keyword_routing"

        if not selected:
            selected = list(self.specialist_agent_names)
            reason = "fallback_full_research"

        return {
            "selected_agents": selected,
            "available_agents": list(self.specialist_agent_names),
            "routing_reason": reason,
            "analysis_focus": agent_params.get("analysis_focus") or agent_params.get("analysis_type"),
            "question": agent_params.get("question") or agent_params.get("user_question"),
        }

    def _create_agent_router_node(self):
        """Create a LangGraph node that records the per-request routing decision."""

        def agent_router_node(state: AgentState) -> Dict[str, Any]:
            routing_decision = self._select_active_agents(state)
            selected = routing_decision["selected_agents"]
            info(
                "Agent router selected "
                f"{len(selected)}/{len(self.specialist_agent_names)} specialists: {', '.join(selected)}"
            )
            return {
                "active_agent_names": selected,
                "retry_agent_names": [],
                "routing_decision": routing_decision,
                "workflow_trace": [
                    self._trace_event(
                        "agent_router",
                        "completed",
                        selected_agents=selected,
                        routing_reason=routing_decision.get("routing_reason"),
                    )
                ],
            }

        return agent_router_node

    @staticmethod
    def _route_specialist_agents(state: AgentState) -> List[str]:
        """Return specialist node names for LangGraph conditional fan-out."""
        return list(state.get("active_agent_names") or [])

    @staticmethod
    def _route_after_quality_gate(state: AgentState) -> str:
        """Route to a graph-level retry when the quality gate requested it."""
        return "retry" if state.get("retry_agent_names") else "continue"

    def _create_quality_gate_node(self):
        """Create a graph-level validation node for specialist outputs."""

        def quality_gate_node(state: AgentState) -> Dict[str, Any]:
            agent_results = state.get("agent_results", {}) or {}
            active_agents = state.get("active_agent_names") or list(agent_results.keys())
            issues: List[Dict[str, Any]] = []
            passed_agents: List[str] = []
            retryable_agents: List[str] = []

            for agent_name in active_agents:
                result = agent_results.get(agent_name)
                if result is None:
                    issues.append({"agent": agent_name, "type": "missing_result", "severity": "high"})
                    retryable_agents.append(agent_name)
                    continue
                if not result.success:
                    issues.append({
                        "agent": agent_name,
                        "type": "agent_failed",
                        "severity": "high",
                        "message": result.error_message,
                    })
                    retryable_agents.append(agent_name)
                    continue
                data = result.result or {}
                agent_issues = []
                if not data.get("key_findings"):
                    agent_issues.append({"type": "empty_key_findings", "severity": "medium"})
                    retryable_agents.append(agent_name)
                try:
                    if float(result.confidence_score or 0.0) <= 0.1:
                        agent_issues.append({"type": "low_confidence", "severity": "medium"})
                        retryable_agents.append(agent_name)
                except (TypeError, ValueError):
                    agent_issues.append({"type": "invalid_confidence", "severity": "medium"})
                    retryable_agents.append(agent_name)

                quality_level = str(data.get("data_quality_level", "")).lower()
                if data.get("is_simulated") or quality_level in {"simulated", "estimated", "unavailable", "partial"}:
                    agent_issues.append({
                        "type": f"data_quality_{quality_level or 'limited'}",
                        "severity": "high" if quality_level in {"simulated", "unavailable"} else "medium",
                        "message": data.get("data_note", ""),
                    })

                if agent_issues:
                    for issue in agent_issues:
                        issues.append({"agent": agent_name, **issue})
                else:
                    passed_agents.append(agent_name)

            retry_count = int(state.get("quality_retry_count") or 0)
            retry_agent_names = sorted(set(retryable_agents)) if retry_count < QUALITY_GATE_MAX_RETRIES else []
            next_retry_count = retry_count + 1 if retry_agent_names else retry_count
            quality_gate = {
                "passed": not any(issue.get("severity") == "high" for issue in issues),
                "passed_agents": passed_agents,
                "issues": issues,
                "issue_count": len(issues),
                "high_severity_count": sum(1 for issue in issues if issue.get("severity") == "high"),
                "retry_agent_names": retry_agent_names,
                "retry_count": next_retry_count,
                "max_retries": QUALITY_GATE_MAX_RETRIES,
            }
            return {
                "quality_gate": quality_gate,
                "retry_agent_names": retry_agent_names,
                "quality_retry_count": next_retry_count,
                "workflow_trace": [
                    self._trace_event(
                        "agent_quality_gate",
                        "completed",
                        passed=quality_gate["passed"],
                        issue_count=quality_gate["issue_count"],
                        high_severity_count=quality_gate["high_severity_count"],
                        retry_agent_names=retry_agent_names,
                        retry_count=next_retry_count,
                    )
                ],
            }

        return quality_gate_node

    def _create_conflict_analyzer_node(self):
        """Analyze specialist result divergence before the chief synthesis step."""

        def conflict_analyzer_node(state: AgentState) -> Dict[str, Any]:
            agent_results = state.get("agent_results", {}) or {}
            scores: Dict[str, float] = {}
            failed_agents: List[str] = []
            data_gaps: List[str] = []
            score_divergences: List[str] = []

            for name, result_obj in agent_results.items():
                if name == "ChiefStrategyAgent":
                    continue
                if not result_obj.success:
                    failed_agents.append(name)
                    continue

                score = self._compute_agent_score(
                    name,
                    {"success": result_obj.success, "result": result_obj.result},
                )
                if score is not None:
                    scores[name] = score

                result_data = result_obj.result or {}
                if not result_data.get("key_findings"):
                    data_gaps.append(f"{name}: key_findings is empty")
                if result_data.get("data_available") is False:
                    note = result_data.get("data_note", "data unavailable")
                    data_gaps.append(f"{name}: {str(note)[:80]}")
                quality_level = str(result_data.get("data_quality_level", "")).lower()
                if result_data.get("is_simulated") is True or quality_level == "simulated":
                    data_gaps.append(f"{name}: simulated data was used")
                elif quality_level in {"partial", "estimated", "unavailable"}:
                    note = result_data.get("data_note") or f"data_quality_level={quality_level}"
                    data_gaps.append(f"{name}: {str(note)[:80]}")

            names = list(scores.keys())
            for idx, left_name in enumerate(names):
                for right_name in names[idx + 1 :]:
                    diff = abs(scores[left_name] - scores[right_name])
                    if diff > 30:
                        score_divergences.append(
                            f"{left_name}({scores[left_name]:.0f}) vs "
                            f"{right_name}({scores[right_name]:.0f}): gap {diff:.0f}"
                        )

            if scores:
                average_score = sum(scores.values()) / len(scores)
                bullish_count = sum(1 for score in scores.values() if score >= 60)
                bearish_count = sum(1 for score in scores.values() if score <= 40)
                if bullish_count >= len(scores) * 0.6:
                    consensus = "bullish"
                elif bearish_count >= len(scores) * 0.6:
                    consensus = "bearish"
                elif score_divergences:
                    consensus = "divergent"
                else:
                    consensus = "neutral"
            else:
                average_score = 50.0
                consensus = "insufficient_data"

            conflict_result = {
                "has_conflicts": bool(score_divergences or len(failed_agents) >= 2),
                "consensus_direction": consensus,
                "average_score": round(average_score, 1),
                "agent_scores": {name: round(score, 1) for name, score in scores.items()},
                "score_divergences": score_divergences,
                "failed_agents": failed_agents,
                "data_gaps": data_gaps,
                "conflict_summary": (
                    f"Consensus: {consensus}; average score {average_score:.0f}. "
                    f"Divergences: {len(score_divergences)}. "
                    f"Failed agents: {len(failed_agents)}. "
                    f"Data gaps: {len(data_gaps)}."
                ),
            }
            info(f"Conflict analysis complete: {conflict_result['conflict_summary']}")
            return {
                "conflict_analysis": conflict_result,
                "workflow_trace": [
                    self._trace_event(
                        "conflict_analyzer",
                        "completed",
                        consensus_direction=consensus,
                        average_score=round(average_score, 1),
                        divergences=len(score_divergences),
                        failed_agents=len(failed_agents),
                        data_gaps=len(data_gaps),
                    )
                ],
            }

        return conflict_analyzer_node

    def _create_agent_node(self, agent_name: str):
        """Create a specialist node with timeout and reflection retry."""
        agent_timeout = AGENT_TIMEOUTS.get(agent_name, 60)

        async def agent_node(state: AgentState) -> Dict[str, Any]:
            stock_code = state.get("stock_code", "")
            time_range = state.get("time_range", "1y")
            agent_params = state.get("agent_params", {}) or {}
            agent = self.agents[agent_name]
            last_result: Optional[AgentResult] = None

            for attempt in range(REFLECTION_MAX_RETRIES + 1):
                if attempt > 0:
                    debug(f"{agent_name} reflection retry {attempt}")

                try:
                    call = functools.partial(agent.analyze, stock_code, time_range, **agent_params)
                    loop = asyncio.get_event_loop()
                    async with self._llm_semaphore:
                        result = await asyncio.wait_for(
                            loop.run_in_executor(None, call),
                            timeout=agent_timeout,
                        )
                except asyncio.TimeoutError:
                    error(f"{agent_name} timed out after {agent_timeout}s on attempt {attempt}")
                    result = AgentResult(
                        agent_name=agent_name,
                        success=False,
                        result={},
                        error_message=f"Timed out after {agent_timeout}s",
                        confidence_score=0.0,
                    )
                except Exception as exc:
                    error(f"{agent_name} failed on attempt {attempt}: {exc}")
                    result = AgentResult(
                        agent_name=agent_name,
                        success=False,
                        result={},
                        error_message=str(exc),
                        confidence_score=0.0,
                    )

                last_result = result
                validation_error = agent._validate_output(result)
                if validation_error is None:
                    if attempt > 0:
                        info(f"{agent_name} produced a valid result after {attempt} retries")
                    agent.clear_reflection_hint()
                    break

                warning(f"{agent_name} validation failed on attempt {attempt}: {validation_error}")
                if attempt < REFLECTION_MAX_RETRIES:
                    agent.set_reflection_hint(validation_error)
                else:
                    warning(f"{agent_name} reached max retries; using final result")
                    agent.clear_reflection_hint()

            if last_result is None:
                last_result = AgentResult(
                    agent_name=agent_name,
                    success=False,
                    result={},
                    error_message="Agent did not produce a result",
                    confidence_score=0.0,
                )

            debug(f"{agent_name} node completed; success={last_result.success}")
            return {
                "agent_results": {agent_name: last_result},
                "workflow_trace": [
                    self._trace_event(
                        agent_name,
                        "completed",
                        success=last_result.success,
                        attempts=attempt + 1,
                        confidence_score=last_result.confidence_score,
                        error=last_result.error_message if not last_result.success else None,
                    )
                ],
            }

        return agent_node

    def _create_chief_node(self):
        """Create the chief strategy synthesis node."""

        def validate_chief_result(result: AgentResult) -> Optional[str]:
            if not result.success:
                return result.error_message or "ChiefStrategyAgent returned a failed result"
            data = result.result or {}
            if not data:
                return "ChiefStrategyAgent result is empty"
            confidence = data.get("confidence_score", result.confidence_score)
            try:
                if confidence is not None and float(confidence) <= 0.1:
                    return f"confidence_score={confidence} is too low"
            except (TypeError, ValueError):
                return f"confidence_score={confidence!r} is not numeric"
            recommendation = data.get("investment_recommendation")
            if not recommendation:
                return "Missing investment_recommendation"
            allowed = set(getattr(self.agents["ChiefStrategyAgent"], "investment_recommendations", {}).keys())
            if allowed and recommendation not in allowed:
                return f"Unsupported investment_recommendation={recommendation!r}"
            findings = data.get("key_findings")
            if not isinstance(findings, list) or not findings:
                return "Missing non-empty key_findings"
            return None

        async def chief_node(state: AgentState) -> Dict[str, Any]:
            stock_code = state.get("stock_code", "")
            agent_results = state.get("agent_results", {}) or {}
            conflict_analysis = state.get("conflict_analysis", {}) or {}
            chief_params = state.get("chief_params", {}) or {}
            chief_timeout = AGENT_TIMEOUTS.get("ChiefStrategyAgent", 120)
            chief_agent = self.agents["ChiefStrategyAgent"]

            async def run_chief():
                try:
                    loop = asyncio.get_event_loop()
                    call = functools.partial(
                        chief_agent.analyze,
                        stock_code,
                        agent_results,
                        conflict_analysis=conflict_analysis,
                        **chief_params,
                    )
                    async with self._llm_semaphore:
                        return await loop.run_in_executor(None, call)
                except Exception as exc:
                    error(f"ChiefStrategyAgent failed: {exc}")
                    return AgentResult(
                        agent_name="ChiefStrategyAgent",
                        success=False,
                        result={},
                        error_message=str(exc),
                        confidence_score=0.0,
                    )

            result: Optional[AgentResult] = None
            for attempt in range(REFLECTION_MAX_RETRIES + 1):
                if attempt > 0:
                    debug(f"ChiefStrategyAgent reflection retry {attempt}")
                try:
                    result = await asyncio.wait_for(run_chief(), timeout=chief_timeout)
                except asyncio.TimeoutError:
                    error(f"ChiefStrategyAgent timed out after {chief_timeout}s on attempt {attempt}")
                    result = AgentResult(
                        agent_name="ChiefStrategyAgent",
                        success=False,
                        result={},
                        error_message=f"Timed out after {chief_timeout}s",
                        confidence_score=0.0,
                    )

                validation_error = validate_chief_result(result)
                if validation_error is None:
                    chief_agent.clear_reflection_hint()
                    break

                warning(f"ChiefStrategyAgent validation failed on attempt {attempt}: {validation_error}")
                if attempt < REFLECTION_MAX_RETRIES:
                    chief_agent.set_reflection_hint(validation_error)
                else:
                    chief_agent.clear_reflection_hint()

            debug("ChiefStrategyAgent node completed")
            return {
                "final_result": result,
                "workflow_trace": [
                    self._trace_event(
                        "ChiefStrategyAgent",
                        "completed",
                        success=result.success if result else False,
                        attempts=attempt + 1,
                        confidence_score=result.confidence_score if result else 0.0,
                        error=result.error_message if result and not result.success else None,
                    )
                ],
            }

        return chief_node

    def _create_initial_state(
        self,
        stock_code: str,
        time_range: str,
        analysis_start_time: str,
        kwargs: Dict[str, Any],
    ) -> AgentState:
        """Build the initial LangGraph state shared by invoke and streaming paths."""
        return AgentState(
            stock_code=stock_code,
            time_range=time_range,
            agent_params=kwargs.get("agent_params", {}),
            chief_params=kwargs.get("chief_params", {}),
            active_agent_names=[],
            retry_agent_names=[],
            quality_retry_count=0,
            routing_decision={},
            quality_gate={},
            agent_results={},
            workflow_trace=[],
            conflict_analysis=None,
            final_result=None,
            analysis_start_time=analysis_start_time,
        )

    def _build_workflow_config(
        self,
        stock_code: str,
        analysis_start_time: str,
        kwargs: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Build optional LangGraph runnable config, including checkpoint thread id."""
        workflow_config = kwargs.get("workflow_config")
        if self.checkpoint_enabled:
            thread_id = kwargs.get("thread_id") or f"{stock_code}-{analysis_start_time}"
            workflow_config = {
                **(workflow_config or {}),
                "configurable": {
                    **((workflow_config or {}).get("configurable", {})),
                    "thread_id": thread_id,
                },
            }
        return workflow_config

    @staticmethod
    def _safe_event_value(value: Any) -> Any:
        """Convert event values into JSON-friendly diagnostics."""
        if isinstance(value, AgentResult):
            return {
                "agent_name": value.agent_name,
                "success": value.success,
                "confidence_score": value.confidence_score,
                "error_message": value.error_message,
            }
        if isinstance(value, dict):
            return {key: AgentCoordinator._safe_event_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [AgentCoordinator._safe_event_value(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def _normalize_stream_event(self, raw_event: Dict[str, Any]) -> Dict[str, Any]:
        """Reduce LangGraph stream events to a stable UI/API event shape."""
        metadata = raw_event.get("metadata") or {}
        data = raw_event.get("data") or {}
        return {
            "event": raw_event.get("event", "unknown"),
            "name": raw_event.get("name", ""),
            "node": metadata.get("langgraph_node") or raw_event.get("name", ""),
            "timestamp": datetime.now().isoformat(),
            "run_id": raw_event.get("run_id"),
            "data": self._safe_event_value(data),
        }

    async def stream_analysis_events_async(
        self,
        stock_code: str,
        time_range: str = "1y",
        **kwargs,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream normalized LangGraph execution events without changing analyze_async."""
        analysis_start_time = datetime.now().isoformat()
        initial_state = self._create_initial_state(stock_code, time_range, analysis_start_time, kwargs)
        workflow_config = self._build_workflow_config(stock_code, analysis_start_time, kwargs)
        final_state: Optional[Dict[str, Any]] = None

        yield {
            "event": "workflow_started",
            "name": "AgentCoordinator",
            "node": "START",
            "timestamp": analysis_start_time,
            "stock_code": stock_code,
            "time_range": time_range,
        }

        try:
            stream_kwargs = {"version": "v2"}
            if workflow_config:
                stream_kwargs["config"] = workflow_config

            async for raw_event in self.graph.astream_events(initial_state, **stream_kwargs):
                normalized = self._normalize_stream_event(raw_event)
                yield normalized

                output = (raw_event.get("data") or {}).get("output")
                if isinstance(output, dict) and (
                    "final_result" in output or "agent_results" in output or "conflict_analysis" in output
                ):
                    final_state = output

            finished_at = datetime.now().isoformat()
            if final_state and final_state.get("final_result") is not None:
                report = self._generate_analysis_report(final_state)
                report["analysis_time"] = finished_at
                report["elapsed_time_seconds"] = (
                    datetime.fromisoformat(finished_at)
                    - datetime.fromisoformat(analysis_start_time)
                ).total_seconds()
                yield {
                    "event": "workflow_finished",
                    "name": "AgentCoordinator",
                    "node": "END",
                    "timestamp": finished_at,
                    "report": report,
                }
            else:
                yield {
                    "event": "workflow_finished",
                    "name": "AgentCoordinator",
                    "node": "END",
                    "timestamp": finished_at,
                }
        except Exception as exc:
            yield {
                "event": "workflow_error",
                "name": "AgentCoordinator",
                "node": "ERROR",
                "timestamp": datetime.now().isoformat(),
                "error": str(exc),
            }

    async def analyze_async(self, stock_code: str, time_range: str = "1y", **kwargs) -> Dict[str, Any]:
        """Run the complete async analysis workflow."""
        try:
            debug(f"Starting coordinated analysis for {stock_code}")
            analysis_start_time = datetime.now().isoformat()
            initial_state = self._create_initial_state(stock_code, time_range, analysis_start_time, kwargs)

            @performance_logger(f"Run coordinated analysis - {stock_code}")
            async def execute_workflow():
                workflow_config = self._build_workflow_config(stock_code, analysis_start_time, kwargs)
                if workflow_config:
                    return await self.graph.ainvoke(initial_state, config=workflow_config)
                return await self.graph.ainvoke(initial_state)

            final_state = await execute_workflow()
            report = self._generate_analysis_report(final_state)

            analysis_end_time = datetime.now().isoformat()
            report["analysis_time"] = analysis_end_time
            report["elapsed_time_seconds"] = (
                datetime.fromisoformat(analysis_end_time)
                - datetime.fromisoformat(analysis_start_time)
            ).total_seconds()

            debug(f"Coordinated analysis complete for {stock_code}: {report['elapsed_time_seconds']:.2f}s")
            return report

        except Exception as exc:
            error(f"Coordinated analysis failed: {exc}")
            print_log_exception()
            return {
                "success": False,
                "error": str(exc),
                "stock_code": stock_code,
                "analysis_time": datetime.now().isoformat(),
            }

    def analyze(self, stock_code: str, time_range: str = "1y", **kwargs) -> Dict[str, Any]:
        """Run the complete analysis workflow from synchronous callers."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                task = loop.create_task(self.analyze_async(stock_code, time_range, **kwargs))
                return loop.run_until_complete(task)
            return loop.run_until_complete(self.analyze_async(stock_code, time_range, **kwargs))
        except Exception as exc:
            error(f"Synchronous analysis failed: {exc}")
            return {"success": False, "error": str(exc), "stock_code": stock_code}

    def _generate_analysis_report(self, final_state: Dict[str, Any]) -> Dict[str, Any]:
        """Convert the final LangGraph state into the UI/API report shape."""
        report = {
            "success": True,
            "stock_code": final_state.get("stock_code", ""),
            "analysis_start_time": final_state.get("analysis_start_time", ""),
            "agent_execution_status": {},
            "final_recommendation": {},
            "detailed_results": {},
            "conflict_analysis": final_state.get("conflict_analysis", {}) or {},
            "quality_gate": final_state.get("quality_gate", {}) or {},
            "workflow_trace": final_state.get("workflow_trace", []) or [],
            "workflow_metadata": {
                "specialist_agents": self.specialist_agent_names,
                "active_agent_names": final_state.get("active_agent_names", []) or [],
                "routing_decision": final_state.get("routing_decision", {}) or {},
                "join_node": "conflict_analyzer",
                "final_node": "ChiefStrategyAgent",
                "execution_model": "conditional_specialist_routing_then_synthesis",
                "topology": self.get_workflow_topology(),
                "checkpointing_enabled": bool(getattr(self, "checkpoint_enabled", False)),
                "checkpointer_type": type(getattr(self, "checkpointer", None)).__name__ if getattr(self, "checkpointer", None) else None,
                "checkpoint_backend": getattr(self, "checkpoint_backend_active", None),
                "checkpoint_path": getattr(self, "checkpoint_path_active", None),
            },
        }

        agent_results = final_state.get("agent_results", {}) or {}
        for agent_name, result in agent_results.items():
            report["agent_execution_status"][agent_name] = {
                "success": result.success,
                "confidence_score": result.confidence_score,
                "error": result.error_message if not result.success else None,
            }
            if result.success:
                report["detailed_results"][agent_name] = result.result

        final_result = final_state.get("final_result")
        if final_result:
            report["agent_execution_status"]["ChiefStrategyAgent"] = {
                "success": final_result.success,
                "confidence_score": final_result.confidence_score,
                "error": final_result.error_message if not final_result.success else None,
            }

        if final_result and final_result.success:
            report["final_recommendation"] = final_result.result
            report["success"] = True
        else:
            report["success"] = False
            report["error"] = final_result.error_message if final_result else "ChiefStrategyAgent failed"

        return report

    def get_agent_status(self) -> Dict[str, Any]:
        """Return the currently initialized agent list."""
        return {
            agent_name: {
                "name": agent.agent_name,
                "description": agent.description,
                "is_initialized": True,
            }
            for agent_name, agent in self.agents.items()
        }

    def update_agent_config(self, agent_name: str, config: Dict[str, Any]):
        """Update one agent config and rebuild the graph when needed."""
        try:
            if agent_name not in self.agents:
                warning(f"Agent does not exist: {agent_name}")
                return

            agent = self.agents[agent_name]
            if hasattr(agent, "config") and isinstance(agent.config, AgentConfig):
                new_config = agent.config.model_copy(update=config)
                agent.config = new_config

                if "enable_memory" in config:
                    if config["enable_memory"] and not agent.memory:
                        agent.memory = agent._init_memory()
                    elif not config["enable_memory"] and agent.memory:
                        agent.memory = None

                if "model_name" in config or "embedding_model" in config:
                    agent.langchain_llm = agent._get_langchain_llm()
                    if agent.memory:
                        agent.memory = agent._init_memory()

            if "model_name" in config or "enable_memory" in config:
                self.build_workflow()

            info(f"Updated agent config: {agent_name}")
        except Exception as exc:
            error(f"Failed to update agent config: {exc}")
            raise

    def execute_single_agent(self, agent_name: str, stock_code: str, time_range: str = "1y", **kwargs) -> AgentResult:
        """Run a single specialist agent, mainly for tests and diagnostics."""
        try:
            if agent_name not in self.agents:
                warning(f"Agent does not exist: {agent_name}")
                return AgentResult(
                    agent_name=agent_name,
                    success=False,
                    result={},
                    error_message="Agent does not exist",
                    confidence_score=0.0,
                )

            agent = self.agents[agent_name]
            use_memory = kwargs.pop("use_memory", agent.config.enable_memory if hasattr(agent, "config") else True)
            original_enable_memory = None

            if hasattr(agent, "config") and hasattr(agent, "memory"):
                original_enable_memory = agent.config.enable_memory
                agent.config.enable_memory = use_memory
                if use_memory and not agent.memory:
                    agent.memory = agent._init_memory()

            result = agent.analyze(stock_code, time_range, **kwargs)

            if original_enable_memory is not None:
                agent.config.enable_memory = original_enable_memory

            info(f"Single agent complete: {agent_name}; success={result.success}; memory={use_memory}")
            return result

        except Exception as exc:
            error(f"Single agent execution failed: {exc}")
            return AgentResult(
                agent_name=agent_name,
                success=False,
                result={},
                error_message=str(exc),
                confidence_score=0.0,
            )
