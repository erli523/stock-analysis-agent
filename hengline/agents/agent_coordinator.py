#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LangGraph coordinator for the stock analysis agents."""

import asyncio
import functools
import os
from datetime import datetime
from typing import Annotated, Any, Dict, List, Optional, TypedDict

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


REFLECTION_MAX_RETRIES = 2

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
    agent_results: Annotated[Dict[str, AgentResult], merge_dicts]
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

    def initialize_agents(self):
        """Initialize all specialist agents and the chief strategy agent."""
        try:
            debug("Initializing stock analysis agents")

            llm_config = get_ai_config()
            embedding_config = get_embedding_config()
            model_name = llm_config.get("model_name", "gpt-4")
            model_type = llm_config.get("provider", "openai").lower()
            enable_memory = embedding_config.get("enable_memory", True)

            agent_configs = {
                "FundamentalAgent": AgentConfig(
                    agent_name="Fundamental Analysis Agent",
                    description="Analyzes financial statements, valuation, profitability, and company fundamentals.",
                    model_name=model_name,
                    model_type=model_type,
                    enable_memory=enable_memory,
                    llm_config=llm_config,
                    embedding_config=embedding_config,
                ),
                "TechnicalAgent": AgentConfig(
                    agent_name="Technical Analysis Agent",
                    description="Analyzes price action, volume, trend, and technical indicators.",
                    model_name=model_name,
                    model_type=model_type,
                    enable_memory=enable_memory,
                    llm_config=llm_config,
                    embedding_config=embedding_config,
                ),
                "IndustryMacroAgent": AgentConfig(
                    agent_name="Industry Macro Agent",
                    description="Analyzes industry trends, macro conditions, and market context.",
                    model_name=model_name,
                    model_type=model_type,
                    enable_memory=enable_memory,
                    llm_config=llm_config,
                    embedding_config=embedding_config,
                ),
                "SentimentAgent": AgentConfig(
                    agent_name="Sentiment Agent",
                    description="Analyzes news, market sentiment, and investor behavior.",
                    model_name=model_name,
                    model_type=model_type,
                    enable_memory=enable_memory,
                    llm_config=llm_config,
                    embedding_config=embedding_config,
                ),
                "FundFlowAgent": AgentConfig(
                    agent_name="Fund Flow Agent",
                    description="Analyzes capital flow, institutional activity, and volume behavior.",
                    model_name=model_name,
                    model_type=model_type,
                    enable_memory=enable_memory,
                    llm_config=llm_config,
                    embedding_config=embedding_config,
                ),
                "ESGRiskAgent": AgentConfig(
                    agent_name="ESG Risk Agent",
                    description="Analyzes ESG, governance, controversy, and sustainability risk.",
                    model_name=model_name,
                    model_type=model_type,
                    enable_memory=enable_memory,
                    llm_config=llm_config,
                    embedding_config=embedding_config,
                ),
                "ChiefStrategyAgent": AgentConfig(
                    agent_name="Chief Strategy Agent",
                    description="Synthesizes specialist outputs into a final investment recommendation.",
                    model_name=model_name,
                    model_type=model_type,
                    enable_memory=enable_memory,
                    memory_top_k=5,
                    llm_config=llm_config,
                    embedding_config=embedding_config,
                ),
            }

            for agent_name, specific_config in self.config.get("agents", {}).items():
                if agent_name in agent_configs:
                    agent_configs[agent_name] = agent_configs[agent_name].model_copy(update=specific_config)

            self.agents = {
                "FundamentalAgent": FundamentalAgent(agent_configs["FundamentalAgent"]),
                "TechnicalAgent": TechnicalAgent(agent_configs["TechnicalAgent"]),
                "IndustryMacroAgent": IndustryMacroAgent(agent_configs["IndustryMacroAgent"]),
                "SentimentAgent": SentimentAgent(agent_configs["SentimentAgent"]),
                "FundFlowAgent": FundFlowAgent(agent_configs["FundFlowAgent"]),
                "ESGRiskAgent": ESGRiskAgent(agent_configs["ESGRiskAgent"]),
                "ChiefStrategyAgent": ChiefStrategyAgent(agent_configs["ChiefStrategyAgent"]),
            }

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

            for name in self.specialist_agent_names:
                graph.add_node(name, self._create_agent_node(name))
                graph.add_edge(START, name)
                graph.add_edge(name, "conflict_analyzer")

            graph.add_node("conflict_analyzer", self._create_conflict_analyzer_node())
            graph.add_node("ChiefStrategyAgent", self._create_chief_node())
            graph.add_edge("conflict_analyzer", "ChiefStrategyAgent")
            graph.add_edge("ChiefStrategyAgent", END)

            self.graph = graph.compile()
            debug("LangGraph workflow built")

        except Exception as exc:
            error(f"Failed to build LangGraph workflow: {exc}")
            raise

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
                    consensus = "偏多"
                elif bearish_count >= len(scores) * 0.6:
                    consensus = "偏空"
                elif score_divergences:
                    consensus = "分歧"
                else:
                    consensus = "中性"
            else:
                average_score = 50.0
                consensus = "数据不足"

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
            return {"conflict_analysis": conflict_result}

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
            return {"agent_results": {agent_name: last_result}}

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
            return {"final_result": result}

        return chief_node

    async def analyze_async(self, stock_code: str, time_range: str = "1y", **kwargs) -> Dict[str, Any]:
        """Run the complete async analysis workflow."""
        try:
            debug(f"Starting coordinated analysis for {stock_code}")
            analysis_start_time = datetime.now().isoformat()
            initial_state = AgentState(
                stock_code=stock_code,
                time_range=time_range,
                agent_params=kwargs.get("agent_params", {}),
                chief_params=kwargs.get("chief_params", {}),
                agent_results={},
                conflict_analysis=None,
                final_result=None,
                analysis_start_time=analysis_start_time,
            )

            @performance_logger(f"Run coordinated analysis - {stock_code}")
            async def execute_workflow():
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
            "workflow_metadata": {
                "specialist_agents": self.specialist_agent_names,
                "join_node": "conflict_analyzer",
                "final_node": "ChiefStrategyAgent",
                "execution_model": "parallel_specialists_then_synthesis",
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
