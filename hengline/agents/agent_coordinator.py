#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@FileName: agent_coordinator.py
@Description: 智能体协调器，负责协调各专业智能体的工作流程，实现并行推理和结果整合
@Author: HengLine
@Time: 2025/11/10
"""

import asyncio
import functools
import os
from datetime import datetime
from typing import Dict, Any, Optional, TypedDict, Annotated, Callable

import nest_asyncio
from langgraph.graph import StateGraph, END

from config.config import get_ai_config, get_embedding_config
from hengline.agents.base_agent import AgentResult, AgentConfig
from hengline.agents.chief_strategy_agent import ChiefStrategyAgent
from hengline.agents.esg_risk_agent import ESGRiskAgent
from hengline.agents.fund_flow_agent import FundFlowAgent
from hengline.agents.fundamental_agent import FundamentalAgent
from hengline.agents.industry_macro_agent import IndustryMacroAgent
from hengline.agents.sentiment_agent import SentimentAgent
from hengline.agents.technical_agent import TechnicalAgent
from hengline.logger import debug, info, error, warning, performance_logger
from hengline.stock.stock_manage import StockDataManager
from utils.log_utils import print_log_exception

# 应用nest_asyncio以支持嵌套事件循环
nest_asyncio.apply()


def merge_dicts(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """
    自定义字典合并函数，用于处理agent_results的并行更新
    """
    merged = left.copy() if left else {}
    merged.update(right)
    return merged

# Reflection Loop 最大重试次数（每个 Agent 最多额外重试 N 次）
REFLECTION_MAX_RETRIES = 2

# 各 Agent 差异化超时（秒）
AGENT_TIMEOUTS: Dict[str, int] = {
    "FundamentalAgent":   int(os.getenv("FUNDAMENTAL_TIMEOUT",   "120")),
    "TechnicalAgent":     int(os.getenv("TECHNICAL_TIMEOUT",      "90")),
    "IndustryMacroAgent": int(os.getenv("INDUSTRY_TIMEOUT",       "90")),
    "SentimentAgent":     int(os.getenv("SENTIMENT_TIMEOUT",      "60")),
    "FundFlowAgent":      int(os.getenv("FUNDFLOW_TIMEOUT",       "60")),
    "ESGRiskAgent":       int(os.getenv("ESG_TIMEOUT",            "45")),
    "ChiefStrategyAgent": int(os.getenv("CHIEF_TIMEOUT",         "120")),
}


class AgentState(TypedDict):
    """
    LangGraph 工作流状态。
    支持 Reflection Loop（重试追踪）和 ConflictAnalyzer（冲突分析）。
    """
    stock_code: str
    time_range: str
    agent_params: Dict[str, Any]
    chief_params: Dict[str, Any]
    agent_results: Annotated[Dict[str, Any], merge_dicts]
    conflict_analysis: Optional[Dict[str, Any]]   # ConflictAnalyzer 写入，Chief 读取
    final_result: Optional[Dict[str, Any]]


class AgentCoordinator:
    """
    智能体协调器，管理多智能体工作流和并行推理
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化智能体协调器
        
        Args:
            config: 协调器配置
        """
        self.config = config or {}
        self.agents = {}
        self.graph = None
        self.initialize_agents()
        self.build_workflow()

    def _get_timeout(self, env_name: str, default: int) -> int:
        try:
            return int(os.getenv(env_name, str(default)))
        except ValueError:
            warning(f"Invalid timeout value for {env_name}; using {default}s")
            return default

    def initialize_agents(self):
        """
        初始化所有专业智能体
        """
        try:
            debug("开始初始化专业智能体")

            # 获取全局配置
            llm_config = get_ai_config()
            embedding_config = get_embedding_config()
            model_name = llm_config.get("model_name", "gpt-4")
            model_type = llm_config.get("provider", "openai").lower()
            enable_memory = embedding_config.get("enable_memory", True)

            # 初始化各专业智能体的配置
            agent_configs = {
                "FundamentalAgent": AgentConfig(
                    agent_name="基本面分析智能体",
                    description="专业的股票基本面分析师，擅长分析公司财务状况、盈利能力和估值水平",
                    model_name=model_name,
                    model_type=model_type,
                    enable_memory=enable_memory,
                    llm_config=llm_config,
                    embedding_config=embedding_config
                ),
                "TechnicalAgent": AgentConfig(
                    agent_name="技术面分析智能体",
                    description="专业的股票技术分析师，擅长分析价格走势、交易量和技术指标",
                    model_name=model_name,
                    model_type=model_type,
                    enable_memory=enable_memory,
                    llm_config=llm_config,
                    embedding_config=embedding_config
                ),
                "IndustryMacroAgent": AgentConfig(
                    agent_name="行业宏观分析智能体",
                    description="专业的行业和宏观经济分析师，擅长分析行业趋势和宏观经济环境",
                    model_name=model_name,
                    model_type=model_type,
                    enable_memory=enable_memory,
                    llm_config=llm_config,
                    embedding_config=embedding_config
                ),
                "SentimentAgent": AgentConfig(
                    agent_name="舆情情绪分析智能体",
                    description="专业的市场情绪分析师，擅长分析新闻舆情和市场情绪",
                    model_name=model_name,
                    model_type=model_type,
                    enable_memory=enable_memory,
                    llm_config=llm_config,
                    embedding_config=embedding_config
                ),
                "FundFlowAgent": AgentConfig(
                    agent_name="资金流分析智能体",
                    description="专业的资金流向分析师，擅长分析机构持仓和资金流动",
                    model_name=model_name,
                    model_type=model_type,
                    enable_memory=enable_memory,
                    llm_config=llm_config,
                    embedding_config=embedding_config
                ),
                "ESGRiskAgent": AgentConfig(
                    agent_name="ESG风险分析智能体",
                    description="专业的ESG和公司治理分析师，擅长评估企业可持续发展风险",
                    model_name=model_name,
                    model_type=model_type,
                    enable_memory=enable_memory,
                    llm_config=llm_config,
                    embedding_config=embedding_config
                ),
                "ChiefStrategyAgent": AgentConfig(
                    agent_name="首席策略官智能体",
                    description="资深投资策略专家，负责整合各方面分析并提供最终投资建议",
                    model_name=model_name,
                    model_type=model_type,
                    enable_memory=enable_memory,
                    memory_top_k=5,  # 策略官需要更多的历史记忆
                    llm_config=llm_config,
                    embedding_config=embedding_config
                )
            }

            # 应用每个智能体的特定配置
            for agent_name, specific_config in self.config.get("agents", {}).items():
                if agent_name in agent_configs:
                    agent_configs[agent_name] = agent_configs[agent_name].model_copy(update=specific_config)

            # 初始化各专业智能体
            self.agents = {
                "FundamentalAgent": FundamentalAgent(agent_configs["FundamentalAgent"]),
                "TechnicalAgent": TechnicalAgent(agent_configs["TechnicalAgent"]),
                "IndustryMacroAgent": IndustryMacroAgent(agent_configs["IndustryMacroAgent"]),
                "SentimentAgent": SentimentAgent(agent_configs["SentimentAgent"]),
                "FundFlowAgent": FundFlowAgent(agent_configs["FundFlowAgent"]),
                "ESGRiskAgent": ESGRiskAgent(agent_configs["ESGRiskAgent"]),
                "ChiefStrategyAgent": ChiefStrategyAgent(agent_configs["ChiefStrategyAgent"])
            }

            # 创建共享 StockDataManager 并注入所有子 Agent，避免重复拉取同只股票数据
            shared_stock_manager = StockDataManager()
            for agent_name, agent in self.agents.items():
                if hasattr(agent, 'inject_stock_manager'):
                    agent.inject_stock_manager(shared_stock_manager)
                    debug(f"已向 {agent_name} 注入共享 StockDataManager")
            debug("共享 StockDataManager 注入完成")

            # 记录初始化的智能体数量
            info(f"成功初始化 {len(self.agents)} 个智能体")
            info(f"记忆功能: {'已启用' if enable_memory else '已禁用'}")

        except Exception as e:
            error(f"初始化智能体失败: {str(e)}")
            raise

    def build_workflow(self):
        """
        构建智能体工作流程。
        图结构：[6个Agent带Reflection循环] → conflict_analyzer → Chief → END
        """
        try:
            debug("开始构建智能体工作流（含 Reflection Loop）")

            self.graph = StateGraph(AgentState)

            # 注册各专业 Agent 节点（含内部 Reflection Loop）
            specialist_names = [n for n in self.agents if n != "ChiefStrategyAgent"]
            for agent_name in specialist_names:
                self.graph.add_node(agent_name, self._create_agent_node(agent_name))

            # 注册冲突检测节点
            self.graph.add_node("conflict_analyzer", self._create_conflict_analyzer_node())

            # 注册首席策略官节点
            self.graph.add_node("ChiefStrategyAgent", self._create_chief_node())

            # 所有专业 Agent 从 START 并行出发
            for name in specialist_names:
                self.graph.add_edge("__start__", name)

            # 所有专业 Agent 完成后汇入 conflict_analyzer
            for name in specialist_names:
                self.graph.add_edge(name, "conflict_analyzer")

            # conflict_analyzer → Chief → END
            self.graph.add_edge("conflict_analyzer", "ChiefStrategyAgent")
            self.graph.add_edge("ChiefStrategyAgent", END)

            self.graph = self.graph.compile()
            debug("工作流构建完成（Reflection Loop + ConflictAnalyzer）")

        except Exception as e:
            error(f"构建工作流失败: {str(e)}")
            raise

    # ── 冲突检测：纯 Python 逻辑，无需 LLM ─────────────────────────────
    @staticmethod
    def _compute_agent_score(agent_name: str, result_dict: dict) -> Optional[float]:
        """从 agent 结果 dict 提取 0-100 评分。"""
        if not result_dict or not result_dict.get("success", False):
            return None
        d = result_dict.get("result", {})
        if agent_name == "FundamentalAgent":
            raw = d.get("overall_score", 0)
            return float(raw) * 10 if raw and raw <= 10 else (float(raw) if raw else None)
        if agent_name == "TechnicalAgent":
            sig = d.get("signal_strength", "neutral")
            mapping = {"strong_bullish": 85, "bullish": 70, "weak_bullish": 60,
                       "neutral": 50, "weak_bearish": 40, "bearish": 30, "strong_bearish": 15}
            base = mapping.get(sig, 50)
            conf = d.get("confidence_score", 0.5)
            return base * 0.7 + conf * 100 * 0.3
        if agent_name == "IndustryMacroAgent":
            i_score = d.get("industry_analysis", {}).get("industry_score", 50)
            m_score = d.get("macro_analysis", {}).get("economic_score", 50)
            return (float(i_score) + float(m_score)) / 2
        if agent_name == "SentimentAgent":
            sm = d.get("sentiment_metrics", {})
            ns = sm.get("news_sentiment", {})
            pos, neg = ns.get("positive", 0), ns.get("negative", 0)
            return (pos / (pos + neg) * 100) if (pos + neg) > 0 else 50.0
        if agent_name == "FundFlowAgent":
            fc = d.get("key_metrics", {}).get("flow_classification", "neutral")
            return {"strong_inflow": 90, "moderate_inflow": 75, "weak_inflow": 60,
                    "neutral": 50, "weak_outflow": 40, "moderate_outflow": 25,
                    "strong_outflow": 10}.get(fc, 50.0)
        if agent_name == "ESGRiskAgent":
            return float(d.get("esg_metrics", {}).get("overall_score", 50) or 50)
        return 50.0

    def _create_conflict_analyzer_node(self):
        """
        冲突检测节点：分析各 Agent 结果之间的评分分歧、数据缺口和方向矛盾。
        纯 Python 逻辑，不调用 LLM，执行极快。
        """
        def conflict_analyzer_node(state: AgentState) -> Dict[str, Any]:
            agent_results = state.get("agent_results", {})

            scores: Dict[str, float] = {}
            failed_agents: List[str] = []
            data_gaps: List[str] = []
            score_divergences: List[str] = []

            for name, result_obj in agent_results.items():
                if name == "ChiefStrategyAgent":
                    continue
                result_dict = {
                    "success": result_obj.success,
                    "result": result_obj.result,
                }
                if not result_obj.success:
                    failed_agents.append(name)
                    continue
                score = self._compute_agent_score(name, result_dict)
                if score is not None:
                    scores[name] = score
                r = result_obj.result or {}
                if not r.get("key_findings"):
                    data_gaps.append(f"{name}: key_findings 为空")
                if r.get("data_available") is False:
                    note = r.get("data_note", "数据不可用")
                    data_gaps.append(f"{name}: {str(note)[:60]}")

            names_list = list(scores.keys())
            for i in range(len(names_list)):
                for j in range(i + 1, len(names_list)):
                    a, b = names_list[i], names_list[j]
                    diff = abs(scores[a] - scores[b])
                    if diff > 30:
                        score_divergences.append(
                            f"{a}({scores[a]:.0f}分) vs {b}({scores[b]:.0f}分)：差距 {diff:.0f} 分"
                        )

            if scores:
                avg = sum(scores.values()) / len(scores)
                bullish_count = sum(1 for s in scores.values() if s >= 60)
                bearish_count = sum(1 for s in scores.values() if s <= 40)
                if bullish_count >= len(scores) * 0.6:
                    consensus = "偏多"
                elif bearish_count >= len(scores) * 0.6:
                    consensus = "偏空"
                elif score_divergences:
                    consensus = "分歧"
                else:
                    consensus = "中性"
            else:
                avg = 50.0
                consensus = "数据不足"

            has_conflicts = bool(score_divergences or len(failed_agents) >= 2)

            conflict_result = {
                "has_conflicts": has_conflicts,
                "consensus_direction": consensus,
                "average_score": round(avg, 1),
                "agent_scores": {k: round(v, 1) for k, v in scores.items()},
                "score_divergences": score_divergences,
                "failed_agents": failed_agents,
                "data_gaps": data_gaps,
                "conflict_summary": (
                    f"共识方向：{consensus}（均分 {avg:.0f}）。"
                    + (f" 存在 {len(score_divergences)} 处显著评分分歧。" if score_divergences else " 各维度无显著分歧。")
                    + (f" {len(failed_agents)} 个 Agent 执行失败。" if failed_agents else "")
                    + (f" {len(data_gaps)} 处数据缺口。" if data_gaps else "")
                )
            }

            conflict_summary_msg = conflict_result["conflict_summary"]
            info(f"冲突分析完成：{conflict_summary_msg}")
            return {"conflict_analysis": conflict_result}

        return conflict_analyzer_node

    def _create_agent_node(self, agent_name: str):
        """
        创建带 Reflection Loop 的智能体节点。
        - 每次执行后验证输出结构
        - 验证失败时将错误注入 Agent._reflection_hint，最多重试 REFLECTION_MAX_RETRIES 次
        """
        agent_timeout = AGENT_TIMEOUTS.get(agent_name, 60)

        async def agent_node(state: AgentState) -> Dict[str, Any]:
            stock_code   = state.get("stock_code", "")
            time_range   = state.get("time_range", "1y")
            agent_params = state.get("agent_params", {})
            agent        = self.agents[agent_name]

            last_result = None

            for attempt in range(REFLECTION_MAX_RETRIES + 1):
                if attempt > 0:
                    debug(f"{agent_name} Reflection 重试第 {attempt} 次")

                try:
                    call = functools.partial(agent.analyze, stock_code, time_range, **agent_params)
                    loop = asyncio.get_event_loop()
                    result = await asyncio.wait_for(
                        loop.run_in_executor(None, call),
                        timeout=agent_timeout
                    )
                except asyncio.TimeoutError:
                    error(f"{agent_name} 超时（{agent_timeout}s），attempt={attempt}")
                    result = AgentResult(
                        agent_name=agent_name, success=False, result={},
                        error_message=f"超时（{agent_timeout}s）", confidence_score=0.0
                    )
                except Exception as exc:
                    error(f"{agent_name} 执行异常 attempt={attempt}: {exc}")
                    result = AgentResult(
                        agent_name=agent_name, success=False, result={},
                        error_message=str(exc), confidence_score=0.0
                    )

                last_result = result
                validation_error = agent._validate_output(result)

                if validation_error is None:
                    if attempt > 0:
                        info(f"{agent_name} 经过 {attempt} 次 Reflection 后输出有效")
                    agent.clear_reflection_hint()
                    break

                warning(f"{agent_name} attempt={attempt} 验证失败: {validation_error}")
                if attempt < REFLECTION_MAX_RETRIES:
                    agent.set_reflection_hint(validation_error)
                else:
                    warning(f"{agent_name} 已达最大重试次数，使用最后一次结果")
                    agent.clear_reflection_hint()

            status_str = "成功" if last_result.success else "失败"
            debug(f"{agent_name} 节点完成，状态: {status_str}")
            return {"agent_results": {agent_name: last_result}}

        return agent_node


    def _create_chief_node(self):
        """
        创建首席策略官节点。
        从 State 中读取 conflict_analysis 并传给 Chief。
        """
        async def chief_node(state: AgentState) -> Dict[str, Any]:
            stock_code        = state.get('stock_code', '')
            agent_results     = state.get('agent_results', {})
            conflict_analysis = state.get('conflict_analysis', {})
            chief_timeout     = AGENT_TIMEOUTS.get('ChiefStrategyAgent', 120)

            async def run_chief():
                try:
                    chief_agent = self.agents['ChiefStrategyAgent']
                    loop = asyncio.get_event_loop()
                    call = functools.partial(
                        chief_agent.analyze,
                        stock_code,
                        agent_results,
                        conflict_analysis=conflict_analysis
                    )
                    return await loop.run_in_executor(None, call)
                except Exception as e:
                    error(f'首席策略官执行失败: {str(e)}')
                    return AgentResult(
                        agent_name='ChiefStrategyAgent',
                        success=False, result={},
                        error_message=str(e), confidence_score=0.0
                    )

            try:
                result = await asyncio.wait_for(run_chief(), timeout=chief_timeout)
            except asyncio.TimeoutError:
                error(f'ChiefStrategyAgent 超时（{chief_timeout}s）')
                result = AgentResult(
                    agent_name='ChiefStrategyAgent',
                    success=False, result={},
                    error_message=f'超时（{chief_timeout}s）', confidence_score=0.0
                )

            debug('首席策略官分析完成')
            return {'final_result': result}

        return chief_node


    async def analyze_async(self, stock_code: str, time_range: str = "1y", **kwargs) -> Dict[str, Any]:
        """
        异步执行完整分析流程
        
        Args:
            stock_code: 股票代码
            time_range: 时间范围
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        try:
            debug(f"开始对股票 {stock_code} 进行综合分析")

            # 记录分析开始时间（不放入状态中）
            analysis_start_time = datetime.now().isoformat()

            # 准备初始状态，严格符合AgentState类型定义
            initial_state = AgentState(
                stock_code=stock_code,
                time_range=time_range,
                agent_params=kwargs.get("agent_params", {}),  # 移除chief_params
                chief_params=kwargs.get("chief_params", {}),    # 添加单独的chief_params字段
                agent_results={}  # 初始化必需的agent_results字段
            )

            # 执行工作流
            @performance_logger(f"执行完整分析流程 - {stock_code}")
            async def execute_workflow():
                return await self.graph.ainvoke(initial_state)

            final_state = await execute_workflow()

            # 生成分析报告
            report = self._generate_analysis_report(final_state)

            # 记录分析完成时间
            analysis_end_time = datetime.now().isoformat()
            report["analysis_time"] = analysis_end_time
            report["elapsed_time_seconds"] = (
                    datetime.fromisoformat(analysis_end_time) -
                    datetime.fromisoformat(analysis_start_time)
            ).total_seconds()

            debug(f"股票 {stock_code} 综合分析完成，耗时: {report['elapsed_time_seconds']:.2f}秒")

            return report

        except Exception as e:
            error(f"综合分析失败: {str(e)}")
            print_log_exception()
            return {
                "success": False,
                "error": str(e),
                "stock_code": stock_code,
                "analysis_time": datetime.now().isoformat()
            }

    def analyze(self, stock_code: str, time_range: str = "1y", **kwargs) -> Dict[str, Any]:
        """
        同步执行完整分析流程
        
        Args:
            stock_code: 股票代码
            time_range: 时间范围
            **kwargs: 其他参数
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        # 创建事件循环并运行异步分析
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 如果已有运行中的事件循环，使用它
                task = loop.create_task(self.analyze_async(stock_code, time_range, **kwargs))
                return loop.run_until_complete(task)
            else:
                # 创建新的事件循环
                return loop.run_until_complete(self.analyze_async(stock_code, time_range, **kwargs))
        except Exception as e:
            error(f"同步分析失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "stock_code": stock_code
            }

    def _generate_analysis_report(self, final_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成综合分析报告
        
        Args:
            final_state: 最终状态
            
        Returns:
            Dict[str, Any]: 综合报告
        """
        report = {
            "success": True,
            "stock_code": final_state.get("stock_code", ""),
            "analysis_start_time": final_state.get("analysis_start_time", ""),
            "agent_execution_status": {},
            "final_recommendation": {},
            "detailed_results": {}
        }

        # 记录各智能体执行状态
        agent_results = final_state.get("agent_results", {})
        for agent_name, result in agent_results.items():
            report["agent_execution_status"][agent_name] = {
                "success": result.success,
                "confidence_score": result.confidence_score,
                "error": result.error_message if not result.success else None
            }

            # 保存详细结果
            if result.success:
                report["detailed_results"][agent_name] = result.result

        # 添加最终建议
        final_result = final_state.get("final_result")
        if final_result and final_result.success:
            report["final_recommendation"] = final_result.result
            report["success"] = True
        else:
            report["success"] = False
            report["error"] = final_result.error_message if final_result else "首席策略官执行失败"

        return report

    def get_agent_status(self) -> Dict[str, Any]:
        """
        获取各智能体状态
        
        Returns:
            Dict[str, Any]: 智能体状态信息
        """
        status = {}
        for agent_name, agent in self.agents.items():
            status[agent_name] = {
                "name": agent.agent_name,
                "description": agent.description,
                "is_initialized": True
            }
        return status

    def update_agent_config(self, agent_name: str, config: Dict[str, Any]):
        """
        更新指定智能体的配置
        
        Args:
            agent_name: 智能体名称
            config: 新的配置参数
        """
        try:
            if agent_name in self.agents:
                agent = self.agents[agent_name]

                # 如果智能体有config属性，使用Pydantic的model_copy更新
                if hasattr(agent, 'config') and isinstance(agent.config, AgentConfig):
                    new_config = agent.config.model_copy(update=config)
                    agent.config = new_config

                    # 如果启用/禁用了记忆，需要重新初始化记忆
                    if "enable_memory" in config:
                        if config["enable_memory"] and not agent.memory:
                            agent.memory = agent._init_memory()
                            info(f"为智能体 {agent_name} 启用了记忆功能")
                        elif not config["enable_memory"] and agent.memory:
                            agent.memory = None
                            info(f"为智能体 {agent_name} 禁用了记忆功能")

                    # 如果更新了模型或嵌入模型，重新获取LLM
                    if "model_name" in config or "embedding_model" in config:
                        agent.langchain_llm = agent._get_langchain_llm()
                        if agent.memory:
                            agent.memory = agent._init_memory()

                # 如果是关键配置变更，重建工作流
                if "model_name" in config or "enable_memory" in config:
                    self.build_workflow()

                info(f"更新智能体配置成功: {agent_name}")
            else:
                warning(f"智能体不存在: {agent_name}")
        except Exception as e:
            error(f"更新智能体配置失败: {str(e)}")
            raise

    def execute_single_agent(self, agent_name: str, stock_code: str, time_range: str = "1y", **kwargs) -> AgentResult:
        """
        执行单个智能体分析
        
        Args:
            agent_name: 智能体名称
            stock_code: 股票代码
            time_range: 时间范围
            **kwargs: 额外参数
            
        Returns:
            AgentResult: 分析结果
        """
        try:
            if agent_name in self.agents:
                agent = self.agents[agent_name]

                # 可以从kwargs中获取特定的记忆相关参数
                use_memory = kwargs.pop("use_memory", agent.config.enable_memory if hasattr(agent, 'config') else True)

                # 保存原始的enable_memory设置
                original_enable_memory = None
                if hasattr(agent, 'config') and hasattr(agent, 'memory'):
                    original_enable_memory = agent.config.enable_memory
                    agent.config.enable_memory = use_memory

                    # 如果临时启用记忆但当前没有，初始化记忆
                    if use_memory and not agent.memory:
                        agent.memory = agent._init_memory()

                # 执行分析
                result = agent.analyze(stock_code, time_range, **kwargs)

                # 恢复原始设置
                if original_enable_memory is not None:
                    agent.config.enable_memory = original_enable_memory
                    if not original_enable_memory and agent.memory:
                        # 可选：是否在分析后清理临时启用的记忆
                        # agent.memory = None
                        pass

                # 记录执行结果
                status = "成功" if result.success else "失败"
                memory_status = "使用记忆" if use_memory else "不使用记忆"
                info(f"执行 {agent_name} 分析完成，状态: {status}，{memory_status}")

                return result
            else:
                warning(f"智能体不存在: {agent_name}")
                return AgentResult(
                    agent_name=agent_name,
                    success=False,
                    result={},
                    error_message="智能体不存在",
                    confidence_score=0.0
                )
        except Exception as e:
            error(f"执行智能体分析失败: {str(e)}")
            return AgentResult(
                agent_name=agent_name,
                success=False,
                result={},
                error_message=str(e),
                confidence_score=0.0
            )
