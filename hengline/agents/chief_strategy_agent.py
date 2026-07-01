#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@FileName: chief_strategy_agent.py
@Description: 首席策略智能体，负责整合其他专业智能体的分析结果并生成最终的投资建议
@Author: HengLine
@Time: 2025/11/10
"""

import json
from datetime import datetime
from typing import Dict, Any, List, Tuple

from hengline.agents.base_agent import BaseAgent, AgentConfig, AgentResult
from hengline.logger import debug, error, warning


class ChiefStrategyAgent(BaseAgent):
    """首席策略官智能体"""
    
    def __init__(self, config: AgentConfig = None):
        """
        初始化首席策略官智能体
        
        Args:
            config: 智能体配置
        """
        super().__init__(config)
        
        # 投资建议类型
        self.investment_recommendations = {
            "强烈买入": {
                "description": "强烈建议买入该股票，预期收益显著高于风险",
                "confidence": "非常高",
                "suitable_for": "激进型、积极型投资者",
                "signal_strength": 5
            },
            "买入": {
                "description": "建议买入该股票，预期收益大于风险",
                "confidence": "高",
                "suitable_for": "积极型、平衡型投资者",
                "signal_strength": 4
            },
            "持有": {
                "description": "建议持有现有仓位，收益与风险相对平衡",
                "confidence": "中等",
                "suitable_for": "平衡型、保守型投资者",
                "signal_strength": 3
            },
            "谨慎观望": {
                "description": "建议暂时观望，等待更明确的买入信号",
                "confidence": "低",
                "suitable_for": "保守型投资者",
                "signal_strength": 2
            },
            "卖出": {
                "description": "建议卖出该股票，风险显著高于预期收益",
                "confidence": "高",
                "suitable_for": "所有类型投资者",
                "signal_strength": 1
            }
        }
        
        # 风险等级定义
        self.risk_levels = {
            "低风险": {"range": (0, 20), "color": "green"},
            "中低风险": {"range": (21, 40), "color": "light-green"},
            "中等风险": {"range": (41, 60), "color": "yellow"},
            "中高风险": {"range": (61, 80), "color": "orange"},
            "高风险": {"range": (81, 100), "color": "red"}
        }
        
        # 各智能体权重配置
        self.agent_weights = {
            "FundamentalAgent": 0.25,      # 基本面分析权重
            "TechnicalAgent": 0.20,        # 技术面分析权重
            "IndustryMacroAgent": 0.15,    # 行业与宏观分析权重
            "SentimentAgent": 0.15,        # 舆情与情绪分析权重
            "FundFlowAgent": 0.15,         # 股东与资金流分析权重
            "ESGRiskAgent": 0.10           # ESG与治理风险分析权重
        }
    
    def analyze(self, stock_code: str, agent_results: Dict[str, AgentResult], conflict_analysis: Dict = None, **kwargs) -> AgentResult:
        """
        整合各智能体分析结果，生成最终投资建议
        
        Args:
            stock_code: 股票代码
            agent_results: 各智能体的分析结果字典
            **kwargs: 其他参数
            
        Returns:
            AgentResult: 最终分析结果
        """
        try:
            debug(f"首席策略官开始整合股票 {stock_code} 的分析结果")
            
            # 验证并过滤有效结果
            filtered_results = self._filter_valid_results(agent_results)

            # 评估数据质量，后续用于降低置信度和限制过强建议
            data_quality = self._assess_data_quality(filtered_results, agent_results, conflict_analysis or {})
            
            # 计算综合评分
            composite_score = self._calculate_composite_score(filtered_results)
            
            # 评估整体风险
            overall_risk = self._assess_overall_risk(filtered_results)
            
            # 识别关键优势与风险
            key_strengths, key_risks = self._identify_key_factors(filtered_results)
            
            # 检索相关知识库信息
            knowledge = self._retrieve_knowledge(f"投资决策 组合管理 风险评估 市场时机")
            
            # 生成分析提示词（包含冲突分析）
            prompt = self._generate_analysis_prompt(
                stock_code, filtered_results, composite_score,
                overall_risk, key_strengths, key_risks,
                conflict_analysis=conflict_analysis,
                data_quality=data_quality,
            )
            
            # 使用LLM生成综合分析
            llm_analysis = self._generate_analysis(prompt, knowledge)
            
            # 结构化最终结果
            result = self._structure_result(
                stock_code, llm_analysis, filtered_results, 
                composite_score, overall_risk, data_quality, conflict_analysis
            )
            
            debug(f"首席策略官完成股票 {stock_code} 的综合分析")
            
            return AgentResult(
                agent_name=self.agent_name,
                success=True,
                result=result,
                confidence_score=result.get("confidence_score", 0.90)
            )
            
        except Exception as e:
            error(f"首席策略官分析失败: {str(e)}")
            return AgentResult(
                agent_name=self.agent_name,
                success=False,
                result=self.get_result_template(),
                error_message=str(e),
                confidence_score=0.0
            )
    
    def _filter_valid_results(self, agent_results: Dict[str, AgentResult]) -> Dict[str, Dict[str, Any]]:
        """
        过滤有效分析结果
        
        Args:
            agent_results: 各智能体的分析结果字典
            
        Returns:
            Dict[str, Dict[str, Any]]: 有效分析结果
        """
        filtered = {}
        
        for agent_name, result in agent_results.items():
            if result.success and result.result:
                filtered[agent_name] = result.result
                debug(f"成功包含 {agent_name} 的分析结果，置信度: {result.confidence_score}")
            else:
                warning(f"跳过 {agent_name} 的分析结果，原因: {result.error_message or '无效结果'}")
        
        return filtered

    @staticmethod
    def _is_simulated_result(result: Dict[str, Any]) -> bool:
        """Return True when an agent result is based on mock/simulated data."""
        if not isinstance(result, dict):
            return False
        if result.get("is_simulated") is True:
            return True
        if str(result.get("data_source", "")).lower() == "mock":
            return True
        if str(result.get("data_quality_level", "")).lower() == "simulated":
            return True
        return any(
            isinstance(value, dict) and ChiefStrategyAgent._is_simulated_result(value)
            for value in result.values()
        )

    def _assess_data_quality(
        self,
        filtered_results: Dict[str, Dict[str, Any]],
        agent_results: Dict[str, AgentResult],
        conflict_analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Summarize data reliability across successful, failed, and simulated agents."""
        failed_agents = list(conflict_analysis.get("failed_agents") or [])
        for agent_name, result in agent_results.items():
            if not result.success and agent_name not in failed_agents:
                failed_agents.append(agent_name)

        simulated_agents: List[str] = []
        unavailable_agents: List[str] = []
        partial_agents: List[str] = []
        notes: List[str] = []

        for agent_name, result in filtered_results.items():
            quality_level = str(result.get("data_quality_level", "")).lower()
            if self._is_simulated_result(result):
                simulated_agents.append(agent_name)
            if result.get("data_available") is False or quality_level == "unavailable":
                unavailable_agents.append(agent_name)
            elif result.get("data_note") or quality_level in {"partial", "estimated"}:
                partial_agents.append(agent_name)
            if result.get("data_note"):
                notes.append(f"{agent_name}: {str(result.get('data_note'))[:120]}")

        data_gaps = list(conflict_analysis.get("data_gaps") or [])
        score = 100
        score -= 30 * len(simulated_agents)
        score -= 20 * len(unavailable_agents)
        score -= 10 * len(partial_agents)
        score -= 15 * len(failed_agents)
        score -= min(20, 5 * len(data_gaps))
        score = max(0, min(100, score))

        if simulated_agents:
            level = "simulated"
        elif unavailable_agents or failed_agents:
            level = "limited"
        elif partial_agents or data_gaps:
            level = "partial"
        else:
            level = "verified"

        return {
            "level": level,
            "score": score,
            "simulated_agents": simulated_agents,
            "unavailable_agents": unavailable_agents,
            "partial_agents": partial_agents,
            "failed_agents": failed_agents,
            "data_gaps": data_gaps,
            "notes": notes[:8],
            "decision_policy": (
                "模拟数据或关键维度缺失时，最终建议不得高于谨慎观望。"
                if level in {"simulated", "limited"}
                else "部分数据缺口时，限制强烈买入并降低置信度。"
                if level == "partial"
                else "数据质量正常。"
            ),
        }
    
    def _calculate_composite_score(self, filtered_results: Dict[str, Dict[str, Any]]) -> float:
        """
        计算综合评分
        
        Args:
            filtered_results: 有效分析结果
            
        Returns:
            float: 综合评分 (0-100)
        """
        if not filtered_results:
            return 50.0  # 默认中性评分
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for agent_name, result in filtered_results.items():
            # 获取智能体权重
            weight = self.agent_weights.get(agent_name, 0.1)
            
            # 从结果中提取评分信息（这里需要根据不同智能体的结果结构定制）
            score = self._extract_agent_score(agent_name, result)
            
            # 加权求和
            weighted_sum += score * weight
            total_weight += weight
        
        # 避免除零错误
        if total_weight == 0:
            return 50.0
        
        # 计算加权平均分
        composite_score = weighted_sum / total_weight
        
        # 确保分数在0-100范围内
        return max(0, min(100, composite_score))
    
    def _extract_agent_score(self, agent_name: str, result: Dict[str, Any]) -> float:
        """
        从不同智能体的结果中提取评分
        
        Args:
            agent_name: 智能体名称
            result: 智能体分析结果
            
        Returns:
            float: 评分 (0-100)
        """
        # 根据不同智能体的实际输出结构提取评分（统一映射到 0-100）
        if agent_name == "FundamentalAgent":
            # LLM 输出 overall_score: 0-10，转换到 0-100
            raw = result.get("overall_score", 0)
            if raw and raw > 0:
                return float(raw) * 10 if raw <= 10 else float(raw)
            # 回退：用 confidence_score 粗估
            return result.get("confidence_score", 0.5) * 100

        elif agent_name == "TechnicalAgent":
            # TechnicalAgent 通过 signal_strength 和 confidence_score 估算评分
            signal_map = {
                "strong_bullish": 85, "bullish": 70, "weak_bullish": 60,
                "neutral": 50,
                "weak_bearish": 40, "bearish": 30, "strong_bearish": 15
            }
            signal = result.get("signal_strength", "neutral")
            base = signal_map.get(signal, 50)
            confidence = result.get("confidence_score", 0.5)
            # 根据置信度调整（高置信度时分值靠近极端，低置信度时向中位收敛）
            return base * 0.7 + confidence * 100 * 0.3

        elif agent_name == "IndustryMacroAgent":
            # 行业与宏观评分
            industry_score = result.get("industry_analysis", {}).get("industry_score", 50.0)
            macro_score = result.get("macro_analysis", {}).get("economic_score", 50.0)
            return (float(industry_score) + float(macro_score)) / 2

        elif agent_name == "SentimentAgent":
            # 从 sentiment_metrics.news_sentiment 计算正向比例
            sentiment_metrics = result.get("sentiment_metrics", {})
            news_sentiment = sentiment_metrics.get("news_sentiment", {})
            pos = news_sentiment.get("positive", 0)
            neg = news_sentiment.get("negative", 0)
            total = pos + neg
            if total > 0:
                return (pos / total) * 100
            # 无新闻数据时，从市场情绪指标尝试
            market_sent = sentiment_metrics.get("market_sentiment", {})
            fear_greed = market_sent.get("current", 0)
            if fear_greed:
                return float(fear_greed)
            return 50.0

        elif agent_name == "FundFlowAgent":
            # 根据资金流向分类映射评分
            key_metrics = result.get("key_metrics", {})
            flow_classification = key_metrics.get("flow_classification", "neutral")
            flow_scores = {
                "strong_inflow": 90,
                "moderate_inflow": 75,
                "weak_inflow": 60,
                "neutral": 50,
                "weak_outflow": 40,
                "moderate_outflow": 25,
                "strong_outflow": 10
            }
            return float(flow_scores.get(flow_classification, 50))

        elif agent_name == "ESGRiskAgent":
            # ESG 综合评分（0-100，已标准化）
            esg_metrics = result.get("esg_metrics", {})
            return float(esg_metrics.get("overall_score", 50.0))

        # 默认评分
        return 50.0
    
    def _assess_overall_risk(self, filtered_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        评估整体风险
        
        Args:
            filtered_results: 有效分析结果
            
        Returns:
            Dict[str, Any]: 风险评估结果
        """
        overall_risk = {
            "risk_score": 50.0,  # 默认中等风险
            "risk_level": "中等风险",
            "risk_dimensions": {},
            "risk_factors": [],
            "risk_mitigation": []
        }
        
        risk_scores = []
        risk_factors = []
        
        # 从各智能体结果中提取风险信息
        for agent_name, result in filtered_results.items():
            if agent_name == "FundamentalAgent":
                # 基本面风险
                financial_risk = result.get("financial_risks", [])
                risk_factors.extend(financial_risk)
                risk_scores.append(100 - self._extract_agent_score(agent_name, result))
                
            elif agent_name == "TechnicalAgent":
                # 技术面风险
                technical_risks = result.get("risk_signals", [])
                risk_factors.extend(technical_risks)
                risk_scores.append(100 - self._extract_agent_score(agent_name, result))
                
            elif agent_name == "ESGRiskAgent":
                # ESG风险
                esg_metrics = result.get("esg_metrics", {})
                esg_risk_level = esg_metrics.get("risk_level", "")
                if esg_risk_level:
                    risk_factors.append(f"ESG风险等级: {esg_risk_level}")
                    
                # ESG风险转换为风险分数
                esg_risk_scores = {
                    "low": 20,
                    "medium": 50,
                    "high": 70,
                    "severe": 90
                }
                risk_scores.append(esg_risk_scores.get(esg_risk_level, 50.0))
        
        # 计算平均风险分数
        if risk_scores:
            overall_risk["risk_score"] = sum(risk_scores) / len(risk_scores)
        
        # 确定风险等级
        risk_score = overall_risk["risk_score"]
        for level, info in self.risk_levels.items():
            min_risk, max_risk = info["range"]
            if min_risk <= risk_score <= max_risk:
                overall_risk["risk_level"] = level
                break
        
        # 添加风险因素
        overall_risk["risk_factors"] = list(set(risk_factors))[:5]  # 去重并保留前5个
        
        # 生成风险缓解建议
        overall_risk["risk_mitigation"] = self._generate_risk_mitigation(overall_risk["risk_level"])
        
        return overall_risk
    
    def _generate_risk_mitigation(self, risk_level: str) -> List[str]:
        """
        生成风险缓解建议
        
        Args:
            risk_level: 风险等级
            
        Returns:
            List[str]: 风险缓解建议
        """
        mitigation_advice = {
            "低风险": ["可考虑适当增加仓位", "关注公司业绩持续改善", "长期持有策略可能适合"],
            "中低风险": ["适度配置，密切关注关键指标变化", "可考虑分批建仓策略", "设置止损位保护收益"],
            "中等风险": ["建议仓位适中，控制在投资组合20%以内", "设置止损位在8-10%", "定期重新评估基本面变化"],
            "中高风险": ["建议谨慎配置，控制在投资组合10%以内", "设置严格止损位在5-7%", "密切关注风险信号变化"],
            "高风险": ["建议极小仓位试探或暂时观望", "如有持仓考虑逐步减仓", "等待风险因素明朗后再评估"]
        }
        
        return mitigation_advice.get(risk_level, ["建议谨慎对待，根据个人风险承受能力决策"])
    
    def _identify_key_factors(self, filtered_results: Dict[str, Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        """
        识别关键优势和风险因素
        
        Args:
            filtered_results: 有效分析结果
            
        Returns:
            Tuple[List[str], List[str]]: (关键优势, 关键风险)
        """
        strengths = []
        risks = []
        
        # 从各智能体结果中提取关键发现
        for agent_name, result in filtered_results.items():
            # 提取优势
            key_findings = result.get("key_findings", [])
            for finding in key_findings:
                # 简单的文本分类，实际项目中应该使用更复杂的NLP
                if any(keyword in finding.lower() for keyword in ["优势", "增长", "改善", "强劲", "正面", "上升", "创新"]):
                    strengths.append(finding)
                elif any(keyword in finding.lower() for keyword in ["风险", "挑战", "下降", "负面", "担忧", "问题", "不确定性"]):
                    risks.append(finding)
            
            # 专门提取风险信息
            if "risk_signals" in result:
                risks.extend(result["risk_signals"])
            
            if "financial_risks" in result:
                risks.extend(result["financial_risks"])
        
        # 去重并限制数量
        strengths = list(set(strengths))[:5]
        risks = list(set(risks))[:5]
        
        return strengths, risks
    
    def _generate_analysis_prompt(self, stock_code: str, filtered_results: Dict[str, Dict[str, Any]],
                                 composite_score: float, overall_risk: Dict[str, Any],
                                 key_strengths: List[str], key_risks: List[str],
                                 conflict_analysis: Dict = None,
                                 data_quality: Dict[str, Any] = None) -> str:
        """
        生成综合分析提示词
        
        Args:
            stock_code: 股票代码
            filtered_results: 有效分析结果
            composite_score: 综合评分
            overall_risk: 整体风险评估
            key_strengths: 关键优势
            key_risks: 关键风险
            
        Returns:
            str: 分析提示词
        """
        # 导入langchain相关组件和提示词管理器
        from langchain_core.prompts import ChatPromptTemplate
        from hengline.prompts.prompt_manage import get_prompt
        
        # 从提示词管理器获取提示模板
        template = get_prompt('chief_strategy_agent', 'analysis')
        
        # 如果获取失败，使用备用模板
        if not template:
            template = """
作为资深投资策略专家，请对股票代码 {stock_code} 进行全面、平衡的综合分析并提供最终投资建议。

综合评分：{composite_score}/100
整体风险评估：{risk_level} (风险分数: {risk_score})

请提供详细分析和明确的投资建议。
"""
        
        # 创建提示模板实例
        prompt_template = ChatPromptTemplate.from_template(template)
        
        # 格式化并返回提示内容
        # 构建冲突分析文本，供首席参考
        conflict_text = ""
        if conflict_analysis:
            summary = conflict_analysis.get("conflict_summary", "")
            divergences = conflict_analysis.get("score_divergences", [])
            data_gaps = conflict_analysis.get("data_gaps", [])
            failed = conflict_analysis.get("failed_agents", [])
            parts = [f"[多维度冲突分析]\n{summary}"]
            if divergences:
                parts.append("评分分歧详情：" + "；".join(divergences))
            if data_gaps:
                parts.append("数据缺口：" + "；".join(data_gaps[:3]))
            if failed:
                failed_str = "、".join(failed)
                parts.append(f"执行失败的维度：{failed_str}（请在建议中注明缺失维度）")
            conflict_text = "\n".join(parts)

        prompt = prompt_template.format(
            stock_code=stock_code,
            composite_score=f"{composite_score:.2f}",
            risk_level=overall_risk['risk_level'],
            risk_score=f"{overall_risk['risk_score']:.2f}",
            key_strengths=json.dumps(key_strengths, indent=2, ensure_ascii=False),
            key_risks=json.dumps(key_risks, indent=2, ensure_ascii=False),
            filtered_results=json.dumps(filtered_results, indent=2, ensure_ascii=False)
        )

        # 将冲突分析追加到提示词末尾
        if conflict_text:
            prompt += f"\n\n{conflict_text}\n\n请在最终建议中明确说明各维度的分歧和不确定性。"

        if data_quality:
            prompt += (
                "\n\n[数据质量与合规约束]\n"
                f"数据质量等级：{data_quality.get('level')}，评分：{data_quality.get('score')}/100。\n"
                f"模拟数据维度：{data_quality.get('simulated_agents', [])}。\n"
                f"不可用/失败维度：{data_quality.get('unavailable_agents', []) + data_quality.get('failed_agents', [])}。\n"
                f"策略约束：{data_quality.get('decision_policy')}\n"
                "如果数据质量不足，请降低建议强度、降低置信度，并明确说明结论仅供研究参考。"
            )

        return prompt

    @staticmethod
    def _recommendation_rank(recommendation: str) -> int:
        order = {
            "卖出": 1,
            "谨慎观望": 2,
            "持有": 3,
            "买入": 4,
            "强烈买入": 5,
        }
        return order.get(recommendation, 3)

    @staticmethod
    def _cap_recommendation(recommendation: str, max_recommendation: str) -> str:
        if ChiefStrategyAgent._recommendation_rank(recommendation) > ChiefStrategyAgent._recommendation_rank(max_recommendation):
            return max_recommendation
        return recommendation

    def _apply_decision_guardrails(
        self,
        recommendation: str,
        confidence_score: float,
        key_findings: List[str],
        risk_disclosure: str,
        data_quality: Dict[str, Any],
        overall_risk: Dict[str, Any],
    ) -> Tuple[str, float, List[str], str, List[str]]:
        """Apply deterministic investment-safety rules after LLM synthesis."""
        guardrail_notes: List[str] = []
        quality_level = (data_quality or {}).get("level", "verified")
        capped = recommendation
        confidence_cap = 0.90

        if quality_level in {"simulated", "limited"}:
            capped = self._cap_recommendation(capped, "谨慎观望")
            confidence_cap = min(confidence_cap, 0.55)
            guardrail_notes.append("数据质量不足或存在模拟/失败维度，系统已限制最终建议强度。")
        elif quality_level == "partial":
            capped = self._cap_recommendation(capped, "买入")
            confidence_cap = min(confidence_cap, 0.70)
            guardrail_notes.append("存在部分数据缺口，系统已降低建议置信度。")

        risk_score = float((overall_risk or {}).get("risk_score", 50.0) or 50.0)
        if risk_score >= 70:
            capped = self._cap_recommendation(capped, "持有")
            confidence_cap = min(confidence_cap, 0.65)
            guardrail_notes.append("整体风险评分偏高，系统已限制买入类建议。")

        confidence_score = max(0.0, min(float(confidence_score or 0.0), confidence_cap))
        if capped != recommendation:
            guardrail_notes.append(f"LLM 原始建议为“{recommendation}”，护栏调整为“{capped}”。")

        if guardrail_notes:
            key_findings = list(key_findings or [])
            key_findings.extend(note for note in guardrail_notes if note not in key_findings)
            extra = "；".join(guardrail_notes)
            risk_disclosure = f"{risk_disclosure}；{extra}" if risk_disclosure else extra

        return capped, confidence_score, key_findings, risk_disclosure, guardrail_notes

    def _assess_human_review(
        self,
        recommendation: str,
        confidence_score: float,
        data_quality: Dict[str, Any],
        overall_risk: Dict[str, Any],
        guardrail_notes: List[str],
    ) -> Dict[str, Any]:
        """Decide whether the final recommendation should be reviewed by a human."""
        reasons: List[str] = []
        quality_level = (data_quality or {}).get("level", "verified")
        risk_score = float((overall_risk or {}).get("risk_score", 50.0) or 50.0)

        if recommendation in {"强烈买入", "买入", "卖出"} and float(confidence_score or 0.0) >= 0.75:
            reasons.append("输出包含高强度买入/卖出建议且模型置信度较高。")
        if quality_level in {"simulated", "limited", "partial"}:
            reasons.append(f"数据质量等级为 {quality_level}，存在不可直接用于决策的数据边界。")
        if risk_score >= 70:
            reasons.append(f"整体风险评分为 {risk_score:.0f}，属于偏高风险区间。")
        if guardrail_notes:
            reasons.append("系统已触发投资建议护栏，需复核调整原因是否合理。")

        required = bool(reasons)
        return {
            "required": required,
            "review_level": "required" if required else "optional",
            "reasons": reasons,
            "policy": (
                "需要人工复核后再用于任何真实交易或仓位决策。"
                if required
                else "未触发强制复核规则，但仍建议结合个人风险承受能力人工判断。"
            ),
        }

    def _build_position_plan(
        self,
        recommendation: str,
        confidence_score: float,
        data_quality: Dict[str, Any],
        overall_risk: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build a deterministic position sizing plan from risk, quality, and signal strength."""
        risk_score = float((overall_risk or {}).get("risk_score", 50.0) or 50.0)
        quality_level = (data_quality or {}).get("level", "verified")
        quality_score = float((data_quality or {}).get("score", 100) or 100)

        base_ranges = {
            "强烈买入": (0.15, 0.25),
            "买入": (0.08, 0.15),
            "持有": (0.03, 0.08),
            "谨慎观望": (0.00, 0.03),
            "卖出": (0.00, 0.00),
        }
        min_position, max_position = base_ranges.get(recommendation, (0.00, 0.05))

        if risk_score >= 80:
            risk_multiplier = 0.25
        elif risk_score >= 65:
            risk_multiplier = 0.50
        elif risk_score >= 50:
            risk_multiplier = 0.75
        else:
            risk_multiplier = 1.00

        if quality_level in {"simulated", "limited"}:
            quality_multiplier = 0.25
        elif quality_level == "partial":
            quality_multiplier = 0.60
        else:
            quality_multiplier = max(0.70, min(1.0, quality_score / 100))

        confidence_multiplier = max(0.50, min(1.0, float(confidence_score or 0.0)))
        multiplier = min(risk_multiplier, quality_multiplier) * confidence_multiplier
        adjusted_min = round(min_position * multiplier, 4)
        adjusted_max = round(max_position * multiplier, 4)

        if recommendation == "卖出":
            action = "reduce_or_exit"
            entry_plan = "不建议新增仓位；已有仓位按风险承受能力分批减仓或退出。"
        elif adjusted_max <= 0.03:
            action = "watch_or_probe"
            entry_plan = "仅适合观察或极小仓位试探，不建议一次性建仓。"
        elif recommendation in {"强烈买入", "买入"}:
            action = "scale_in"
            entry_plan = "如需参与，建议分 2-3 批建仓，并等待量价与数据质量继续确认。"
        else:
            action = "hold_or_small_adjust"
            entry_plan = "以持有和小幅调仓为主，避免在信息不完整时显著加仓。"

        stop_loss_pct = 0.05 if risk_score >= 70 else 0.08 if risk_score >= 50 else 0.10
        take_profit_pct = 0.12 if risk_score >= 70 else 0.18 if risk_score >= 50 else 0.25

        return {
            "action": action,
            "suggested_position_range": {
                "min_pct": adjusted_min,
                "max_pct": adjusted_max,
                "display": f"{adjusted_min * 100:.0f}%-{adjusted_max * 100:.0f}%",
            },
            "base_position_range": {
                "min_pct": min_position,
                "max_pct": max_position,
            },
            "risk_multiplier": round(risk_multiplier, 2),
            "quality_multiplier": round(quality_multiplier, 2),
            "confidence_multiplier": round(confidence_multiplier, 2),
            "entry_plan": entry_plan,
            "risk_controls": {
                "stop_loss_pct": stop_loss_pct,
                "take_profit_review_pct": take_profit_pct,
                "max_single_stock_pct": adjusted_max,
                "rebalance_trigger": "当数据质量、风险评分或主要 Agent 结论发生明显变化时重新评估。",
            },
            "notes": [
                "仓位区间为研究辅助输出，不代表适合所有账户。",
                "真实交易需结合个人资金规模、组合集中度、流动性和风险承受能力人工确认。",
            ],
        }

    def _build_decision_boundaries(
        self,
        recommendation: str,
        data_quality: Dict[str, Any],
        overall_risk: Dict[str, Any],
        conflict_analysis: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Create structured evidence gaps, invalidation conditions, and reverse risks."""
        risk_level = (overall_risk or {}).get("risk_level", "中等风险")
        risk_score = float((overall_risk or {}).get("risk_score", 50.0) or 50.0)
        quality_level = (data_quality or {}).get("level", "verified")
        conflict_analysis = conflict_analysis or {}

        missing_data = []
        for key in ("simulated_agents", "unavailable_agents", "partial_agents", "failed_agents", "data_gaps"):
            items = (data_quality or {}).get(key) or []
            if items:
                missing_data.append({"type": key, "items": items})

        invalidation_conditions = [
            "后续核心数据源修复后，如果关键财务、行情或新闻数据与当前结论明显相反，应重新运行分析。",
            "若主要技术趋势跌破关键均线或成交量异常放大且价格走弱，应重新评估短线结论。",
            "若公告、财报、监管问询、减持或解禁事件改变基本面假设，应重新评估中长期结论。",
        ]
        if recommendation in {"强烈买入", "买入"}:
            invalidation_conditions.append("如果综合评分下降到 60 以下或风险评分升至 70 以上，买入类结论失效。")
        elif recommendation == "卖出":
            invalidation_conditions.append("如果风险评分回落且多维度 Agent 重新形成偏多共识，卖出结论需复核。")

        reverse_risks = [
            f"当前整体风险等级为 {risk_level}，风险评分 {risk_score:.0f}。",
            f"数据质量等级为 {quality_level}，该等级会影响结论可靠性。",
        ]
        if conflict_analysis.get("has_conflicts"):
            reverse_risks.append("专业 Agent 之间存在明显分歧，最终建议需降低确定性。")
        if missing_data:
            reverse_risks.append("部分维度存在模拟、估算、不可用或失败数据，结论可能随数据修复而变化。")

        return {
            "evidence_quality": {
                "level": quality_level,
                "score": (data_quality or {}).get("score"),
                "missing_or_limited_data": missing_data,
            },
            "reverse_risks": reverse_risks,
            "invalidation_conditions": invalidation_conditions,
            "recheck_triggers": [
                "重大公告、财报或监管事件发布后",
                "价格单日大幅波动或成交量异常放大后",
                "数据源从模拟/估算切换为真实数据后",
                "持仓比例接近建议上限或组合集中度明显升高时",
            ],
        }
    
    def _structure_result(self, stock_code: str, llm_analysis: Dict[str, Any],
                         filtered_results: Dict[str, Dict[str, Any]], composite_score: float,
                         overall_risk: Dict[str, Any],
                         data_quality: Dict[str, Any] = None,
                         conflict_analysis: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        结构化最终结果
        
        Args:
            stock_code: 股票代码
            llm_analysis: LLM生成的分析
            filtered_results: 各智能体分析结果
            composite_score: 综合评分
            overall_risk: 整体风险评估
            
        Returns:
            Dict[str, Any]: 结构化的结果
        """
        result = self.get_result_template()
        
        # 获取建议详情
        recommendation = llm_analysis.get("investment_recommendation", "持有")
        key_findings = llm_analysis.get("key_findings", [])
        if not isinstance(key_findings, list):
            key_findings = [str(key_findings)]
        if not key_findings:
            summary = llm_analysis.get("analysis_summary") or llm_analysis.get("recommendation_details")
            if summary:
                key_findings = [str(summary)]
            else:
                key_findings = [f"综合评分 {composite_score:.2f}，最终建议为{recommendation}。"]
        risk_disclosure = llm_analysis.get("risk_disclosure", "")
        confidence_score = llm_analysis.get("confidence_score", 0.90)
        recommendation, confidence_score, key_findings, risk_disclosure, guardrail_notes = self._apply_decision_guardrails(
            recommendation,
            confidence_score,
            key_findings,
            risk_disclosure,
            data_quality or {},
            overall_risk,
        )
        recommendation_details = self.investment_recommendations.get(recommendation, {})
        human_review = self._assess_human_review(
            recommendation,
            confidence_score,
            data_quality or {},
            overall_risk,
            guardrail_notes,
        )
        position_plan = self._build_position_plan(
            recommendation,
            confidence_score,
            data_quality or {},
            overall_risk,
        )
        decision_boundaries = self._build_decision_boundaries(
            recommendation,
            data_quality or {},
            overall_risk,
            conflict_analysis or {},
        )
        
        result.update({
            "stock_code": stock_code,
            "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "investment_recommendation": recommendation,
            "recommendation_description": recommendation_details.get("description", ""),
            "recommendation_confidence": recommendation_details.get("confidence", ""),
            "signal_strength": recommendation_details.get("signal_strength", 3),
            "recommendation_details": llm_analysis.get("recommendation_details", ""),
            "suitable_investors": llm_analysis.get("suitable_investors", ""),
            "position_suggestion": llm_analysis.get("position_suggestion", ""),
            "key_monitoring_metrics": llm_analysis.get("key_monitoring_metrics", []),
            "risk_disclosure": risk_disclosure,
            "confidence_score": confidence_score,
            "key_findings": key_findings,
            "analysis_summary": llm_analysis.get("analysis_summary", ""),
            "analysis_horizon": {
                "short_term": llm_analysis.get("short_term_outlook", "短线结论以技术面和成交量为主，需结合实时行情确认。"),
                "medium_term": llm_analysis.get("medium_term_outlook", "中线结论需结合基本面、估值和行业景气度。"),
                "long_term": llm_analysis.get("long_term_outlook", "长线结论应以财务质量、竞争力和行业周期为核心。"),
            },
            "data_quality": data_quality or {},
            "guardrail_notes": guardrail_notes,
            "human_review": human_review,
            "position_plan": position_plan,
            "decision_boundaries": decision_boundaries,
            "compliance_disclaimer": "本系统输出仅供研究和学习参考，不构成任何投资建议或收益承诺。",
            "detailed_analysis": {
                "recommendation_details": llm_analysis.get("recommendation_details", ""),
                "suitable_investors": llm_analysis.get("suitable_investors", ""),
                "position_suggestion": llm_analysis.get("position_suggestion", ""),
                "risk_disclosure": risk_disclosure,
                "position_plan": position_plan,
                "decision_boundaries": decision_boundaries,
            },
            "comprehensive_metrics": {
                "composite_score": round(composite_score, 2),
                "risk_score": round(overall_risk["risk_score"], 2),
                "risk_level": overall_risk["risk_level"],
                "agents_used": list(filtered_results.keys())
            },
            "agent_contributions": {}
        })
        
        # 添加各智能体的贡献摘要
        for agent_name, agent_result in filtered_results.items():
            result["agent_contributions"][agent_name] = {
                "key_findings": agent_result.get("key_findings", []),
                "confidence": agent_result.get("confidence_score", 0.0)
            }
        
        # 验证结果
        if not self.validate_result(result):
            warning("首席策略官分析结果验证失败，使用默认结构")
            result = self.get_result_template()
        
        return result
