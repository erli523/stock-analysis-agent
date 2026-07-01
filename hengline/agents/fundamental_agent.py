#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@FileName: fundamental_agent.py
@Description: 基本面分析智能体，负责分析股票的财务状况、业绩和估值指标
@Author: HengLine
@Time: 2025/11/10
"""

import json
import pandas as pd
from datetime import datetime
from typing import Dict, Any

# 从stock_manage统一获取数据
from hengline.stock.stock_manage import StockDataManager

from hengline.agents.base_agent import BaseAgent, AgentConfig, AgentResult
from hengline.agents.result_utils import build_data_quality_fields
from hengline.logger import debug, error, warning


class FundamentalAgent(BaseAgent):
    """基本面分析智能体"""

    def __init__(self, config: AgentConfig = None):
        """
        初始化基本面分析智能体
        
        Args:
            config: 智能体配置
        """
        super().__init__(config)

        # 初始化股票数据管理器（优先使用协调器注入的共享实例）
        if self.stock_manager is None:
            self.stock_manager = StockDataManager()

        # 基本面分析关键维度
        self.analysis_dimensions = [
            "financial_health",  # 财务健康度
            "profitability",  # 盈利能力
            "growth",  # 成长能力
            "valuation",  # 估值水平
            "competitive_advantage",  # 竞争优势
            "management_quality"  # 管理质量
        ]

    def analyze(self, stock_code: str, time_range: str = "1y", **kwargs) -> AgentResult:
        """
        执行基本面分析
        
        Args:
            stock_code: 股票代码
            time_range: 时间范围
            **kwargs: 其他参数
            
        Returns:
            AgentResult: 分析结果
        """
        try:
            debug(f"开始对股票 {stock_code} 进行基本面分析")

            # 获取财务数据
            financial_data = self._get_financial_data(stock_code)

            # 检索相关知识库信息
            knowledge = self._retrieve_knowledge(f"股票基本面分析 {stock_code}")

            # 生成分析提示词
            prompt = self._generate_analysis_prompt(stock_code, financial_data, time_range)

            # 使用LLM生成分析
            llm_analysis = self._generate_analysis(prompt, knowledge)

            # 结构化分析结果
            result = self._structure_result(stock_code, llm_analysis, financial_data)

            debug(f"股票 {stock_code} 基本面分析完成")

            return AgentResult(
                agent_name=self.agent_name,
                success=True,
                result=result,
                confidence_score=result.get("confidence_score", 0.85)
            )

        except Exception as e:
            error(f"基本面分析失败: {str(e)}")
            return AgentResult(
                agent_name=self.agent_name,
                success=False,
                result=self.get_result_template(),
                error_message=str(e),
                confidence_score=0.0
            )

    def _get_financial_data(self, stock_code: str) -> Dict[str, Any]:
        """
        获取股票的财务数据
        
        Args:
            stock_code: 股票代码
            
        Returns:
            Dict[str, Any]: 财务数据
        """
        financial_data = {
            "company_info": {},
            "balance_sheet": {},
            "income_statement": {},
            "cash_flow": {},
            "financial_ratios": {},
            "valuation_metrics": {}
        }

        try:
            def _num(value: Any, default: float = 0.0) -> float:
                try:
                    if pd.isna(value):
                        return default
                    return float(value)
                except (TypeError, ValueError):
                    return default

            def _first_number(*values: Any, default: float = 0.0) -> float:
                for value in values:
                    number = _num(value, default=None)
                    if number is not None and number != 0:
                        return number
                return default

            # 获取股票基本信息
            stock_info = self.stock_manager.get_stock_info(stock_code)
            financial_data["company_info"] = {
                "name": stock_info.get("name", ""),
                "sector": stock_info.get("sector", ""),
                "industry": stock_info.get("industry", ""),
                "description": stock_info.get("description", ""),
                "employees": stock_info.get("employees", 0)
            }

            # 获取财务数据
            financial_reports = self.stock_manager.get_financial_data(stock_code)

            report_valuation = financial_reports.get("valuation_metrics", {})
            financial_data["valuation_metrics"] = {
                "market_cap": stock_info.get("market_cap") or report_valuation.get("market_cap", 0),
                "pe_ratio": _first_number(stock_info.get("pe_ratio"), report_valuation.get("pe_ratio")),
                "pb_ratio": _first_number(stock_info.get("pb_ratio"), report_valuation.get("pb_ratio")),
                "ps_ratio": _first_number(stock_info.get("ps_ratio"), report_valuation.get("ps_ratio")),
                "peg_ratio": _first_number(stock_info.get("peg_ratio"), report_valuation.get("peg_ratio")),
            }
            financial_data["valuation_comparison"] = self._build_valuation_comparison(
                financial_data["company_info"],
                financial_data["valuation_metrics"],
            )
            
            # 财务报表数据
            financial_data["balance_sheet"] = financial_reports.get("balance_sheet", {})
            financial_data["income_statement"] = financial_reports.get("income_statement", {})
            financial_data["cash_flow"] = financial_reports.get("cash_flow", {})

            # 财务比率
            financial_data["financial_ratios"] = {
                # 盈利能力
                "profit_margin": financial_reports.get("financial_ratios", {}).get("profit_margin", 0),
                "operating_margin": financial_reports.get("financial_ratios", {}).get("operating_margin", 0),
                "return_on_equity": financial_reports.get("financial_ratios", {}).get("return_on_equity", 0),
                
                # 成长能力
                "revenue_growth": financial_reports.get("financial_ratios", {}).get("revenue_growth", 0),
                "earnings_growth": financial_reports.get("financial_ratios", {}).get("earnings_growth", 0),
                
                # 偿债能力
                "debt_to_equity": financial_reports.get("financial_ratios", {}).get("debt_to_equity", 0),
                "current_ratio": financial_reports.get("financial_ratios", {}).get("current_ratio", 0),
                "quick_ratio": financial_reports.get("financial_ratios", {}).get("quick_ratio", 0),
                
                # 估值指标
                "pe_ratio": financial_data["valuation_metrics"].get("pe_ratio", 0),
                "pb_ratio": financial_data["valuation_metrics"].get("pb_ratio", 0),
                "ps_ratio": financial_data["valuation_metrics"].get("ps_ratio", 0),
                "peg_ratio": financial_data["valuation_metrics"].get("peg_ratio", 0)
            }

            # 尝试直接从financial_reports获取收入报表数据
            if "income_statement" in financial_reports:
                try:
                    # 处理不同格式的收入报表数据
                    income_stmt = financial_reports["income_statement"]
                    if isinstance(income_stmt, dict):
                        financial_data["income_statement"] = {
                            "total_revenue": float(income_stmt.get("totalRevenue", 0) or income_stmt.get("Total Revenue", 0)),
                            "gross_profit": float(income_stmt.get("grossProfit", 0) or income_stmt.get("Gross Profit", 0)),
                            "operating_income": float(income_stmt.get("operatingIncome", 0) or income_stmt.get("Operating Income", 0)),
                            "net_income": float(income_stmt.get("netIncome", 0) or income_stmt.get("Net Income", 0))
                        }
                    elif isinstance(income_stmt, pd.DataFrame) and not income_stmt.empty:
                        # 处理DataFrame格式的数据
                        try:
                            latest = income_stmt.iloc[:, 0] if income_stmt.shape[1] > 0 else {}
                            financial_data["income_statement"] = {
                                "total_revenue": float(latest.get("totalRevenue", 0) or latest.get("Total Revenue", 0) or 0),
                                "gross_profit": float(latest.get("grossProfit", 0) or latest.get("Gross Profit", 0) or 0),
                                "operating_income": float(latest.get("operatingIncome", 0) or latest.get("Operating Income", 0) or 0),
                                "net_income": float(latest.get("netIncome", 0) or latest.get("Net Income", 0) or 0)
                            }
                        except Exception as inner_e:
                            debug(f"处理DataFrame格式的收入报表时出错: {str(inner_e)}")
                except Exception as e:
                    debug(f"处理收入报表数据时出错: {str(e)}")

            # 尝试直接从financial_reports获取现金流数据
            if "cash_flow" in financial_reports:
                try:
                    # 处理不同格式的现金流数据
                    cash_flow_data = financial_reports["cash_flow"]
                    if isinstance(cash_flow_data, dict):
                        financial_data["cash_flow"] = {
                            "operating_cash_flow": float(cash_flow_data.get("operatingCashFlow", 0) or cash_flow_data.get("Operating Cash Flow", 0)),
                            "investing_cash_flow": float(cash_flow_data.get("investingCashFlow", 0) or cash_flow_data.get("Investing Cash Flow", 0)),
                            "financing_cash_flow": float(cash_flow_data.get("financingCashFlow", 0) or cash_flow_data.get("Financing Cash Flow", 0)),
                            "free_cash_flow": float(cash_flow_data.get("freeCashFlow", 0) or cash_flow_data.get("Free Cash Flow", 0))
                        }
                    elif isinstance(cash_flow_data, pd.DataFrame) and not cash_flow_data.empty:
                        # 处理DataFrame格式的数据
                        try:
                            latest = cash_flow_data.iloc[:, 0] if cash_flow_data.shape[1] > 0 else {}
                            financial_data["cash_flow"] = {
                                "operating_cash_flow": float(latest.get("operatingCashFlow", 0) or latest.get("Operating Cash Flow", 0) or 0),
                                "investing_cash_flow": float(latest.get("investingCashFlow", 0) or latest.get("Investing Cash Flow", 0) or 0),
                                "financing_cash_flow": float(latest.get("financingCashFlow", 0) or latest.get("Financing Cash Flow", 0) or 0),
                                "free_cash_flow": float(latest.get("freeCashFlow", 0) or latest.get("Free Cash Flow", 0) or 0)
                            }
                        except Exception as inner_e:
                            debug(f"处理DataFrame格式的现金流时出错: {str(inner_e)}")
                except Exception as e:
                    debug(f"处理现金流数据时出错: {str(e)}")
            
            income_frame = financial_reports.get("income_statement")
            if isinstance(income_frame, pd.DataFrame) and not income_frame.empty:
                income_latest = income_frame.sort_values("statDate").iloc[-1] if "statDate" in income_frame.columns else income_frame.iloc[-1]
                net_income = _num(income_latest.get("netProfit"))
                net_margin = _num(income_latest.get("npMargin"))
                gross_margin = _num(income_latest.get("gpMargin"))
                revenue = net_income / net_margin if net_income and net_margin else 0.0
                financial_data["income_statement"] = {
                    "total_revenue": revenue,
                    "gross_profit": revenue * gross_margin if revenue and gross_margin else 0.0,
                    "operating_income": net_income,
                    "net_income": net_income,
                    "eps_ttm": _num(income_latest.get("epsTTM")),
                    "report_date": str(income_latest.get("statDate", ""))
                }

            cash_frame = financial_reports.get("cash_flow")
            if isinstance(cash_frame, pd.DataFrame) and not cash_frame.empty:
                cash_latest = cash_frame.sort_values("statDate").iloc[-1] if "statDate" in cash_frame.columns else cash_frame.iloc[-1]
                net_income = financial_data["income_statement"].get("net_income", 0)
                operating_cash_flow = _num(cash_latest.get("CFOToNP")) * net_income if net_income else 0.0
                financial_data["cash_flow"] = {
                    "operating_cash_flow": operating_cash_flow,
                    "investing_cash_flow": 0.0,
                    "financing_cash_flow": 0.0,
                    "free_cash_flow": operating_cash_flow,
                    "cash_flow_to_revenue": _num(cash_latest.get("CFOToOR")),
                    "report_date": str(cash_latest.get("statDate", ""))
                }

            ratio_frame = financial_reports.get("financial_ratios")
            growth_frame = financial_reports.get("growth")
            balance_frame = financial_reports.get("balance_sheet")
            ratio_latest = ratio_frame.sort_values("statDate").iloc[-1] if isinstance(ratio_frame, pd.DataFrame) and not ratio_frame.empty and "statDate" in ratio_frame.columns else (ratio_frame.iloc[-1] if isinstance(ratio_frame, pd.DataFrame) and not ratio_frame.empty else None)
            growth_latest = growth_frame.sort_values("statDate").iloc[-1] if isinstance(growth_frame, pd.DataFrame) and not growth_frame.empty and "statDate" in growth_frame.columns else (growth_frame.iloc[-1] if isinstance(growth_frame, pd.DataFrame) and not growth_frame.empty else None)
            balance_latest = balance_frame.sort_values("statDate").iloc[-1] if isinstance(balance_frame, pd.DataFrame) and not balance_frame.empty and "statDate" in balance_frame.columns else (balance_frame.iloc[-1] if isinstance(balance_frame, pd.DataFrame) and not balance_frame.empty else None)

            if ratio_latest is not None or growth_latest is not None or balance_latest is not None:
                current_ratios = financial_data.get("financial_ratios", {}) if isinstance(financial_data.get("financial_ratios"), dict) else {}
                total_assets = _num(balance_latest.get("totalAssets")) if balance_latest is not None else 0.0
                total_liabilities = _num(balance_latest.get("totalLiability")) if balance_latest is not None else 0.0
                if total_liabilities == 0 and balance_latest is not None:
                    total_liabilities = _num(balance_latest.get("totalLiabilities"))
                total_equity = _num(balance_latest.get("totalEquity")) if balance_latest is not None else 0.0
                debt_to_equity = total_liabilities / total_equity if total_liabilities and total_equity else 0.0
                debt_to_assets = total_liabilities / total_assets if total_liabilities and total_assets else 0.0
                current_ratios.update({
                    "profit_margin": _num(financial_data["income_statement"].get("net_income")) / _num(financial_data["income_statement"].get("total_revenue"), 1.0) if _num(financial_data["income_statement"].get("total_revenue")) else 0.0,
                    "operating_margin": _num(financial_data["income_statement"].get("operating_income")) / _num(financial_data["income_statement"].get("total_revenue"), 1.0) if _num(financial_data["income_statement"].get("total_revenue")) else 0.0,
                    "return_on_equity": _num(ratio_latest.get("dupontROE")) if ratio_latest is not None else 0.0,
                    "revenue_growth": _first_number(growth_latest.get("YOYEquity") if growth_latest is not None else None),
                    "earnings_growth": _num(growth_latest.get("YOYNI")) if growth_latest is not None else 0.0,
                    "debt_to_equity": debt_to_equity,
                    "debt_to_assets": debt_to_assets,
                    "current_ratio": _num(balance_latest.get("currentRatio")) if balance_latest is not None else 0.0,
                    "quick_ratio": _num(balance_latest.get("quickRatio")) if balance_latest is not None else 0.0
                })
                financial_data["financial_ratios"] = current_ratios

            if not financial_data["income_statement"] or not financial_data["income_statement"].get("total_revenue", 0):
                warning(f"股票 {stock_code} 的财务报表数据不完整，将告知 LLM 数据缺失")
                financial_data["data_available"] = False
                financial_data["data_note"] = (
                    "财务报表数据暂时无法获取（可能为新股/数据源限制），"
                    "LLM 将基于可用的估值指标（PE/PB/市值）进行分析，"
                    "请用户在正式决策前自行核查最新财报数据。"
                )
                
        except Exception as e:
            error(f"获取财务数据失败: {str(e)}")
            financial_data["data_available"] = False
            financial_data["data_note"] = (
                f"财务数据获取异常：{str(e)}。"
                "LLM 将基于有限可用数据进行分析，结论仅供参考。"
            )

        return financial_data

    def _generate_analysis_prompt(self, stock_code: str, financial_data: Dict[str, Any], time_range: str) -> str:
        """
        生成分析提示词
        
        Args:
            stock_code: 股票代码
            financial_data: 财务数据
            time_range: 时间范围
            
        Returns:
            str: 分析提示词
        """
        # 导入langchain相关组件和提示词管理器
        from langchain_core.prompts import ChatPromptTemplate
        from hengline.prompts.prompt_manage import get_prompt

        # 从提示词管理器获取提示模板
        template = get_prompt('fundamental_agent', 'analysis')

        # 如果获取失败，使用备用模板
        if not template:
            template = """
请对股票代码 {stock_code} 进行全面的基本面分析。

公司基本信息：
{company_info}

财务比率：
{financial_ratios}

估值指标：
{valuation_metrics}

请提供详细分析。
"""

        # 创建提示模板实例
        prompt_template = ChatPromptTemplate.from_template(template)

        # 格式化并返回提示内容
        prompt = prompt_template.format(
            stock_code=stock_code,
            company_info=json.dumps(financial_data['company_info'], indent=2, ensure_ascii=False),
            financial_ratios=json.dumps(financial_data['financial_ratios'], indent=2),
            valuation_metrics=json.dumps(financial_data['valuation_metrics'], indent=2)
        )

        return prompt

    def _structure_result(self, stock_code: str, llm_analysis: Dict[str, Any], financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        结构化分析结果
        
        Args:
            stock_code: 股票代码
            llm_analysis: LLM生成的分析
            financial_data: 财务数据
            
        Returns:
            Dict[str, Any]: 结构化的结果
        """
        result = self.get_result_template()
        data_available = financial_data.get("data_available", True)
        is_simulated = bool(financial_data.get("is_simulated") or financial_data.get("__metadata__", {}).get("is_simulated"))
        quality_fields = build_data_quality_fields(
            data_available=data_available,
            data_note=financial_data.get("data_note", ""),
            is_simulated=is_simulated,
        )
        result.update({
            "stock_code": stock_code,
            "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "company_name": financial_data["company_info"].get("name", ""),
            "sector": financial_data["company_info"].get("sector", ""),
            "industry": financial_data["company_info"].get("industry", ""),
            "key_findings": llm_analysis.get("key_findings", []),
            "detailed_analysis": llm_analysis.get("detailed_analysis", {}),
            "overall_score": llm_analysis.get("overall_score", 0),
            "risk_factors": llm_analysis.get("risk_factors", []),
            "investment_implications": llm_analysis.get("investment_implications", []),
            "confidence_score": llm_analysis.get("confidence_score", 0.85),
            **quality_fields,
            "financial_summary": {
                "pe_ratio": financial_data["financial_ratios"].get("pe_ratio", 0),
                "pb_ratio": financial_data["financial_ratios"].get("pb_ratio", 0),
                "market_cap": financial_data["valuation_metrics"].get("market_cap", 0),
                "revenue": financial_data["income_statement"].get("total_revenue", 0),
                "net_income": financial_data["income_statement"].get("net_income", 0)
            },
            "valuation_comparison": financial_data.get("valuation_comparison", {}),
        })

        # 验证结果
        if not self.validate_result(result):
            warning("基本面分析结果验证失败，使用默认结构")
            result = self.get_result_template()

        return result

    def _build_valuation_comparison(
        self,
        company_info: Dict[str, Any],
        valuation_metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compare PE/PB/PS against coarse industry reference bands."""
        def _num(value: Any, default: float = 0.0) -> float:
            try:
                number = float(value)
                return number if pd.notna(number) else default
            except (TypeError, ValueError):
                return default

        industry_text = " ".join(str(company_info.get(key, "")) for key in ("sector", "industry"))
        profiles = [
            {
                "name": "科技成长",
                "keywords": ["电子", "通信", "计算机", "软件", "半导体", "人工智能", "云"],
                "pe": (20, 35, 55),
                "pb": (2.0, 4.0, 7.0),
                "ps": (2.0, 5.0, 10.0),
            },
            {
                "name": "先进制造",
                "keywords": ["制造", "设备", "新能源", "汽车", "机械", "电气"],
                "pe": (15, 25, 40),
                "pb": (1.5, 3.0, 5.0),
                "ps": (1.0, 3.0, 6.0),
            },
            {
                "name": "消费医药",
                "keywords": ["消费", "食品", "医药", "医疗", "生物"],
                "pe": (18, 32, 50),
                "pb": (2.0, 4.5, 8.0),
                "ps": (2.0, 5.0, 9.0),
            },
        ]
        profile = next(
            (item for item in profiles if any(keyword in industry_text for keyword in item["keywords"])),
            {
                "name": "全市场参考",
                "keywords": [],
                "pe": (12, 22, 35),
                "pb": (1.2, 2.5, 4.5),
                "ps": (1.0, 2.5, 5.0),
            },
        )

        def _assess(metric_name: str, value: float) -> Dict[str, Any]:
            low, median, high = profile[metric_name]
            if value <= 0:
                bucket = "unavailable"
                percentile = None
                note = "当前数据源未返回有效估值指标。"
            elif value <= low:
                bucket = "low"
                percentile = 25
                note = "低于行业参考低位，需结合盈利质量确认是否低估。"
            elif value <= median:
                bucket = "reasonable"
                percentile = 50
                note = "接近行业参考中枢。"
            elif value <= high:
                bucket = "elevated"
                percentile = 75
                note = "高于行业参考中枢，需匹配成长性。"
            else:
                bucket = "expensive"
                percentile = 90
                note = "显著高于行业参考高位，估值容错率偏低。"
            return {
                "value": value,
                "reference_band": {"low": low, "median": median, "high": high},
                "bucket": bucket,
                "approx_percentile": percentile,
                "note": note,
            }

        comparison = {
            "profile": profile["name"],
            "method": "static_industry_reference_band",
            "limitations": "当前为行业粗粒度参考区间；接入同业样本池后可升级为实时分位数。",
            "metrics": {
                "pe": _assess("pe", _num(valuation_metrics.get("pe_ratio"))),
                "pb": _assess("pb", _num(valuation_metrics.get("pb_ratio"))),
                "ps": _assess("ps", _num(valuation_metrics.get("ps_ratio"))),
            },
        }
        expensive_count = sum(
            1 for item in comparison["metrics"].values()
            if item["bucket"] in {"elevated", "expensive"}
        )
        comparison["valuation_risk"] = (
            "high" if expensive_count >= 2 else
            "medium" if expensive_count == 1 else
            "low"
        )
        return comparison

