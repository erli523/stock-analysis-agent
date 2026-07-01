#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Streamlit stock data dashboard."""

import os
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
load_dotenv(".env")

from hengline.stock.stock_manage import (  # noqa: E402
    get_financial_data,
    get_stock_info,
    get_stock_news,
    get_stock_price_data,
)
from hengline.agents.agent_coordinator import AgentCoordinator  # noqa: E402


@st.cache_resource(show_spinner="Initializing agent system (first time only)...")
def get_coordinator(use_memory: bool) -> AgentCoordinator:
    """缓存 AgentCoordinator 单例，避免每次点击重建 7 个 Agent 实例。
    use_memory 变化时自动重建。"""
    return AgentCoordinator(
        {
            "agents": {
                name: {"enable_memory": use_memory}
                for name in [
                    "FundamentalAgent",
                    "TechnicalAgent",
                    "IndustryMacroAgent",
                    "SentimentAgent",
                    "FundFlowAgent",
                    "ESGRiskAgent",
                    "ChiefStrategyAgent",
                ]
            }
        }
    )


PERIOD_OPTIONS = {
    "1D": "1d",
    "1W": "1w",
    "1M": "1m",
    "3M": "3m",
    "6M": "6m",
    "1Y": "1y",
    "5Y": "5y",
    "10Y": "10y",
    "MAX": "max",
}


def setup_page():
    st.set_page_config(page_title="Stock Data Analysis", page_icon="📈", layout="wide")
    st.markdown(
        """
        <style>
        .block-container { padding-top: 2rem; }
        .main-title { color: #07539f; font-weight: 800; font-size: 2.2rem; }
        .recommendation-box {
            border: 1px solid #e5e7eb;
            border-left: 0.45rem solid #0f766e;
            border-radius: 0.5rem;
            padding: 1rem 1.2rem;
            background: #f8fafc;
            margin: 0.5rem 0 1rem 0;
        }
        .recommendation-box h4 { margin: 0 0 0.4rem 0; color: #111827; }
        .muted-small { color: #6b7280; font-size: 0.86rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def stock_display_name(stock_info: dict, ticker: str) -> str:
    return (
        stock_info.get("name")
        or stock_info.get("company_name")
        or stock_info.get("long_name")
        or stock_info.get("symbol")
        or ticker
    )


def normalize_price_data(price_data: pd.DataFrame) -> pd.DataFrame:
    if price_data is None or price_data.empty:
        return pd.DataFrame()
    data = price_data.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date")
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    return data.dropna(subset=["Open", "High", "Low", "Close"])


def format_volume(value) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if number >= 100000000:
        return f"{number / 100000000:.2f}B shares"
    if number >= 10000:
        return f"{number / 10000:.2f}W shares"
    return f"{number:,.0f} shares"


def format_optional(value, fallback: str = "N/A") -> str:
    if value in (None, "", [], {}):
        return fallback
    return str(value)


def format_score(value) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "N/A"


def normalize_agent_name(name: str) -> str:
    labels = {
        "TechnicalAgent": "Technical",
        "FundamentalAgent": "Fundamental",
        "SentimentAgent": "Sentiment",
        "FundFlowAgent": "Fund Flow",
        "ESGRiskAgent": "ESG Risk",
        "IndustryMacroAgent": "Industry & Macro",
        "ChiefStrategyAgent": "Chief Strategy",
    }
    return labels.get(name, name)


def first_non_empty(*values, fallback=""):
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return fallback


def show_price_summary(price_data: pd.DataFrame):
    if price_data.empty:
        st.warning("No price data available.")
        return

    latest = price_data.iloc[-1]
    previous = price_data.iloc[-2] if len(price_data) > 1 else latest
    day_change = latest["Close"] - previous["Close"]
    day_change_pct = day_change / previous["Close"] * 100 if previous["Close"] else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest Close", f"{latest['Close']:.2f}", f"{day_change:+.2f} ({day_change_pct:+.2f}%)")
    c2.metric("High", f"{latest['High']:.2f}")
    c3.metric("Low", f"{latest['Low']:.2f}")
    c4.metric("Volume", format_volume(latest.get("Volume", 0)))


def show_candlestick(ticker: str, price_data: pd.DataFrame):
    if price_data.empty:
        st.warning("No chart data available.")
        return

    # 将日期转为字符串，配合 category x 轴使用，可自动去除非交易日（周末/节假日）空白
    date_strs = price_data["Date"].dt.strftime("%Y-%m-%d").tolist()

    # 根据数据量动态控制 x 轴刻度密度，避免标签过密
    n = len(date_strs)
    if n > 120:
        step = max(1, n // 20)
        tick_vals = date_strs[::step]
    elif n > 40:
        step = max(1, n // 10)
        tick_vals = date_strs[::step]
    else:
        tick_vals = date_strs

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=date_strs,
                open=price_data["Open"],
                high=price_data["High"],
                low=price_data["Low"],
                close=price_data["Close"],
                name=ticker,
            )
        ]
    )
    fig.update_layout(
        title=f"{ticker} Candlestick",
        xaxis_title="Date",
        yaxis_title="Price",
        height=560,
        xaxis=dict(
            type="category",
            tickangle=45,
            tickmode="array",
            tickvals=tick_vals,
        ),
    )
    st.plotly_chart(fig, width="stretch")

    if "Volume" in price_data.columns:
        volume_data = price_data.copy()
        volume_data["DateStr"] = date_strs
        volume_data["VolumeWan"] = volume_data["Volume"] / 10000
        volume_fig = px.bar(volume_data, x="DateStr", y="VolumeWan", title="Volume")
        volume_fig.update_layout(
            height=260,
            xaxis_title="Date",
            yaxis_title="Volume (10k shares)",
            xaxis=dict(
                type="category",
                tickangle=45,
                tickmode="array",
                tickvals=tick_vals,
            ),
        )
        st.plotly_chart(volume_fig, width="stretch")


def show_overview(ticker: str, stock_info: dict, price_data: pd.DataFrame, news_data: list):
    name = stock_display_name(stock_info, ticker)
    symbol = stock_info.get("symbol") or stock_info.get("code") or ticker
    st.markdown(f"### {name} ({symbol})")

    description = stock_info.get("description") or "Company profile is not available from the current data source."
    st.write(description)

    show_price_summary(price_data)
    show_candlestick(ticker, price_data)

    st.markdown("#### Basic Information")
    info_items = {
        "Market Cap": stock_info.get("market_cap"),
        "IPO Date": stock_info.get("ipo_date") or stock_info.get("list_date"),
        "Exchange": stock_info.get("market") or stock_info.get("primary_exchange") or stock_info.get("exchange"),
        "PE (TTM)": str(stock_info.get("pe_ratio")) if stock_info.get("pe_ratio") is not None else None,
        "PB (MRQ)": str(stock_info.get("pb_ratio")) if stock_info.get("pb_ratio") is not None else None,
    }
    cols = st.columns(len(info_items))
    for col, (label, value) in zip(cols, info_items.items()):
        col.metric(label, value if value not in (None, "") else "N/A")

    sector = stock_info.get("sector") or stock_info.get("industry")
    industry = stock_info.get("industry")
    if sector or industry:
        si_cols = st.columns(2)
        si_cols[0].metric("Sector / Industry", sector or "N/A")
        if industry and industry != sector:
            si_cols[1].metric("Industry", industry)

    st.markdown("#### Latest News")
    if not news_data:
        st.info("Stock-specific news is not available from the current configured sources.")
    for item in news_data:
        title = item.get("title", "Untitled")
        source = item.get("source") or item.get("publisher") or "Unknown"
        summary = item.get("summary") or ""
        with st.expander(f"{title} - {source}"):
            st.write(summary)
            link = item.get("link")
            if link:
                st.link_button("Open article", link)


def show_technical(ticker: str, price_data: pd.DataFrame):
    if price_data.empty:
        st.warning("No technical data available.")
        return

    data = price_data.copy()
    data["MA5"] = data["Close"].rolling(5).mean()
    data["MA20"] = data["Close"].rolling(20).mean()
    data["MA60"] = data["Close"].rolling(60).mean()

    date_strs = data["Date"].dt.strftime("%Y-%m-%d").tolist()
    n = len(date_strs)
    step = max(1, n // 20) if n > 120 else (max(1, n // 10) if n > 40 else 1)
    tick_vals = date_strs[::step]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=date_strs, y=data["Close"], name="Close"))
    fig.add_trace(go.Scatter(x=date_strs, y=data["MA5"], name="MA5"))
    fig.add_trace(go.Scatter(x=date_strs, y=data["MA20"], name="MA20"))
    fig.add_trace(go.Scatter(x=date_strs, y=data["MA60"], name="MA60"))
    fig.update_layout(
        title=f"{ticker} Price and Moving Averages",
        xaxis_title="Date",
        yaxis_title="Price",
        height=560,
        xaxis=dict(type="category", tickangle=45, tickmode="array", tickvals=tick_vals),
    )
    st.plotly_chart(fig, width="stretch")


def show_comparison(tickers: list[str], period: str):
    st.markdown("### Stock Comparison")
    fig = go.Figure()

    all_date_strs = None
    for ticker in tickers:
        data = normalize_price_data(get_stock_price_data(ticker, period=period))
        if data.empty:
            st.warning(f"No comparable data for {ticker}.")
            continue
        norm_close = data["Close"] / data["Close"].iloc[0] * 100
        date_strs = data["Date"].dt.strftime("%Y-%m-%d").tolist()
        fig.add_trace(go.Scatter(x=date_strs, y=norm_close, name=ticker))
        if all_date_strs is None or len(date_strs) > len(all_date_strs):
            all_date_strs = date_strs

    if all_date_strs:
        n = len(all_date_strs)
        step = max(1, n // 20) if n > 120 else (max(1, n // 10) if n > 40 else 1)
        tick_vals = all_date_strs[::step]
        xaxis_cfg = dict(type="category", tickangle=45, tickmode="array", tickvals=tick_vals)
    else:
        xaxis_cfg = {}

    fig.update_layout(
        title="Normalized Price Comparison (first day = 100)",
        xaxis_title="Date",
        yaxis_title="Index",
        height=520,
        xaxis=xaxis_cfg,
    )
    st.plotly_chart(fig, width="stretch")


def show_financial_export(ticker: str):
    try:
        financial_data = get_financial_data(ticker)
        if not financial_data:
            st.info("Financial data is not available from the current data source.")
            return

        st.markdown("### Financial Data")
        st.caption("Financial statements and ratios returned by the configured market data sources.")

        table_names = {
            "income_statement": "Income Statement",
            "balance_sheet": "Balance Sheet",
            "cash_flow": "Cash Flow",
            "financial_ratios": "Financial Ratios",
            "growth": "Growth",
            "operation": "Operations",
            "valuation_metrics": "Valuation Metrics",
        }
        available_tables = []
        for key, value in financial_data.items():
            if isinstance(value, pd.DataFrame) and not value.empty:
                available_tables.append((key, value))
            elif isinstance(value, dict) and value:
                available_tables.append((key, pd.DataFrame([value])))

        if not available_tables:
            st.info("Financial data is not available from the current configured sources.")
            return

        tabs = st.tabs([table_names.get(key, key.replace("_", " ").title()) for key, _ in available_tables])
        for tab, (key, frame) in zip(tabs, available_tables):
            with tab:
                st.dataframe(frame, width="stretch", hide_index=True)
                st.download_button(
                    label=f"Download {table_names.get(key, key)} CSV",
                    data=frame.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"{ticker}_{key}.csv",
                    mime="text/csv",
                )
    except Exception as exc:
        st.warning(f"Unable to load financial data: {exc}")


def render_recommendation(recommendation: dict):
    action = format_optional(recommendation.get("investment_recommendation"), "No recommendation")
    description = format_optional(recommendation.get("recommendation_description"), "")
    confidence = format_optional(recommendation.get("recommendation_confidence"), "N/A")
    signal_strength = format_optional(recommendation.get("signal_strength"), "N/A")
    summary = first_non_empty(
        recommendation.get("analysis_summary"),
        recommendation.get("recommendation_details"),
        fallback="No written summary returned by the chief strategy agent.",
    )

    st.markdown("#### Final Recommendation")
    st.markdown(
        f"""
        <div class="recommendation-box">
            <h4>{action}</h4>
            <div>{description}</div>
            <div class="muted-small">Generated at {format_optional(recommendation.get("analysis_time"))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recommendation", action)
    c2.metric("Confidence", confidence)
    c3.metric("Signal Strength", signal_strength)
    c4.metric("Model Confidence", format_score(recommendation.get("confidence_score")))

    metrics = recommendation.get("comprehensive_metrics") or {}
    if metrics:
        m1, m2, m3 = st.columns(3)
        m1.metric("Composite Score", format_score(metrics.get("composite_score")))
        m2.metric("Risk Score", format_score(metrics.get("risk_score")))
        m3.metric("Risk Level", format_optional(metrics.get("risk_level")))

    st.markdown("#### Analysis Summary")
    st.write(summary)

    details = recommendation.get("recommendation_details")
    if details and details != summary:
        with st.expander("Recommendation Rationale", expanded=False):
            st.write(details)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Position Suggestion")
        st.write(format_optional(recommendation.get("position_suggestion"), "No position sizing suggestion returned."))
        st.markdown("#### Suitable Investors")
        st.write(format_optional(recommendation.get("suitable_investors"), "No investor profile returned."))
    with c2:
        st.markdown("#### Key Monitoring Metrics")
        metrics_to_watch = recommendation.get("key_monitoring_metrics") or []
        if metrics_to_watch:
            for item in metrics_to_watch:
                st.markdown(f"- {item}")
        else:
            st.info("No monitoring metrics returned.")

    risk_disclosure = recommendation.get("risk_disclosure")
    if risk_disclosure:
        with st.expander("Risk Disclosure", expanded=False):
            st.write(risk_disclosure)


def render_agent_status(status: dict):
    if not status:
        return
    st.markdown("#### Agent Status")

    failed_agents = [
        name for name, item in status.items() if not item.get("success")
    ]
    if failed_agents:
        failed_labels = ", ".join(normalize_agent_name(n) for n in failed_agents)
        st.warning(
            f"**{len(failed_agents)} 个 Agent 分析失败**：{failed_labels}。"
            " 以下维度数据缺失，最终建议仅基于成功的 Agent，请谨慎参考。",
            icon="⚠️",
        )

    rows = []
    for agent_name, item in status.items():
        success = item.get("success", False)
        rows.append(
            {
                "Agent": normalize_agent_name(agent_name),
                "Status": "✅ 成功" if success else "❌ 失败",
                "Confidence": f"{item.get('confidence_score', 0):.0%}" if item.get('confidence_score') is not None else "N/A",
                "Issue": item.get("error") or ("" if success else "分析未完成"),
            }
        )
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_agent_details(details: dict, status: dict = None):
    if not details and not status:
        return

    # 合并 details 和 status，确保失败的 Agent 也显示在 Tab 中
    all_agent_names = list(details.keys())
    if status:
        for name in status:
            if name not in all_agent_names and name != "ChiefStrategyAgent":
                all_agent_names.append(name)

    if not all_agent_names:
        return

    st.markdown("#### Agent Findings")
    tabs = st.tabs([normalize_agent_name(name) for name in all_agent_names])

    for tab, agent_name in zip(tabs, all_agent_names):
        with tab:
            agent_status = (status or {}).get(agent_name, {})
            agent_result = details.get(agent_name, {})
            agent_success = agent_status.get("success", bool(agent_result))

            if not agent_success:
                err_msg = agent_status.get("error") or "该维度分析未完成"
                st.error(
                    f"**{normalize_agent_name(agent_name)} 分析失败**：{err_msg}\n\n"
                    "此维度数据缺失，最终投资建议中该维度权重已被跳过。",
                    icon="❌",
                )
                continue

            st.metric("Confidence", format_score(agent_result.get("confidence_score")))
            findings = agent_result.get("key_findings") or []
            if findings:
                st.markdown("**Key Findings**")
                for finding in findings[:8]:
                    st.markdown(f"- {finding}")
            else:
                st.info("This agent did not return key findings.")

            summary_keys = [
                "short_term_outlook",
                "medium_term_outlook",
                "industry_outlook",
                "potential_impact",
                "risk_assessment",
                "sentiment_summary",
                "technical_summary",
                "financial_health",
                "valuation_assessment",
            ]
            shown = False
            for key in summary_keys:
                value = agent_result.get(key)
                if value not in (None, "", [], {}):
                    with st.expander(key.replace("_", " ").title(), expanded=False):
                        st.write(value)
                    shown = True
            if not shown:
                with st.expander("Structured Result", expanded=False):
                    st.write(agent_result)


def show_agent_analysis(ticker: str, period: str):
    st.markdown("### AI Agent Analysis")
    st.caption("Runs the coordinated multi-agent workflow and summarizes the final recommendation.")

    use_memory = st.checkbox("Use agent memory", value=False)
    if not st.button("Run AI analysis", type="primary"):
        st.info("Click the button to start the agent workflow. This can take a little while.")
        return

    with st.spinner("Running agent workflow..."):
        try:
            coordinator = get_coordinator(use_memory)
            result = coordinator.analyze(ticker, time_range=period)
        except Exception as exc:
            st.error(f"Agent workflow failed to start: {exc}")
            return

    elapsed = result.get("elapsed_time_seconds")
    if not result.get("success"):
        st.error(result.get("error") or "Agent workflow failed.")
    else:
        if isinstance(elapsed, (int, float)):
            st.success(f"Agent workflow completed in {elapsed:.1f}s.")
        else:
            st.success("Agent workflow completed.")

    recommendation = result.get("final_recommendation") or {}
    if recommendation:
        render_recommendation(recommendation)

    status = result.get("agent_execution_status") or {}
    render_agent_status(status)

    details = result.get("detailed_results") or {}
    render_agent_details(details, status)


setup_page()

with st.sidebar:
    st.markdown("## Stock Query")
    ticker = st.text_input("Ticker / Stock Code", value="300502").strip().upper()
    period_display = st.selectbox("Period", list(PERIOD_OPTIONS.keys()), index=2)
    period = PERIOD_OPTIONS[period_display]

    st.markdown("---")
    st.markdown("## Compare")
    comparison_enabled = st.checkbox("Enable comparison", value=False)
    compare_tickers = []
    if comparison_enabled:
        compare_tickers.append(st.text_input("Compare 1", value="300308").strip().upper())
        compare_tickers.append(st.text_input("Compare 2", value="NVDA").strip().upper())
        compare_tickers = [code for code in compare_tickers if code]

    st.markdown("---")
    st.markdown("## View")
    view_mode = st.selectbox("View mode", ["Overview", "Price Chart", "Technical", "Financial", "AI Analysis"])

    st.markdown("---")
    if st.button("🔄 Clear Cache & Reload", help="Clear all cached data and reload fresh from data sources"):
        st.cache_data.clear()
        st.rerun()

    # ── 知识库 RAG 状态 ──────────────────────────────────
    st.markdown("---")
    st.markdown("## 📚 Knowledge Base")
    _kb_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "knowledge_base")
    _embed_dir_raw = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "embeddings")
    _vs_file = os.path.join(_embed_dir_raw, "default__vector_store.json")

    # 统计知识库文档
    _doc_count = sum(1 for r, _, fs in os.walk(_kb_dir) for f in fs if f.endswith((".txt", ".md", ".pdf")))

    # 检查向量索引状态
    _vec_count = 0
    _index_ready = False
    if os.path.exists(_vs_file):
        try:
            import json as _json
            with open(_vs_file, "r", encoding="utf-8") as _f:
                _vs = _json.load(_f)
            _vec_count = len(_vs.get("embedding_dict", {}))
            _index_ready = _vec_count > 0
        except Exception:
            pass

    if _index_ready:
        st.success(f"✅ Index ready ({_vec_count} chunks)")
    else:
        st.warning(f"⚠️ Index not built yet")

    st.caption(f"Docs: {_doc_count} files")

    if st.button("🔨 Build / Rebuild Index", help="向量化知识库文档（需调用 Embedding API，约1~3分钟）"):
        with st.spinner("Building RAG index... (calling DashScope text-embedding-v4)"):
            try:
                from hengline.tools.llama_index_tool import create_index_from_directory
                from config.config import get_data_embeddings_path
                _idx = create_index_from_directory(
                    directory_path=_kb_dir,
                    index_name="stock_knowledge_base",
                    storage_dir=get_data_embeddings_path(),
                    recursive=True,
                    rebuild=True,
                )
                st.success("✅ Knowledge base index built successfully!")
                st.rerun()
            except Exception as _e:
                st.error(f"Build failed: {_e}")
    # ────────────────────────────────────────────────────

st.markdown('<div class="main-title">Stock Data Analysis Platform</div>', unsafe_allow_html=True)


@st.cache_data(ttl=600, show_spinner=False)
def load_market_data(ticker: str, period: str):
    """缓存行情数据 10 分钟，避免切换视图时重复拉取（首次加载约 15~20s）。
    注意：先拉 stock_info（含市值/PE/PB），再拉大量K线，避免BaoStock session被大查询占满。
    """
    # 1. 先拉基本信息（_get_market_metrics 需要在 BaoStock 连接干净时调用）
    info = get_stock_info(ticker) or {}
    # 2. 再拉价格数据（可能涉及大量5分钟K线）
    price_df = normalize_price_data(get_stock_price_data(ticker, period=period))
    # 3. 最后拉新闻
    news = get_stock_news(ticker) or []
    return price_df, info, news


with st.spinner("Loading data... (first load may take ~15 s while fetching market metrics)"):
    try:
        price_data, stock_info, news_data = load_market_data(ticker, period)

        if view_mode == "Overview":
            show_overview(ticker, stock_info, price_data, news_data)
        elif view_mode == "Price Chart":
            show_price_summary(price_data)
            show_candlestick(ticker, price_data)
        elif view_mode == "Technical":
            show_price_summary(price_data)
            show_technical(ticker, price_data)
        elif view_mode == "Financial":
            show_financial_export(ticker)
        elif view_mode == "AI Analysis":
            show_agent_analysis(ticker, period)

        if comparison_enabled and compare_tickers:
            st.markdown("---")
            show_comparison([ticker] + compare_tickers, period)

        st.markdown("---")
        st.markdown("## Export")
        if not price_data.empty:
            st.download_button(
                label="Download price data (CSV)",
                data=price_data.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{ticker}_price_data.csv",
                mime="text/csv",
            )
        else:
            st.info("No price data to export.")
    except Exception as exc:
        st.error(f"Failed to load data: {exc}")
        st.info("Try 300502, 300308, NVDA, or AAPL.")

