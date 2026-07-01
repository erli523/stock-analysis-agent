#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Streamlit stock data dashboard."""

import os
import sys
from datetime import date, timedelta

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
from hengline.streamlit.st_product_features import (  # noqa: E402
    AGENT_LABELS,
    add_favorite,
    answer_followup_question,
    build_markdown_report,
    load_favorites,
    normalize_agent_selection,
    remove_favorite,
    render_advanced_technical_charts,
    render_alerts_page,
    render_backtest_page,
    render_financial_visuals,
    render_history_page,
    render_portfolio_page,
    render_report_downloads,
    render_screener_page,
    render_watchlist_page,
    save_analysis_result,
)


# 全局页面样式（从 setup_page 内联 <style> 提取为模块常量，便于集中维护）
APP_CSS = """
<style>
.block-container {
    padding-top: 1.4rem;
    padding-bottom: 2.5rem;
    max-width: 1500px;
}
[data-testid="stSidebar"] {
    background: #f4f7fb;
    border-right: 1px solid #dde4ee;
}
.app-header {
    border-bottom: 1px solid #e5e7eb;
    padding: 0.4rem 0 1.1rem 0;
    margin-bottom: 1.2rem;
}
.main-title {
    color: #073b74;
    font-weight: 800;
    font-size: 2.15rem;
    line-height: 1.15;
    margin: 0;
}
.sub-title {
    color: #667085;
    font-size: 0.95rem;
    margin-top: 0.45rem;
}
.context-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 0.85rem;
}
.context-pill {
    border: 1px solid #d7dee8;
    border-radius: 999px;
    padding: 0.25rem 0.7rem;
    color: #344054;
    background: #ffffff;
    font-size: 0.85rem;
}
.section-title {
    color: #111827;
    font-weight: 750;
    font-size: 1.15rem;
    margin: 1.25rem 0 0.75rem 0;
}
.data-note {
    color: #667085;
    font-size: 0.88rem;
    margin-bottom: 0.75rem;
}
.insight-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 0.75rem;
    margin: 0.8rem 0 1.1rem 0;
}
.insight-card {
    background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
    border: 1px solid #dfe7f1;
    border-radius: 0.5rem;
    padding: 0.85rem 0.95rem;
    min-height: 6.2rem;
}
.insight-label {
    color: #667085;
    font-size: 0.78rem;
    margin-bottom: 0.35rem;
}
.insight-value {
    color: #0f172a;
    font-size: 1.22rem;
    line-height: 1.2;
    font-weight: 780;
}
.insight-note {
    color: #526071;
    font-size: 0.82rem;
    line-height: 1.35;
    margin-top: 0.42rem;
}
.callout {
    border: 1px solid #dbe5f2;
    border-radius: 0.5rem;
    padding: 0.9rem 1rem;
    background: #f8fafc;
    margin: 0.75rem 0;
}
.callout strong { color: #0f172a; }
.split-panel {
    border: 1px solid #e2e8f0;
    border-radius: 0.5rem;
    background: #ffffff;
    padding: 0.9rem 1rem;
}
div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e6ebf2;
    border-radius: 0.5rem;
    padding: 0.7rem 0.85rem;
}
div[data-testid="stMetricLabel"] {
    color: #667085;
}
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
"""


@st.cache_resource(show_spinner="Initializing agent system (first time only)...")
def get_coordinator(use_memory: bool, enabled_agents: tuple[str, ...]) -> AgentCoordinator:
    """缓存 AgentCoordinator 单例，避免每次点击重建 7 个 Agent 实例。
    use_memory 或选择的 Agent 变化时自动重建。"""
    return AgentCoordinator(
        {
            "enabled_agents": list(enabled_agents),
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

VIEW_OPTIONS = {
    "Overview": "概览",
    "Price Chart": "K线行情",
    "Technical": "技术分析",
    "Financial": "财务数据",
    "AI Analysis": "智能体分析",
    "Knowledge QA": "知识库问答",
    "Watchlist": "自选股",
    "History": "历史分析",
    "Screener": "股票筛选",
    "Backtest": "策略回测",
    "Portfolio": "投资组合",
    "Alerts": "预警配置",
}

CHART_LAYOUT = {
    "template": "plotly_white",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"family": "Arial, sans-serif", "color": "#344054"},
    "margin": {"l": 48, "r": 24, "t": 58, "b": 48},
    "hovermode": "x unified",
}


def _tick_step(n: int) -> int:
    """按数据量选择 x 轴刻度抽样步长，避免标签过密。"""
    if n > 120:
        return max(1, n // 20)
    if n > 40:
        return max(1, n // 10)
    return 1


def _category_axis(date_strs: list[str]) -> dict:
    """构造 category 型 x 轴配置：按交易日压缩横轴、去除非交易日空白。"""
    if not date_strs:
        return {}
    tick_vals = date_strs[::_tick_step(len(date_strs))]
    return dict(type="category", tickangle=45, tickmode="array", tickvals=tick_vals)


def setup_page():
    st.set_page_config(page_title="股票数据分析平台", page_icon="📈", layout="wide")
    st.markdown(APP_CSS, unsafe_allow_html=True)


def stock_display_name(stock_info: dict, ticker: str) -> str:
    return (
        stock_info.get("name")
        or stock_info.get("company_name")
        or stock_info.get("long_name")
        or stock_info.get("symbol")
        or ticker
    )


def render_app_header(ticker: str, stock_info: dict, period_label: str, view_label: str):
    name = stock_display_name(stock_info, ticker)
    symbol = stock_info.get("symbol") or stock_info.get("code") or ticker
    exchange = stock_info.get("market") or stock_info.get("primary_exchange") or stock_info.get("exchange")
    source_hint = stock_info.get("data_source") or stock_info.get("source") or "configured market sources"
    st.markdown(
        f"""
        <div class="app-header">
            <div class="main-title">股票数据分析平台</div>
            <div class="sub-title">{name} <strong>{symbol}</strong></div>
            <div class="context-row">
                <span class="context-pill">视图：{view_label}</span>
                <span class="context-pill">周期：{period_label}</span>
                <span class="context-pill">市场：{format_optional(exchange, "N/A")}</span>
                <span class="context-pill">数据源：{format_optional(source_hint)}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(title: str, note: str = ""):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if note:
        st.markdown(f'<div class="data-note">{note}</div>', unsafe_allow_html=True)


def normalize_price_data(price_data: pd.DataFrame) -> pd.DataFrame:
    if price_data is None or price_data.empty:
        return pd.DataFrame()
    data = price_data.copy()
    data.attrs.update(getattr(price_data, "attrs", {}) or {})
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date")
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    return data.dropna(subset=["Open", "High", "Low", "Close"])


def render_data_quality_warning(stock_info: dict, price_data: pd.DataFrame, financial_data: dict = None):
    simulated_sources = []
    if stock_info.get("is_simulated"):
        simulated_sources.append("基本信息")
    if getattr(price_data, "attrs", {}).get("is_simulated"):
        simulated_sources.append("价格行情")
    if financial_data and financial_data.get("__metadata__", {}).get("is_simulated"):
        simulated_sources.append("财务数据")
    if not simulated_sources:
        return
    st.error(
        " / ".join(simulated_sources)
        + " 使用了模拟数据：在线数据源获取失败，当前结果仅用于界面预览和流程调试，不能作为投资分析依据。",
        icon="⚠️",
    )


def format_volume(value) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if number >= 100000000:
        return f"{number / 100000000:.2f}亿股"
    if number >= 10000:
        return f"{number / 10000:.2f}万股"
    return f"{number:,.0f}股"


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
        "TechnicalAgent": "技术面",
        "FundamentalAgent": "基本面",
        "SentimentAgent": "情绪面",
        "FundFlowAgent": "资金流",
        "ESGRiskAgent": "ESG 风险",
        "IndustryMacroAgent": "行业宏观",
        "ChiefStrategyAgent": "首席策略",
    }
    return labels.get(name, name)


def first_non_empty(*values, fallback=""):
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return fallback


def pct_text(value) -> str:
    try:
        return f"{float(value):+.2f}%"
    except (TypeError, ValueError):
        return "N/A"


def number_text(value, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "N/A"


def render_insight_cards(cards: list[dict]):
    for start in range(0, len(cards), 4):
        chunk = cards[start:start + 4]
        cols = st.columns(len(chunk))
        for col, card in zip(cols, chunk):
            col.metric(card.get("label", ""), card.get("value", "N/A"))
            note = card.get("note")
            if note:
                col.caption(note)


def compute_price_insights(price_data: pd.DataFrame) -> dict:
    if price_data is None or price_data.empty or len(price_data) < 2:
        return {}

    data = price_data.copy()
    close = data["Close"].astype(float)
    high = data["High"].astype(float)
    low = data["Low"].astype(float)
    volume = data["Volume"].astype(float) if "Volume" in data.columns else pd.Series(dtype=float)
    returns = close.pct_change().dropna()

    latest_close = float(close.iloc[-1])
    start_close = float(close.iloc[0])
    period_return = (latest_close / start_close - 1) * 100 if start_close else 0
    period_high = float(high.max())
    period_low = float(low.min())
    distance_to_high = (latest_close / period_high - 1) * 100 if period_high else 0
    distance_to_low = (latest_close / period_low - 1) * 100 if period_low else 0

    rolling_peak = close.cummax()
    drawdown = close / rolling_peak - 1
    max_drawdown = float(drawdown.min() * 100)
    volatility = float(returns.std() * (252 ** 0.5) * 100) if not returns.empty else 0

    ma5 = close.rolling(5).mean().iloc[-1] if len(close) >= 5 else None
    ma20 = close.rolling(20).mean().iloc[-1] if len(close) >= 20 else None
    ma60 = close.rolling(60).mean().iloc[-1] if len(close) >= 60 else None

    latest_volume = float(volume.iloc[-1]) if not volume.empty else None
    avg_volume = float(volume.tail(min(20, len(volume))).mean()) if not volume.empty else None
    volume_ratio = latest_volume / avg_volume if latest_volume and avg_volume else None

    if ma20 and latest_close > ma20 and (not ma5 or ma5 >= ma20):
        trend_label = "偏强"
    elif ma20 and latest_close < ma20 and (not ma5 or ma5 <= ma20):
        trend_label = "偏弱"
    else:
        trend_label = "震荡"

    if volume_ratio is None:
        volume_label = "暂无量能判断"
    elif volume_ratio >= 1.5:
        volume_label = "明显放量"
    elif volume_ratio <= 0.7:
        volume_label = "明显缩量"
    else:
        volume_label = "量能平稳"

    risk_label = "高波动" if volatility >= 45 else ("中等波动" if volatility >= 25 else "低波动")
    position_label = "接近区间高位" if distance_to_high > -5 else ("接近区间低位" if distance_to_low < 8 else "区间中部")

    return {
        "latest_close": latest_close,
        "period_return": period_return,
        "period_high": period_high,
        "period_low": period_low,
        "distance_to_high": distance_to_high,
        "distance_to_low": distance_to_low,
        "max_drawdown": max_drawdown,
        "volatility": volatility,
        "ma5": float(ma5) if ma5 is not None and pd.notna(ma5) else None,
        "ma20": float(ma20) if ma20 is not None and pd.notna(ma20) else None,
        "ma60": float(ma60) if ma60 is not None and pd.notna(ma60) else None,
        "latest_volume": latest_volume,
        "avg_volume": avg_volume,
        "volume_ratio": volume_ratio,
        "trend_label": trend_label,
        "volume_label": volume_label,
        "risk_label": risk_label,
        "position_label": position_label,
        "trading_days": len(data),
    }


def render_market_brief(price_data: pd.DataFrame):
    insights = compute_price_insights(price_data)
    if not insights:
        st.info("数据点不足，暂无法生成行情摘要。")
        return

    cards = [
        {
            "label": "区间涨跌幅",
            "value": pct_text(insights["period_return"]),
            "note": f"基于 {insights['trading_days']} 个交易日的首尾收盘价。",
        },
        {
            "label": "最大回撤",
            "value": pct_text(insights["max_drawdown"]),
            "note": "衡量区间内从阶段高点回落的最大幅度。",
        },
        {
            "label": "年化波动率",
            "value": pct_text(insights["volatility"]),
            "note": insights["risk_label"],
        },
        {
            "label": "量能状态",
            "value": insights["volume_label"],
            "note": (
                f"最新成交量约为20日均量的 {number_text(insights['volume_ratio'], 2)} 倍。"
                if insights.get("volume_ratio") is not None else "当前数据缺少成交量。"
            ),
        },
    ]
    render_insight_cards(cards)

    st.markdown(
        f"""
        <div class="callout">
            <strong>行情解读：</strong>
            当前趋势判断为 <strong>{insights["trend_label"]}</strong>，
            价格位置为 <strong>{insights["position_label"]}</strong>。
            距区间高点 {pct_text(insights["distance_to_high"])}，
            距区间低点 {pct_text(insights["distance_to_low"])}。
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_price_distribution(price_data: pd.DataFrame):
    if price_data.empty or len(price_data) < 5:
        return
    data = price_data.copy()
    data["ReturnPct"] = data["Close"].pct_change() * 100
    dist = data.dropna(subset=["ReturnPct"])
    if dist.empty:
        return
    fig = px.histogram(dist, x="ReturnPct", nbins=24, title="日收益分布")
    fig.update_layout(**CHART_LAYOUT, height=300, xaxis_title="日收益率（%）", yaxis_title="频次")
    st.plotly_chart(fig, width="stretch")


def render_news_brief(news_data: list):
    if not news_data:
        st.info("当前配置的数据源暂未返回该股票相关新闻。")
        return

    sources = {}
    keywords = {}
    for item in news_data:
        source = item.get("source") or item.get("publisher") or "Unknown"
        sources[source] = sources.get(source, 0) + 1
        title = str(item.get("title", ""))
        for token in ["融资", "基金", "ETF", "业绩", "AI", "科技", "通信", "机构", "成交", "市值"]:
            if token.lower() in title.lower():
                keywords[token] = keywords.get(token, 0) + 1

    top_source = max(sources.items(), key=lambda pair: pair[1])[0] if sources else "N/A"
    top_keywords = "、".join([key for key, _ in sorted(keywords.items(), key=lambda pair: pair[1], reverse=True)[:5]]) or "暂无明显关键词"
    render_insight_cards(
        [
            {"label": "新闻数量", "value": str(len(news_data)), "note": "当前数据源返回的相关新闻条数。"},
            {"label": "主要来源", "value": top_source, "note": f"覆盖 {len(sources)} 个新闻来源。"},
            {"label": "标题关键词", "value": top_keywords, "note": "按标题中的高频业务/市场词粗略提取。"},
        ]
    )


def show_basic_information(stock_info: dict, ticker: str):
    section_title("基本信息", "来自当前可用数据源的公司与估值摘要。")

    rows = [
        ("股票代码", stock_info.get("symbol") or stock_info.get("code") or ticker),
        ("公司名称", stock_display_name(stock_info, ticker)),
        ("总市值", stock_info.get("market_cap")),
        ("交易所", stock_info.get("market") or stock_info.get("primary_exchange") or stock_info.get("exchange")),
        ("上市日期", stock_info.get("ipo_date") or stock_info.get("list_date")),
        ("PE (TTM)", stock_info.get("pe_ratio")),
        ("PB (MRQ)", stock_info.get("pb_ratio")),
    ]

    info_df = pd.DataFrame(
        [{"字段": label, "值": format_optional(value, "暂无数据")} for label, value in rows]
    )
    st.table(info_df)

    industry_parts = []
    sector = stock_info.get("sector")
    industry = stock_info.get("industry")
    sub_industry = stock_info.get("sub_industry")
    if sector and sector != industry:
        industry_parts.append(("板块", sector))
    if industry:
        industry_parts.append(("行业", industry))
    if sub_industry and sub_industry != industry:
        industry_parts.append(("细分行业", sub_industry))

    if industry_parts:
        cols = st.columns(len(industry_parts))
        for col, (label, value) in zip(cols, industry_parts):
            col.markdown(f"**{label}**")
            col.write(value)
    else:
        st.caption("当前数据源暂未提供行业分类。")


def show_price_summary(price_data: pd.DataFrame):
    if price_data.empty:
        st.warning("暂无价格数据。")
        return

    latest = price_data.iloc[-1]
    previous = price_data.iloc[-2] if len(price_data) > 1 else latest
    day_change = latest["Close"] - previous["Close"]
    day_change_pct = day_change / previous["Close"] * 100 if previous["Close"] else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("最新收盘", f"{latest['Close']:.2f}", f"{day_change:+.2f} ({day_change_pct:+.2f}%)")
    c2.metric("最高价", f"{latest['High']:.2f}")
    c3.metric("最低价", f"{latest['Low']:.2f}")
    c4.metric("成交量", format_volume(latest.get("Volume", 0)))


def show_candlestick(ticker: str, price_data: pd.DataFrame, include_analysis: bool = True):
    if price_data.empty:
        st.warning("暂无图表数据。")
        return

    # 将日期转为字符串，配合 category x 轴使用，可自动去除非交易日（周末/节假日）空白
    date_strs = price_data["Date"].dt.strftime("%Y-%m-%d").tolist()
    xaxis_cfg = _category_axis(date_strs)

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
    section_title("K线与成交量", "横轴按真实交易日展示，周末和节假日不会留出空白。")
    fig.update_layout(
        **CHART_LAYOUT,
        title=f"{ticker} K线",
        xaxis_title="日期",
        yaxis_title="价格",
        height=560,
        xaxis=xaxis_cfg,
    )
    st.plotly_chart(fig, width="stretch")

    if "Volume" in price_data.columns:
        volume_data = price_data.copy()
        volume_data["DateStr"] = date_strs
        volume_data["VolumeWan"] = volume_data["Volume"] / 10000
        volume_fig = px.bar(volume_data, x="DateStr", y="VolumeWan", title="成交量")
        volume_fig.update_layout(
            **CHART_LAYOUT,
            height=260,
            xaxis_title="日期",
            yaxis_title="成交量（万股）",
            xaxis=xaxis_cfg,
        )
        st.plotly_chart(volume_fig, width="stretch")

    if include_analysis:
        section_title("价格行为摘要")
        render_market_brief(price_data)
        render_price_distribution(price_data)


def show_overview(ticker: str, stock_info: dict, price_data: pd.DataFrame, news_data: list):
    name = stock_display_name(stock_info, ticker)
    symbol = stock_info.get("symbol") or stock_info.get("code") or ticker
    section_title(f"{name} ({symbol})")

    description = stock_info.get("description") or "当前数据源暂未提供公司简介。"
    st.write(description)

    show_price_summary(price_data)
    section_title("市场快照", "用本地行情数据自动生成，不额外消耗 LLM 调用。")
    render_market_brief(price_data)
    show_candlestick(ticker, price_data, include_analysis=False)

    show_basic_information(stock_info, ticker)

    section_title("最新新闻")
    render_news_brief(news_data)
    for item in news_data:
        title = item.get("title", "Untitled")
        source = item.get("source") or item.get("publisher") or "Unknown"
        summary = item.get("summary") or ""
        with st.expander(f"{title} - {source}"):
            st.write(summary)
            link = item.get("link")
            if link:
                st.link_button("打开原文", link)


def show_technical(ticker: str, price_data: pd.DataFrame):
    if price_data.empty:
        st.warning("暂无技术分析数据。")
        return

    data = price_data.copy()
    data["MA5"] = data["Close"].rolling(5).mean()
    data["MA20"] = data["Close"].rolling(20).mean()
    data["MA60"] = data["Close"].rolling(60).mean()

    date_strs = data["Date"].dt.strftime("%Y-%m-%d").tolist()
    xaxis_cfg = _category_axis(date_strs)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=date_strs, y=data["Close"], name="Close"))
    fig.add_trace(go.Scatter(x=date_strs, y=data["MA5"], name="MA5"))
    fig.add_trace(go.Scatter(x=date_strs, y=data["MA20"], name="MA20"))
    fig.add_trace(go.Scatter(x=date_strs, y=data["MA60"], name="MA60"))
    section_title("技术分析", "均线图同样按交易日压缩横轴，避免非交易日空档。")
    fig.update_layout(
        **CHART_LAYOUT,
        title=f"{ticker} 收盘价与移动均线",
        xaxis_title="日期",
        yaxis_title="价格",
        height=560,
        xaxis=xaxis_cfg,
    )
    st.plotly_chart(fig, width="stretch")

    _render_technical_insights(price_data)

    section_title("MACD / RSI / 布林带", "补充动量、超买超卖和波动区间指标，辅助验证均线结论。")
    render_advanced_technical_charts(price_data)


def _render_technical_insights(price_data: pd.DataFrame):
    """渲染技术面解读卡片与技术结论文案。"""
    insights = compute_price_insights(price_data)
    if not insights:
        return

    section_title("技术面解读")
    render_insight_cards([
        {"label": "趋势状态", "value": insights["trend_label"],
         "note": f"MA5 {number_text(insights.get('ma5'))} / MA20 {number_text(insights.get('ma20'))}"},
        {"label": "价格位置", "value": insights["position_label"],
         "note": f"距高点 {pct_text(insights['distance_to_high'])}，距低点 {pct_text(insights['distance_to_low'])}"},
        {"label": "波动风险", "value": insights["risk_label"],
         "note": f"年化波动率 {pct_text(insights['volatility'])}"},
        {"label": "量价确认", "value": insights["volume_label"],
         "note": "放量上涨/放量下跌需要结合最新K线方向继续判断。"},
    ])

    ma_confirmed = insights.get("ma5") and insights.get("ma20") and insights["ma5"] >= insights["ma20"]
    trend_note = "短期均线仍在中期均线上方，动能相对占优。" if ma_confirmed else "短期均线未明显占优，趋势确认度一般。"
    st.markdown(
        f"""
        <div class="callout">
            <strong>技术结论：</strong>{trend_note}
            当前最大回撤 {pct_text(insights["max_drawdown"])}，
            更适合结合支撑/压力位观察突破或回踩确认。
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_comparison(tickers: list[str], period: str):
    section_title("股票对比")
    fig = go.Figure()

    all_date_strs = None
    for ticker in tickers:
        data = normalize_price_data(get_stock_price_data(ticker, period=period))
        if data.empty:
            st.warning(f"{ticker} 暂无可对比数据。")
            continue
        norm_close = data["Close"] / data["Close"].iloc[0] * 100
        date_strs = data["Date"].dt.strftime("%Y-%m-%d").tolist()
        fig.add_trace(go.Scatter(x=date_strs, y=norm_close, name=ticker))
        if all_date_strs is None or len(date_strs) > len(all_date_strs):
            all_date_strs = date_strs

    xaxis_cfg = _category_axis(all_date_strs or [])

    fig.update_layout(
        **CHART_LAYOUT,
        title="标准化价格对比（首日 = 100）",
        xaxis_title="日期",
        yaxis_title="Index",
        height=520,
        xaxis=xaxis_cfg,
    )
    st.plotly_chart(fig, width="stretch")


FINANCIAL_TABLE_NAMES = {
    "income_statement": "利润表",
    "balance_sheet": "资产负债表",
    "cash_flow": "现金流量表",
    "financial_ratios": "财务比率",
    "growth": "成长能力",
    "operation": "运营能力",
    "valuation_metrics": "估值指标",
}


def _collect_financial_tables(financial_data: dict) -> list:
    """从原始财务数据中提取可展示的 (key, DataFrame) 列表。"""
    available_tables = []
    for key, value in financial_data.items():
        if key.startswith("__"):
            continue
        if isinstance(value, pd.DataFrame) and not value.empty:
            available_tables.append((key, value))
        elif isinstance(value, dict) and value:
            available_tables.append((key, pd.DataFrame([value])))
    return available_tables


def _render_financial_table_picker(ticker: str, available_tables: list):
    """渲染财务表选择器、预览与 CSV 下载。"""
    table_lookup = {key: frame for key, frame in available_tables}
    table_labels = {
        key: FINANCIAL_TABLE_NAMES.get(key, key.replace("_", " ").title())
        for key, _ in available_tables
    }
    selected_key = st.selectbox(
        "选择财务表",
        options=list(table_lookup.keys()),
        format_func=lambda key: table_labels.get(key, key),
        key=f"{ticker}_financial_table_selector",
    )
    selected_frame = table_lookup[selected_key]

    rows_count, cols_count = selected_frame.shape
    table_cols = st.columns(3)
    table_cols[0].metric("当前表", table_labels[selected_key])
    table_cols[1].metric("行数", str(rows_count))
    table_cols[2].metric("字段数", str(cols_count))

    st.dataframe(selected_frame, width="stretch", hide_index=True)
    st.download_button(
        label=f"下载 {table_labels[selected_key]} CSV",
        data=selected_frame.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{ticker}_{selected_key}.csv",
        mime="text/csv",
        key=f"{ticker}_{selected_key}_financial_download",
    )


def show_financial_export(ticker: str):
    try:
        financial_data = get_financial_data(ticker)
        if not financial_data:
            st.info("当前数据源暂未提供财务数据。")
            return
        if financial_data.get("__metadata__", {}).get("is_simulated"):
            st.error(
                "财务数据使用了模拟数据：在线数据源获取失败，当前表格仅用于界面预览，不能作为投资分析依据。",
                icon="⚠️",
            )

        section_title("财务数据", "按数据源返回的报表、比率与估值字段分组展示。")
        render_financial_visuals(financial_data)

        available_tables = _collect_financial_tables(financial_data)
        if not available_tables:
            st.info("当前配置的数据源暂未返回可展示的财务表。")
            return

        section_title("财务数据覆盖")
        render_insight_cards([
            {"label": "可用数据表", "value": str(len(available_tables)),
             "note": "当前数据源可展示的财务数据分组数量。"},
            {"label": "导出状态", "value": "可导出", "note": "每个分组均支持 CSV 下载。"},
            {"label": "数据用途", "value": "估值/盈利/成长",
             "note": "适合与行情走势、AI Agent 结论交叉验证。"},
        ])

        _render_financial_table_picker(ticker, available_tables)
    except Exception as exc:
        st.warning(f"无法加载财务数据: {exc}")


def render_recommendation(recommendation: dict):
    action = format_optional(recommendation.get("investment_recommendation"), "暂无建议")
    description = format_optional(recommendation.get("recommendation_description"), "")
    confidence = format_optional(recommendation.get("recommendation_confidence"), "N/A")
    signal_strength = format_optional(recommendation.get("signal_strength"), "N/A")
    summary = first_non_empty(
        recommendation.get("analysis_summary"),
        recommendation.get("recommendation_details"),
        fallback="首席策略智能体未返回文字摘要。",
    )

    section_title("最终建议")
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
    c1.metric("建议", action)
    c2.metric("置信度", confidence)
    c3.metric("信号强度", signal_strength)
    c4.metric("模型置信度", format_score(recommendation.get("confidence_score")))

    metrics = recommendation.get("comprehensive_metrics") or {}
    if metrics:
        m1, m2, m3 = st.columns(3)
        m1.metric("综合评分", format_score(metrics.get("composite_score")))
        m2.metric("风险评分", format_score(metrics.get("risk_score")))
        m3.metric("风险等级", format_optional(metrics.get("risk_level")))

    section_title("分析摘要")
    st.write(summary)

    details = recommendation.get("recommendation_details")
    if details and details != summary:
        with st.expander("建议依据", expanded=False):
            st.write(details)

    c1, c2 = st.columns(2)
    with c1:
        section_title("仓位建议")
        st.write(format_optional(recommendation.get("position_suggestion"), "暂未返回仓位建议。"))
        section_title("适合投资者")
        st.write(format_optional(recommendation.get("suitable_investors"), "暂未返回适合投资者画像。"))
    with c2:
        section_title("重点监控指标")
        metrics_to_watch = recommendation.get("key_monitoring_metrics") or []
        if metrics_to_watch:
            for item in metrics_to_watch:
                st.markdown(f"- {item}")
        else:
            st.info("暂未返回监控指标。")

    risk_disclosure = recommendation.get("risk_disclosure")
    if risk_disclosure:
        with st.expander("风险提示", expanded=False):
            st.write(risk_disclosure)


def render_agent_status(status: dict):
    if not status:
        return
    section_title("智能体状态")

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
            "智能体": normalize_agent_name(agent_name),
            "状态": "✅ 成功" if success else "❌ 失败",
            "置信度": f"{item.get('confidence_score', 0):.0%}" if item.get('confidence_score') is not None else "N/A",
            "问题": item.get("error") or ("" if success else "分析未完成"),
            }
        )
    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch", hide_index=True)


def render_workflow_diagnostics(conflict_analysis: dict, workflow_metadata: dict = None, workflow_trace: list = None):
    if not conflict_analysis and not workflow_metadata and not workflow_trace:
        return

    section_title("工作流诊断", "展示 LangGraph 汇总前的共识方向、分歧和数据缺口。")
    if workflow_metadata:
        specialists = workflow_metadata.get("specialist_agents") or []
        execution_model = workflow_metadata.get("execution_model") or "unknown"
        topology = workflow_metadata.get("topology") or {}
        edge_count = len(topology.get("edges") or [])
        st.caption(
            f"执行模型：{execution_model}；专家节点：{len(specialists)}；"
            f"汇总节点：{workflow_metadata.get('join_node', 'N/A')} -> {workflow_metadata.get('final_node', 'N/A')}；"
            f"LangGraph 边数：{edge_count}"
        )
        if topology:
            with st.expander("LangGraph 拓扑", expanded=False):
                edge_rows = topology.get("edges") or []
                if edge_rows:
                    st.dataframe(pd.DataFrame(edge_rows), width="stretch", hide_index=True)
                limitations = topology.get("current_limitations") or []
                if limitations:
                    st.caption("当前边界：" + "；".join(limitations))
    if workflow_trace:
        with st.expander("节点执行轨迹", expanded=False):
            st.dataframe(pd.DataFrame(workflow_trace), width="stretch", hide_index=True)
    if not conflict_analysis:
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("共识方向", format_optional(conflict_analysis.get("consensus_direction")))
    c2.metric("平均评分", format_score(conflict_analysis.get("average_score")))
    c3.metric("存在分歧", "是" if conflict_analysis.get("has_conflicts") else "否")

    summary = conflict_analysis.get("conflict_summary")
    if summary:
        st.caption(summary)

    agent_scores = conflict_analysis.get("agent_scores") or {}
    if agent_scores:
        score_df = pd.DataFrame(
            [{"智能体": normalize_agent_name(name), "评分": score} for name, score in agent_scores.items()]
        )
        st.dataframe(score_df, width="stretch", hide_index=True)

    gaps = conflict_analysis.get("data_gaps") or []
    divergences = conflict_analysis.get("score_divergences") or []
    if gaps or divergences:
        with st.expander("数据缺口与评分分歧", expanded=False):
            for item in gaps:
                st.markdown(f"- 数据缺口: {item}")
            for item in divergences:
                st.markdown(f"- 评分分歧: {item}")


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

    section_title("智能体发现")
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

            st.metric("置信度", format_score(agent_result.get("confidence_score")))
            findings = agent_result.get("key_findings") or []
            if findings:
                st.markdown("**核心发现**")
                for finding in findings[:8]:
                    st.markdown(f"- {finding}")
            else:
                st.info("该智能体未返回核心发现。")

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
                with st.expander("结构化结果", expanded=False):
                    st.write(agent_result)


def show_agent_analysis(ticker: str, period: str):
    section_title("智能体分析", "并行运行多个专业 Agent，再由 Chief Strategy Agent 汇总最终建议。")

    render_insight_cards(
        [
            {"label": "基本面 Agent", "value": "财务 / 估值", "note": "检查盈利、成长、估值和财务健康度。"},
            {"label": "技术面 Agent", "value": "趋势 / 量价", "note": "分析均线、动量、成交量和价格位置。"},
            {"label": "行业宏观 Agent", "value": "产业 / 宏观", "note": "评估行业景气、政策和宏观环境。"},
            {"label": "首席策略 Agent", "value": "综合建议", "note": "汇总多 Agent 输出并给出最终建议。"},
        ]
    )

    use_memory = st.checkbox("使用智能体记忆", value=False)
    selected_labels = st.multiselect(
        "选择分析维度",
        options=list(AGENT_LABELS.values()),
        default=list(AGENT_LABELS.values()),
        help="只运行选中的专业 Agent，可减少等待时间和 LLM token 消耗。",
    )
    enabled_agents = tuple(normalize_agent_selection(selected_labels) or list(AGENT_LABELS.keys()))
    if not st.button("运行智能分析", type="primary"):
        st.info("点击按钮后会启动完整 LangGraph 工作流。")
        return

    with st.spinner("正在运行智能体工作流..."):
        try:
            coordinator = get_coordinator(use_memory, enabled_agents)
            result = coordinator.analyze(ticker, time_range=period)
        except Exception as exc:
            st.error(f"智能体工作流启动失败: {exc}")
            return

    _render_analysis_results(ticker, result)


def _render_analysis_results(ticker: str, result: dict):
    """渲染一次智能体工作流的完整结果（状态、建议、诊断、明细、追问）。"""
    elapsed = result.get("elapsed_time_seconds")
    if not result.get("success"):
        st.error(result.get("error") or "智能体工作流执行失败。")
    else:
        if isinstance(elapsed, (int, float)):
            st.success(f"智能体工作流完成，用时 {elapsed:.1f}s。")
        else:
            st.success("智能体工作流完成。")
        saved_path = save_analysis_result(result, ticker)
        if saved_path:
            st.caption(f"分析结果已保存：{saved_path}")

    recommendation = result.get("final_recommendation") or {}
    if recommendation:
        render_recommendation(recommendation)

    status = result.get("agent_execution_status") or {}
    render_agent_status(status)
    render_workflow_diagnostics(
        result.get("conflict_analysis") or {},
        result.get("workflow_metadata") or {},
        result.get("workflow_trace") or [],
    )
    render_agent_details(result.get("detailed_results") or {}, status)

    if result:
        section_title("报告导出")
        render_report_downloads(result, ticker)
        _render_followup_qa(result)


def _render_followup_qa(result: dict):
    """基于本次分析结果的对话式追问区块。"""
    section_title("对话式追问", "基于本次 AI 分析结果继续提问。")
    question = st.text_input("追问内容", placeholder="例如：为什么建议持有？最大的风险是什么？")
    if st.button("提交追问", disabled=not question.strip()):
        with st.spinner("正在基于本次分析回答..."):
            try:
                answer = answer_followup_question(result, question)
                st.session_state.setdefault("analysis_followups", []).append(
                    {"question": question, "answer": answer}
                )
            except Exception as exc:
                st.error(f"追问失败: {exc}")
    for item in reversed(st.session_state.get("analysis_followups", [])[-5:]):
        with st.expander(item["question"], expanded=False):
            st.write(item["answer"])


setup_page()

with st.sidebar:
    st.markdown("## 股票查询")
    favorites = load_favorites()
    quick_symbol = ""
    if favorites:
        quick_symbol = st.selectbox("自选股快捷切换", [""] + favorites, format_func=lambda item: item or "不使用自选")
    default_ticker = quick_symbol or "300502"
    ticker = st.text_input("股票代码", value=default_ticker).strip().upper()
    fav_cols = st.columns(2)
    if fav_cols[0].button("加入自选", disabled=not ticker):
        add_favorite(ticker)
        st.rerun()
    if fav_cols[1].button("移除自选", disabled=not ticker):
        remove_favorite(ticker)
        st.rerun()
    period_display = st.selectbox("周期", list(PERIOD_OPTIONS.keys()), index=2)
    period = PERIOD_OPTIONS[period_display]
    custom_date_range = st.checkbox("使用自定义日期范围", value=False)
    start_date = end_date = None
    if custom_date_range:
        date_cols = st.columns(2)
        start_date = date_cols[0].date_input("开始日期", value=date.today() - timedelta(days=180))
        end_date = date_cols[1].date_input("结束日期", value=date.today())
        period_display = f"{start_date} 至 {end_date}"
        period = "max"

    st.markdown("---")
    st.markdown("## 对比")
    comparison_enabled = st.checkbox("启用对比", value=False)
    compare_tickers = []
    if comparison_enabled:
        compare_tickers.append(st.text_input("对比标的 1", value="300308").strip().upper())
        compare_tickers.append(st.text_input("对比标的 2", value="NVDA").strip().upper())
        compare_tickers = [code for code in compare_tickers if code]

    st.markdown("---")
    st.markdown("## 视图")
    view_label = st.selectbox("查看模式", list(VIEW_OPTIONS.values()))
    view_mode = next(key for key, label in VIEW_OPTIONS.items() if label == view_label)

    st.markdown("---")
    if st.button("刷新数据缓存", help="清空本地缓存并重新拉取数据"):
        st.cache_data.clear()
        st.rerun()

    # ── 知识库 RAG 状态 ──────────────────────────────────
    st.markdown("---")
    st.markdown("## 知识库")
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
        st.success(f"索引就绪（{_vec_count} chunks）")
    else:
        st.warning("索引尚未构建")

    st.caption(f"文档：{_doc_count} 个文件")

    if st.button("构建 / 重建索引", help="向量化知识库文档（需调用 Embedding API，约1~3分钟）"):
        with st.spinner("正在构建 RAG 索引..."):
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
                st.success("知识库索引构建成功。")
                st.rerun()
            except Exception as _e:
                st.error(f"构建失败: {_e}")
    # ────────────────────────────────────────────────────


@st.cache_data(ttl=600, show_spinner=False)
def load_market_data(ticker: str, period: str, start_date=None, end_date=None):
    """缓存行情数据 10 分钟，避免切换视图时重复拉取（首次加载约 15~20s）。
    注意：先拉 stock_info（含市值/PE/PB），再拉大量K线，避免BaoStock session被大查询占满。
    """
    # 1. 先拉基本信息（_get_market_metrics 需要在 BaoStock 连接干净时调用）
    info = get_stock_info(ticker) or {}
    # 2. 再拉价格数据（可能涉及大量5分钟K线）
    price_df = normalize_price_data(get_stock_price_data(ticker, period=period))
    if start_date and end_date and not price_df.empty:
        start_ts = pd.to_datetime(start_date)
        end_ts = pd.to_datetime(end_date)
        price_df = price_df[(price_df["Date"] >= start_ts) & (price_df["Date"] <= end_ts)]
    # 3. 最后拉新闻
    news = get_stock_news(ticker) or []
    return price_df, info, news


if view_mode in {"Knowledge QA", "Watchlist", "History", "Screener", "Portfolio", "Alerts"}:
    render_app_header(ticker, {"symbol": ticker, "data_source": "local app state"}, period_display, VIEW_OPTIONS.get(view_mode, view_mode))
    if view_mode == "Knowledge QA":
        try:
            from hengline.streamlit.st_qa import show_qa_view

            show_qa_view()
        except Exception as exc:
            st.error(f"知识库问答加载失败: {exc}")
    elif view_mode == "Watchlist":
        selected_watch = render_watchlist_page(ticker)
        if selected_watch and selected_watch != ticker:
            st.info(f"已选择 {selected_watch}。请在侧边栏股票代码中切换后查看行情。")
    elif view_mode == "History":
        render_history_page()
    elif view_mode == "Screener":
        render_screener_page(get_stock_info, get_stock_price_data)
    elif view_mode == "Portfolio":
        render_portfolio_page(get_stock_info, get_stock_price_data)
    elif view_mode == "Alerts":
        render_alerts_page(get_stock_price_data)
else:
    with st.spinner("Loading data... (first load may take ~15 s while fetching market metrics)"):
        try:
            price_data, stock_info, news_data = load_market_data(ticker, period, start_date, end_date)
            render_app_header(ticker, stock_info, period_display, VIEW_OPTIONS.get(view_mode, view_mode))
            render_data_quality_warning(stock_info, price_data)

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
            elif view_mode == "Backtest":
                show_price_summary(price_data)
                render_backtest_page(ticker, price_data)

            if comparison_enabled and compare_tickers:
                st.markdown("---")
                show_comparison([ticker] + compare_tickers, period)

            st.markdown("---")
            section_title("数据导出")
            if not price_data.empty:
                st.download_button(
                    label="下载价格数据（CSV）",
                    data=price_data.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"{ticker}_price_data.csv",
                    mime="text/csv",
                )
            else:
                st.info("当前没有可导出的价格数据。")
        except Exception as exc:
            st.error(f"数据加载失败: {exc}")
            st.info("可以尝试 300502、300308、NVDA 或 AAPL。")
