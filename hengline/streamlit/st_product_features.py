#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Product-level Streamlit features for the stock dashboard."""

from __future__ import annotations

import json
import os
import html
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
USER_DATA_DIR = DATA_DIR / "user"
OUTPUT_DIR = DATA_DIR / "output"
FAVORITES_FILE = USER_DATA_DIR / "favorites.json"
PORTFOLIO_FILE = USER_DATA_DIR / "portfolio.json"
ALERTS_FILE = USER_DATA_DIR / "alerts.json"


AGENT_LABELS = {
    "FundamentalAgent": "基本面",
    "TechnicalAgent": "技术面",
    "IndustryMacroAgent": "行业宏观",
    "SentimentAgent": "情绪面",
    "FundFlowAgent": "资金流",
    "ESGRiskAgent": "ESG 风险",
}


def ensure_user_data_dir() -> None:
    USER_DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_favorites() -> List[str]:
    ensure_user_data_dir()
    if not FAVORITES_FILE.exists():
        return []
    try:
        data = json.loads(FAVORITES_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, list):
        return []
    return [str(item).strip().upper() for item in data if str(item).strip()]


def save_favorites(favorites: Iterable[str]) -> None:
    ensure_user_data_dir()
    cleaned = []
    for item in favorites:
        symbol = str(item).strip().upper()
        if symbol and symbol not in cleaned:
            cleaned.append(symbol)
    FAVORITES_FILE.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json_list(path: Path) -> List[Dict[str, Any]]:
    ensure_user_data_dir()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []


def save_json_list(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_user_data_dir()
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def add_favorite(symbol: str) -> List[str]:
    favorites = load_favorites()
    symbol = symbol.strip().upper()
    if symbol and symbol not in favorites:
        favorites.append(symbol)
        save_favorites(favorites)
    return favorites


def remove_favorite(symbol: str) -> List[str]:
    symbol = symbol.strip().upper()
    favorites = [item for item in load_favorites() if item != symbol]
    save_favorites(favorites)
    return favorites


def normalize_agent_selection(selected_labels: List[str]) -> List[str]:
    reverse = {label: name for name, label in AGENT_LABELS.items()}
    return [reverse[label] for label in selected_labels if label in reverse]


def calculate_technical_indicators(price_data: pd.DataFrame) -> pd.DataFrame:
    data = price_data.copy()
    if data.empty:
        return data
    data["Date"] = pd.to_datetime(data["Date"])
    close = pd.to_numeric(data["Close"], errors="coerce")
    high = pd.to_numeric(data["High"], errors="coerce")
    low = pd.to_numeric(data["Low"], errors="coerce")

    data["MA5"] = close.rolling(5).mean()
    data["MA20"] = close.rolling(20).mean()
    data["MA60"] = close.rolling(60).mean()

    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, pd.NA)
    data["RSI14"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    data["MACD"] = ema12 - ema26
    data["MACD_SIGNAL"] = data["MACD"].ewm(span=9, adjust=False).mean()
    data["MACD_HIST"] = data["MACD"] - data["MACD_SIGNAL"]

    middle = close.rolling(20).mean()
    std = close.rolling(20).std()
    data["BOLL_MID"] = middle
    data["BOLL_UPPER"] = middle + 2 * std
    data["BOLL_LOWER"] = middle - 2 * std
    data["HIGH"] = high
    data["LOW"] = low
    return data


def _category_dates(data: pd.DataFrame) -> List[str]:
    return pd.to_datetime(data["Date"]).dt.strftime("%Y-%m-%d").tolist()


def render_advanced_technical_charts(price_data: pd.DataFrame) -> None:
    if price_data.empty or len(price_data) < 26:
        st.info("数据点不足，至少需要 26 个交易日才能展示 MACD / RSI / 布林带。")
        return

    data = calculate_technical_indicators(price_data)
    date_strs = _category_dates(data)

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
        row_heights=[0.42, 0.18, 0.2, 0.2],
        subplot_titles=("价格与布林带", "成交量", "RSI(14)", "MACD"),
    )

    fig.add_trace(go.Scatter(x=date_strs, y=data["Close"], name="收盘价", line=dict(color="#0f766e")), row=1, col=1)
    fig.add_trace(go.Scatter(x=date_strs, y=data["BOLL_UPPER"], name="BOLL上轨", line=dict(color="#64748b", dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=date_strs, y=data["BOLL_MID"], name="BOLL中轨", line=dict(color="#94a3b8")), row=1, col=1)
    fig.add_trace(go.Scatter(x=date_strs, y=data["BOLL_LOWER"], name="BOLL下轨", line=dict(color="#64748b", dash="dot")), row=1, col=1)

    if "Volume" in data.columns:
        fig.add_trace(go.Bar(x=date_strs, y=data["Volume"] / 10000, name="成交量(万股)", marker_color="#2563eb"), row=2, col=1)

    fig.add_trace(go.Scatter(x=date_strs, y=data["RSI14"], name="RSI14", line=dict(color="#7c3aed")), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#dc2626", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#16a34a", row=3, col=1)

    hist_colors = ["#16a34a" if value >= 0 else "#dc2626" for value in data["MACD_HIST"].fillna(0)]
    fig.add_trace(go.Bar(x=date_strs, y=data["MACD_HIST"], name="MACD柱", marker_color=hist_colors), row=4, col=1)
    fig.add_trace(go.Scatter(x=date_strs, y=data["MACD"], name="DIF", line=dict(color="#0f172a")), row=4, col=1)
    fig.add_trace(go.Scatter(x=date_strs, y=data["MACD_SIGNAL"], name="DEA", line=dict(color="#f59e0b")), row=4, col=1)

    step = max(1, len(date_strs) // 12)
    fig.update_xaxes(type="category", tickangle=45, tickmode="array", tickvals=date_strs[::step])
    fig.update_layout(
        template="plotly_white",
        height=880,
        hovermode="x unified",
        margin=dict(l=48, r=24, t=72, b=48),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, width="stretch")


def _numeric_columns(df: pd.DataFrame) -> List[str]:
    columns = []
    for column in df.columns:
        converted = pd.to_numeric(df[column], errors="coerce")
        if converted.notna().any():
            columns.append(column)
    return columns


def render_financial_visuals(financial_data: Dict[str, Any]) -> None:
    available_frames = {
        key: value for key, value in financial_data.items()
        if isinstance(value, pd.DataFrame) and not value.empty and not key.startswith("__")
    }
    if not available_frames:
        st.info("当前财务数据不足，暂无法生成趋势图或雷达图。")
        return

    income_df = available_frames.get("income_statement")
    if income_df is not None:
        numeric_cols = _numeric_columns(income_df)
        preferred = [
            col for col in ["operatingRevenue", "totalRevenue", "netProfit", "netIncome", "grossProfit"]
            if col in numeric_cols
        ]
        selected = preferred[:3] or numeric_cols[:3]
        if selected:
            fig = go.Figure()
            x_values = income_df.index.astype(str).tolist()
            for column in selected:
                fig.add_trace(go.Scatter(
                    x=x_values,
                    y=pd.to_numeric(income_df[column], errors="coerce"),
                    mode="lines+markers",
                    name=column,
                ))
            fig.update_layout(
                template="plotly_white",
                title="利润表关键指标趋势",
                xaxis_title="报告期",
                yaxis_title="数值",
                height=420,
                hovermode="x unified",
            )
            st.plotly_chart(fig, width="stretch")

    ratios = {}
    for key in ("financial_ratios", "valuation_metrics"):
        item = financial_data.get(key)
        if isinstance(item, dict):
            ratios.update(item)
        elif isinstance(item, pd.DataFrame) and not item.empty:
            ratios.update(item.iloc[-1].to_dict())

    radar_items = {}
    for key, value in ratios.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if pd.notna(number) and number > 0:
            radar_items[key] = min(number, 100.0)
        if len(radar_items) >= 8:
            break

    if len(radar_items) >= 3:
        categories = list(radar_items.keys())
        values = list(radar_items.values())
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=values + values[:1], theta=categories + categories[:1], fill="toself", name="指标"))
        fig.update_layout(
            template="plotly_white",
            title="财务/估值指标雷达图",
            polar=dict(radialaxis=dict(visible=True)),
            height=440,
            margin=dict(l=32, r=32, t=56, b=32),
        )
        st.plotly_chart(fig, width="stretch")


def save_analysis_result(result: Dict[str, Any], stock_code: str) -> Optional[Path]:
    if not result:
        return None
    stock_code = str(stock_code).strip().upper() or "UNKNOWN"
    output_dir = OUTPUT_DIR / stock_code
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"analysis_{stock_code}_{timestamp}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


def load_analysis_history(limit: int = 200) -> List[Dict[str, Any]]:
    if not OUTPUT_DIR.exists():
        return []
    records = []
    for file_path in OUTPUT_DIR.glob("*/*.json"):
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        records.append({
            "stock_code": data.get("stock_code") or file_path.parent.name,
            "analysis_time": data.get("analysis_time") or data.get("analysis_start_time") or "",
            "success": data.get("success", False),
            "recommendation": (data.get("final_recommendation") or {}).get("investment_recommendation", ""),
            "risk_level": ((data.get("final_recommendation") or {}).get("comprehensive_metrics") or {}).get("risk_level", ""),
            "file_path": str(file_path),
            "data": data,
            "mtime": file_path.stat().st_mtime,
        })
    records.sort(key=lambda item: item["mtime"], reverse=True)
    return records[:limit]


def build_markdown_report(result: Dict[str, Any]) -> str:
    recommendation = result.get("final_recommendation") or {}
    metrics = recommendation.get("comprehensive_metrics") or {}
    lines = [
        f"# 股票 AI 分析报告 - {result.get('stock_code', 'UNKNOWN')}",
        "",
        f"- 生成时间：{result.get('analysis_time') or result.get('analysis_start_time') or ''}",
        f"- 执行状态：{'成功' if result.get('success') else '失败'}",
        f"- 最终建议：{recommendation.get('investment_recommendation', '暂无')}",
        f"- 综合评分：{metrics.get('composite_score', 'N/A')}",
        f"- 风险等级：{metrics.get('risk_level', 'N/A')}",
        "",
        "## 分析摘要",
        recommendation.get("analysis_summary") or recommendation.get("recommendation_details") or "暂无摘要。",
        "",
        "## 核心发现",
    ]
    for finding in recommendation.get("key_findings") or []:
        lines.append(f"- {finding}")

    details = result.get("detailed_results") or {}
    if details:
        lines.extend(["", "## 各 Agent 结论"])
        for agent_name, agent_result in details.items():
            lines.append(f"### {AGENT_LABELS.get(agent_name, agent_name)}")
            for finding in agent_result.get("key_findings") or []:
                lines.append(f"- {finding}")

    conflict = result.get("conflict_analysis") or {}
    if conflict:
        lines.extend([
            "",
            "## 工作流诊断",
            f"- 共识方向：{conflict.get('consensus_direction', 'N/A')}",
            f"- 平均评分：{conflict.get('average_score', 'N/A')}",
            f"- 数据缺口：{len(conflict.get('data_gaps') or [])}",
        ])
    return "\n".join(lines).strip() + "\n"


def build_html_report(result: Dict[str, Any]) -> str:
    markdown = build_markdown_report(result)
    body_lines = []
    for line in markdown.splitlines():
        escaped = html.escape(line)
        if line.startswith("# "):
            body_lines.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            body_lines.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("### "):
            body_lines.append(f"<h3>{html.escape(line[4:])}</h3>")
        elif line.startswith("- "):
            body_lines.append(f"<li>{html.escape(line[2:])}</li>")
        elif line.strip():
            body_lines.append(f"<p>{escaped}</p>")
        else:
            body_lines.append("")
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Stock AI Report</title>"
        "<style>body{font-family:Arial,'Microsoft YaHei',sans-serif;max-width:920px;margin:40px auto;"
        "line-height:1.65;color:#111827}h1,h2,h3{color:#073b74}li{margin:.35rem 0}"
        "@media print{body{margin:20px}}</style></head><body>"
        + "\n".join(body_lines)
        + "</body></html>"
    )


def render_report_downloads(result: Dict[str, Any], stock_code: str) -> None:
    if not result:
        return
    markdown = build_markdown_report(result)
    st.download_button(
        "下载 AI 分析报告（Markdown）",
        data=markdown.encode("utf-8-sig"),
        file_name=f"{stock_code}_ai_report.md",
        mime="text/markdown",
    )
    st.download_button(
        "下载 AI 分析报告（HTML，可打印 PDF）",
        data=build_html_report(result).encode("utf-8"),
        file_name=f"{stock_code}_ai_report.html",
        mime="text/html",
    )
    st.download_button(
        "下载 AI 分析结果（JSON）",
        data=json.dumps(result, ensure_ascii=False, indent=2, default=str).encode("utf-8"),
        file_name=f"{stock_code}_ai_result.json",
        mime="application/json",
    )


def answer_followup_question(result: Dict[str, Any], question: str) -> str:
    from config.config import get_ai_config
    from hengline.client.client_factory import get_langchain_llm

    if not question.strip():
        return "请输入追问内容。"
    context = json.dumps(result, ensure_ascii=False, default=str)[:12000]
    config = get_ai_config()
    llm = get_langchain_llm(provider=config.get("provider"), config=config)
    if llm is None:
        return "当前 LLM 未初始化，无法追问。"
    messages = [
        {
            "role": "system",
            "content": "你是股票投研助手。请只基于给定分析结果回答用户追问，说明依据和不确定性，不构成投资建议。",
        },
        {
            "role": "user",
            "content": f"分析结果JSON：\n{context}\n\n用户追问：{question}",
        },
    ]
    response = llm.invoke(messages)
    return getattr(response, "content", str(response))


def render_history_page() -> None:
    st.markdown("### 历史分析")
    records = load_analysis_history()
    if not records:
        st.info("暂无历史分析记录。运行一次智能体分析后，这里会自动显示保存的 JSON 结果。")
        return

    stock_filter = st.text_input("按股票代码筛选", value="").strip().upper()
    filtered = [record for record in records if not stock_filter or stock_filter in record["stock_code"].upper()]
    if not filtered:
        st.info("没有匹配的历史记录。")
        return

    table = pd.DataFrame([
        {
            "股票": item["stock_code"],
            "时间": item["analysis_time"],
            "状态": "成功" if item["success"] else "失败",
            "建议": item["recommendation"] or "N/A",
            "风险": item["risk_level"] or "N/A",
        }
        for item in filtered
    ])
    st.dataframe(table, width="stretch", hide_index=True)

    labels = [f"{item['stock_code']} | {item['analysis_time']} | {item['recommendation'] or 'N/A'}" for item in filtered]
    selected_label = st.selectbox("查看历史详情", labels)
    selected = filtered[labels.index(selected_label)]
    st.json(selected["data"], expanded=False)
    render_report_downloads(selected["data"], selected["stock_code"])


def render_watchlist_page(current_ticker: str) -> Optional[str]:
    st.markdown("### 自选股")
    favorites = load_favorites()
    col1, col2 = st.columns([2, 1])
    with col1:
        new_symbol = st.text_input("添加股票代码", value=current_ticker.strip().upper())
    with col2:
        st.write("")
        st.write("")
        if st.button("加入自选", type="primary"):
            favorites = add_favorite(new_symbol)
            st.success(f"已加入 {new_symbol.strip().upper()}")

    if not favorites:
        st.info("自选股列表为空。")
        return None

    selected = st.radio("点击选择自选股", favorites, horizontal=True)
    if st.button("从自选中移除"):
        remove_favorite(selected)
        st.rerun()

    st.caption(f"自选股文件：{FAVORITES_FILE}")
    return selected


def render_screener_page(get_stock_info_func, get_stock_price_data_func) -> None:
    st.markdown("### 股票筛选器")
    st.caption("输入候选股票池后，按估值、区间涨跌和趋势条件筛选。")

    default_pool = ",".join(load_favorites() or ["300502", "300308", "NVDA", "AAPL"])
    raw_codes = st.text_area("候选股票代码（逗号或换行分隔）", value=default_pool, height=90)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        max_pe = st.number_input("最大 PE", min_value=0.0, value=120.0, step=5.0)
    with col2:
        max_pb = st.number_input("最大 PB", min_value=0.0, value=80.0, step=2.0)
    with col3:
        min_return = st.number_input("最小区间涨跌幅(%)", value=-100.0, step=5.0)
    with col4:
        require_uptrend = st.checkbox("要求收盘价高于 MA20", value=False)

    if not st.button("运行筛选", type="primary"):
        return

    codes = []
    for token in raw_codes.replace("\n", ",").split(","):
        code = token.strip().upper()
        if code and code not in codes:
            codes.append(code)

    rows = []
    progress = st.progress(0, text="正在筛选股票...")
    for idx, code in enumerate(codes, 1):
        try:
            info = get_stock_info_func(code) or {}
            price = get_stock_price_data_func(code, period="3m")
            if price is None or price.empty:
                continue
            price = price.copy()
            price["Date"] = pd.to_datetime(price["Date"])
            close = pd.to_numeric(price["Close"], errors="coerce")
            latest_close = float(close.iloc[-1])
            period_return = (latest_close / float(close.iloc[0]) - 1) * 100 if float(close.iloc[0]) else 0.0
            ma20 = close.rolling(20).mean().iloc[-1] if len(close) >= 20 else None
            pe = _safe_float(info.get("pe_ratio"))
            pb = _safe_float(info.get("pb_ratio"))
            uptrend_ok = not require_uptrend or (ma20 is not None and pd.notna(ma20) and latest_close >= float(ma20))
            if pe <= max_pe and pb <= max_pb and period_return >= min_return and uptrend_ok:
                rows.append({
                    "股票": code,
                    "名称": info.get("name") or info.get("company_name") or code,
                    "最新价": round(latest_close, 2),
                    "区间涨跌幅%": round(period_return, 2),
                    "PE": round(pe, 2),
                    "PB": round(pb, 2),
                    "趋势": "强于MA20" if ma20 is not None and pd.notna(ma20) and latest_close >= float(ma20) else "弱于MA20",
                })
        except Exception as exc:
            rows.append({"股票": code, "名称": "筛选失败", "最新价": None, "区间涨跌幅%": None, "PE": None, "PB": None, "趋势": str(exc)[:80]})
        progress.progress(idx / max(len(codes), 1), text=f"已处理 {idx}/{len(codes)}")
    progress.empty()

    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    else:
        st.info("没有股票满足当前筛选条件。")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        if pd.isna(number):
            return default
        return number
    except (TypeError, ValueError):
        return default


def render_backtest_page(ticker: str, price_data: pd.DataFrame) -> None:
    st.markdown("### 简化回测")
    st.caption("当前版本提供均线交叉策略回测，用来快速验证技术信号是否具备历史解释力。")
    if price_data.empty or len(price_data) < 60:
        st.info("回测至少需要 60 个交易日数据，请切换到 3M/6M/1Y 或更长周期。")
        return

    fast = st.slider("短均线", min_value=3, max_value=30, value=10)
    slow = st.slider("长均线", min_value=20, max_value=120, value=30)
    if fast >= slow:
        st.warning("短均线周期必须小于长均线周期。")
        return

    data = price_data.copy().sort_values("Date")
    close = pd.to_numeric(data["Close"], errors="coerce")
    data["fast_ma"] = close.rolling(fast).mean()
    data["slow_ma"] = close.rolling(slow).mean()
    data["signal"] = (data["fast_ma"] > data["slow_ma"]).astype(int)
    data["strategy_return"] = data["signal"].shift(1).fillna(0) * close.pct_change().fillna(0)
    data["buy_hold_curve"] = (1 + close.pct_change().fillna(0)).cumprod()
    data["strategy_curve"] = (1 + data["strategy_return"]).cumprod()

    total_strategy = (float(data["strategy_curve"].iloc[-1]) - 1) * 100
    total_hold = (float(data["buy_hold_curve"].iloc[-1]) - 1) * 100
    trades = int((data["signal"].diff().abs() == 1).sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("策略收益", f"{total_strategy:+.2f}%")
    c2.metric("买入持有", f"{total_hold:+.2f}%")
    c3.metric("交易次数", str(trades))

    date_strs = pd.to_datetime(data["Date"]).dt.strftime("%Y-%m-%d").tolist()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=date_strs, y=data["strategy_curve"], name="均线策略"))
    fig.add_trace(go.Scatter(x=date_strs, y=data["buy_hold_curve"], name="买入持有"))
    fig.update_layout(template="plotly_white", title=f"{ticker} 回测收益曲线", height=480, hovermode="x unified")
    st.plotly_chart(fig, width="stretch")


def render_portfolio_page(get_stock_info_func, get_stock_price_data_func) -> None:
    st.markdown("### 投资组合管理")
    holdings = load_json_list(PORTFOLIO_FILE)

    with st.form("portfolio_add_form"):
        cols = st.columns(4)
        symbol = cols[0].text_input("股票代码").strip().upper()
        quantity = cols[1].number_input("持仓数量", min_value=0.0, value=0.0, step=100.0)
        cost = cols[2].number_input("成本价", min_value=0.0, value=0.0, step=1.0)
        note = cols[3].text_input("备注")
        submitted = st.form_submit_button("添加/更新持仓")
    if submitted and symbol:
        holdings = [row for row in holdings if row.get("symbol") != symbol]
        holdings.append({"symbol": symbol, "quantity": quantity, "cost": cost, "note": note})
        save_json_list(PORTFOLIO_FILE, holdings)
        st.rerun()

    if not holdings:
        st.info("暂无持仓。")
        return

    rows = []
    for row in holdings:
        symbol = str(row.get("symbol", "")).upper()
        try:
            info = get_stock_info_func(symbol) or {}
            price = get_stock_price_data_func(symbol, period="1m")
            latest = _safe_float(price["Close"].iloc[-1]) if price is not None and not price.empty else 0.0
        except Exception:
            info, latest = {}, 0.0
        quantity = _safe_float(row.get("quantity"))
        cost = _safe_float(row.get("cost"))
        market_value = latest * quantity
        pnl = (latest - cost) * quantity if cost else 0.0
        rows.append({
            "股票": symbol,
            "名称": info.get("name") or info.get("company_name") or symbol,
            "数量": quantity,
            "成本价": cost,
            "最新价": round(latest, 2),
            "市值": round(market_value, 2),
            "浮动盈亏": round(pnl, 2),
            "行业": info.get("industry") or info.get("sector") or "N/A",
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch", hide_index=True)
    st.metric("组合总市值", f"{df['市值'].sum():,.2f}")
    st.metric("组合浮动盈亏", f"{df['浮动盈亏'].sum():+,.2f}")

    remove_symbol = st.selectbox("删除持仓", [""] + [row["股票"] for row in rows])
    if remove_symbol and st.button("确认删除"):
        save_json_list(PORTFOLIO_FILE, [row for row in holdings if row.get("symbol") != remove_symbol])
        st.rerun()


def render_alerts_page(get_stock_price_data_func) -> None:
    st.markdown("### 预警配置")
    alerts = load_json_list(ALERTS_FILE)

    with st.form("alert_add_form"):
        cols = st.columns(4)
        symbol = cols[0].text_input("股票代码").strip().upper()
        above = cols[1].number_input("高于价格", min_value=0.0, value=0.0, step=1.0)
        below = cols[2].number_input("低于价格", min_value=0.0, value=0.0, step=1.0)
        enabled = cols[3].checkbox("启用", value=True)
        submitted = st.form_submit_button("保存预警")
    if submitted and symbol:
        alerts = [row for row in alerts if row.get("symbol") != symbol]
        alerts.append({"symbol": symbol, "above": above, "below": below, "enabled": enabled})
        save_json_list(ALERTS_FILE, alerts)
        st.rerun()

    if not alerts:
        st.info("暂无预警。")
        return

    rows = check_alerts(get_stock_price_data_func, alerts)
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption(f"预警配置文件：{ALERTS_FILE}。可用 `python main.py alerts-check` 进行命令行检查，并交给 Windows 任务计划或 cron 定时执行。")


def check_alerts(get_stock_price_data_func, alerts: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    alerts = load_json_list(ALERTS_FILE) if alerts is None else alerts
    rows = []
    for alert in alerts:
        symbol = str(alert.get("symbol", "")).upper()
        latest = 0.0
        try:
            price = get_stock_price_data_func(symbol, period="1m")
            latest = _safe_float(price["Close"].iloc[-1]) if price is not None and not price.empty else 0.0
        except Exception:
            pass
        above = _safe_float(alert.get("above"))
        below = _safe_float(alert.get("below"))
        triggered = bool(alert.get("enabled")) and ((above and latest >= above) or (below and latest <= below))
        rows.append({
            "股票": symbol,
            "最新价": round(latest, 2),
            "高于": above or "",
            "低于": below or "",
            "启用": bool(alert.get("enabled")),
            "状态": "触发" if triggered else "未触发",
        })
    return rows
