"""
itda GNN 사기 탐지 대시보드 — Streamlit Cloud 배포용
torch/torch_geometric 없이 사전 저장된 추론 결과 사용
"""

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from pathlib import Path

BASE    = Path(__file__).parent
DATA    = BASE / "data"
REPORTS = BASE / "reports"

C = dict(
    bg       = "#0E1117",
    surface  = "#1C2333",
    surface2 = "#242B3D",
    teal     = "#00C9A7",
    blue     = "#4F9CF9",
    amber    = "#F0A500",
    red      = "#FF4B4B",
    green    = "#21C55D",
    purple   = "#A855F7",
    text     = "#E2E8F0",
    muted    = "#8892A4",
)

CHART_TEMPLATE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    font         =dict(family="Noto Sans KR, sans-serif", color=C["text"]),
    margin       =dict(l=10, r=10, t=40, b=10),
    hoverlabel   =dict(bgcolor=C["surface2"], font_color=C["text"]),
)

st.set_page_config(
    page_title="itda GNN | 사기 탐지 시스템",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;600;700&display=swap');
  html, body, [class*="css"] {{ font-family: 'Noto Sans KR', sans-serif; }}
  .main {{ background:{C['bg']}; }}
  .kpi-card {{
      background: linear-gradient(135deg, {C['surface']}, {C['surface2']});
      border-radius: 12px; padding: 18px 20px;
      border-left: 4px solid; margin-bottom: 8px;
  }}
  .kpi-val  {{ font-size: 28px; font-weight: 700; margin: 4px 0; }}
  .kpi-lbl  {{ font-size: 12px; color: {C['muted']}; text-transform: uppercase; letter-spacing:.8px; }}
  .kpi-sub  {{ font-size: 12px; margin-top: 2px; }}
  .sec-title {{
      font-size: 18px; font-weight: 700; color: {C['text']};
      border-bottom: 2px solid {C['teal']};
      padding-bottom: 6px; margin-bottom: 14px;
  }}
  .review-box {{
      background:{C['surface2']}; border-radius:8px; padding:14px 16px;
      border-left:3px solid {C['blue']}; font-size:14px; line-height:1.7; color:{C['text']};
  }}
  section[data-testid="stSidebar"] > div:first-child {{
      background: linear-gradient(180deg, #141926 0%, #0E1117 100%);
  }}
  .stTabs [aria-selected="true"] {{
      color:{C['teal']} !important;
      border-bottom-color:{C['teal']} !important;
  }}
  .badge-amber {{ background:#f0a50022; color:{C['amber']}; border:1px solid {C['amber']}44;
                  display:inline-block; padding:2px 10px; border-radius:99px; font-size:11px; }}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    probs     = np.load(DATA / "all_probs.npy")
    labels    = np.load(DATA / "all_labels.npy")
    test_mask = np.load(DATA / "test_mask.npy")
    burst_ei  = np.load(DATA / "burst_ei.npy")
    burst_dt  = np.load(DATA / "burst_dt.npy")
    sim_ei    = np.load(DATA / "sim_ei.npy")
    rur_ei    = np.load(DATA / "rur_ei.npy")
    df        = pd.read_parquet(DATA / "df_sampled.parquet")
    return probs, labels, test_mask, burst_ei, burst_dt, sim_ei, rur_ei, df

@st.cache_data
def load_results():
    log     = pd.read_csv(DATA / "experiment_log.csv")
    log_ind = pd.read_csv(DATA / "experiment_log_inductive.csv")
    camps   = pd.read_csv(DATA / "fraud_campaigns.csv")
    if "actual_spam_ratio" in camps.columns:
        camps = camps.rename(columns={"actual_spam_ratio": "spam_ratio"})
    if "detected" in camps.columns:
        camps = camps.rename(columns={"detected": "is_campaign"})
    attr = pd.read_csv(DATA / "xai_edge_attribution.csv")
    return log, log_ind, camps, attr

def load_json(name):
    p = DATA / name
    return json.load(open(p, encoding="utf-8")) if p.exists() else {}


# 사이드바
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center; padding:16px 0 8px'>
      <div style='font-size:32px'>🛡️</div>
      <div style='font-size:16px; font-weight:700; color:{C["teal"]}'>itda GNN</div>
      <div style='font-size:11px; color:{C["muted"]}; margin-top:4px'>사기 리뷰 탐지 시스템</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    page = st.radio("내비게이션", [
        "🏠  개요", "🎯  실시간 탐지", "🕸  캠페인 분석", "📊  모델 비교", "🔬  XAI 분석"
    ], label_visibility="collapsed")
    st.divider()
    st.markdown(f"<div style='font-size:12px;color:{C['muted']};'>최종 성능 (3-way Ensemble)</div>", unsafe_allow_html=True)
    st.metric("PR-AUC",   "0.9419", delta="+0.077 vs baseline")
    st.metric("Macro F1", "0.9386", delta="+0.099 vs baseline")
    st.metric("인덕티브", "0.7748", delta="4-way Ensemble")
    st.divider()
    st.caption("YelpZip · 30K 노드 · 963K 엣지")


probs, labels, test_mask, burst_ei, burst_dt, sim_ei, rur_ei, df = load_data()
log, log_ind, camps, attr = load_results()
eda   = load_json("eda_summary.json")
csum  = load_json("campaign_summary.json")
kfold = load_json("inductive_kfold_result.json")
e4    = load_json("ensemble_4way_result.json")


# ─────────────────────────────────────────────────────────────────────────────
# 🏠 개요
# ─────────────────────────────────────────────────────────────────────────────
if "개요" in page:
    st.markdown(f"""
    <div style='padding:24px 0 16px'>
      <h1 style='font-size:28px; font-weight:800; margin:0;
                 background:linear-gradient(90deg,{C["teal"]},{C["blue"]});
                 -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
        헤테로 시간 인식 GNN 기반 조직적 리뷰 어뷰징 탐지
      </h1>
      <p style='color:{C["muted"]}; margin-top:6px; font-size:13px;'>
        YelpZip 608K 리뷰 · 5종 도메인 특화 엣지 · Bochner 시간 인코딩 · DRAGWave 융합 아키텍처
      </p>
    </div>
    """, unsafe_allow_html=True)

    k1, k2, k3, k4, k5 = st.columns(5)
    kpis = [
        (k1, "PR-AUC",    "0.9419", "+9.4%p",    C["teal"],   "3-way Ensemble"),
        (k2, "Macro F1",  "0.9386", "+9.9%p",    C["blue"],   "3-way Ensemble"),
        (k3, "인덕티브",  "0.7748", "+18.7%p",   C["purple"], "4-way Ensemble"),
        (k4, "캠페인 탐지", f"{csum.get('suspected_campaigns',91)}개", "91개 클러스터", C["amber"], "R-Sim-R + Burst"),
        (k5, "총 리뷰",   "608,458", "스팸 13.2%", C["red"],  "YelpZip 원본"),
    ]
    for col, lbl, val, sub, color, tip in kpis:
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="border-left-color:{color}">
              <div class="kpi-lbl">{lbl}</div>
              <div class="kpi-val" style="color:{color}">{val}</div>
              <div class="kpi-sub" style="color:{C['muted']}">{sub}</div>
            </div>""", unsafe_allow_html=True)
            st.caption(tip)

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown(f"<div class='sec-title'>📐 5종 엣지 기여도 분석</div>", unsafe_allow_html=True)
        if not attr.empty:
            edge_d = {
                "엣지 타입": attr["edge_type"].tolist(),
                "엣지 수":   attr["n_edges"].tolist(),
                "기여도(%)": attr["contribution_pct"].tolist(),
            }
            edf = pd.DataFrame(edge_d)
            fig_e = make_subplots(specs=[[{"secondary_y": True}]])
            colors = [C["purple"], C["teal"], C["blue"], C["amber"], C["muted"]]
            fig_e.add_trace(go.Bar(
                name="엣지 수", x=edf["엣지 타입"], y=edf["엣지 수"],
                marker_color=colors, opacity=0.8,
                text=edf["엣지 수"].apply(lambda v: f"{v:,}"), textposition="outside",
            ), secondary_y=False)
            fig_e.add_trace(go.Scatter(
                name="XAI 기여도(%)", x=edf["엣지 타입"], y=edf["기여도(%)"],
                mode="lines+markers", line=dict(color=C["red"], width=2.5),
                marker=dict(size=9, color=C["red"]),
            ), secondary_y=True)
            fig_e.update_layout(**CHART_TEMPLATE, height=300,
                                legend=dict(orientation="h", y=1.12))
            fig_e.update_yaxes(title_text="엣지 수", secondary_y=False,
                               showgrid=True, gridcolor="rgba(255,255,255,0.06)")
            fig_e.update_yaxes(title_text="기여도 (%)", secondary_y=True)
            st.plotly_chart(fig_e, use_container_width=True)

    with right:
        st.markdown(f"<div class='sec-title'>🍕 데이터 분포</div>", unsafe_allow_html=True)
        rating_dist = eda.get("rating_분포", {"1.0":930,"2.0":1887,"3.0":5016,"4.0":11517,"5.0":10650})
        fig_pie = go.Figure(go.Pie(
            labels=[f"{k}점" for k in rating_dist], values=list(rating_dist.values()),
            hole=0.55, marker_colors=[C["red"],C["amber"],C["muted"],C["blue"],C["teal"]],
            textfont_size=11,
        ))
        fig_pie.update_layout(**CHART_TEMPLATE, height=190,
                              annotations=[dict(text="별점", x=0.5, y=0.5, showarrow=False,
                                               font_size=13, font_color=C["text"])])
        st.plotly_chart(fig_pie, use_container_width=True)

        fig_spam = go.Figure(go.Pie(
            labels=["정상 (86.8%)", "스팸 (13.2%)"], values=[86.8, 13.2],
            hole=0.55, marker_colors=[C["blue"], C["red"]], textfont_size=11,
        ))
        fig_spam.update_layout(**CHART_TEMPLATE, height=190,
                               annotations=[dict(text="클래스", x=0.5, y=0.5, showarrow=False,
                                               font_size=13, font_color=C["text"])])
        st.plotly_chart(fig_spam, use_container_width=True)

    # Stage 카드
    st.markdown(f"<div class='sec-title'>🚀 모델 발전 4단계</div>", unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    stages = [
        (s1, "Stage 1", "정적 GNN 베이스라인",  "HeteroSAGE / GAT / BWGNN", "PR-AUC 0.83", C["muted"]),
        (s2, "Stage 2", "시간 인식 (TGATLite)",  "Bochner 시간 인코딩",       "Δt 연속 신호", C["blue"]),
        (s3, "Stage 3", "Dual Memory (V2)",       "User + Product 분리",       "Recall 향상",  C["purple"]),
        (s4, "Stage 4 ★","DRAGWave (제안)",       "DRAG × BWGAT 융합",         "PR-AUC 0.9419", C["teal"]),
    ]
    for col, stage, title, sub, perf, color in stages:
        with col:
            st.markdown(f"""
            <div style='background:{C["surface"]}; border-radius:10px; padding:14px;
                        border-top:3px solid {color}; min-height:150px;'>
              <div style='font-size:11px; color:{color}; font-weight:700;'>{stage}</div>
              <div style='font-size:13px; font-weight:700; margin:5px 0 3px;'>{title}</div>
              <div style='font-size:11px; color:{C["muted"]}; margin-bottom:8px;'>{sub}</div>
              <div style='font-size:12px; color:{color}; font-weight:600;'>{perf}</div>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 🎯 실시간 탐지
# ─────────────────────────────────────────────────────────────────────────────
elif "탐지" in page:
    st.markdown(f"<h2 style='color:{C['teal']}'>🎯 실시간 리뷰 사기 탐지 데모</h2>", unsafe_allow_html=True)
    st.caption("Test Set 리뷰를 선택하면 GNN 사전 추론 결과로 사기 확률과 탐지 근거를 분석합니다.")

    ctrl, result = st.columns([1, 2], gap="large")
    with ctrl:
        st.markdown(f"<div class='sec-title'>🔧 탐지 설정</div>", unsafe_allow_html=True)
        lf = st.selectbox("라벨 필터", ["전체", "실제 스팸만", "실제 정상만"])
        if st.button("🎲 랜덤 리뷰 선택", use_container_width=True, type="primary"):
            idx = np.where(test_mask)[0]
            if lf == "실제 스팸만":   idx = idx[labels[idx] == 1]
            elif lf == "실제 정상만": idx = idx[labels[idx] == 0]
            st.session_state["sel"] = int(np.random.choice(idx))
        nid = st.number_input("노드 ID 직접 입력", 0, len(df)-1, 0)
        if st.button("이 노드 분석", use_container_width=True):
            st.session_state["sel"] = int(nid)
        st.divider()
        test_p = probs[test_mask]
        c1, c2 = st.columns(2)
        c1.metric("탐지율 (>0.5)", f"{(test_p>0.5).mean():.1%}")
        c2.metric("평균 사기 확률", f"{test_p.mean():.3f}")

    with result:
        sel = st.session_state.get("sel")
        if sel is None:
            st.markdown(f"""
            <div style='display:flex;align-items:center;justify-content:center;
                        height:250px; background:{C["surface"]}; border-radius:12px;
                        border:2px dashed {C["muted"]}44;'>
              <div style='text-align:center; color:{C["muted"]}'>
                <div style='font-size:48px'>🔍</div>
                <div style='margin-top:12px;'>왼쪽에서 리뷰를 선택하세요</div>
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            fp = float(probs[sel])
            is_spam  = fp >= 0.5
            real_spam = int(labels[sel]) == 1
            color = C["red"] if is_spam else C["green"]

            fig_g = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=fp * 100,
                delta={"reference": 50, "valueformat": ".1f"},
                number={"suffix": "%", "font": {"size": 36, "color": color}},
                title={"text": "사기 확률", "font": {"size": 14, "color": C["muted"]}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": C["muted"]},
                    "bar":  {"color": color, "thickness": 0.25},
                    "bgcolor": C["surface2"],
                    "bordercolor": C["surface"],
                    "steps": [
                        {"range": [0,  30], "color": "rgba(33,197,93,0.08)"},
                        {"range": [30, 60], "color": "rgba(240,165,0,0.08)"},
                        {"range": [60,100], "color": "rgba(255,75,75,0.08)"},
                    ],
                    "threshold": {"line": {"color": C["red"], "width": 3}, "value": 50},
                },
            ))
            fig_g.update_layout(**CHART_TEMPLATE, height=230)
            st.plotly_chart(fig_g, use_container_width=True)

            ok  = is_spam == real_spam
            cols = st.columns(3)
            for col, lbl, val, clr in [
                (cols[0], "GNN 판정",  "🚨 사기" if is_spam else "✅ 정상",    C["red"] if is_spam else C["green"]),
                (cols[1], "실제 라벨", "🚨 스팸" if real_spam else "✅ 정상",  C["red"] if real_spam else C["green"]),
                (cols[2], "판정 결과", "✓ 정확" if ok else "✗ 오분류",        C["green"] if ok else C["red"]),
            ]:
                with col:
                    st.markdown(f"""
                    <div style='background:{clr}22; border-radius:8px; padding:10px;
                                border:1px solid {clr}44; text-align:center;'>
                      <div style='font-size:16px; font-weight:700; color:{clr};'>{val}</div>
                      <div style='font-size:11px; color:{C["muted"]}; margin-top:4px;'>{lbl}</div>
                    </div>""", unsafe_allow_html=True)

    if sel is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<div class='sec-title'>🔎 노드 {sel} 탐지 근거</div>", unsafe_allow_html=True)
        nb  = np.where((burst_ei[0]==sel)|(burst_ei[1]==sel))[0]
        ns  = np.where((sim_ei[0]==sel)|(sim_ei[1]==sel))[0]
        nu  = np.where((rur_ei[0]==sel)|(rur_ei[1]==sel))[0]
        avg_dt = float(burst_dt[nb].mean()) if len(nb) > 0 else 0.0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("R-Burst-R 연결", f"{len(nb)}개")
        m2.metric("평균 Δt", f"{avg_dt:.1f}h", delta="짧을수록 의심", delta_color="inverse")
        m3.metric("R-Sim-R 연결", f"{len(ns)}개")
        m4.metric("R-U-R 연결",  f"{len(nu)}개")

        dl, dr = st.columns([2, 1])
        with dl:
            row = df.iloc[sel] if sel < len(df) else None
            if row is not None:
                st.markdown("**리뷰 텍스트**")
                st.markdown(f"<div class='review-box'>{str(row.get('text',''))[:400]}</div>",
                            unsafe_allow_html=True)
        with dr:
            st.markdown("**신호 강도**")
            sig_data = [("R-U-R", len(nu), C["purple"]),
                        ("R-Burst-R", len(nb), C["teal"]),
                        ("R-Sim-R", len(ns), C["amber"])]
            fig_sig = go.Figure([go.Bar(
                y=[n for n,_,_ in sig_data], x=[c for _,c,_ in sig_data],
                orientation="h",
                marker_color=[cl for _,_,cl in sig_data], opacity=0.85,
                text=[f"{c}개" for _,c,_ in sig_data], textposition="outside",
            )])
            fig_sig.update_layout(**CHART_TEMPLATE, height=150,
                                  showlegend=False, xaxis_range=[0,max([c for _,c,_ in sig_data])*1.4+1])
            st.plotly_chart(fig_sig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# 🕸 캠페인 분석
# ─────────────────────────────────────────────────────────────────────────────
elif "캠페인" in page:
    st.markdown(f"<h2 style='color:{C['amber']}'>🕸 사기 캠페인 탐지기</h2>", unsafe_allow_html=True)
    kc1, kc2, kc3, kc4 = st.columns(4)
    kc1.metric("전체 컴포넌트",   f"{csum.get('total_components',1584):,}개")
    kc2.metric("의심 캠페인",     f"{csum.get('suspected_campaigns',91)}개", delta="스팸 ≥ 50%")
    kc3.metric("캠페인 내 리뷰",  f"{csum.get('total_reviews_in_campaigns',464)}건")
    kc4.metric("최대 캠페인 크기",f"{csum.get('max_campaign_size',90)}개")

    cl, cr = st.columns([2, 1], gap="large")
    with cl:
        st.markdown(f"<div class='sec-title'>🗂 의심 캠페인 목록</div>", unsafe_allow_html=True)
        if not camps.empty:
            spam_col = "spam_ratio" if "spam_ratio" in camps.columns else camps.columns[2]
            flag_col  = "is_campaign" if "is_campaign" in camps.columns else None
            dc = camps[camps[flag_col] == True].copy() if flag_col else camps[camps[spam_col] >= 0.5].copy()
            dc["스팸 비율"] = dc[spam_col].apply(lambda v: f"{v:.0%}")
            show_cols = [c for c in ["campaign_id","n_nodes","스팸 비율","avg_fraud_prob"] if c in dc.columns]
            col_map = {"campaign_id":"캠페인 ID","n_nodes":"리뷰 수","avg_fraud_prob":"평균 사기 확률"}
            st.dataframe(dc[show_cols].rename(columns=col_map).head(25),
                         use_container_width=True, height=380)

    with cr:
        st.markdown(f"<div class='sec-title'>📊 캠페인 크기 분포</div>", unsafe_allow_html=True)
        if not camps.empty and "n_nodes" in camps.columns:
            fig_h = go.Figure(go.Histogram(
                x=camps["n_nodes"], nbinsx=20, marker_color=C["amber"], opacity=0.85))
            fig_h.update_layout(**CHART_TEMPLATE, height=220, xaxis_title="리뷰 수", yaxis_title="빈도")
            st.plotly_chart(fig_h, use_container_width=True)

        spam_col2 = "spam_ratio" if "spam_ratio" in camps.columns else None
        if not camps.empty and spam_col2:
            dc2 = camps.dropna(subset=[spam_col2]).copy()
            dc2["spam_ratio"] = dc2[spam_col2]
            vals = [
                (dc2["spam_ratio"]==1.0).sum(),
                ((dc2["spam_ratio"]>=0.75)&(dc2["spam_ratio"]<1.0)).sum(),
                ((dc2["spam_ratio"]>=0.5)&(dc2["spam_ratio"]<0.75)).sum(),
                (dc2["spam_ratio"]<0.5).sum(),
            ]
            fig_d = go.Figure(go.Pie(
                labels=["100%","75~99%","50~74%","50% 미만"], values=vals, hole=0.6,
                marker_colors=[C["red"],C["amber"],C["blue"],C["muted"]], textfont_size=11))
            fig_d.update_layout(**CHART_TEMPLATE, height=200,
                                annotations=[dict(text="스팸 비율",x=0.5,y=0.5,showarrow=False,
                                               font_size=12,font_color=C["text"])])
            st.plotly_chart(fig_d, use_container_width=True)

    viz_path = REPORTS / "fraud_network_viz.html"
    if viz_path.exists():
        st.markdown(f"<div class='sec-title'>🌐 인터랙티브 사기 네트워크</div>", unsafe_allow_html=True)
        with open(viz_path, encoding="utf-8") as f:
            st.components.v1.html(f.read(), height=580)


# ─────────────────────────────────────────────────────────────────────────────
# 📊 모델 비교
# ─────────────────────────────────────────────────────────────────────────────
elif "모델" in page:
    st.markdown(f"<h2 style='color:{C['blue']}'>📊 모델 성능 비교</h2>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["🏆 트랜스덕티브", "🌍 인덕티브 (배포 환경)", "📈 학습 곡선"])

    with tab1:
        lf = log.sort_values("pr_auc", ascending=True).tail(14)
        fig_bar = make_subplots(rows=1, cols=2, subplot_titles=["PR-AUC", "Macro F1"])
        for i, metric in enumerate(["pr_auc", "macro_f1"]):
            lfs = lf.sort_values(metric, ascending=True)
            bar_colors = [C["teal"] if v == lfs[metric].max() else C["blue"] for v in lfs[metric]]
            fig_bar.add_trace(go.Bar(
                y=lfs["model"], x=lfs[metric], orientation="h",
                marker_color=bar_colors,
                text=lfs[metric].apply(lambda v: f"{v:.4f}"), textposition="outside",
                showlegend=False), row=1, col=i+1)
        fig_bar.update_xaxes(range=[0.6, 1.02], row=1, col=1)
        fig_bar.update_xaxes(range=[0.6, 1.02], row=1, col=2)
        fig_bar.update_layout(**CHART_TEMPLATE, height=450, title="모델별 성능 (상위 14개)")
        st.plotly_chart(fig_bar, use_container_width=True)

        disp = log[["model","pr_auc","macro_f1","params","notes"]].copy()
        disp.columns = ["모델","PR-AUC","Macro-F1","파라미터","비고"]
        st.dataframe(disp, use_container_width=True, height=300)

    with tab2:
        st.markdown("Train→Test 연결 엣지를 제거하고 Test 노드끼리만 연결된 서브그래프에서 재평가.")
        st.markdown("<span class='badge-amber'>⚠ 실제 배포 환경 시뮬레이션</span>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # 4-way 앙상블 + 개별 모델 추가
        extra = []
        for name, sc in e4.get("individual_scores", {}).items():
            if name not in log_ind["model"].values:
                extra.append({"model": name, "pr_auc": sc["pr_auc"], "macro_f1": sc["macro_f1"]})
        extra.append({
            "model": "4-way Ensemble ★",
            "pr_auc": e4.get("inductive_pr_auc", 0.7748),
            "macro_f1": e4.get("inductive_macro_f1", 0.8158)
        })
        li_all = pd.concat([log_ind, pd.DataFrame(extra)], ignore_index=True)
        li_all = li_all.drop_duplicates("model").sort_values("pr_auc", ascending=True)

        bar_colors = [
            C["teal"] if v >= 0.75 else (C["blue"] if v >= 0.65 else (C["muted"] if v >= 0.5 else C["red"]))
            for v in li_all["pr_auc"]
        ]
        fig_ind = go.Figure()
        fig_ind.add_trace(go.Bar(
            y=li_all["model"], x=li_all["pr_auc"], orientation="h",
            marker_color=bar_colors, opacity=0.9,
            text=li_all["pr_auc"].apply(lambda v: f"{v:.4f}"), textposition="outside",
        ))
        fig_ind.add_vline(x=0.7,    line_dash="dot",  line_color=C["amber"], annotation_text="0.7 기준선")
        fig_ind.add_vline(x=0.7748, line_dash="dash", line_color=C["teal"],
                          annotation_text="4-way 0.7748", annotation_position="bottom right")
        fig_ind.update_layout(**CHART_TEMPLATE, height=max(340, len(li_all)*28+80),
                              xaxis_range=[0, 1.1], title="인덕티브 PR-AUC — 4-way Ensemble 포함")
        st.plotly_chart(fig_ind, use_container_width=True)

        merged = pd.merge(
            log[["model","pr_auc"]].rename(columns={"pr_auc":"transductive"}),
            log_ind[["model","pr_auc"]].rename(columns={"pr_auc":"inductive"}),
            on="model", how="inner"
        )
        fig_gap = go.Figure([
            go.Bar(x=merged["model"], y=merged["transductive"], name="트랜스덕티브",
                   marker_color=C["blue"], opacity=0.8),
            go.Bar(x=merged["model"], y=merged["inductive"], name="인덕티브",
                   marker_color=C["teal"], opacity=0.8),
        ])
        fig_gap.update_layout(**CHART_TEMPLATE, height=320, barmode="group",
                              yaxis_range=[0,1.1], title="트랜스덕 vs 인덕티브 Gap",
                              legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_gap, use_container_width=True)

    with tab3:
        st.markdown("주요 모델의 학습 진행 PR-AUC 수렴 곡선")
        st.info("학습 곡선 데이터는 로컬 환경에서만 제공됩니다.")


# ─────────────────────────────────────────────────────────────────────────────
# 🔬 XAI 분석
# ─────────────────────────────────────────────────────────────────────────────
elif "XAI" in page:
    st.markdown(f"<h2 style='color:{C['purple']}'>🔬 XAI — 설명 가능한 사기 탐지</h2>", unsafe_allow_html=True)

    if not attr.empty:
        xa1, xa2 = st.columns([3, 2], gap="large")
        with xa1:
            st.markdown(f"<div class='sec-title'>📊 엣지 기여도</div>", unsafe_allow_html=True)
            a2 = attr.sort_values("contribution_pct", ascending=True)
            bar_c = [C["purple"] if v == a2["contribution_pct"].max() else C["blue"] for v in a2["contribution_pct"]]
            fig_xai = go.Figure(go.Bar(
                y=a2["edge_type"], x=a2["contribution_pct"], orientation="h",
                marker_color=bar_c, opacity=0.9,
                text=a2["contribution_pct"].apply(lambda v: f"{v:.2f}%"), textposition="outside",
            ))
            fig_xai.update_layout(**CHART_TEMPLATE, height=280, xaxis_title="기여도 (%)")
            st.plotly_chart(fig_xai, use_container_width=True)
            st.dataframe(attr[["edge_type","n_edges","base_pr_auc","ablated_pr_auc",
                                "delta_pr_auc","contribution_pct"]].rename(columns={
                "edge_type":"엣지","n_edges":"엣지 수","base_pr_auc":"기준 PR-AUC",
                "ablated_pr_auc":"제거 후","delta_pr_auc":"하락폭","contribution_pct":"기여도(%)"}),
                use_container_width=True)

        with xa2:
            st.markdown(f"<div class='sec-title'>🍩 기여도 비율</div>", unsafe_allow_html=True)
            fig_d = go.Figure(go.Pie(
                labels=attr["edge_type"], values=attr["contribution_pct"], hole=0.55,
                marker_colors=[C["purple"],C["teal"],C["blue"],C["amber"],C["muted"]], textfont_size=11))
            fig_d.update_layout(**CHART_TEMPLATE, height=260,
                                annotations=[dict(text="XAI 기여도",x=0.5,y=0.5,showarrow=False,
                                               font_size=13,font_color=C["text"])])
            st.plotly_chart(fig_d, use_container_width=True)
            st.markdown(f"""
            <div style='background:{C["surface"]}; border-radius:10px; padding:14px;
                        border-left:3px solid {C["purple"]};'>
              <b style='color:{C["purple"]}'>💡 핵심 인사이트</b><br><br>
              <span style='font-size:12px; line-height:1.9;'>
              • <b style='color:{C["purple"]}'>R-U-R 19.97%</b> — 동일 계정 반복이 최대 신호<br>
              • <b style='color:{C["teal"]}'>R-Burst-R 5.63%</b> — 72h 집중 2위<br>
              • <b style='color:{C["muted"]}'>R-S-R 0.20%</b> — 317K 엣지지만 기여 최저<br>
              → DRAGWave가 이 비대칭을 자동 학습
              </span>
            </div>""", unsafe_allow_html=True)

    # K-Fold 어뷰저 진화
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<div class='sec-title'>⏳ 어뷰저 전략 진화 분석</div>", unsafe_allow_html=True)
    if kfold and "kfold" in kfold:
        kdf = pd.DataFrame(kfold["kfold"]).dropna(subset=["pr_auc"])
        kdf["spam_pct"] = kdf["spam_ratio"] * 100
        fig_kf = make_subplots(specs=[[{"secondary_y": True}]])
        fig_kf.add_trace(go.Bar(
            x=[f"Fold {r['fold']}" for _,r in kdf.iterrows()], y=kdf["pr_auc"],
            name="인덕티브 PR-AUC", opacity=0.85,
            marker_color=[C["teal"] if v>=0.7 else (C["amber"] if v>=0.4 else C["red"]) for v in kdf["pr_auc"]],
            text=kdf["pr_auc"].apply(lambda v: f"{v:.3f}"), textposition="outside",
        ), secondary_y=False)
        fig_kf.add_trace(go.Scatter(
            x=[f"Fold {r['fold']}" for _,r in kdf.iterrows()], y=kdf["spam_pct"],
            name="스팸 밀도 (%)", mode="lines+markers",
            line=dict(color=C["purple"], width=2.5, dash="dot"),
            marker=dict(size=9, color=C["purple"]),
        ), secondary_y=True)
        fig_kf.update_yaxes(title_text="인덕티브 PR-AUC", secondary_y=False,
                            range=[0,1.1], gridcolor="rgba(255,255,255,0.06)")
        fig_kf.update_yaxes(title_text="스팸 밀도 (%)", secondary_y=True, range=[0,40])
        fig_kf.update_layout(**CHART_TEMPLATE, height=320,
                             title="시간 구간별 성능 vs 스팸 밀도",
                             legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_kf, use_container_width=True)

        st.markdown(f"""
        <div style='background:{C["surface"]}; border-radius:10px; padding:14px;
                    border-left:3px solid {C["amber"]};'>
          <b style='color:{C["amber"]}'>🔍 발견: 어뷰저 전략 진화</b><br>
          <span style='font-size:12px; line-height:1.8;'>
            초기(스팸 29.4%) → 후기(6~12%) 밀도 급감. 재학습 포함 3가지 완화 시도 모두 PR-AUC 개선 없음.
            <b>플랫폼 탐지 강화에 적응한 어뷰저 전략 진화의 증거.</b>
          </span>
        </div>""", unsafe_allow_html=True)
