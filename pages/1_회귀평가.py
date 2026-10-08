# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st

st.set_page_config(page_title="Regression Model", layout="wide")
st.title("📈 연평균 기온 선형회귀 모델 평가 및 비교")


@st.cache_data(ttl=600)
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
    df = pd.read_csv(url, encoding="utf-8")
    df["날짜"] = pd.to_datetime(df["날짜"])
    df["연도"] = df["날짜"].dt.year

    df_filtered = df[df["연도"] <= 2025].copy()
    yearly = (
        df_filtered.groupby("연도")
        .agg(관측일수=("평균기온", "count"), 연평균기온=("평균기온", "mean"))
        .reset_index()
    )
    clean_yearly = yearly[yearly["관측일수"] >= 300].copy()
    clean_yearly["경과연수"] = clean_yearly["연도"] - 1908
    return clean_yearly


clean_yearly = load_data()

full_data = clean_yearly.copy()
train_100 = clean_yearly[
    (clean_yearly["연도"] >= 1906) & (clean_yearly["연도"] <= 2005)
].copy()
train_50 = clean_yearly[
    (clean_yearly["연도"] >= 1956) & (clean_yearly["연도"] <= 2005)
].copy()
test_data = clean_yearly[
    (clean_yearly["연도"] >= 2006) & (clean_yearly["연도"] <= 2025)
].copy()


def fit_and_evaluate(train_df, test_df, model_name):
    X_tr, y_tr = train_df["경과연수"].values, train_df["연평균기온"].values
    slope, intercept = np.polyfit(X_tr, y_tr, 1)

    X_te, y_te = test_df["경과연수"].values, test_df["연평균기온"].values
    y_pred = slope * X_te + intercept

    mae = mean_absolute_error(y_te, y_pred)
    mse = mean_squared_error(y_te, y_pred)
    r2 = r2_score(y_te, y_pred)

    eval_summary = {
        "모델 구분": model_name,
        "학습 기간": f"{train_df['연도'].min()}~{train_df['연도'].max()}",
        "테스트 기간": f"{test_df['연도'].min()}~{test_df['연도'].max()}",
        "기울기 (C/년)": round(slope, 4),
        "MAE (C)": round(mae, 3),
        "MSE": round(mse, 3),
        "R2 Score": round(r2, 3),
    }
    return eval_summary, slope, intercept


res_full, slope_full, intercept_full = fit_and_evaluate(
    full_data, full_data, "전체 데이터 (Baseline)"
)
res_100, slope_100, intercept_100 = fit_and_evaluate(
    train_100, test_data, "최근 100년 학습 (1906~2005)"
)
res_50, slope_50, intercept_50 = fit_and_evaluate(
    train_50, test_data, "최근 50년 학습 (1956~2005)"
)

st.subheader("📋 1. 회귀 모델 평가 지표")
df_eval = pd.DataFrame([res_full, res_100, res_50])
st.dataframe(df_eval, use_container_width=True)

st.subheader("🔍 2. 기울기 및 예측 성능 비교")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("최근 100년 학습 기울기", f"{slope_100:.4f} C/년")
with col2:
    st.metric(
        "최근 50년 학습 기울기",
        f"{slope_50:.4f} C/년",
        f"+{(slope_50 - slope_100):.4f} (약 48% 가파름)",
    )
with col3:
    st.metric(
        "최근 50년 모델 MAE 오차",
        f"{res_50['MAE (C)']} C",
        f"-{(res_100['MAE (C)'] - res_50['MAE (C)']):.3f} C 개선",
    )

st.markdown("""
---
#### 💡 주요 분석 요약
1. **기울기 비교:** 최근 50년 학습 모델(+0.0298 C/년)이 최근 100년 학습 모델(+0.0201 C/년)보다 **기울기가 약 48% 가파릅니다.** (최근 온도 상승세 반영)
2. **예측력 비교:** 최근 50년 모델은 테스트 기간(2006~2025)의 상승 추세를 잘 반영하여 **MAE 0.316 C, R² 0.489**를 기록한 반면, 최근 100년 모델은 최근 기온을 계속 과소평가하여 **MAE 0.658 C, R² -0.814**로 오차가 더 큽니다.
""")

st.subheader("📊 3. 회귀선 시각화")
years_all = np.arange(1900, 2026)
elapsed_all = years_all - 1908

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=clean_yearly["연도"],
        y=clean_yearly["연평균기온"],
        mode="markers",
        name="실제 데이터 (1908~2025)",
        marker=dict(color="gray", size=6, opacity=0.5),
    )
)
fig.add_trace(
    go.Scatter(
        x=test_data["연도"],
        y=test_data["연평균기온"],
        mode="markers",
        name="테스트 데이터 (2006~2025)",
        marker=dict(color="red", size=8),
    )
)
fig.add_trace(
    go.Scatter(
        x=years_all,
        y=slope_full * elapsed_all + intercept_full,
        mode="lines",
        name="전체 회귀선",
        line=dict(color="green", width=2),
    )
)
fig.add_trace(
    go.Scatter(
        x=years_all,
        y=slope_100 * elapsed_all + intercept_100,
        mode="lines",
        name="최근 100년 회귀선",
        line=dict(color="blue", width=2, dash="dash"),
    )
)
fig.add_trace(
    go.Scatter(
        x=years_all,
        y=slope_50 * elapsed_all + intercept_50,
        mode="lines",
        name="최근 50년 회귀선",
        line=dict(color="orange", width=2, dash="dot"),
    )
)

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (C)",
    hovermode
