import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st

st.set_page_config(page_title="선형회귀 모델 평가", layout="wide")
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
        "기울기 (도/년)": round(slope, 4),
        "MAE (도)": round(mae, 3),
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

st.subheader("📋 1. 회귀 모델 평가 지표 (MAE, MSE, R2)")
df_eval = pd.DataFrame([res_full, res_100, res_50])
st.dataframe(df
