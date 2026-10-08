import traceback
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="선형회귀 모델 평가", layout="wide")
st.title("📈 연평균 기온 선형회귀 모델 평가")

try:
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )

    # 1. 데이터 로드 및 전처리
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

    with st.spinner("데이터를 불러오는 중입니다..."):
        clean_yearly = load_data()

    # 2. 데이터 분할
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

    # 3. 모델 평가 함수
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
            "테스트 기간": f"{test_df['연도'].
