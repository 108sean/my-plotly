import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st

st.title("📈 회귀 모델 성능 평가")


# 데이터 로드
@st.cache_data
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

# 데이터 분할
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


def evaluate(train_df, test_df, name):
    X_tr, y_tr = train_df["경과연수"].values, train_df["연평균기온"].values
    slope, intercept = np.polyfit(X_tr, y_tr, 1)

    X_te, y_te = test_df["경과연수"].values, test_df["연평균기온"].values
    y_pred = slope * X_te + intercept

    return {
        "모델": name,
        "기울기": round(slope, 4),
        "MAE": round(mean_absolute_error(y_te, y_pred), 3),
        "MSE": round(mean_squared_error(y_te, y_pred), 3),
        "R²": round(r2_score(y_te, y_pred), 3),
    }


res = [
    evaluate(full_data, full_data, "전체 (1908~2025)"),
    evaluate(train_100, test_data, "최근 100년 (1906~2005)"),
    evaluate(train_50, test_data, "최근 50년 (1956~2005)"),
]

st.dataframe(pd.DataFrame(res), use_container_width=True)
