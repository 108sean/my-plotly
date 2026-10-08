import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st

st.set_page_config(page_title="다항 회귀 모델 평가", layout="wide")
st.title("🌊 다항 회귀(곡선) 모델 평가 및 2050년 예측")


# 1. 데이터 로드 및 전처리
@st.cache_data(ttl=600)
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
    df = pd.read_csv(url, encoding="utf-8")
    df["날짜"] = pd.to_datetime(df["날짜"])
    df["연도"] = df["날짜"].dt.year

    # 2025년 이하 & 관측일수 300일 이상 필터링
    df_filtered = df[df["연도"] <= 2025].copy()
    yearly = (
        df_filtered.groupby("연도")
        .agg(관측일수=("평균기온", "count"), 연평균기온=("평균기온", "mean"))
        .reset_index()
    )
    clean_yearly = yearly[yearly["관측일수"] >= 300].copy()

    # 연도 숫자가 커서 발생하는 수치 불안정 방지 (1908년 기준 0으로 스케일링)
    clean_yearly["경과연수"] = clean_yearly["연도"] - 1908
    return clean_yearly


clean_yearly = load_data()

# 2. 데이터 분할: 2005년 이전(훈련용), 2005년 이후(테스트용: 2005~2025)
train_df = clean_yearly[clean_yearly["연도"] < 2005].copy()
test_df = clean_yearly[clean_yearly["연도"] >= 2005].copy()

n_train = len(train_df)
n_test = len(test_df)

# 데이터 개수 안내 출력
st.info(
    f"📌 **데이터 분할 안내**: 훈련 데이터 **{n_train}개 연도** (< 2005년) | 테스트 데이터 **{n_test}개 연도** (2005년~2025년)"
)

# 3. 모델 피팅 및 평가 (1차, 3차, 9차)
degrees = [1, 3, 9]
results = []
models = {}

X_tr = train_df["경과연수"].values
y_tr = train_df["연평균기온"].values

X_te = test_df["경과연수"].values
y_te = test_df["연평균기온"].values

# 2050년 계산용 경과연수 (2050 - 1908 = 142)
x_2050 = 2050 - 1908

for deg in degrees:
    # 훈련용 데이터로만 피팅
    poly_coeffs = np.polyfit(X_tr, y_tr, deg)
    poly_func = np.poly1d(poly_coeffs)

    models[deg] = poly_func

    # 테스트 데이터 예측 및 평가
    y_pred = poly_func(X_te)
    mae = mean_absolute_error(y_te, y_pred)
    mse = mean_squared_error(y_te, y_pred)
    r2 = r2_score(y_te, y_pred)

    # 2050년 예측값 계산
    pred_2050 = poly_func(x_2050)

    results.append(
        {
            "차수": f"{deg}차 모델",
            "MAE (평균 빗나간 정도)": f"{mae:.3f} °C",
            "MSE": round(mse, 3),
            "R² Score": round(r2, 3),
            "2050년 예측기온": f"{pred_2050:.2f} °C",
        }
    )

# 4. 평가 요약표 출력
st.subheader("📋 테스트 데이터 성능 평가 및 2050년 예측 비교")
st.dataframe(pd.DataFrame(results), use_container_width=True)

# 5. 시각화 (Plotly)
st.subheader("📊 차수별 다항 회귀 곡선 및 2050년 추세 예측")

years_plot = np.arange(1908, 2051)
elapsed_plot = years_plot - 1908

fig = go.Figure()

# 훈련 데이터 점
fig.add_trace(
    go.Scatter(
        x=train_df["연도"],
        y=train_
