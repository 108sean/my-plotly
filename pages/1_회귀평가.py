import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. 데이터 로드 및 전처리
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
clean_yearly["경과연수"] = clean_yearly["연도"] - 1908

# 2. 데이터셋 분할
# A. 전체 데이터 (Baseline)
full_data = clean_yearly.copy()

# B. 훈련 데이터: 최근 100년 (1906~2005)
train_100 = clean_yearly[
    (clean_yearly["연도"] >= 1906) & (clean_yearly["연도"] <= 2005)
].copy()

# C. 훈련 데이터: 최근 50년 (1956~2005)
train_50 = clean_yearly[
    (clean_yearly["연도"] >= 1956) & (clean_yearly["연도"] <= 2005)
].copy()

# D. 공통 테스트 데이터: 최근 20년 (2006~2025)
test_data = clean_yearly[
    (clean_yearly["연도"] >= 2006) & (clean_yearly["연도"] <= 2025)
].copy()


# 3. 회귀 모델 학습 및 평가 함수
def evaluate_model(train_df, test_df, model_name):
    X_train = train_df["경과연수"].values
    y_train = train_df["연평균기온"].values

    # 선형 회귀 (기울기, 절편)
    slope, intercept = np.polyfit(X_train, y_train, 1)

    # 테스트 데이터 예측
    X_test = test_df["경과연수"].values
    y_test = test_df["연평균기온"].values
    y_pred = slope * X_test + intercept

    # 평가 지표 계산
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        "모델 구분": model_name,
        "학습 연도 범위": f"{train_df['연도'].min()}~{train_df['연도'].max()}",
        "기울기 (°C/년)": slope,
        "절편": intercept,
        "MAE (°C)": mae,
        "MSE": mse,
        "R² Score": r2,
    }


# 4. 결과 집계
results = []

# 전체 데이터 (In-sample 평가)
results.append(evaluate_model(full_data, full_data, "전체 데이터 (Overall)"))

# 최근 100년 학습 -> 최근 20년 예측
results.append(
    evaluate_model(train_100, test_data, "최근 100년 학습 (1906~2005)")
)

# 최근 50년 학습 -> 최근 20년 예측
results.append(
    evaluate_model(train_50, test_data, "최근 50년 학습 (1956~2005)")
)

# 결과 출력
df_res = pd.DataFrame(results)
print(df_res.to_string(index=False))
