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


# 3. 모델 학습 및 평가 함수
def evaluate_model(train_df, test_df, model_name):
    X_train = train_df["경과연수"].values
    y_train = train_df["연평균기온"].values

    slope, intercept = np.polyfit(X_train, y_train, 1)

    X_test = test_df["경과연수"].values
    y_test = test_df["연평균기온"].values
    y_pred = slope * X_test + intercept

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        "모델 구분": model_name,
        "학습 기간": f"{train_df['연도'].min()}~{train_df['연도'].max()}",
        "기울기 (°C/년)": round(slope, 4),
        "MAE (°C)": round(mae, 3),
        "MSE": round(mse, 3),
        "R² Score": round(r2, 3),
    }


# 4. 결과 집계
results = [
    evaluate_model(full_data, full_data, "전체 데이터 (Baseline)"),
    evaluate_model(train_100, test_data, "최근 100년 학습"),
    evaluate_model(train_50, test_data, "최근 50년 학습"),
]

df_res = pd.DataFrame(results)
print(df_res.to_string(index=False))
