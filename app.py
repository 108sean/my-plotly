import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# 페이지 설정
st.set_page_config(page_title="서울 연평균 기온 예측기", layout="wide")
st.title("🌡️ 서울 연평균 기온 예측기")


# 1. 데이터 로드 및 전처리
@st.cache_data
def load_and_process_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
    df = pd.read_csv(url, encoding="utf-8")

    # 날짜 컬럼을 datetime으로 변환 및 연도 추출
    df["날짜"] = pd.to_datetime(df["날짜"])
    df["연도"] = df["날짜"].dt.year

    # 1) 기준 기간 Filter: 2025년 이하 데이터만 사용
    df_filtered = df[df["연도"] <= 2025].copy()

    # 2) 각 연도별 관측일 수 및 평균기온 계산
    yearly_summary = (
        df_filtered.groupby("연도")
        .agg(관측일수=("평균기온", "count"), 연평균기온=("평균기온", "mean"))
        .reset_index()
    )

    # 3) 관측일수 300일 이상인 연도만 남기기
    clean_yearly = yearly_summary[yearly_summary["관측일수"] >= 300].copy()

    # 4) 회귀 분석을 위한 독립변수 (1908년부터 경과한 연수)
    clean_yearly["경과연수"] = clean_yearly["연도"] - 1908

    return clean_yearly


df_clean = load_and_process_data()

# 2. 선형 회귀 모델 계산 (1계 다항식)
# X: 1908년 기준 경과연수, Y: 연평균기온
x = df_clean["경과연수"].values
y = df_clean["연평균기온"].values

# 회귀 계수 (기울기, 절편)
slope, intercept = np.polyfit(x, y, 1)

# 상관계수 계산
corr_matrix = np.corrcoef(x, y)
correlation = corr_matrix[0, 1]

# 데이터 통계 정보 추출
data_count = len(df_clean)
start_year = int(df_clean["연도"].min())
end_year = int(df_clean["연도"].max())

# 3. 사이드바 / 상단 정보 표시
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📌 데이터 분석 정보")
    st.markdown(
        f"""
    - **분석에 사용된 연도 개수**: `{data_count}`개 해
    - **분석 시작 연도**: `{start_year}`년
    - **분석 종료 연도**: `{end_year}`년
    - **상관계수 (Pearson r)**: `{correlation:.4f}`
    """
    )

    st.subheader("🔮 기온 예측 (슬라이더)")
    target_year = st.slider("연도를 선택하세요", min_value=1900, max_value=2100, value=2026, step=1)

    # Target Year 예상 기온 계산
    target_elapsed = target_year - 1908
    predicted_temp = slope * target_elapsed + intercept

    st.metric(
        label=f"{target_year}년 예상 연평균 기온",
        value=f"{predicted_temp:.2f} °C",
        delta=f"1908년 대비 {predicted_temp - (slope * 0 + intercept):+.2f} °C 변화 추정",
    )

# 4. Plotly 시각화
with col2:
    st.subheader("📊 연도별 연평균기온 및 회귀 직선")

    # 추세선을 위한 전체 X축 범위 설정 (1900~2100년 반영)
    years_range = np.arange(1900, 2101)
    elapsed_range = years_range - 1908
    trendline_y = slope * elapsed_range + intercept

    fig = go.Figure()

    # 관측 데이터 산점도
    fig.add_trace(
        go.Scatter(
            x=df_clean["연도"],
            y=df_clean["연평균기온"],
            mode="markers",
            name="실제 연평균기온",
            marker=dict(size=8, color="#1f77b4", opacity=0.8),
            hovertemplate="%{x}년: %{y:.2f}°C<extra></extra>",
        )
    )

    # 회귀 직선
    fig.add_trace(
        go.Scatter(
            x=years_range,
            y=trendline_y,
            mode="lines",
            name="회귀 직선",
            line=dict(color="#ff7f0e", width=2, dash="dash"),
            hovertemplate="예측값(%{x}년): %{y:.2f}°C<extra></extra>",
        )
    )

    # 사용자 선택 연도 강조 표시
    fig.add_trace(
        go.Scatter(
            x=[target_year],
            y=[predicted_temp],
            mode="markers",
            name=f"선택 연도 ({target_year})",
            marker=dict(size=14, color="red", symbol="star"),
            hovertemplate=f"<b>선택된 연도: {target_year}년</b><br>예상 기온: {predicted_temp:.2f}°C<extra></extra>",
        )
    )

    # 레이아웃 설정
    fig.update_layout(
        xaxis_title="연도",
        yaxis_title="연평균기온 (°C)",
        hovermode="closest",
        legend=dict(orient="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=40, b=20),
    )

    st.plotly_chart(fig, use_container_width=True)
