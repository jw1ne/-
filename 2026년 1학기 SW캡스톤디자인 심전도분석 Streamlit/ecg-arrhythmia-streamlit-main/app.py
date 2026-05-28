import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time
import os

# ===============================
# 기본 설정
# ===============================
st.set_page_config(
    page_title="ECG 부정맥 탐지 시스템",
    layout="wide"
)

# ===============================
# 판정 기준
# ===============================
NORMAL_LIMIT = 0.40
ARRHYTHMIA_LIMIT = 0.60

# ===============================
# CSS
# ===============================
st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: 900;
    text-align: center;
    margin-bottom: 5px;
}
.sub-title {
    text-align: center;
    color: #9ca3af;
    font-size: 18px;
    margin-bottom: 30px;
}
.metric-card {
    border-radius: 18px;
    padding: 28px;
    text-align: center;
    margin-top: 10px;
}
.normal-card {
    border: 2px solid #22c55e;
    background-color: #ecfdf5;
}
.warning-card {
    border: 2px solid #f59e0b;
    background-color: #fffbeb;
}
.abnormal-card {
    border: 2px solid #ef4444;
    background-color: #fef2f2;
}
.normal-text {
    color: #16a34a;
    font-size: 44px;
    font-weight: 900;
}
.warning-text {
    color: #d97706;
    font-size: 44px;
    font-weight: 900;
}
.abnormal-text {
    color: #dc2626;
    font-size: 44px;
    font-weight: 900;
}
.info-card {
    border-radius: 14px;
    padding: 20px;
    background-color: #ffffff;
    border: 1px solid #e5e7eb;
    color: #111827;
    margin-top: 16px;
}
.info-row {
    display: flex;
    justify-content: space-between;
    border-bottom: 1px solid #e5e7eb;
    padding: 8px 0;
    font-size: 15px;
}
.info-row:last-child {
    border-bottom: none;
}
.info-label {
    font-weight: 700;
    color: #374151;
}
.info-value {
    color: #111827;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<div class='main-title'>💙 실시간 ECG 부정맥 탐지 시스템</div>",
    unsafe_allow_html=True
)
st.markdown(
    "<div class='sub-title'>CNN 모델을 활용하여 ECG 신호를 실시간으로 시각화하고 부정맥 위험도를 분석합니다.</div>",
    unsafe_allow_html=True
)

# ===============================
# 데이터 로드
# ===============================
X = np.load("X_ecg.npy")
y = np.load("y_ecg.npy")

normal_indices = np.where(y == 0)[0]
arrhythmia_indices = np.where(y == 1)[0]

# ===============================
# 모델 로드
# ===============================
model = None
if os.path.exists("ecg_cnn_10sec_tuned.keras"):
    from tensorflow.keras.models import load_model
    model = load_model("ecg_cnn_10sec_tuned.keras")

# ===============================
# 상태 저장
# ===============================
if "status" not in st.session_state:
    st.session_state.status = "stopped"

if "current_i" not in st.session_state:
    st.session_state.current_i = 80

if "sample_index" not in st.session_state:
    st.session_state.sample_index = 0

if "last_sample_index" not in st.session_state:
    st.session_state.last_sample_index = 0

# ===============================
# 사이드바
# ===============================
st.sidebar.header("설정")

sample_index = st.sidebar.slider(
    "ECG 샘플 선택",
    0,
    len(X) - 1,
    st.session_state.sample_index
)

st.session_state.sample_index = sample_index

col_a, col_b = st.sidebar.columns(2)

if col_a.button("정상 샘플"):
    st.session_state.sample_index = int(normal_indices[0])
    st.session_state.status = "stopped"
    st.session_state.current_i = 80
    st.rerun()

if col_b.button("부정맥 샘플"):
    st.session_state.sample_index = int(arrhythmia_indices[0])
    st.session_state.status = "stopped"
    st.session_state.current_i = 80
    st.rerun()

speed = st.sidebar.slider(
    "재생 속도",
    0.001,
    0.10,
    0.01
)

step_size = st.sidebar.slider(
    "그래프 진행 단위",
    20,
    250,
    120,
    10
)

if sample_index != st.session_state.last_sample_index:
    st.session_state.status = "stopped"
    st.session_state.current_i = 80
    st.session_state.last_sample_index = sample_index

st.sidebar.markdown("---")

btn1, btn2 = st.sidebar.columns(2)
btn3, btn4 = st.sidebar.columns(2)

start_button = btn1.button("▶ 시작")
pause_button = btn2.button("⏸ 일시정지")
resume_button = btn3.button("⏵ 재개")
stop_button = btn4.button("⏹ 정지")

if start_button:
    st.session_state.status = "running"
    st.session_state.current_i = 80

if pause_button:
    st.session_state.status = "paused"

if resume_button:
    if st.session_state.status == "paused":
        st.session_state.status = "running"

if stop_button:
    st.session_state.status = "stopped"
    st.session_state.current_i = 80

st.sidebar.markdown("---")
st.sidebar.subheader("데이터 정보")
st.sidebar.write("총 샘플 수:", len(X))
st.sidebar.write("정상 샘플 수:", int(np.sum(y == 0)))
st.sidebar.write("부정맥 샘플 수:", int(np.sum(y == 1)))
st.sidebar.write("현재 상태:", st.session_state.status)

# ===============================
# ECG 선택
# ===============================
ecg = X[st.session_state.sample_index]

if len(ecg.shape) > 1:
    ecg = ecg.reshape(-1)

# ===============================
# 예측 함수
# ===============================
def predict_ecg(signal):
    if model is None:
        return "모델 없음", 0.0, "분석 불가"

    signal_norm = (signal - signal.mean()) / (signal.std() + 1e-8)
    input_data = signal_norm.reshape(1, -1, 1)

    pred = model.predict(input_data, verbose=0)
    risk_score = float(pred[0][0])

    if risk_score < NORMAL_LIMIT:
        return "정상", risk_score, "낮음"
    elif risk_score < ARRHYTHMIA_LIMIT:
        return "관찰 필요", risk_score, "중간"
    else:
        return "부정맥 의심", risk_score, "높음"

# ===============================
# 그래프 함수
# ===============================
def draw_ecg(signal):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=signal,
        mode="lines",
        name="ECG Signal",
        line=dict(width=2)
    ))

    fig.update_layout(
        title="Real-time ECG Signal",
        xaxis_title="Time (sample)",
        yaxis_title="Amplitude",
        height=560,
        margin=dict(l=30, r=30, t=50, b=30),
        showlegend=False
    )

    return fig

# ===============================
# 결과 카드
# ===============================
# ===============================
# 결과 카드
# ===============================
# ===============================
# 결과 카드
# ===============================
# ===============================
# 결과 카드
# ===============================
def show_result(label, risk, level):
    percent = risk * 100

    result_area.empty()

    with result_area.container():

        if label == "정상":
            st.success("✅ 정상")
            st.metric("부정맥 위험도", f"{percent:.1f}%")
            st.write(f"위험 수준: {level}")

        elif label == "관찰 필요":
            st.warning("⚠️ 관찰 필요")
            st.metric("부정맥 위험도", f"{percent:.1f}%")
            st.write(f"위험 수준: {level}")

        elif label == "부정맥 의심":
            st.error("🚨 부정맥 의심")
            st.metric("부정맥 위험도", f"{percent:.1f}%")
            st.write(f"위험 수준: {level}")

        else:
            st.warning("모델 파일이 없어 예측을 수행할 수 없습니다.")
# ===============================
# 화면 배치
# ===============================
left_col, right_col = st.columns([2.7, 1])

with left_col:
    st.subheader("● Real-time ECG Signal")
    graph_area = st.empty()
    message_area = st.empty()

with right_col:
    st.subheader("분석 결과")
    result_area = st.empty()
    st.subheader("상세 정보")
    detail_area = st.empty()

# ===============================
# 그래프 출력
# ===============================
end_i = min(st.session_state.current_i, len(ecg))
current_signal = ecg[:end_i]

graph_area.plotly_chart(draw_ecg(current_signal), use_container_width=True)

# ===============================
# 상태별 동작
# ===============================
if st.session_state.status == "running":
    message_area.info("ECG 모니터링이 진행 중입니다.")

    if st.session_state.current_i < len(ecg):
        st.session_state.current_i += step_size
        time.sleep(speed)
        st.rerun()
    else:
        st.session_state.status = "stopped"

elif st.session_state.status == "paused":
    message_area.warning("모니터링이 일시정지되었습니다. 재개 버튼을 누르면 이어서 진행됩니다.")

else:
    message_area.info("왼쪽 사이드바에서 시작 버튼을 누르세요.")

# ===============================
# 분석 결과
# ===============================
prediction_start_ratio = 0.4

result_label = "분석 중"
risk_score = 0.0
risk_level = "분석 중"

if st.session_state.current_i > len(ecg) * prediction_start_ratio:
    result_label, risk_score, risk_level = predict_ecg(ecg)
    show_result(result_label, risk_score, risk_level)
else:
    result_area.info("ECG 신호가 40% 이상 입력되면 분석 결과가 표시됩니다.")

progress = min(st.session_state.current_i, len(ecg))

detail_area.markdown(f"""
<div class="info-card">
    <div class="info-row">
        <span class="info-label">선택 샘플</span>
        <span class="info-value">{st.session_state.sample_index}</span>
    </div>
    <div class="info-row">
        <span class="info-label">진행 상태</span>
        <span class="info-value">{st.session_state.status}</span>
    </div>
    <div class="info-row">
        <span class="info-label">진행 샘플</span>
        <span class="info-value">{progress} / {len(ecg)}</span>
    </div>
    <div class="info-row">
        <span class="info-label">분석 결과</span>
        <span class="info-value">{result_label}</span>
    </div>
    <div class="info-row">
        <span class="info-label">위험도</span>
        <span class="info-value">{risk_score * 100:.1f}%</span>
    </div>
    <div class="info-row">
        <span class="info-label">위험 수준</span>
        <span class="info-value">{risk_level}</span>
    </div>
    <div class="info-row">
        <span class="info-label">모델</span>
        <span class="info-value">CNN</span>
    </div>
    <div class="info-row">
        <span class="info-label">신호 길이</span>
        <span class="info-value">{len(ecg)}</span>
    </div>
</div>
""", unsafe_allow_html=True)