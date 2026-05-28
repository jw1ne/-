# ECG Arrhythmia Streamlit

ECG 데이터를 기반으로 부정맥을 분석하는 AI 웹 시스템입니다.  
MIT-BIH Arrhythmia Database를 활용하여 ECG 데이터를 학습하였으며,  
CNN 기반 딥러닝 모델을 통해 정상/부정맥 여부를 분류합니다.

## Features

- ECG 신호 시각화
- 실시간 ECG 그래프 출력
- 심박수 측정
- 정상 / 부정맥 분류
- ECG 파일 업로드 기능
- Streamlit 기반 웹 구현

## Tech Stack

- Python
- TensorFlow / Keras
- Streamlit
- NumPy
- Pandas
- Matplotlib
- WFDB
 
## Dataset

- MIT-BIH Arrhythmia Database 사용
- ECG 데이터를 10초 단위로 분할하여 학습 진행
- PVC 개수를 기준으로 라벨링 수행
 
## Model

- Conv1D 기반 CNN 모델 사용
- ECG 시계열 데이터 분류 수행
- Accuracy, Recall, F1-score, AUC 기반 성능 평가

## Run

bash
pip install -r requirements.txt
streamlit run app.py
