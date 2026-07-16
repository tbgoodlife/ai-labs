# 『인공지능 개론』 실습 노트북

『인공지능 개론』(김정수 저)의 장별 실습 노트북 공식 저장소입니다.
책 지면에는 핵심 코드만 실려 있으며, 전체 코드·실행 결과·심화 실험은
이 저장소에서 제공합니다.

## 시작하기

- **Colab(권장)**: 아래 목차의 "Colab" 배지를 누르면 설치 없이 브라우저에서
  바로 실행됩니다. 처음이라면 책의 **부록 B.1**(Colab 시작하기)을 먼저 읽으세요.
- **로컬**: Python 3.10 이상에서 `pip install -r requirements.txt` 후
  Jupyter로 엽니다(책 부록 B.1.5의 "더 알아보기" 참고).

## 노트북 읽는 법

- `[셀 n] … 📖` — 책 지면 코드와 **1:1 대응**하는 셀
- `[준비]` — 저장소 전용: 환경 설정·한글 폰트·버전 출력 (**가장 먼저 실행**)
- `[보조 n]` — 저장소 전용: 상세 시각화·확장 실험
- `[심화 n]` — 장말 연습문제 심화 문항과 연계된 변형 실험 뼈대(`TODO` 채우기)

모든 노트북은 전체 재실행(Restart & Run All) 검증을 거쳐 **출력이 포함된
상태**로 제공되며, 난수 시드가 고정되어 있어 책의 수치가 재현됩니다.

## 실습 목차

| 실습 | 노트북 | 환경 | Colab |
|---|---|---|---|
| 3-1 미로 탐색 — BFS/DFS | `ch03/lab-03-01_maze-search.ipynb` | CPU | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tbgoodlife/ai-labs/blob/main/ch03/lab-03-01_maze-search.ipynb) |
| 3-2 8-퍼즐 A* | `ch03/lab-03-02_eight-puzzle-astar.ipynb` | CPU | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tbgoodlife/ai-labs/blob/main/ch03/lab-03-02_eight-puzzle-astar.ipynb) |
| 4-1 전방 연쇄 추론 엔진 | `ch04/lab-04-01_forward-chaining.ipynb` | CPU | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tbgoodlife/ai-labs/blob/main/ch04/lab-04-01_forward-chaining.ipynb) |
| 5-1 나이브 베이즈 스팸 분류 | `ch05/lab-05-01_naive-bayes-spam.ipynb` | CPU | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tbgoodlife/ai-labs/blob/main/ch05/lab-05-01_naive-bayes-spam.ipynb) |
| 6-1 과적합 시각화 | `ch06/lab-06-01_overfitting-curves.ipynb` | CPU | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tbgoodlife/ai-labs/blob/main/ch06/lab-06-01_overfitting-curves.ipynb) |
| 7-1 붓꽃 분류 4종 비교 | `ch07/lab-07-01_iris-classifiers.ipynb` | CPU | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tbgoodlife/ai-labs/blob/main/ch07/lab-07-01_iris-classifiers.ipynb) |
| 8-1 k-평균 고객 세분화 | `ch08/lab-08-01_kmeans-customers.ipynb` | CPU | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tbgoodlife/ai-labs/blob/main/ch08/lab-08-01_kmeans-customers.ipynb) |
| 8-2 PCA로 MNIST 시각화 | `ch08/lab-08-02_pca-mnist.ipynb` | CPU | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tbgoodlife/ai-labs/blob/main/ch08/lab-08-02_pca-mnist.ipynb) |
| 9-1 퍼셉트론과 XOR | `ch09/lab-09-01_perceptron-xor.ipynb` | CPU | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tbgoodlife/ai-labs/blob/main/ch09/lab-09-01_perceptron-xor.ipynb) |
| 9-2 Keras MNIST 드롭아웃 | `ch09/lab-09-02_keras-mnist.ipynb` | GPU 권장 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tbgoodlife/ai-labs/blob/main/ch09/lab-09-02_keras-mnist.ipynb) |
| 10-1 CNN CIFAR-10 | `ch10/lab-10-01_cnn-cifar10.ipynb` | GPU 권장 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tbgoodlife/ai-labs/blob/main/ch10/lab-10-01_cnn-cifar10.ipynb) |
| 10-2 전이학습 | `ch10/lab-10-02_transfer-learning.ipynb` | GPU 권장 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tbgoodlife/ai-labs/blob/main/ch10/lab-10-02_transfer-learning.ipynb) |
| 10-3 확산 모델 체험 † | `ch10/lab-10-03_diffusion-demo.ipynb` | GPU 권장 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tbgoodlife/ai-labs/blob/main/ch10/lab-10-03_diffusion-demo.ipynb) |
| 11-1 LSTM 긴 문맥·기울기 | `ch11/lab-11-01_lstm-long-context.ipynb` | CPU | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tbgoodlife/ai-labs/blob/main/ch11/lab-11-01_lstm-long-context.ipynb) |
| 11-2 어텐션 히트맵 | `ch11/lab-11-02_attention-heatmap.ipynb` | CPU | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tbgoodlife/ai-labs/blob/main/ch11/lab-11-02_attention-heatmap.ipynb) |
| 12-1 HF NLP 파이프라인 | `ch12/lab-12-01_hf-pipeline.ipynb` | CPU | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tbgoodlife/ai-labs/blob/main/ch12/lab-12-01_hf-pipeline.ipynb) |
| 12-2 프롬프트 Before/After | `ch12/lab-12-02_prompt-before-after.ipynb` | GPU 권장 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tbgoodlife/ai-labs/blob/main/ch12/lab-12-02_prompt-before-after.ipynb) |
| 12-3 RAG 미니 체험 † | `ch12/lab-12-03_rag-mini.ipynb` | CPU | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tbgoodlife/ai-labs/blob/main/ch12/lab-12-03_rag-mini.ipynb) |
| 13-1 FrozenLake Q-러닝 | `ch13/lab-13-01_frozenlake-qlearning.ipynb` | CPU | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tbgoodlife/ai-labs/blob/main/ch13/lab-13-01_frozenlake-qlearning.ipynb) |
| 14-1 편향과 공정성 지표 | `ch14/lab-14-01_fairness-metrics.ipynb` | CPU | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tbgoodlife/ai-labs/blob/main/ch14/lab-14-01_fairness-metrics.ipynb) |

† 표시는 저장소 전용 실습(책 본문에는 안내만 있음)입니다.
"CPU"인 실습도 GPU에서 더 빨리 돕니다. 12-2의 1순위 경로(무료 웹 챗봇)는
코드 없이 브라우저만으로 진행합니다 — 책 12.3의 안내를 따르세요.

모든 배지는 `https://github.com/tbgoodlife/ai-labs` 저장소를 가리킵니다.

## 폴더 구성

| 경로 | 내용 |
|---|---|
| `ch03/ … ch14/` | 장별 실습 노트북 |
| `utils/` | 공용 모듈 — 도해 스타일(`plot_style`), 시각화(`viz`), 데이터 로더(`data`) |
| `data/` | 소형 동봉 데이터(5장 한국어 스팸 미니 코퍼스 — 자체 제작) |
| `requirements.txt` | 로컬 실행용 패키지 목록 |

## 문제가 생겼을 때

책 **부록 B.4.4**의 공통 처방 다섯 가지를 먼저 확인하세요.
해결되지 않으면 이 저장소의 Issues에 남겨 주세요.

## 라이선스

- **코드**(노트북·`utils/`·스크립트): MIT License — [LICENSE](LICENSE) 참조
- **동봉 데이터**(`data/spam_mini.csv`): 이 책을 위해 자체 제작 — 코드와 동일 조건
- 실습이 내려받는 외부 데이터셋(MNIST, CIFAR-10, tf_flowers, UCI Adult 등)과
  사전학습 모델은 각 배포처의 라이선스를 따릅니다.
