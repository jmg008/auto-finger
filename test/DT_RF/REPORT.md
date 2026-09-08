# Stride별 DT·RF 비교 결과

## 실험 조건

- 데이터: 프로젝트 루트의 `none.txt`, `rock.txt`, `paper.txt`, `middle.txt`, `thumb.txt`
- 샘플링: 1,000 Hz, 창 길이 200 ms(200개 샘플)
- stride: 50/100/150/200 ms
- 특성: 기존 `learn.py`와 동일한 MAV/RMS/WL 및 2채널 관계 특성 15개
- 분할: 각 stride에서 클래스별 시간 순서를 유지해 앞 75% 학습, 뒤 25% 테스트
- DT와 RF는 각 stride 안에서 완전히 동일한 학습·테스트 창 사용
- DT: `DecisionTreeClassifier(max_depth=5, random_state=42)`
- RF: `RandomForestClassifier(n_estimators=300, max_depth=5, random_state=42, n_jobs=1)`

## 성능 요약

| stride | overlap | 학습/테스트 창 | DT 정확도 | DT Macro F1 | RF 정확도 | RF Macro F1 | 정확도 차이(RF-DT) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 ms | 150 ms (75%) | 662/224 | 87.50% | 87.29% | 85.27% | 85.08% | -2.23%p |
| 100 ms | 100 ms (50%) | 331/113 | 92.04% | 91.91% | 82.30% | 81.89% | -9.73%p |
| 150 ms | 50 ms (25%) | 220/76 | 82.89% | 82.43% | 84.21% | 83.98% | +1.32%p |
| 200 ms | 0 ms (0%) | 167/57 | 89.47% | 89.06% | 80.70% | 79.96% | -8.77%p |

전체 8개 조합 중 최고 결과는 **100 ms DT**로, 정확도 92.04%, macro F1 91.91%입니다.
DT의 최고 stride는 **100 ms**(정확도 92.04%), RF의 최고 stride는 **50 ms**(정확도 85.27%)입니다.

## 동일 테스트 창에서의 직접 비교

| stride | 둘 다 정답 | DT만 정답 | RF만 정답 | 둘 다 오답 | exact McNemar p |
|---:|---:|---:|---:|---:|---:|
| 50 ms | 186 | 10 | 5 | 23 | 0.301758 |
| 100 ms | 92 | 12 | 1 | 8 | 0.003418 |
| 150 ms | 61 | 2 | 3 | 10 | 1.000000 |
| 200 ms | 46 | 5 | 0 | 6 | 0.062500 |

## 클래스별 F1

| stride | 모델 | rock | paper | none | middle | thumb |
|---:|---|---:|---:|---:|---:|---:|
| 50 ms | DT | 71.60% | 93.18% | 100.00% | 94.12% | 77.55% |
| 50 ms | RF | 71.43% | 90.53% | 100.00% | 89.74% | 73.68% |
| 100 ms | DT | 82.93% | 95.65% | 100.00% | 95.24% | 85.71% |
| 100 ms | RF | 69.77% | 86.27% | 100.00% | 81.08% | 72.34% |
| 150 ms | DT | 75.00% | 85.71% | 100.00% | 80.00% | 71.43% |
| 150 ms | RF | 73.33% | 87.50% | 100.00% | 85.71% | 73.33% |
| 200 ms | DT | 84.21% | 88.00% | 100.00% | 84.21% | 88.89% |
| 200 ms | RF | 72.73% | 81.48% | 100.00% | 70.59% | 75.00% |

## 혼동행렬

행은 실제 클래스, 열은 예측 클래스입니다.

### 50 ms · DT

| actual \ predicted | rock | paper | none | middle | thumb |
|---|---:|---:|---:|---:|---:|
| rock | 29 | 0 | 0 | 0 | 15 |
| paper | 1 | 41 | 0 | 2 | 0 |
| none | 0 | 0 | 48 | 0 | 0 |
| middle | 0 | 3 | 0 | 40 | 0 |
| thumb | 7 | 0 | 0 | 0 | 38 |

### 50 ms · RF

| actual \ predicted | rock | paper | none | middle | thumb |
|---|---:|---:|---:|---:|---:|
| rock | 30 | 0 | 0 | 0 | 14 |
| paper | 0 | 43 | 0 | 0 | 1 |
| none | 0 | 0 | 48 | 0 | 0 |
| middle | 0 | 8 | 0 | 35 | 0 |
| thumb | 10 | 0 | 0 | 0 | 35 |

### 100 ms · DT

| actual \ predicted | rock | paper | none | middle | thumb |
|---|---:|---:|---:|---:|---:|
| rock | 17 | 0 | 0 | 0 | 5 |
| paper | 0 | 22 | 0 | 0 | 0 |
| none | 0 | 0 | 24 | 0 | 0 |
| middle | 0 | 2 | 0 | 20 | 0 |
| thumb | 2 | 0 | 0 | 0 | 21 |

### 100 ms · RF

| actual \ predicted | rock | paper | none | middle | thumb |
|---|---:|---:|---:|---:|---:|
| rock | 15 | 0 | 0 | 0 | 7 |
| paper | 0 | 22 | 0 | 0 | 0 |
| none | 0 | 0 | 24 | 0 | 0 |
| middle | 0 | 7 | 0 | 15 | 0 |
| thumb | 6 | 0 | 0 | 0 | 17 |

### 150 ms · DT

| actual \ predicted | rock | paper | none | middle | thumb |
|---|---:|---:|---:|---:|---:|
| rock | 12 | 0 | 0 | 0 | 3 |
| paper | 0 | 15 | 0 | 0 | 0 |
| none | 0 | 0 | 16 | 0 | 0 |
| middle | 0 | 5 | 0 | 10 | 0 |
| thumb | 5 | 0 | 0 | 0 | 10 |

### 150 ms · RF

| actual \ predicted | rock | paper | none | middle | thumb |
|---|---:|---:|---:|---:|---:|
| rock | 11 | 0 | 0 | 0 | 4 |
| paper | 0 | 14 | 0 | 1 | 0 |
| none | 0 | 0 | 16 | 0 | 0 |
| middle | 0 | 3 | 0 | 12 | 0 |
| thumb | 4 | 0 | 0 | 0 | 11 |

### 200 ms · DT

| actual \ predicted | rock | paper | none | middle | thumb |
|---|---:|---:|---:|---:|---:|
| rock | 8 | 0 | 0 | 0 | 3 |
| paper | 0 | 11 | 0 | 0 | 0 |
| none | 0 | 0 | 12 | 0 | 0 |
| middle | 0 | 3 | 0 | 8 | 0 |
| thumb | 0 | 0 | 0 | 0 | 12 |

### 200 ms · RF

| actual \ predicted | rock | paper | none | middle | thumb |
|---|---:|---:|---:|---:|---:|
| rock | 8 | 0 | 0 | 0 | 3 |
| paper | 0 | 11 | 0 | 0 | 0 |
| none | 0 | 0 | 12 | 0 | 0 |
| middle | 0 | 5 | 0 | 6 | 0 |
| thumb | 3 | 0 | 0 | 0 | 9 |

## 해석 시 주의사항

- stride가 짧을수록 같은 원시 기록에서 서로 겹치는 창이 더 많이 생성됩니다. 창 수 증가는 새로운 독립 데이터가 늘어난 것이 아닙니다.
- 기존 방식처럼 창을 만든 뒤 80/20으로 분할했기 때문에, 50/100/150 ms 조건에서는 분할 경계의 마지막 학습 창과 첫 테스트 창이 일부 원시 샘플을 공유할 수 있습니다. 200 ms 조건은 창이 겹치지 않습니다.
- stride마다 테스트 창 개수가 다르므로 정확도의 1개 오차가 차지하는 비율도 다릅니다. 모델 간 직접 비교는 같은 stride 안의 DT/RF 결과가 가장 타당합니다.
- McNemar 값은 겹치는 창들의 독립성이 보장되지 않으므로 참고용 paired 지표로만 해석해야 합니다.
- 한 수집 세션의 고정 holdout 결과입니다. 최종 선정에는 별도 세션이나 사용자 단위 분리 평가가 필요합니다.

## 생성 파일

- `results.json`: 설정과 전체 세부 결과
- `summary.csv`: stride·모델별 핵심 지표
- `paired_comparison.csv`: 동일 테스트 창의 DT/RF 직접 비교
- `predictions_*ms.csv`: 테스트 창별 실제값과 두 모델 예측
- `confusion_*_*.csv`: 모델·stride별 혼동행렬
