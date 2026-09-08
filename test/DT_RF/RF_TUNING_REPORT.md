# Random Forest 설정 탐색 결과

## 실험 설계

- 데이터·특성: 기존 `learn.py`와 동일한 200 ms 창 및 15개 특성
- stride: 50/100/150/200 ms
- 최종 분할: 클래스별 시간순 앞 75% 학습, 뒤 25% 테스트
- 설정 선택: 75% 학습 구간 내부를 다시 시간순 75/25 적합·검증으로 분할
- 후보: baseline 1개와 고정 시드 무작위 조합 31개, 총 32개
- 선택 기준: 검증 macro F1, 동률이면 검증 정확도
- 최종 테스트셋은 설정 선택에 사용하지 않음
- DT reference: `DecisionTreeClassifier(max_depth=5, random_state=42)`

### 탐색 범위

| 설정 | 후보값 |
|---|---|
| `n_estimators` | `100`, `300`, `500` |
| `criterion` | `gini`, `entropy` |
| `max_depth` | `None`, `3`, `5`, `8`, `12` |
| `max_features` | `sqrt`, `log2`, `0.5`, `1.0` |
| `min_samples_split` | `2`, `5`, `10` |
| `min_samples_leaf` | `1`, `2`, `4` |
| `bootstrap` | `True`, `False` |
| `class_weight` | `None`, `balanced` |

## stride별 선택 설정

| stride | 후보 | 검증 Macro F1 | n_estimators | criterion | max_depth | max_features | min_split | min_leaf | bootstrap | class_weight |
|---:|---:|---:|---:|---|---:|---|---:|---:|---|---|
| 50 ms | 28 | 100.00% | 500 | gini | 5 | 1.0 | 10 | 2 | True | None |
| 100 ms | 10 | 100.00% | 100 | gini | None | 0.5 | 10 | 4 | False | None |
| 150 ms | 1 | 98.25% | 500 | gini | 12 | log2 | 2 | 1 | True | None |
| 200 ms | 11 | 100.00% | 100 | gini | None | 1.0 | 2 | 1 | True | balanced |

### 전체 stride 공통 설정

네 stride의 검증 macro F1 평균이 가장 높은 후보는 **11번**입니다.

`n_estimators=100, criterion=gini, max_depth=None, max_features=1.0, min_samples_split=2, min_samples_leaf=1, bootstrap=True, class_weight=balanced`

평균 검증 macro F1은 99.09%, 가장 낮은 stride의 검증 macro F1은 98.18%입니다.

## 최종 테스트 결과

| stride | 학습/테스트 창 | DT reference | RF baseline | RF stride-tuned | 변화 | RF global-tuned |
|---:|---:|---:|---:|---:|---:|---:|
| 50 ms | 662/224 | 87.50% | 85.27% | 89.29% | +4.02%p | 88.84% |
| 100 ms | 331/113 | 92.04% | 82.30% | 89.38% | +7.08%p | 88.50% |
| 150 ms | 220/76 | 82.89% | 84.21% | 89.47% | +5.26%p | 89.47% |
| 200 ms | 167/57 | 89.47% | 80.70% | 87.72% | +7.02%p | 87.72% |

### Macro F1

| stride | DT reference | RF baseline | RF stride-tuned | 변화 | RF global-tuned |
|---:|---:|---:|---:|---:|---:|
| 50 ms | 87.29% | 85.08% | 89.17% | +4.10%p | 88.74% |
| 100 ms | 91.91% | 81.89% | 89.23% | +7.34%p | 88.38% |
| 150 ms | 82.43% | 83.98% | 89.32% | +5.35%p | 89.29% |
| 200 ms | 89.06% | 79.96% | 87.72% | +7.76%p | 87.72% |

## 클래스별 F1

| stride | 모델 | rock | paper | none | middle | thumb |
|---:|---|---:|---:|---:|---:|---:|
| 50 ms | DT reference | 71.60% | 93.18% | 100.00% | 94.12% | 77.55% |
| 50 ms | RF baseline | 71.43% | 90.53% | 100.00% | 89.74% | 73.68% |
| 50 ms | RF stride-tuned | 74.07% | 96.63% | 100.00% | 97.62% | 77.55% |
| 50 ms | RF global-tuned | 73.17% | 96.63% | 100.00% | 97.62% | 76.29% |
| 100 ms | DT reference | 82.93% | 95.65% | 100.00% | 95.24% | 85.71% |
| 100 ms | RF baseline | 69.77% | 86.27% | 100.00% | 81.08% | 72.34% |
| 100 ms | RF stride-tuned | 73.17% | 97.78% | 100.00% | 97.67% | 77.55% |
| 100 ms | RF global-tuned | 71.43% | 97.78% | 100.00% | 97.67% | 75.00% |
| 150 ms | DT reference | 75.00% | 85.71% | 100.00% | 80.00% | 71.43% |
| 150 ms | RF baseline | 73.33% | 87.50% | 100.00% | 85.71% | 73.33% |
| 150 ms | RF stride-tuned | 77.42% | 96.77% | 100.00% | 96.55% | 75.86% |
| 150 ms | RF global-tuned | 75.00% | 100.00% | 100.00% | 100.00% | 71.43% |
| 200 ms | DT reference | 84.21% | 88.00% | 100.00% | 84.21% | 88.89% |
| 200 ms | RF baseline | 72.73% | 81.48% | 100.00% | 70.59% | 75.00% |
| 200 ms | RF stride-tuned | 72.73% | 95.65% | 100.00% | 95.24% | 75.00% |
| 200 ms | RF global-tuned | 72.73% | 95.65% | 100.00% | 95.24% | 75.00% |

## 해석 및 주의사항

- baseline 대비 가장 큰 테스트 정확도 변화는 100 ms의 +7.08%p입니다.
- stride별 최적 설정은 내부 검증셋 기준으로 선택했습니다. 최종 테스트 성능이 항상 baseline보다 좋아진다는 보장은 없으며, 나빠진 경우도 그대로 기록합니다.
- 후보 수가 32개이므로 탐색하지 않은 조합이 많습니다. 이 결과는 해당 탐색 공간의 제한된 무작위 탐색 결과입니다.
- 짧은 stride의 겹치는 창은 독립 표본이 아니며, 설정 탐색 과정에서도 같은 수집 세션의 시간적 변화에 과적합될 수 있습니다.
- 실제 모델 선정 전에는 별도 수집 세션이나 사용자 단위의 외부 검증이 필요합니다.
- RF의 트리 수·깊이·특성 수 증가는 임베디드 메모리와 추론 비용을 늘릴 수 있으므로 정확도와 별도로 배포 비용을 확인해야 합니다.

## 생성 파일

- `rf_validation_candidates.csv`: 4개 stride × 32개 후보의 내부 검증 결과
- `rf_final_summary.csv`: DT·baseline RF·stride별 튜닝 RF·공통 튜닝 RF의 최종 테스트 결과
- `rf_tuning_results.json`: 설정, 세부 지표, 혼동행렬, 선택 정보
- `rf_tuning_predictions_*ms.csv`: 최종 테스트 창별 네 모델 예측
