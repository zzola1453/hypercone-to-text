# ITS: Hypercone 기반 텍스트 분류 + 관련 없는 문장 거르기

HyperSphereDiff의 hypercone과 vMF를 이용해서, 텍스트를 정해진 의도(intent)로 분류하면서
학습하지 않은 관련 없는 문장은 거르는 방법을 실험한다. 연구 배경, novelty, 계획은 `research_proposal.md`.

## 아이디어 요약
1. 고정된 문장 임베딩(`all-mpnet-base-v2`)을 MLP에 통과시킨 뒤 정규화해서 단위 구 위에 놓는다.
2. 클래스마다 원뿔(중심 μ_k, 반지름 θ_k)을 둔다. 어떤 원뿔 안이면 그 클래스, 어느 원뿔에도 없으면 거부한다.
3. (확장) 관련 없는 문장 데이터 없이, 각 원뿔 경계 바로 밖 띠 `[r_k, r_k + δ]`로 잘라낸 vMF
   (`hsdiff.sphere.sample_tvmf`)에서 가상 이상치를 뽑아 원뿔 반지름을 학습한다.
   κ는 원뿔 반지름에서 `A_d(κ) = cos r_k`로 계산한다. 효과가 작아 보조 기여로 둔다.

## 파일
| 파일 | 내용 |
|---|---|
| `data.py` | CLINC150(`clinc/clinc_oos`, plus), BANKING77(`mteb/banking77`, train의 10%를 val로 사용), ADB식 known/unknown 분할 |
| `embed.py` | 문장 임베딩 추출과 캐시(`its/cache/`) |
| `methods.py` | MSP, ADB(재구현), hypercone 분류기와 가상 이상치(tvMF / vMF / Gaussian) |
| `metrics.py` | Acc, F1(all/known/unknown), AUROC, FPR95 |
| `run.py` | 전체 실험 실행, 결과는 `its/results/<tag>.json`, `<tag>.md` |
| `tune.py` | validation만 쓰는 하이퍼파라미터 튜닝 (가짜 unknown 클래스) |
| `plot_sphere.py` | 설명용 그림 `figures/sphere_embeddings.png` (3차원 모델의 실제 출력) |
| `plot_results.py` | 결과 그림 `figures/results_main.png`, `figures/results_ablation.png` |
| `examples.py` | 실제 문장 판정 사례 `results/examples.md` (MSP vs Hypercone+tvMF) |
| `DATA.md` | 데이터셋 설명 (출처, 의도, 크기, 예시 문장, known/unknown 분할) |
| `EXPERIMENTS.md` | 실험별 목적 · 설정 · 결과 · 결론 기록 |
| `brief_for_professor.md` | 교수님께 드리는 연구 요약 |

## 실행
```
python -m its.run --datasets clinc banking --ratios 0.25 0.5 0.75 --seeds 0 1 2 --tag main
```

## 비교 방법
| 이름 | 내용 |
|---|---|
| `msp` | softmax 분류기, 최대 확률이 validation 기준(known 95% 통과)보다 낮으면 거부 |
| `energy` | logit의 logsumexp가 낮으면 거부 (Liu et al., 2020) |
| `maha` | 클래스 평균·공유 공분산 기준 Mahalanobis 거리가 멀면 거부 (Lee et al., 2018) |
| `knn` | 정규화 feature에서 k번째(10) 이웃까지 거리가 멀면 거부 (Sun et al., 2022) |
| `cq_msp`, `cq_maha`, `cq_maha_raw`, `cq_knn` | cone_q와 같은 cosine softmax 신경망 위의 MSP / Mahalanobis / kNN (판정 방식만 비교) |
| `adb` | ADB loss로 클래스별 유클리드 공 반지름 학습 (같은 임베딩/MLP 위 재구현, 원 논문 BERT 파이프라인 아님) |
| `cone_q` | 이상치 없이 cosine-softmax로 학습, 원뿔 반지름 = 학습 각도의 95% 분위수 (HyperSphereDiff의 HCR 원뿔) |
| `cone_gauss` | VOS식 Gaussian 저확률 가상 이상치 |
| `cone_vmf` | 자르지 않은 vMF 가상 이상치 |
| `cone_tvmf` | **제안 방법**: 원뿔 밖 띠로 잘라낸 vMF 가상 이상치 |

`cone_*`는 원뿔 기준 그대로 판정한 `_raw` 행과, MSP와 같은 규칙(validation known 95% 통과)으로
전역 기준값 하나만 맞춘 행을 함께 기록한다. 어느 쪽도 unknown 데이터는 쓰지 않는다.

## 결과 (18개 설정 = 2 데이터셋 × known 25/50/75% × seed 3)
해석은 `brief_for_professor.md`, 실험별 질문과 결론은 `EXPERIMENTS.md`(E0~E6). 전체 표:
`results/main.md`(메인, feature 공간 이상치), `results/input.md`(입력 공간 이상치, `--space input --delta 0.15`),
`results/baselines.md`(Energy·Mahalanobis·kNN), `results/cosine_rules.md`(같은 cosine 신경망에서 판정 방식만 비교),
튜닝 `results/tune.json`.

| 방법 | 거부 기준 | AUROC | FPR95 | F1(all) |
|---|---|---|---|---|
| msp | validation 보정 | 89.77 | 48.11 | 76.34 |
| energy | validation 보정 | 89.01 | 43.36 | 76.23 |
| knn | validation 보정 | 91.45 | 42.77 | 76.74 |
| adb | validation 보정 | 88.00 | 53.06 | 74.53 |
| maha | validation 보정 | 92.62 | 35.35 | 81.19 |
| cone_q | validation 보정 | 93.39 | 32.80 | 81.00 |
| cone_q_raw | 원뿔 경계 그대로 | 93.39 | 32.80 | 83.24 |
| cone_tvmf (입력 공간), raw | 원뿔 경계 그대로 | 93.12 | 31.35 | 84.07 |

같은 cosine 신경망 위(E6, AUROC): `cq_msp` 92.28 · `cq_maha` 92.18 · `cq_knn` 93.46 · `cone_q` 93.39.

튜닝(`python -m its.tune`)은 known 클래스의 25%를 가짜 unknown으로 떼어낸 validation으로만 한다.

## 주의
- 임베딩 모델은 학습하지 않는다(고정). 원 논문들의 BERT fine-tuning 결과와 숫자를 직접 비교할 수 없다.
- `mteb/banking77`은 원본(10,003 / 3,080)과 샘플 수가 약간 다르다(9,993 / 3,076).
