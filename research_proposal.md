# 연구 제안서: Hypercone 기반 텍스트 분류와 관련 없는 문장 거르기

> 최종 수정: 예비 실험 E0~E6 결과 반영. 실험 상세는 `EXPERIMENTS.md`, 교수님용 요약은 `brief_for_professor.md`.

## 한 줄 요약
HyperSphereDiff(ICML 2025)의 hypercone을 문장 임베딩에 적용해서, 텍스트를 정해진 의도로 분류하면서 학습하지 않은 관련 없는 문장은 걸러내는 방법을 제안한다. 각도 원뿔은 흔히 쓰는 거르기 방법보다 일관되게 좋고, 가장 강한 방법들(Mahalanobis, kNN)과 같은 수준의 성능을 클래스당 방향 하나와 반지름 하나만으로, 기준선 보정 없이 낸다.

---

## 1. 연구 목표
문장 임베딩을 단위 구 위에 놓고 클래스마다 원뿔(hypercone)을 정의해서, **학습한 의도는 정확히 분류하고 관련 없는 입력은 거부하는 open-set 텍스트 분류 방법**을 만든다. 관련 없는 입력 데이터는 학습에도 설정 선택에도 쓰지 않는다.

## 2. 연구 배경

### 2.1 문제
- softmax 분류기는 관련 없는 입력이 들어와도 학습한 클래스 중 하나로 억지로 분류한다. 예: 은행 챗봇이 "피자 주문해줘"를 "송금"으로 분류
- 그래서 확신도가 낮으면 거부하는 장치를 따로 붙인다. 사용자가 할 엉뚱한 말을 미리 다 모을 수는 없으므로, 학습한 클래스의 데이터만으로 거부 기준을 정해야 한다.

### 2.2 기존 방법
| 가족 | 대표 방법 | 점수를 계산하는 곳 |
|---|---|---|
| 확신도 | MSP (ICLR 2017), Energy (NeurIPS 2020) | 분류기 출력(확률, logit) |
| 거리·밀도 | Mahalanobis (NeurIPS 2018), kNN (ICML 2022), LOF | 중간 feature 공간의 거리 |
| 경계 학습 | ADB (AAAI 2021), DA-ADB (2022) | 클래스별 유클리드 공 |
| 구·각도 | LMCL+LOF (ACL 2019), CIDER (ICLR 2023), HACk-OOD (2025) | 정규화 feature의 각도 |
| 가상·외부 이상치 | Outlier Exposure (ICLR 2019), VOS (ICLR 2022), NPOS (ICLR 2023), DCLOOS (ACL 2021) | "모르는 것" 예시를 만들어 학습 |
| 출발점 | HyperSphereDiff (ICML 2025) | hypercone과 (truncated) vMF로 클래스 구조를 보존하는 이미지 생성 |

### 2.3 공백
텍스트 open-set 분류에서 **클래스별 각도 원뿔을 결정 경계로 직접 쓰고**, 원뿔 반지름 자체를 거부 기준으로 쓰는 방법은 찾지 못했다. HyperSphereDiff의 원뿔·vMF 기하를 생성이 아닌 분류·거르기에 쓴 사례도 찾지 못했다.

---

## 3. 제안 방법

### 3.1 기본: Hypercone 분류기
1. 문장 인코더(`all-mpnet-base-v2`, 고정)로 768차원 단위 벡터를 얻는다.
2. 작은 MLP(768 → 512 → 128)를 지나 정규화해서 구 위의 점 z를 얻는다. MLP는 cosine softmax(`softmax(16·cos(z, μ_k))`)로 학습한다.
3. 클래스 k마다 원뿔을 둔다. 중심 μ_k는 학습한 prototype, 반지름 θ_k는 학습 문장 각도의 95% 분위수.
4. 판정: `m_k = θ_k − ∠(z, μ_k)`. `max_k m_k ≥ 0`이면 그 의도로 분류하고, 모두 0보다 작으면 거부한다. 반지름이 곧 기준이라 기준선 보정이 필요 없다.

### 3.2 확장: 원뿔 밖 tvMF 가상 이상치
- 각 원뿔 경계 바로 밖 띠 `[r_k, r_k + δ]`로 잘라낸 vMF(truncated vMF)에서 가상 이상치를 뽑아, 원뿔 반지름을 학습한다. κ_k는 `A_d(κ_k) = cos r_k`로 계산한다.
- HyperSphereDiff는 tvMF로 생성 샘플을 원뿔 **안에** 가뒀고, 여기서는 잘라내는 영역을 원뿔 **밖** 띠로 뒤집어 쓴다.
- 손실: `L = L_CE + α·relu(∠(z, μ_y) − θ_y) + β·relu(max_k θ_k − ∠(o, μ_k))`
- 이상치를 MLP 뒤(feature 공간, E1)와 MLP 앞(입력 구, E2·E3) 두 위치에서 시험했다.

### 3.3 HyperSphereDiff에서 가져온 것
| 요소 | 코드 | 역할 |
|---|---|---|
| hypercone 정의 | `hsdiff/metrics.py` `Hypercones`, `class_angles` | 원뿔 반지름(95% 분위수), 설명용 그림 |
| vMF 샘플러, tvMF | `hsdiff/sphere.py` `sample_vmf`, `sample_tvmf`, `sample_tvmf_cone` | 가상 이상치 생성 |
| κ–각도 관계 | `hsdiff/sphere.py` `bessel_ratio`, `inverse_bessel_ratio` | 반지름에서 κ 계산 |

---

## 4. 예비 실험 결과 요약
데이터: CLINC150, BANKING77. 클래스의 25/50/75%만 학습, seed 3개 → 18개 설정. 모든 방법이 같은 인코더와 같은 크기의 MLP를 쓴다.

| 방법 (18개 설정 평균, %) | AUROC ↑ | FPR95 ↓ | F1 ↑ |
|---|---|---|---|
| MSP | 89.77 | 48.11 | 76.34 |
| Energy | 89.01 | 43.36 | 76.23 |
| kNN | 91.45 | 42.77 | 76.74 |
| ADB (재구현) | 88.00 | 53.06 | 74.53 |
| Mahalanobis | 92.62 | 35.35 | 81.19 |
| **Hypercone** | **93.39** | 32.80 | 81.00 (보정 없음 83.24) |
| Hypercone + tvMF 이상치 (입력 공간) | 93.12 | **31.35** | 81.49 (보정 없음 **84.07**) |

F1은 validation 기준선 보정 판정, 괄호는 원뿔 경계 그대로 판정.

**같은 cosine 신경망 위에서 판정 방식만 비교 (E6)**: cosine MSP 92.28, Mahalanobis 92.18, kNN 93.46, 원뿔 93.39 (AUROC).

**확인된 것**
1. 원뿔은 MSP·Energy보다 18/18, kNN·ADB보다 17/18 설정에서 AUROC가 높다.
2. Mahalanobis와는 비슷하다(AUROC +0.8%p, 13/18). 같은 신경망에서 비교하면 원뿔이 앞선다(+1.2%p, 14/18). 학습 방식만의 효과가 아니다.
3. MSP 대비 이득(+3.6%p)은 cosine 학습(+2.5)과 원뿔 판정(+1.1)이 나눠 만든다.
4. 같은 feature의 kNN과는 동률이다. kNN은 학습 문장 전체를 저장·탐색하고, 원뿔은 클래스당 방향·반지름만 쓴다.
5. tvMF 이상치로 학습한 원뿔은 기준선 보정 없이 F1 최고(84.07). feature 공간 Gaussian 이상치는 보정 없이 76.9로 무너지고 tvMF는 82.0으로 안정적이다.

**확인되지 않은 것**
- 가상 이상치의 순위 지표(AUROC) 이득은 ±0.5%p 이내로 작다. tvMF가 Gaussian보다 낫다는 근거도 순위 지표에서는 없다.

---

## 5. Novelty (결과 반영 후)
1. **텍스트 open-set 분류에 클래스별 각도 원뿔을 결정 경계로 적용.** 흔히 쓰는 방법보다 일관되게 좋고, 가장 강한 방법들과 같은 성능을 훨씬 적은 저장량(클래스당 방향·반지름)으로 낸다.
2. **반지름 자체가 거부 기준.** 다른 방법들은 validation으로 기준선을 맞춰야 하지만, 원뿔은 보정 없이 판정할 수 있고 각도로 판정 이유를 설명할 수 있다.
3. **(확장) 원뿔 밖 띠로 잘라낸 vMF 가상 이상치.** HyperSphereDiff의 tvMF를 뒤집어 쓴 것으로, 기준선 보정 없이 판정할 때 안정적이다. 순위 지표 이득은 작아서 보조 기여로 둔다.

### novelty로 주장하지 않는 것
- 텍스트를 구·vMF로 다루기, open-set intent 분류, 각도 margin, 가상 이상치로 거부 학습하기 자체
- "hypersphere가 있어야 거를 수 있다", "원뿔이 모든 방법보다 크게 낫다"

---

## 6. 한계
1. Mahalanobis와의 차이가 작고, 같은 feature의 kNN과는 동률이다.
2. 보정 없이 판정하면 관련 없는 문장을 많이 거르는 대신 학습한 문장도 더 많이 거부한다(CLINC150 75% seed 0: 잘못 거부 14.5% vs MSP 4.7%).
3. 비슷하지만 학습하지 않은 의도(near-OOD, 예: `interest_rate` vs `apr`)는 원뿔 안에 떨어진다.
4. 원뿔은 둥근 모양만 표현한다. 길쭉한 분포는 Mahalanobis가 유리할 수 있다.
5. 임베딩 모델을 고정했다. ADB는 재구현이라 원 논문 수치와 직접 비교할 수 없다.
6. 입력 공간 tvMF의 띠 폭(δ=0.15)은 튜닝에서 FPR95 기준으로 골랐다(원래 기준은 AUROC).

---

## 7. 남은 일
| 시점 | 할 일 |
|---|---|
| 경진대회 전 | 속도·메모리 측정(가벼움의 근거), 기준선 민감도(보정 불필요의 근거), 짝지은 통계 검정, 논문 초안 |
| 학회 논문 전 | StackOverflow·KLUE-YNAT 추가, 다른 임베딩 모델, q·δ 민감도, LLM 직접 판정 비교, 원뿔 겹 위치(HDS)로 신뢰도·오분류 분석, 임베딩 fine-tuning, ADB·DA-ADB 공식 코드 비교 |
| 확장 | 원뿔을 조건부 데이터 생성의 제약으로 사용 (생성 후 필터링 → 난이도별 생성 → 생성 중 tvMF 제약) |

---

## 참고 문헌

### 출발점
- Dosi et al., *Harmonizing Geometry and Uncertainty: Diffusion with Hyperspheres*, ICML 2025 — https://arxiv.org/abs/2506.10576

### 거르기 (OOD / open-set) 기본 방법
- Hendrycks & Gimpel, *A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks* (MSP), ICLR 2017 — https://arxiv.org/abs/1610.02136
- Liu et al., *Energy-based Out-of-distribution Detection*, NeurIPS 2020 — https://arxiv.org/abs/2010.03759
- Lee et al., *A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks* (Mahalanobis), NeurIPS 2018 — https://arxiv.org/abs/1807.03888
- Sun et al., *Out-of-Distribution Detection with Deep Nearest Neighbors* (kNN), ICML 2022 — https://arxiv.org/abs/2204.06507

### Open intent / 텍스트
- Zhang et al., *Deep Open Intent Classification with Adaptive Decision Boundary* (ADB), AAAI 2021 — https://arxiv.org/abs/2012.10209
- Zhang et al., *Learning Discriminative Representations and Decision Boundaries for Open Intent Detection* (DA-ADB), 2022 — https://arxiv.org/abs/2203.05823
- Lin & Xu, *Deep Unknown Intent Detection with Margin Loss*, ACL 2019 — https://aclanthology.org/P19-1548/
- Zhan et al., *Out-of-Scope Intent Detection with Self-Supervision and Discriminative Training* (DCLOOS), ACL 2021 — https://aclanthology.org/2021.acl-long.273
- Larson et al., *An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction* (CLINC150), EMNLP 2019
- Casanueva et al., *Efficient Intent Detection with Dual Sentence Encoders* (BANKING77), NLP4ConvAI 2020

### 가상 이상치 / 구 위 OOD
- Lee et al., *Training Confidence-calibrated Classifiers for Detecting Out-of-Distribution Samples*, ICLR 2018 — https://research.google/pubs/training-confidence-calibrated-classifiers-for-detecting-out-of-distribution-samples/
- Hendrycks et al., *Deep Anomaly Detection with Outlier Exposure*, ICLR 2019 — https://arxiv.org/abs/1812.04606
- Du et al., *VOS: Learning What You Don't Know by Virtual Outlier Synthesis*, ICLR 2022 — https://arxiv.org/abs/2202.01197
- *Non-Parametric Outlier Synthesis* (NPOS), ICLR 2023 — https://arxiv.org/abs/2303.02966
- Ming et al., *How to Exploit Hyperspherical Embeddings for Out-of-Distribution Detection?* (CIDER), ICLR 2023 — https://arxiv.org/abs/2203.04450
- Vapsi et al., *Hypercone Assisted Contour Generation for Out-of-Distribution Detection* (HACk-OOD), 2025 — https://arxiv.org/abs/2501.10209
- Zhang et al., *MARGIN: Margin-Aware Regularized Geometry for Imbalanced Vulnerability Detection*, 2026 (arXiv) — https://arxiv.org/abs/2605.10240

### 텍스트 + 구 / vMF
- Banerjee et al., *Clustering on the Unit Hypersphere using von Mises-Fisher Distributions*, JMLR 2005 — https://jmlr.org/papers/v6/banerjee05a.html
- Xu & Durrett, *Spherical Latent Spaces for Stable Variational Autoencoders*, EMNLP 2018 — https://arxiv.org/abs/1808.10805
- Deschenaux & Gulcehre, *Language Modeling with Hyperspherical Flows* (S-FLM), 2026 (arXiv) — https://arxiv.org/abs/2605.11125
- Wang & Isola, *Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere*, ICML 2020 — https://proceedings.mlr.press/v119/wang20k.html
- Rajaee & Pilehvar, *A Cluster-based Approach for Improving Isotropy in Contextual Embedding Space*, ACL 2021 — https://arxiv.org/abs/2106.01183

2026년 arXiv 논문(S-FLM, MARGIN)은 검색 요약만 확인했다. 인용 전에 원문 확인 필요.
