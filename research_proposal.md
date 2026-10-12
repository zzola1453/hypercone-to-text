# 연구 제안서: Hypercone 기반 Open-Set 텍스트 분류

## 한 줄 요약
문장 임베딩을 단위 구(hypersphere) 위에 놓고 클래스마다 원뿔(hypercone) 영역을 정의해서, **학습한 클래스의 문장은 정확히 분류하고 관련 없는 문장은 거부하는 텍스트 분류기**를 만든다. 관련 없는 문장 데이터를 따로 모으지 않고, **각 원뿔 경계 바로 바깥 띠로 잘라낸 vMF 분포**에서 가상 이상치를 뽑아 거부 경계를 학습한다.

---

## 1. 연구 목표
문장 임베딩 공간에서 클래스별 hypercone과 원뿔에 맞춘 vMF 가상 이상치를 이용해, **실제 unknown 데이터 없이** 알려진 의도는 정확히 분류하고 관련 없는 입력은 거부하는 **open-set 텍스트 분류 방법**을 제안한다.

### 성공 기준
1. CLINC150, BANKING77, StackOverflow의 25/50/75% known 설정에서 **Unknown F1과 AUROC가 기존 방법(ADB, DA-ADB, DCLOOS, 텍스트에 적용한 VOS)보다 높다.**
2. **Known 정확도는 기존 방법 대비 1%p 이내로 유지된다.**
3. **원뿔에 맞춘 vMF 가상 이상치가 Gaussian 가상 이상치보다 낫다** (ablation).
4. 원뿔 반지름에서 계산한 κ가 **튜닝한 κ와 비슷한 성능**을 낸다 (튜닝이 필요 없다는 근거).

---

## 2. 연구 배경

### 2.1 문제
- 일반 분류기는 관련 없는 입력이 들어와도 학습한 클래스 중 하나로 억지로 분류한다.
  - 예: 은행 챗봇이 "피자 주문해줘"를 "송금"으로 분류함
- 관련 없는 입력을 거부하도록 학습하려면 그런 예시가 필요하지만, 사용자가 할 수 있는 모든 엉뚱한 말을 미리 모을 수는 없다.

### 2.2 기존 연구
| 분류 | 연구 | 내용 | 한계 / 우리와의 차이 |
|---|---|---|---|
| Open intent 경계 | ADB (AAAI 2021), DA-ADB (2022) | 클래스별 유클리드 공(ball) 경계를 학습 | 각도 기하를 쓰지 않음 |
| 각도 margin | LMCL + LOF (ACL 2019) | cosine margin loss로 학습한 뒤 LOF로 unknown 탐지 | 클래스별 영역을 명시적으로 정의하지 않음 |
| 가상 이상치 (이미지) | Lee et al. (ICLR 2018), VOS (ICLR 2022), NPOS (ICLR 2023) | 경계 근처에 가짜 샘플·점을 만들어 거부 학습 | 유클리드 공간, Gaussian 가정이나 임계값 튜닝 필요, 이미지 대상 |
| 가상 이상치 (텍스트) | DCLOOS (ACL 2021) | inlier feature로 pseudo outlier를 만들어 out-of-scope intent 탐지 | 구·원뿔 기하를 쓰지 않음 |
| 실제 외부 이상치 | Outlier Exposure (ICLR 2019) | 다른 데이터셋을 "관련 없음"으로 사용 | 외부 데이터가 필요함 |
| 구 위 OOD | CIDER (ICLR 2023), HACk-OOD (2025) | 클래스를 구 위 vMF로 보거나 hypercone으로 OOD 탐지 | 이미지 대상 |
| 텍스트 + 구/vMF | Banerjee (JMLR 2005), Xu & Durrett (EMNLP 2018), S-FLM (2026), vMF DP 메커니즘 | 군집화, VAE, 생성, 개인정보 보호 | open-set 분류에 쓰지 않음 |
| 출발점 | HyperSphereDiff (ICML 2025) | hypercone과 vMF로 클래스 구조를 보존하는 생성, tvMF로 샘플을 원뿔 **안에** 가둠 | 생성 모델이며 분류·거부에는 쓰지 않음 |

### 2.3 공백
가상 이상치로 거부 경계를 학습하는 방식은 이미 있다. 그러나 **hypersphere 위에서 클래스별 원뿔 경계에 맞춰 vMF로 가상 이상치를 만들고, 그 위치를 원뿔 반지름으로 해석적으로 정하는 방법**은 찾지 못했다. 텍스트 open-set 분류에 클래스별 각도 원뿔 경계를 적용한 사례도 찾지 못했다.

---

## 3. 제안 방법

### 3.1 전체 흐름
1. **임베딩**: 문장 임베딩을 centering한 뒤 정규화해서 hypersphere 위에 놓는다.
2. **원뿔 정의**: 클래스 k마다 원뿔을 만든다.
   - 중심 μ_k: 클래스 평균 방향
   - 반지름 θ_k: 클래스 내 각도 분포의 분위수 (또는 학습)
3. **κ 계산**: 원뿔 반지름에서 vMF 집중도를 계산한다. `A_D(κ_k) = cos θ_k` (A_D는 Bessel 함수 비율)
4. **가상 이상치 생성**: 원뿔 경계 바로 바깥 띠 {θ_k < ∠(z, μ_k) ≤ θ_k + δ}로 잘라낸 vMF(**complement-truncated vMF**)에서 점을 뽑고 "unknown" 라벨을 붙인다.
5. **학습**: 원뿔 안의 진짜 샘플은 해당 클래스로, 경계 밖 가상 이상치는 거부로 학습한다.
6. **추론**: 입력이 어떤 원뿔 안에 있으면 그 클래스로 분류하고, 어느 원뿔에도 없으면 거부한다.

### 3.2 핵심 아이디어
- **고차원 vMF의 각도 집중**: 고차원에서는 vMF 샘플과 중심 사이의 각도가 거의 일정한 값(cos 각도 ≈ A_D(κ))에 몰린다. 그래서 κ 하나로 샘플이 떨어질 **각도 띠를 정밀하게 지정**할 수 있다.
- **원뿔에 맞춘 노이즈**: κ를 원뿔 반지름 θ_k에서 계산하므로, 넓은 원뿔은 멀리, 좁은 원뿔은 가깝게 클래스마다 자동으로 경계 밖에 가상 이상치가 배치된다. VOS처럼 임계값을 튜닝할 필요가 없다.
- **tvMF를 뒤집어 사용**: HyperSphereDiff는 tvMF로 샘플을 원뿔 **안에** 가두어 생성했다. 우리는 잘라내는 영역을 원뿔 **밖** 띠로 바꿔 가상 이상치를 만든다. 가상 이상치가 원뿔 안으로 잘못 들어가는 일이 원천적으로 없다.
- **원뿔 안쪽 증강 (선택)**: 데이터가 적은 클래스는 논문의 tvMF를 그대로 써서 원뿔 안쪽 샘플로 보강한다.

### 3.3 HyperSphereDiff에서 가져오는 것
| 요소 | 저장소 코드 | 역할 |
|---|---|---|
| hypercone 정의 | `hsdiff/metrics.py` `Hypercones.fit`, `class_angles` | 분류기의 본체 |
| vMF 샘플러 | `hsdiff/sphere.py` `sample_vmf` | 가상 이상치 생성 |
| κ–각도 관계 | `hsdiff/sphere.py` `bessel_ratio`, `inverse_bessel_ratio` | θ_k에서 κ_k 계산 |
| 구 위 연산 | `hsdiff/sphere.py` `project`, `angle`, `rotate_towards` 등 | 정규화, 각도 계산 |
| HCR / HDS | `hsdiff/metrics.py` `hcr`, `hds_per_class` | 원뿔 진단, 신뢰도 지표 |
| tvMF 개념 (Sec. 4.3) | 미구현 | 원뿔 밖 띠로 뒤집어 가상 이상치 생성에 사용 |

diffusion의 forward/reverse 과정, 시간 스케줄, UNet 등 생성 관련 부분은 사용하지 않는다.

---

## 4. Novelty
1. **(핵심) 원뿔 기하로 결정되는 가상 이상치**: 고차원 vMF의 각도 집중 성질과 원뿔 반지름으로부터 κ를 해석적으로 정하고, 원뿔 밖 띠로 잘라낸 vMF로 가상 이상치를 생성한다. 기존 가상 이상치 방법(VOS, NPOS, DCLOOS)과 달리 구·원뿔 기하에 기반하며 위치 튜닝이 필요 없다.
2. **텍스트 open-set 분류에 클래스별 각도 원뿔 경계를 적용**: ADB/DA-ADB의 유클리드 공 대신 클래스마다 반지름이 다른 각도 원뿔을 결정 경계로 쓴다.
3. **(보조) 원뿔 내부 위치를 신뢰도 지표로 사용**: HyperSphereDiff의 생성 평가 지표(HCR/HDS)를 분류기의 신뢰도와 난이도 진단으로 바꿔 쓴다.

### 이미 있는 것 (novelty로 주장하지 않음)
- 텍스트를 hypersphere/vMF로 다루기, open-set intent 분류, 각도 margin, 가상 이상치로 거부 학습하기 자체

---

## 5. 실험 계획

### 5.1 데이터와 설정
| 항목 | 내용 |
|---|---|
| 데이터 | CLINC150, BANKING77, StackOverflow (영어 open intent), KLUE-YNAT (한국어 추가 확인) |
| 설정 | 클래스의 25/50/75%만 학습에 쓰고 나머지를 unknown으로 둠 (ADB와 같은 프로토콜) |
| 분할 | 원뿔 학습은 train, 설정 선택은 val, 최종 결과는 test에서만 산출 |
| 반복 | seed 3~5개, 평균 ± 표준편차, paired t-test 또는 bootstrap 신뢰구간 |

### 5.2 지표
| 보여줄 것 | 지표 |
|---|---|
| 아는 내용을 잘 분류 | Accuracy, Macro-F1, Known F1 |
| 관련 없는 내용을 잘 거름 | Unknown F1, AUROC, FPR95 |
| 확신도를 믿을 수 있음 | ECE, risk–coverage 곡선 |
| 원뿔 진단 | 테스트 HCR, 겹별 정확도, 원뿔 겹침 비율 |

### 5.3 비교 대상
- softmax 임계값, LMCL + LOF, ADB, DA-ADB, DCLOOS
- **VOS를 텍스트 임베딩에 적용한 버전** (가상 이상치 방식의 직접 비교)
- 모든 방법을 **같은 문장 임베딩 모델** 위에서 비교해 임베딩 효과와 방법 효과를 분리

### 5.4 Ablation (novelty별 증명)
| 주장 | 실험 | 실패 시 |
|---|---|---|
| Novelty 1 | vMF vs Gaussian 가상 이상치, 계산한 κ vs 튜닝한 κ, 띠 잘라내기 있음 vs 없음, 띠 폭 δ | 핵심 주장이 사라짐 |
| Novelty 2 | 각도 원뿔 vs 유클리드 공, 같은 임베딩에서 비교 | "구 위에서 다시 했다" 수준으로 약해짐 |
| Novelty 3 | 원뿔 겹별 정확도, 오분류 예측 AUROC | 보조 분석으로 내림 |
| 기타 | 클래스당 원뿔 1개 vs 여러 개, centering 유무, θ_k 분위수 vs 학습, 임베딩 모델 교체 | — |

---

## 6. 예상 한계
1. **near-OOD**: 클래스와 비슷하지만 학습하지 않은 의도(예: 은행 챗봇의 "적금 가입")는 임베딩상 원뿔 안에 떨어질 수 있어 경계만으로는 거르기 어렵다.
2. **임베딩 품질이 상한선**: 임베딩 모델이 관련 없는 문장을 클래스 가까이에 놓으면 경계로 해결할 수 없다. 이방성(좁은 원뿔에 몰림)도 구분을 어렵게 한다.
3. **원뿔은 둥근 모양만 표현**: 길쭉한 클래스 분포에서는 빈 공간이 생긴다. 공분산을 쓰는 Gaussian(VOS)이 더 유연할 수 있다.
4. **클래스당 원뿔 하나 가정**: 하위 주제가 갈라진 클래스는 원뿔이 지나치게 넓어진다.
5. **비슷한 클래스끼리 원뿔 겹침**: 겹친 영역의 문장은 애매하고, 가상 이상치가 다른 클래스 영역을 침범할 수 있다.
6. **데이터가 적은 클래스에서 불안정**: θ_k와 그에 따른 κ, 가상 이상치 위치가 흔들린다.
7. **벤치마크와 실제의 차이**: 벤치마크의 unknown은 같은 도메인의 보류 클래스라서 실제 서비스의 관련 없는 입력과 분포가 다를 수 있다.

---

## 7. 진행 계획
| 단계 | 내용 | 판단 기준 |
|---|---|---|
| 1 | CLINC150에서 임베딩 추출, 원뿔 분류기와 평가 스크립트 구현 | 닫힌 집합 정확도가 cosine/logistic 분류기와 비슷한가 |
| 2 | complement-truncated vMF 가상 이상치 추가, ADB·VOS와 비교 | Unknown F1·AUROC에서 앞서는가 |
| 3 | 전체 데이터셋, 25/50/75% 설정, ablation | 성공 기준 1~4 충족 여부 |
| 4 | 논문 작성 | — |

2단계에서 거부 성능의 이점이 없으면, 원뿔 위치를 신뢰도·라벨 노이즈 진단에 쓰는 분석 연구로 방향을 조정한다.

---

## 8. 향후 연구: 조건부 생성의 제약 레이어
원뿔이 클래스를 잘 표현한다는 것이 확인되면, 같은 원뿔을 생성의 제약 조건으로 확장할 수 있다.
1. **생성 후 거르기**: LLM이 만든 학습 데이터 중 해당 클래스 원뿔 안에 들어간 것만 남긴다.
2. **난이도별 생성**: 원뿔의 겹(HDS 계층)으로 쉬운 샘플과 경계 근처의 어려운 샘플을 골라 생성한다.
3. **생성 중 제약**: 연속 공간 생성 모델에서 매 단계 원뿔 밖으로 나간 예측을 원뿔 안으로 되돌리거나(측지선 회전), 논문의 tvMF로 원뿔 안에서만 샘플링한다.

---

## 9. 확인 요청 사항
1. open intent 벤치마크로 충분한지, 한국어 데이터가 꼭 필요한지
2. 비교 대상으로 ADB, DA-ADB, DCLOOS, VOS면 되는지
3. 목표 학회나 저널이 있는지

---

## 참고 문헌

### 출발점
- Dosi et al., *Harmonizing Geometry and Uncertainty: Diffusion with Hyperspheres*, ICML 2025 — https://arxiv.org/abs/2506.10576

### Open intent / open-set 텍스트 분류
- Zhang et al., *Deep Open Intent Classification with Adaptive Decision Boundary* (ADB), AAAI 2021 — https://arxiv.org/abs/2012.10209
- Zhang et al., *Learning Discriminative Representations and Decision Boundaries for Open Intent Detection* (DA-ADB), 2022 — https://arxiv.org/abs/2203.05823
- Lin & Xu, *Deep Unknown Intent Detection with Margin Loss*, ACL 2019 — https://aclanthology.org/P19-1548/
- Zhan et al., *Out-of-Scope Intent Detection with Self-Supervision and Discriminative Training* (DCLOOS), ACL 2021 — https://aclanthology.org/2021.acl-long.273
- *Holistic Sentence Embeddings for Better Out-of-Distribution Detection*, Findings of EMNLP 2022 — https://arxiv.org/abs/2210.07485

### 가상 이상치 / OOD
- Lee et al., *Training Confidence-calibrated Classifiers for Detecting Out-of-Distribution Samples*, ICLR 2018 — https://research.google/pubs/training-confidence-calibrated-classifiers-for-detecting-out-of-distribution-samples/
- Hendrycks et al., *Deep Anomaly Detection with Outlier Exposure*, ICLR 2019 — https://arxiv.org/abs/1812.04606
- Du et al., *VOS: Learning What You Don't Know by Virtual Outlier Synthesis*, ICLR 2022 — https://arxiv.org/abs/2202.01197
- *Non-Parametric Outlier Synthesis* (NPOS), ICLR 2023 — https://arxiv.org/abs/2303.02966
- Ming et al., *How to Exploit Hyperspherical Embeddings for Out-of-Distribution Detection?* (CIDER), ICLR 2023 — https://arxiv.org/abs/2203.04450
- Vapsi et al., *Hypercone Assisted Contour Generation for Out-of-Distribution Detection* (HACk-OOD), 2025 — https://arxiv.org/abs/2501.10209

### 텍스트 + hypersphere / vMF
- Banerjee et al., *Clustering on the Unit Hypersphere using von Mises-Fisher Distributions*, JMLR 2005 — https://jmlr.org/papers/v6/banerjee05a.html
- Xu & Durrett, *Spherical Latent Spaces for Stable Variational Autoencoders*, EMNLP 2018 — https://arxiv.org/abs/1808.10805
- Deschenaux & Gulcehre, *Language Modeling with Hyperspherical Flows* (S-FLM), 2026 — https://arxiv.org/abs/2605.11125
- *A Metric Differential Privacy Mechanism for Sentence Embeddings* — https://dl.acm.org/doi/10.1145/3708321
- *STAMP: Selective Task-Aware Mechanism for Text Privacy*, 2026 — https://arxiv.org/abs/2603.12237

### 임베딩 기하
- Wang & Isola, *Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere*, ICML 2020 — https://proceedings.mlr.press/v119/wang20k.html
- Rajaee & Pilehvar, *A Cluster-based Approach for Improving Isotropy in Contextual Embedding Space*, ACL 2021 — https://arxiv.org/abs/2106.01183
