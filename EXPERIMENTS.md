# 실험 기록

실험마다 **왜 했는지(질문) → 어떻게 했는지 → 결과 → 결론과 다음 행동** 순서로 적는다.
공통 설정: 임베딩 `all-mpnet-base-v2` 고정, 그 위에 MLP(768 → 512 → 128), 30 epoch, Adam lr 1e-3.
평가는 ADB 프로토콜(known 클래스 25/50/75%, 나머지와 CLINC out-of-scope는 unknown).
지표: Acc, F1(all/known/unknown), AUROC, FPR95 (`metrics.py`).

---

## E0. 파이프라인 점검 (smoke test)
- **질문**: 데이터 로딩 → 임베딩 → 학습 → 평가가 끝까지 돌아가는가? 대략 어느 정도 숫자가 나오는가?
- **설정**: CLINC150, known 75%, seed 0, 모든 방법 1회 (`--tag smoke`)
- **결과**: 원뿔 계열이 MSP보다 좋았다(F1 unknown 65.5 → 79~82). 가상 이상치를 넣으면 FPR95는 좋아졌지만 known F1이 떨어졌다. ADB는 F1 59로 원 논문(약 85)보다 크게 낮았다.
- **결론 / 다음**: 두 가지 문제를 확인했다.
  1. 가상 이상치가 원뿔을 학습 데이터 기준으로 너무 좁게 만들어서 테스트의 known 문장까지 거부한다. → 공정 비교를 위해 validation 보정 규칙을 추가 (E1)
  2. ADB가 너무 낮다. → 원인 분석 (E1-a)

## E1-a. ADB가 낮은 원인 분석
- **질문**: ADB 수치가 낮은 게 구현 버그인가, 설정 차이인가?
- **설정**: CLINC150 75% seed 0에서 클래스 중심까지 거리 분포와 학습된 반지름 확인
- **결과**: 거리 중앙값 14.1, 95% 분위수 23.0. ADB 반지름으로 validation known의 **44%만 통과**했다.
- **결론**: ADB loss(`(d>r)(d−r) + (d≤r)(r−d)`)는 반지름을 **거리 중앙값**으로 수렴시킨다. 원 논문의 768차원 BERT feature에서는 거리 분포가 좁아 중앙값 ≈ 95% 분위수지만, 우리 128차원 feature에서는 차이가 커서 known의 절반을 거부한다. 구현은 원 논문 코드와 같은 loss이고 버그는 아니다. → ADB에도 validation 보정을 적용한 `adb` 행을 추가하고 원래 규칙은 `adb_raw`로 남김.

## E1. 메인 비교 (feature 공간 가상 이상치)
- **질문**:
  - Q1. hypercone 분류기가 기존 방법(MSP, ADB)보다 관련 없는 문장을 잘 거르는가?
  - Q2. 원뿔 밖 띠로 자른 vMF(tvMF) 가상 이상치가 이상치 없음, 자르지 않은 vMF, Gaussian(VOS식)보다 나은가?
- **설정**: 2 데이터셋 × known 25/50/75% × seed 3 = 18개 설정. 방법 `msp, adb, cone_q, cone_gauss, cone_vmf, cone_tvmf`. 이상치 하이퍼파라미터는 튜닝 전 기본값(β=1, δ=0.1, q=0.95). 원뿔 계열은 validation 보정 행과 `_raw`(원뿔 경계 그대로) 행을 둘 다 기록.
- **결과** (`results/main.md`):
  - Q1: 원뿔(cone_q)이 MSP 대비 18/18 설정에서 AUROC↑(평균 +3.6%p), FPR95↓(평균 −15.3%p), F1↑(+4.7%p). ADB 대비 17/18 승.
  - Q2: tvMF vs 이상치 없음 AUROC +0.2%p(13/18), tvMF vs Gaussian −0.1%p(7/18). 차이가 대부분 표준편차 안쪽.
  - 원뿔 경계 그대로(`_raw`) 판정에서는 feature 공간 Gaussian이 F1 76.9로 크게 무너지고 tvMF는 82.0으로 안정적.
- **결론 / 다음**: Q1은 강하게 지지. Q2는 지지되지 않음. 원인을 분석해 보니 feature 공간 이상치는 MLP를 통과하지 않아서 gradient가 원뿔 반지름에만 전달되고 표현(feature)을 바꾸지 못한다. → 이상치를 입력 임베딩 공간에서 만들어 MLP에 통과시키는 버전을 추가하고 튜닝 (E2)

## E2. 하이퍼파라미터 튜닝 (테스트셋 미사용)
- **질문**: 가상 이상치를 입력 임베딩 공간에서 만들거나, 손실 가중치·띠 폭·Outlier Exposure 손실을 바꾸면 효과가 생기는가? tvMF와 Gaussian을 같은 예산으로 튜닝하면 어느 쪽이 나은가?
- **설정** (`tune.py`): known 75%, seed 0·1, 2 데이터셋. known 클래스의 25%를 가짜 unknown으로 떼고 validation으로만 평가. 격자: 이상치 {tvMF, Gaussian} × 공간 {feature, input} × β {1, 3} × γ(OE) {0, 1} × δ {0.05, 0.15}(tvMF만) + 이상치 없음 = 25개 설정. 원래 선택 기준은 AUROC.
- **결과** (`results/tune.json`):
  - AUROC 1위 Gaussian feature(92.42)와 이상치 없음(92.29)의 차이는 0.1로 사실상 같음.
  - FPR95 1위는 입력 공간 tvMF(β=1, δ=0.15): 35.35 vs 이상치 없음 38.12.
  - OE 손실(γ=1)을 넣은 설정은 모두 AUROC 84~88로 크게 나빠짐.
- **결론 / 다음**: AUROC 기준으로는 고를 만한 이상치 설정이 없어서, **FPR95 1위 설정(입력 공간 tvMF, δ=0.15)과 같은 공간의 Gaussian을 테스트로 최종 평가**하기로 함. 선택 기준을 AUROC에서 FPR95로 바꾼 점은 논문에 명시해야 함. OE 손실은 제외.

## E3. 입력 공간 가상 이상치 최종 평가
- **질문**: 튜닝에서 고른 입력 공간 설정이 18개 테스트 설정 전체에서도 효과가 있는가?
- **설정**: `--methods cone_q cone_gauss cone_tvmf --space input --delta 0.15 --tag input`. 같은 seed의 cone_q가 E1과 완전히 같은 값인지 확인함(차이 0).
- **결과** (`results/input.md`):
  - validation 보정 판정: 입력 공간 tvMF vs 이상치 없음 AUROC −0.27%p(5/18), FPR95 −1.45%p 개선(10/18).
  - 원뿔 경계 그대로 판정: 입력 공간 tvMF F1 84.07로 전체 1위. 이상치 없음 83.24(+0.83%p, 11/18), 입력 공간 Gaussian 83.93(+0.14%p, 11/18), MSP 76.34(+7.7%p, 18/18).
- **결론**: 가상 이상치의 이점은 "거부 기준값을 따로 맞추지 않아도 된다"는 점에서만 일관되게 보이고, 순위 지표(AUROC)에서는 뚜렷하지 않다. → 논문의 메인은 hypercone 분류기, 가상 이상치는 보조로 정리.

## E4. 구 표면 시각화 (설명용 그림)
- **질문**: 문장 임베딩, 클래스 원뿔, 가상 이상치, 관련 없는 문장이 구 위에서 실제로 어떻게 놓이는가?
- **설정** (`plot_sphere.py` → `figures/sphere_embeddings.png`): 768차원을 3차원으로 압축하면 각도가 왜곡되므로, 모델의 출력 차원을 3으로 줄여 실제로 학습시킨 뒤 S² 위의 점을 그대로 그린다. CLINC150의 3개 의도(weather, play_music, balance)만 known으로 학습. 원뿔은 HyperSphereDiff 정의(클래스 평균 방향 + 학습 각도의 95% 분위수), 가상 이상치는 그 원뿔 밖 띠 [θ, θ+0.25]에서 tvMF로 생성(κ는 A₃(κ) = cos θ). (a) 구 전체, (b) play_music 원뿔을 중심 기준 실제 각도로 내려다본 확대도.
- **결과**: 원뿔 반지름 약 10~13°. 테스트에서 known 문장 92% 통과, 관련 없는 문장(250개 표본) 84% 거부.
- **주의**: 설명용 그림이다. 처음에는 학습한 prototype을 원뿔 중심으로 썼는데, 3차원·3클래스에서는 cosine CE가 prototype을 문장 무리 중심에 두지 않아 문장이 원뿔 가장자리에 몰려 보였다. 그래서 그림에서는 원 논문의 원뿔 정의를 썼다. 128차원 본 실험(학습한 원뿔)과 설정이 다르다.

## E5. hypersphere 없는 거르기 방법과 비교
- **질문**: "모르겠다"는 hypersphere 없이도 할 수 있다. 흔히 쓰는 다른 거르기 방법(Energy, Mahalanobis, kNN)보다도 원뿔이 나은가?
- **설정**: `--methods msp energy maha knn --tag baselines`. 네 방법 모두 MSP와 같은 softmax 신경망과 같은 128차원 feature를 쓰고, 거부 기준은 validation의 학습한 문장 95% 통과로 맞춤. MSP 결과가 E1과 완전히 같게 재현됨을 확인(차이 0). kNN의 k=10은 튜닝하지 않음.
- **결과** (`results/baselines.md`, 18개 설정 평균):
  | 방법 | AUROC | FPR95 | F1 |
  |---|---|---|---|
  | Energy | 89.01 | 43.36 | 76.23 |
  | kNN (k=10) | 91.45 | 42.77 | 76.74 |
  | Mahalanobis | 92.62 | 35.35 | 81.19 |
  | Hypercone (validation 보정) | 93.39 | 32.80 | 81.00 |
  - 원뿔 vs Energy: AUROC +4.4%p (18/18), vs kNN: +1.9%p (17/18)
  - 원뿔 vs **Mahalanobis: AUROC +0.8%p (13/18), FPR95 −2.6%p (11/18), F1 −0.2%p (11/18)**
- **결론**: 원뿔은 Energy·kNN·MSP보다 확실히 낫지만, Mahalanobis와는 차이가 작다. "hypersphere라서 크게 낫다"고 주장할 수는 없고, "각도 원뿔은 Mahalanobis 수준의 성능을 공분산 추정 없이, 기준값 보정 없이 낸다"가 정직한 주장이다. 원뿔 쪽 feature는 cosine softmax로 학습했고 나머지는 일반 softmax라서, 차이의 일부는 학습 방식 차이일 수 있다.

## E6. 같은 cosine softmax 신경망 위에서 판정 방식만 비교
- **질문**: E5에서 원뿔이 Mahalanobis보다 약간 나았던 건 판정 방식 덕분인가, 원뿔 쪽 신경망을 cosine softmax로 학습한 덕분인가?
- **설정**: `--methods cone_q cq_msp cq_maha cq_maha_raw cq_knn --tag cosine_rules`. 원뿔(cone_q)과 같은 신경망 하나를 학습하고, 그 위에서 cosine MSP, Mahalanobis(정규화 feature z / 정규화 전 feature), kNN(k=10, z)을 계산. 거부 기준은 모두 validation 95% 통과. cone_q가 E1과 같게 재현됨을 확인(차이 0).
- **결과** (`results/cosine_rules.md`, 18개 설정 평균):
  | 판정 방식 (같은 cosine 신경망) | AUROC | FPR95 | F1 | 원뿔 승 (AUROC) |
  |---|---|---|---|---|
  | cosine MSP | 92.28 | 38.92 | 79.33 | 18/18 |
  | Mahalanobis (z) | 92.18 | 37.40 | 79.00 | 14/18 |
  | Mahalanobis (정규화 전) | 92.05 | 37.19 | 78.79 | 16/18 |
  | kNN (z) | 93.46 | 32.84 | 81.10 | 9/18 |
  | 원뿔 | 93.39 | 32.80 | 81.00 | — |
  - 학습 방식 효과: MSP는 softmax → cosine 학습으로 AUROC +2.5%p(18/18). Mahalanobis는 오히려 −0.4%p(7/18).
- **결론**: 원뿔이 Mahalanobis보다 나은 건 학습 방식만의 효과가 아니다. 같은 신경망에서도 원뿔이 AUROC +1.2%p, 14/18 설정에서 앞선다. MSP 대비 이득(+3.6%p)은 cosine 학습(+2.5)과 원뿔 판정(+1.1)이 나눠 만든다. 같은 feature의 kNN과는 동률이지만, kNN은 학습 문장 전체를 저장·탐색하고 원뿔은 클래스당 방향 하나와 반지름 하나만 쓴다.

---

## 아직 하지 않은 실험
- StackOverflow, KLUE-YNAT 데이터셋
- 임베딩 모델 fine-tuning (이상치가 표현 자체를 바꾸는지 확인하려면 필요)
- ADB / DA-ADB 공식 코드 비교, DCLOOS·VOS 원본·LMCL+LOF 실행
- LLM에게 직접 판정시키는 비교 (정확도, 속도, 비용)
- 입력 공간에서 자르지 않은 vMF
- 튜닝을 25/50% 설정과 더 많은 seed로 확대
