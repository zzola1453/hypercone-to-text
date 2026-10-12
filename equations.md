# Hypercone 분류기: 핵심 수식

코드(`its/methods.py`)와 1:1로 대응하도록 적었다. 기본 모델은 (1)–(6), 확장(가상 이상치)은 (7)–(10)이다.
차원과 하이퍼파라미터는 기호로 두고, 실험에서 쓴 값은 0절 표에 모았다.

## 0. 기호와 실험 값
| 기호 | 뜻 | 실험 값 |
|---|---|---|
| $x$, $y$ | 입력 문장, 그 의도 라벨 ($y \in \{1,\dots,K\}$) | |
| $K$ | 학습한 의도(클래스) 수 | 데이터셋·분할마다 다름 |
| $\mathbb{S}^{n-1}$ | $n$차원 단위 구 $\{v \in \mathbb{R}^n : \lVert v \rVert = 1\}$ | |
| $D$ | 문장 인코더 출력 차원 (입력 구 $\mathbb{S}^{D-1}$) | 768 (all-mpnet-base-v2) |
| $d$ | 정규화된 feature 차원 (feature 구 $\mathbb{S}^{d-1}$) | 128 (MLP $D \to 512 \to d$) |
| $\angle(a,b) = \arccos(a^\top b)$ | 두 단위 벡터 사이의 각도 | |
| $\mu_k \in \mathbb{S}^{d-1}$, $\theta_k \in (0, \pi)$ | 클래스 $k$ 원뿔의 중심 방향과 반지름(각도) | |
| $s$ | cosine softmax의 scale | 16 |
| $q$ | 원뿔 반지름을 정하는 분위수 | 0.95 |
| $\alpha$, $\beta$ | 확장 손실 가중치 | 1, 1 |
| $\delta$ | 가상 이상치 띠 폭 (라디안) | 0.15 |

## 1. 기본 모델

**(1) 문장을 구 위의 점으로**

$$
e = f_{\text{enc}}(x) \in \mathbb{S}^{D-1}, \qquad
h = g_\varphi(e) \in \mathbb{R}^d, \qquad
z = \frac{h}{\lVert h \rVert} \in \mathbb{S}^{d-1}
$$

$f_{\text{enc}}$는 고정된 문장 인코더, $g_\varphi$는 학습하는 MLP다.
인코더 출력이 정규화되어 있지 않다면 $e$를 정규화해서 쓴다.

**(2) 클래스마다 원뿔 하나**

$$
C_k = \{\, z \in \mathbb{S}^{d-1} : \angle(z, \mu_k) \le \theta_k \,\}, \qquad
\mu_k = \frac{p_k}{\lVert p_k \rVert}
$$

$p_k \in \mathbb{R}^d$는 학습하는 prototype 벡터다. 원뿔은 "중심 방향에서 $\theta_k$ 이내"인 구 위의 영역이다.

**(3) 원뿔 안쪽 여유**

$$
m_k(z) = \theta_k - \angle(z, \mu_k)
$$

$m_k > 0$이면 원뿔 $k$ 안, $m_k < 0$이면 밖이다. 단위는 각도(라디안)이고, "경계에서 몇 도 안쪽인가"로 읽는다.

**(4) 판정 규칙**

$$
\hat{y} =
\begin{cases}
\arg\max_k \, m_k(z) & \text{if } \max_k m_k(z) \ge 0 \\
\text{unknown (거부)} & \text{otherwise}
\end{cases}
$$

반지름 $\theta_k$가 곧 거부 기준이다. 별도의 기준값 $\tau$가 없다.

**(5) 학습: cosine softmax**

$$
\mathcal{L}_{\text{CE}} = -\log
\frac{\exp\!\big(s \cdot \cos(z, \mu_y)\big)}{\sum_{j=1}^{K} \exp\!\big(s \cdot \cos(z, \mu_j)\big)}
$$

$\varphi$와 $p_k$를 학습한다. 같은 의도의 문장은 $\mu_y$ 방향으로 모이고, 다른 의도의 중심과는 멀어진다.

**(6) 반지름: 학습 문장 각도의 $q$ 분위수**

$$
\theta_k = Q_{q}\big(\{\, \angle(z_i, \mu_k) : y_i = k \,\}\big)
$$

학습이 끝난 뒤 한 번 계산한다. 클래스 $k$ 학습 문장의 $q$ 비율을 품는 가장 작은 원뿔이다(HyperSphereDiff의 원뿔 정의).
표현이 넓게 퍼진 의도는 반지름이 크고, 좁은 의도는 작다.

## 2. 확장: 원뿔 밖 truncated vMF 가상 이상치

**(7) 반지름도 학습 파라미터로**

$$
\theta_k = \pi \cdot \sigma(\rho_k)
$$

$\sigma$는 sigmoid다. $\theta_k$가 항상 $(0, \pi)$ 안에 있도록 $\rho_k$를 학습한다.

**(8) 전체 손실**

$$
\mathcal{L} = \mathcal{L}_{\text{CE}}
+ \alpha \cdot \underbrace{\mathrm{relu}\big(\angle(z, \mu_y) - \theta_y\big)}_{\text{진짜 문장은 자기 원뿔 안으로}}
+ \beta \cdot \underbrace{\mathrm{relu}\Big(\max_k \big(\theta_k - \angle(\tilde{o}, \mu_k)\big)\Big)}_{\text{가상 이상치는 모든 원뿔 밖으로}}
$$

- 두 번째 항은 원뿔 밖으로 나간 진짜 문장이 있으면 $\theta_y$를 키우고 $z$를 중심으로 당긴다.
- 세 번째 항은 어떤 원뿔 안에 들어온 가상 이상치 $\tilde{o}$가 있으면 그 원뿔의 $\theta_k$를 줄이고 $\tilde{o}$를 밀어낸다.
- 두 힘이 맞서서, 반지름이 "진짜 문장은 품고 바로 바깥 띠는 내보내는" 크기에 자리 잡는다.

**(9) 가상 이상치: 원뿔 밖 띠로 잘라낸 vMF**

입력 구 $\mathbb{S}^{D-1}$에서 클래스마다 원뿔을 먼저 잡는다.

$$
c_k = \frac{\sum_{i:\,y_i=k} e_i}{\big\lVert \sum_{i:\,y_i=k} e_i \big\rVert}, \qquad
r_k = Q_{q}\big(\{\, \angle(e_i, c_k) : y_i = k \,\}\big)
$$

그 원뿔 바로 바깥 띠 $[r_k, r_k + \delta]$에서만 점을 뽑는다.

$$
o \sim \mathrm{tvMF}(c_k, \kappa_k;\, r_k, r_k + \delta), \qquad
p(o) \propto \exp\!\big(\kappa_k\, c_k^\top o\big) \cdot \mathbb{1}\big[\, r_k \le \angle(o, c_k) \le r_k + \delta \,\big],
\qquad o \in \mathbb{S}^{D-1}
$$

뽑은 점을 MLP에 통과시켜 손실 (8)에 넣는다: $\tilde{o} = g_\varphi(o) / \lVert g_\varphi(o) \rVert \in \mathbb{S}^{d-1}$.
그래서 이상치의 gradient가 반지름뿐 아니라 표현 $\varphi$까지 바꾼다. 학습 배치의 문장 하나마다 그 문장 클래스의 띠에서 이상치 하나를 뽑는다.
(feature 공간 버전은 $c_k = \mu_k$로 두고 $\mathbb{S}^{d-1}$ 위에서 epoch마다 $r_k$를 다시 계산하며, $\tilde{o} = o$라서 MLP를 거치지 않는다. 결과가 더 나빠서 메인은 입력 공간 버전이다.)

샘플링 방법: $\mathbb{S}^{n-1}$ 위의 vMF에서 중심과의 각도 $\phi$의 밀도는 $\propto \exp(\kappa \cos\phi)\, \sin^{n-2}\phi$이다.
이 1차원 밀도를 $[r_k, r_k+\delta]$로 잘라 역CDF로 $\phi$를 뽑고, 방향은 접평면에서 균일하게 뽑는다(`hsdiff.sphere.sample_tvmf`). 입력 공간 버전은 $n = D$다.

**(10) 집중도 κ는 반지름에서 계산**

$$
A_n(\kappa_k) = \cos r_k, \qquad
A_n(\kappa) = \frac{I_{n/2}(\kappa)}{I_{n/2-1}(\kappa)} = \mathbb{E}_{o \sim \mathrm{vMF}(c,\kappa)}\big[\, c^\top o \,\big]
$$

$I_\nu$는 제1종 수정 Bessel 함수, $n$은 이상치를 뽑는 구의 차원이다(입력 공간 $n = D$, feature 공간 $n = d$).
$A_n(\kappa)$는 vMF 샘플이 중심과 이루는 평균 cos 값이다.
그래서 이 식은 "vMF 샘플이 평균적으로 원뿔 경계 근처에 놓이도록 $\kappa$를 고른다"는 뜻이다.
덕분에 잘라낸 띠 안에 확률 질량이 충분히 있고, 사람이 정할 값은 띠 폭 $\delta$ 하나뿐이다.

## 3. 비교: 다른 방법의 판정 규칙
모두 같은 인코더 $f_{\text{enc}}$와 같은 크기의 MLP 위에서 계산한다. $h \in \mathbb{R}^d$는 정규화 전 MLP 출력, $\bar{h}_k$는 클래스 평균이다.

| 방법 | 점수 (클수록 known) | 거부 조건 | 기준값 |
|---|---|---|---|
| MSP | $\max_k \mathrm{softmax}(W h)_k$ | 점수 $< \tau$ | 필요 |
| Energy | $\log \sum_k \exp\big((Wh)_k\big)$ | 점수 $< \tau$ | 필요 |
| Mahalanobis | $-\min_k (h - \bar{h}_k)^\top \Sigma^{-1} (h - \bar{h}_k)$ | 점수 $< \tau$ | 필요 |
| kNN | $-\lVert \hat{h} - \hat{h}_{(k')} \rVert$ ($k'$번째 이웃까지 거리, 실험 $k'=10$) | 점수 $< \tau$ | 필요 |
| ADB | $\max_k \big(\Delta_k - \lVert h - \bar{h}_k \rVert\big)$ | 점수 $< 0$ | 반지름 $\Delta_k$ (유클리드) |
| **Hypercone** | $\max_k \big(\theta_k - \angle(z, \mu_k)\big)$ | 점수 $< 0$ | **반지름 $\theta_k$ (각도)** |

$\tau$는 validation의 학습한 문장 95%가 통과하도록 정한다(원뿔의 $q$와 별개인 값이고, 실험에서 우연히 같은 0.95다). 원뿔도 같은 방식으로 보정할 수 있고(점수 $< \tau$로 거부),
결과 표에서는 "validation 보정"과 "원뿔 경계 그대로"를 둘 다 보고한다.

ADB와 원뿔은 둘 다 클래스별 닫힌 영역이다. 차이는 거리의 종류다.
ADB의 $\lVert h - \bar{h}_k \rVert$는 벡터 길이 차이도 거리로 세지만, 원뿔의 $\angle(z, \mu_k)$는 방향 차이만 본다.
