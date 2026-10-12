# 데이터 설명

두 데이터셋 모두 **의도(intent) 분류** 데이터다. 사용자가 챗봇이나 음성 비서에게 한 짧은 영어 문장 하나에 의도 라벨 하나가 붙어 있다.
이 연구에서는 "학습한 의도는 맞히고, 학습하지 않은 의도나 관련 없는 문장은 거부하는가"를 평가하는 데 쓴다.

## 1. CLINC150
| 항목 | 내용 |
|---|---|
| 출처 | Larson et al., *An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction*, EMNLP 2019. Hugging Face `clinc/clinc_oos` (config `plus`) |
| 내용 | 음성 비서에게 하는 요청. 은행, 신용카드, 여행, 업무, 자동차, 요리, 일상 대화, 스마트홈 등 10개 분야 150개 의도 |
| 특별한 점 | 150개 의도 어디에도 속하지 않는 **out-of-scope(oos)** 문장이 따로 들어 있다. 관련 없는 문장 거르기 평가용으로 만든 데이터셋 |
| 크기 | train 15,250 (의도당 100 + oos 250) / validation 3,100 (의도당 20 + oos 100) / test 5,500 (의도당 30 + oos 1,000) |
| 문장 길이 | 평균 8.3 단어 |
| 클래스 균형 | 완전히 균형 (의도마다 같은 개수) |

**예시 문장**
| 의도 | 문장 |
|---|---|
| whisper_mode | make saound function operate in whisper mode |
| how_old_are_you | your birthday please |
| reminder | give me my reminders |
| order | i'm out of milk so you will order me some more |
| **oos** (관련 없음) | who played ronald weasly |
| **oos** (관련 없음) | how could i get into the guiness book of world records |
| **oos** (관련 없음) | who has the best record in the nba |

원문에 오타("saound", "weasly")가 그대로 있다. 실제 사용자 입력을 흉내 낸 데이터이기 때문이다.

## 2. BANKING77
| 항목 | 내용 |
|---|---|
| 출처 | Casanueva et al., *Efficient Intent Detection with Dual Sentence Encoders*, NLP4ConvAI 2020. Hugging Face `mteb/banking77` |
| 내용 | 온라인 은행 고객센터 질문. **은행 하나의 분야 안에서** 77개 의도로 세밀하게 나뉜다 |
| 특별한 점 | 의도끼리 매우 비슷하다 (예: 카드 도착 문의 / 카드 배송 추적 / 카드 미수령). 그래서 학습 안 한 의도가 학습한 의도와 헷갈리기 쉬운 **어려운 데이터** (near-OOD) |
| 크기 | 원본 train 9,993 / test 3,076. validation이 없어서 **train의 10%를 의도별로 떼어** validation으로 씀 → train 9,028 / validation 965 / test 3,076 |
| 문장 길이 | 평균 11.0 단어 |
| 클래스 균형 | train은 불균형 (의도당 32~169개), test는 거의 균형 (39~40개) |
| 비고 | `mteb/banking77`은 원본 배포(10,003 / 3,080)보다 샘플이 약간 적은 사본. 원본 `PolyAI/banking77`은 현재 `datasets` 버전에서 로딩 스크립트를 지원하지 않아 사용하지 못함 |

**예시 문장**
| 의도 | 문장 |
|---|---|
| top_up_by_bank_transfer_charge | Do I have to pay for topping up by transfer? |
| pending_transfer | What is the cause that the transfer shows as pending? |
| verify_source_of_funds | Why do you need all this information about my Source of Funds? |
| visa_or_mastercard | I would like to apply for a visa card. |

## 3. 실험에서 쓰는 방식 (known / unknown 분할)
ADB 논문(Zhang et al., AAAI 2021)과 같은 프로토콜이다 (`data.py`의 `open_split`).

1. 의도 중 25% / 50% / 75%를 무작위로 골라 **학습한 의도(known)**로 정한다. CLINC의 oos는 항상 제외된다.
2. **train / validation**: known 의도의 문장만 남긴다. 나머지 의도의 문장은 학습에 전혀 쓰지 않는다.
3. **test**: 전체 문장을 그대로 쓰되, known이 아닌 의도의 문장과 oos 문장은 모두 **"관련 없음(unknown)"** 정답으로 바꾼다.
4. 같은 비율에서 seed 3개(0, 1, 2)로 known 의도를 다르게 골라 반복한다.

예: CLINC150, known 75%, seed 0 → 학습한 의도 112개. test 5,500개 중 3,360개는 학습한 의도, 2,140개(학습 안 한 의도 38개 × 30 + oos 1,000)는 관련 없는 문장이다.

따라서 이 실험의 "관련 없는 문장"에는 두 종류가 섞여 있다.
- **완전히 엉뚱한 문장**: CLINC의 oos (예: "who played ronald weasly")
- **같은 분야지만 학습하지 않은 의도**: 예를 들어 known에 `interest_rate`는 있고 `apr`은 없을 때의 "what is my apr rate". 이쪽이 훨씬 어렵다 (`results/examples.md`의 C 사례).

## 4. 임베딩
- 모델: `sentence-transformers/all-mpnet-base-v2` (768차원, 출력이 길이 1로 정규화됨 → 문장이 처음부터 구 위에 있음)
- 학습하지 않고 고정. 한 번 계산해서 `cache/`에 저장
