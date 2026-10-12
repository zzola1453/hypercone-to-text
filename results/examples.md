# 실제 문장 판정 사례: clinc, known 75%, seed 0

학습한 의도 112개, 테스트 문장 5500개 (학습한 의도 3360개, 관련 없는 문장 2140개). Hypercone+tvMF는 입력 공간 tvMF 가상 이상치(δ=0.15)로 학습하고 원뿔 경계 그대로 판정, MSP는 validation known 95% 통과 기준.

## 요약

|  | MSP | Hypercone+tvMF |
|---|---|---|
| 학습한 의도 문장 중 정확히 분류 | 3136개 (93.3%) | 2846개 (84.7%) |
| 학습한 의도 문장 중 잘못 거부 | 158개 (4.7%) | 486개 (14.5%) |
| 관련 없는 문장 중 거부 | 1119개 (52.3%) | 1888개 (88.2%) |
| 관련 없는 문장 중 받아들임 (새어 들어옴) | 1021개 (47.7%) | 252개 (11.8%) |


## A. 관련 없는 문장: MSP는 받아들였지만 Hypercone은 거부 (개선 사례)

전체 827개 중 무작위 10개

| 문장 | 정답 (* = 학습 안 한 의도) | MSP 판정 | Hypercone+tvMF 판정 | 판정 원뿔까지 각도 / 그 원뿔 반지름 |
|---|---|---|---|---|
| when can i get a business card printed locally | oos * | replacement_card_duration | 거부(unknown) | 70.2° / 55.4° (replacement_card_duration) |
| move forward one song | next_song * | play_music | 거부(unknown) | 55.3° / 54.1° (play_music) |
| could you set a timer | timer * | reminder_update | 거부(unknown) | 68.6° / 54.1° (last_maintenance) |
| can you set the alarm for noon | alarm * | reminder_update | 거부(unknown) | 65.9° / 54.0° (time) |
| who is the best architecture | oos * | where_are_you_from | 거부(unknown) | 68.1° / 57.7° (travel_suggestion) |
| what is the lowest amount i can pay for my cellphone bill | min_payment * | bill_balance | 거부(unknown) | 62.3° / 56.1° (bill_balance) |
| change the code of my savings account to be 1234 | pin_change * | routing | 거부(unknown) | 79.1° / 59.7° (balance) |
| give me my wallet location | oos * | current_location | 거부(unknown) | 68.5° / 55.7° (current_location) |
| park the car | oos * | smart_home | 거부(unknown) | 63.9° / 56.7° (smart_home) |
| could you find out if my bank charges transaction fees on my american card if use internationally | international_fees * | credit_limit | 거부(unknown) | 72.3° / 56.1° (bill_balance) |

## B. 관련 없는 문장: 둘 다 거부 (쉬운 사례)

전체 1061개 중 무작위 10개

| 문장 | 정답 (* = 학습 안 한 의도) | MSP 판정 | Hypercone+tvMF 판정 | 판정 원뿔까지 각도 / 그 원뿔 반지름 |
|---|---|---|---|---|
| what do you do if your 4 year old just ate playdoh | oos * | 거부(unknown) | 거부(unknown) | 72.1° / 57.7° (maybe) |
| what are the transaction fees associated with my discover card if i am in rome | international_fees * | 거부(unknown) | 거부(unknown) | 71.3° / 56.8° (new_card) |
| how many kilowatts have i used this month | oos * | 거부(unknown) | 거부(unknown) | 70.0° / 58.1° (pto_used) |
| what have i spent things on | transactions * | 거부(unknown) | 거부(unknown) | 65.7° / 58.1° (pto_used) |
| does travelling to st petersburg require a travel visa | international_visa * | 거부(unknown) | 거부(unknown) | 68.2° / 56.5° (travel_alert) |
| read my to-do list items | todo_list * | 거부(unknown) | 거부(unknown) | 56.3° / 55.3° (todo_list_update) |
| get me a hotel room big enough for 4 people from may 3 to may 5 in boston | book_hotel * | 거부(unknown) | 거부(unknown) | 63.7° / 55.5° (schedule_meeting) |
| what does adulation mean | definition * | 거부(unknown) | 거부(unknown) | 75.4° / 55.3° (calculator) |
| call an uber to take a group of 4 from this bar to the one on maple st | uber * | 거부(unknown) | 거부(unknown) | 60.6° / 54.9° (restaurant_reservation) |
| how do i transfer my 401k to a new firm | rollover_401k * | 거부(unknown) | 거부(unknown) | 61.7° / 53.3° (transfer) |

## C. 관련 없는 문장: Hypercone도 받아들임 (한계 사례, near-OOD)

전체 252개 중 무작위 10개

| 문장 | 정답 (* = 학습 안 한 의도) | MSP 판정 | Hypercone+tvMF 판정 | 판정 원뿔까지 각도 / 그 원뿔 반지름 |
|---|---|---|---|---|
| is it possible i have a meeting with kim today | meeting_schedule * | schedule_meeting | schedule_meeting | 52.8° / 55.5° (schedule_meeting) |
| tell me the timezone for california | timezone * | time | time | 46.9° / 54.0° (time) |
| how many steps have i taken today | oos * | pto_used | pto_used | 56.1° / 58.1° (pto_used) |
| how can i transfer my 401k | rollover_401k * | 거부(unknown) | transfer | 48.0° / 53.3° (transfer) |
| i need what i'm listening to right now to be put on my exercise playlist | update_playlist * | play_music | play_music | 44.4° / 54.1° (play_music) |
| are there any meetings between 2 and 5 pm | meeting_schedule * | schedule_meeting | schedule_meeting | 35.8° / 55.5° (schedule_meeting) |
| what's the annual percentage rate for my credit card | apr * | interest_rate | interest_rate | 38.1° / 52.6° (interest_rate) |
| what is my apr rate | apr * | interest_rate | interest_rate | 35.6° / 52.6° (interest_rate) |
| give me the number of a traffic attorney near here | oos * | traffic | traffic | 52.7° / 52.9° (traffic) |
| look up the lowest amount i can pay on my electricity bill | min_payment * | bill_balance | bill_balance | 54.7° / 56.1° (bill_balance) |

## D. 학습한 의도: Hypercone이 잘못 거부

전체 486개 중 무작위 10개

| 문장 | 정답 (* = 학습 안 한 의도) | MSP 판정 | Hypercone+tvMF 판정 | 판정 원뿔까지 각도 / 그 원뿔 반지름 |
|---|---|---|---|---|
| so what subjects do you like | what_can_i_ask_you | what_can_i_ask_you | 거부(unknown) | 61.1° / 56.5° (what_can_i_ask_you) |
| "i'm fly out from lax to phl this weekend for around $120 | book_flight | book_flight | 거부(unknown) | 57.9° / 52.7° (book_flight) |
| i want to pay my bill through the app | pay_bill | pay_bill | 거부(unknown) | 62.1° / 55.7° (pay_bill) |
| is there a reservation for mark at mimis | confirm_reservation | confirm_reservation | 거부(unknown) | 60.2° / 56.2° (confirm_reservation) |
| let me know the date tomorrow | date | date | 거부(unknown) | 55.9° / 55.0° (date) |
| i would like you to change your accent | change_accent | change_accent | 거부(unknown) | 57.5° / 52.0° (change_accent) |
| tell jim i'm coming home soon | text | 거부(unknown) | 거부(unknown) | 61.6° / 53.9° (text) |
| what should i squirrel away to pay in taxes | taxes | taxes | 거부(unknown) | 52.9° / 50.8° (taxes) |
| is it six o clock yet | time | time | 거부(unknown) | 70.0° / 54.0° (time) |
| i'd like to start calling you friend | change_ai_name | change_ai_name | 거부(unknown) | 57.4° / 55.5° (change_user_name) |

## E. 학습한 의도: Hypercone이 정확히 분류

전체 2846개 중 무작위 10개

| 문장 | 정답 (* = 학습 안 한 의도) | MSP 판정 | Hypercone+tvMF 판정 | 판정 원뿔까지 각도 / 그 원뿔 반지름 |
|---|---|---|---|---|
| who do you take your orders from | who_do_you_work_for | 거부(unknown) | who_do_you_work_for | 51.9° / 55.0° (who_do_you_work_for) |
| has the french laundry gotten good reviews lately | restaurant_reviews | restaurant_reviews | restaurant_reviews | 45.2° / 57.9° (restaurant_reviews) |
| remove fencing from my calendar for may 7th | calendar_update | calendar_update | calendar_update | 43.2° / 56.3° (calendar_update) |
| how much do i have in my bank accounts | balance | balance | balance | 39.7° / 59.7° (balance) |
| what is the highway mpg of this car | mpg | mpg | mpg | 31.5° / 53.4° (mpg) |
| thanks a million | thank_you | thank_you | thank_you | 48.2° / 56.3° (thank_you) |
| i wanna know how low are my tires on air | tire_pressure | tire_pressure | tire_pressure | 42.3° / 53.9° (tire_pressure) |
| where am i on the map | current_location | current_location | current_location | 47.9° / 55.7° (current_location) |
| do i need a vaccine for the congo | vaccines | vaccines | vaccines | 45.4° / 50.5° (vaccines) |
| can you tell me what the traffic is usually like at 7:00 pm from philadelphia to ocean city, new jersey | traffic | traffic | traffic | 44.4° / 52.9° (traffic) |

## F. 관련 없는 문장: MSP는 거부했지만 Hypercone은 받아들임 (역전 사례)

전체 58개 중 무작위 10개

| 문장 | 정답 (* = 학습 안 한 의도) | MSP 판정 | Hypercone+tvMF 판정 | 판정 원뿔까지 각도 / 그 원뿔 반지름 |
|---|---|---|---|---|
| for what time is my meeting with marketing scheduled | meeting_schedule * | 거부(unknown) | schedule_meeting | 51.8° / 55.5° (schedule_meeting) |
| when is my meeting with jack for | meeting_schedule * | 거부(unknown) | schedule_meeting | 50.4° / 55.5° (schedule_meeting) |
| what's a good chili recipe | recipe * | 거부(unknown) | meal_suggestion | 52.8° / 56.0° (meal_suggestion) |
| what's on my todo list | todo_list * | 거부(unknown) | todo_list_update | 42.3° / 55.3° (todo_list_update) |
| i wanna transfer my 401k | rollover_401k * | 거부(unknown) | transfer | 48.5° / 53.3° (transfer) |
| can you find me the apr listed on the statement for my citi mastercard | apr * | 거부(unknown) | interest_rate | 49.5° / 52.6° (interest_rate) |
| check today's calendar for meetings | meeting_schedule * | 거부(unknown) | schedule_meeting | 43.9° / 55.5° (schedule_meeting) |
| what are some of the most visited countries for people looking to travel abroad | oos * | 거부(unknown) | travel_suggestion | 51.4° / 57.7° (travel_suggestion) |
| is the meeting with lisa today | meeting_schedule * | 거부(unknown) | schedule_meeting | 50.1° / 55.5° (schedule_meeting) |
| where can i get 100% merino wool t-shirts on the internet | oos * | 거부(unknown) | order | 55.8° / 57.3° (order) |
