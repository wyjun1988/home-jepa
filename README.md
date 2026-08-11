# Home-JEPA

AI 안경(~1fps) 관측 스트림으로 4D scene graph 를 유지하며, **지금 안 보이는 물체의 현재 위치**
(그리고 미래 위치·정리 관련 다운스트림)를 JEPA 방식 latent 로 추정한다. stock-v2 의 Graph-JEPA
레시피(2-stage, VICReg+imputation 앵커, EMA 타깃)를 가정환경으로 이식한 프로젝트.

- 시작 2026-08-07 · 관측-미관측 추정이라는 문제 정의는 기존 벤치마크에 없음(서베이 §0)
- 원장 문서: [SURVEY](docs/SURVEY_20260807.md) → [DESIGN](docs/DESIGN_20260807.md) →
  [P1](docs/RESULTS_P1_20260807.md) → [AUDIT](docs/AUDIT_20260808.md) → [P2](docs/RESULTS_P2_20260808.md) →
  [P3](docs/RESULTS_P3_20260808.md) → [P4](docs/RESULTS_P4_20260809.md) → [P5+P6](docs/RESULTS_P5_20260809.md) →
  [P7](docs/RESULTS_P7_20260809.md) → [P8](docs/RESULTS_P8_20260810.md) → [P9](docs/RESULTS_P9_20260810.md) →
  [P10](docs/RESULTS_P10_20260811.md) → [P12](docs/RESULTS_P12_20260812.md) · 능력 요약 [CAPABILITY](docs/CAPABILITY_20260810.md)
- 데이터 준비: **[DATA_SETUP](docs/DATA_SETUP.md)** · [ADT 셋업](docs/ADT_SETUP.md)

## 확립된 레시피 (2026-08-11 기준)

| 층 | 내용 |
|---|---|
| 상태 공간 | **receptacle**(가구 표면, 집당 8–30) — 실루틴 이동의 ~92%가 방내라는 실측(가구:방 13:1)이 근거 |
| 데이터 | 그래프 레벨 시뮬레이터(렌더링 없음, 초당 ~3집) + **LLM 행동 레이어**(가구 팩: 페르소나·시그니처 활동·사후 목적지 체인) + 무결성 감사 게이트 |
| 관측 모델 | FOV 원뿔 + receptacle 위치 + p_det(크기·거리) → POS/NEG/SELF 이벤트 (부정 증거는 불완전) |
| 사전학습 (JEPA) | 컨텍스트=인과 로그, 티처=**k번째 다음 재관측까지**(k∈{1,2,4}, 이벤트 단위 지평), 구조화 마스킹(접미사 절단), VICReg+히스토그램 앵커, EMA. **라벨 0** |
| 지각 가정 | **물체 re-id 없음**(연관은 모델이 추론) · 3D 좌표 없음 · 장소=물체 조합 · L1 사람 존재(신원·소지품 없음) |
| readout | 게이트형(anchor one-hot 혼합) + 질의물체 예약 윈도우 + 앵커 어트리뷰트 + 부재증거 게이트 직결 |
| 배포 적응 | **방-이동 한정 유사라벨로 목적지 헤드만** 미세조정 + 게이트는 무라벨 τ-recal — 전부 GT 불필요 |

## 핵심 결과 (전부 감사 PASS 데이터)

- **시뮬 내 (v1, 32-way)**: jepa_ft 종합 1위 — NLL 0.748 / top-1 0.816 (vs scratch 0.768/0.812, 최강 비학습 1.07/0.795)
- **월드모델 스코어카드 (frozen latent + 선형 probe, 7태스크)**: JEPA latent 가 현재 태스크 동률,
  **모든 미래 태스크·tidy 에서 supervised 트렁크 우위** (+6h +3.0pt, tidy +7.5pt). 다운스트림 채택:
  tidy·+1h/+6h(강), nsight·misplaced(중), nmove(보류)
- **실센서 (ADT 36 시퀀스, 모캡 GT)**: 게이트형 모델 zero-shot 무붕괴 (0.976 = last_known 동률)
- **실루틴 전이 (HOMER+ 15.2k 질의)**: zero-shot 무붕괴 + 합성 레시피(생활사체인 시뮬 + recal)가
  전체 top-2 0.912 로 최고. 남은 갭 = 방내 목적지(0.078 vs 시뮬내 0.5)
- **무라벨 적응(s1-continue, 라벨 0)**: 캘리브레이션 대폭 개선(NLL 0.92→0.66), 목적지는 못 메움

## 결론이 뒤집힌 기록 (교훈 로그)

| 시점 | 처음 결론 | 뒤집힌 계기 |
|---|---|---|
| P1→P2 | "2-헤드 구조 필요" | 진짜 원인은 잘린 윈도우(질의 14%가 마지막 목격 못 봄) — flat 이 역전 |
| P2→P5 | "flat > two_head" | 32-way(receptacle)로 가자 게이트 구조가 재역전 |
| P3→P4 | "JEPA=supervised 동률" | 전이 무대(라벨 없음)에서 JEPA 우위 최초 확인 |
| P3→P5 | "supervised 트렁크가 probe 전승" | receptacle 난이도+이벤트지평 목표에서 JEPA latent 역전 |
| P4→P5 | "방 수준으로 충분" | HOMER 실측 가구:방 13:1 → receptacle 재도입 |

프로세스 교훈: ① **무결성 감사를 CI 처럼**(치명 버그 6건을 감사가 적발 — 이동시점 관례, 티처 누수,
피처-앵커 불일치, ADT 초기틱 등) ② 평가는 top-2/탐색수/층화(stayed·moved_within·moved_room) ③
결론은 시드 반복 후에만 ④ 파생 피처는 기준 정의와 같은 소스에서.

## 저장소 지도

```
homejepa/    world(어휘·집 샘플링) routines(스케줄·팩) sim(시뮬+관측+질의) model(EpTensors+백본)
             jepa(2-stage) baselines metrics homer(HOMER+ 변환) adt(ADT 변환)
scripts/     gen_dataset audit_dataset(필수 게이트) train_supervised train_jepa run_baselines
             eval_transfer(+무라벨 recal) probe_multitask(스코어카드) convert_homer convert_adt
             reeval compare diag_* *_chain.sh(재현 체인)
data/        v1,v2(손코딩 체인),v3(LLM 팩) / homer(test+adapt) / adt / packs(가구 팩 JSON)
results/     체크포인트·평가 JSON·체인 로그 (감사 전 결과는 pre_audit_archive/ 격리)
snapshots/   pre_receptacle_20260809 (v0 코드 보존)
```

## 능력 현황 (상세: [CAPABILITY](docs/CAPABILITY_20260810.md))

- 시뮬(표면 21개 중 택1, 찍기 0.048): 안 움직임 0.96 · 내가 옮김 0.87 · **동거인이 옮김 0.17**
  · 탐색비용 12.6→5.0 표면
- 실루틴(HOMER+, 현장 적응 후): 방 이동 top-2 **0.414**, 탐색 10.9→**6.4** 표면, 안 움직임 무손실
- 하나의 frozen latent 로 7개 다운스트림(현재/+1h/+6h 위치, 다음 발견 위치, 지정석, 이탈 감지)

## 진행 중 / 다음

- **P12 완료**: ① 조합 어트리뷰트는 이동(방내 +14%, 자가이동 +8%)엔 도움이나 stayed 를 깎아
  전체는 손해 → **목적지 경로 전용 연결**이 다음 수 ② **JEPA 사전학습은 값을 하지 않음(확정)** —
  조건을 맞춘 3자 비교에서 랜덤초기화가 두 설정 모두에서 우세. Track J 는 주력에서 제외
- 최고 구성: `two_head_v5`(어트리뷰트 OFF·게이트감독·noid·L1사람) + 방-이동 한정 유사라벨 적응
- 미실험: nmove 비선형 헤드, 로컬 Qwen 팩 대량화, ADT 1인 시퀀스 확충, 주~월 단위 실로그 확보

## 재현

**데이터 준비는 [docs/DATA_SETUP.md](docs/DATA_SETUP.md)** — 시뮬은 재생성, HOMER+ 는 클론,
ADT 는 본인 라이선스 동의 후 선별 다운로드(3.8GB). `data/` 와 `results/` 는 저장소에 없다.

각 버전 체인 스크립트가 전체 파이프라인(생성→감사 게이트→학습→평가)을 담는다. 예: `bash scripts/v3_chain.sh`.
환경: `~/work/stock-v2/.venv-mps/bin/python` (py3.9, torch 2.2.2, x86_64; MPS fp32).
주의: 이 머신 VRAM 8GB — MPS 학습 중 영상 재생 불가. 데이터 재생성 후 `scripts/audit_dataset.py` PASS 필수.
