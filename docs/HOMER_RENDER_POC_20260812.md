# HOMER 픽셀 GT 재실행 POC — 판정: 가능 (병합-재구성 경로) (2026-08-12)

질문: HOMER 는 심볼릭뿐인데, VirtualHome 재실행으로 pose/depth/seg 픽셀 GT 를 "만들 수" 있는가.
이 아이맥에서 end-to-end 로 확인했다.

## 확인된 것 (전부 실측)

| 단계 | 결과 |
|---|---|
| Unity 실행체 v2.3.0 (macOS) | ✅ 유니버설 바이너리, 이 인텔 아이맥에서 구동·HTTP 응답 |
| HOMER 그래프 직접 주입 | ❌ 전 씬 널참조 — rail_tasksim 포크의 자체 id 공간(씬과 일치 0/80) |
| **씬 그래프에 병합 주입** | ✅ book·mug 등 배치 성공 (id 는 시뮬레이터가 재부여) |
| 픽셀 GT 렌더 | ✅ RGB(uint8)·depth(float32)·instance seg + **색↔노드 매핑 445개**(`instance_colors`) |
| 원자 행동 재생 | ✅ walk+grab 실행 → 환경 그래프에 `HOLDS_RH` 상태 전이 반영 |

증거: `results/vh_poc_{normal,seg_inst,depth}.png`

## 남은 엔지니어링 (전체 하루 재생까지)

1. **id 매핑 레이어** — 주입 시 id 가 재부여되므로 HOMER id↔씬 id 대응표 유지 필요
2. **프리팹 매핑** — 그래프 어휘 직접 일치는 22/73. HOMER 노드의 `prefab_name` 필드를
   `expand_scene(prefabs_map=...)` 으로 넘기는 연결이 다음 열쇠 (rail_tasksim 포크에 기존
   매핑이 있을 가능성 큼 — 포크 조사 우선)
3. **배치 실패 폴백** — plate/fryingpan/cookingpot 등 일부 프리팹이 transform 없는 배치에서
   까탈 (표면 분산·대체 표면 재시도 로직)
4. 하루치(converted_scripts, 분당 원자행동) 재생 + 우리 합성 관찰자 궤적을 카메라로 → 에고 영상

## 위치

- 실행체·API: `tools/` (git 제외 대상), 포트 8080 HTTP
- 이 경로의 산출물 = "다중 방 × 며칠 지평 × (게임급) 에고 영상 + 완전 GT" — ADT(실영상·2공간·95초)와 상보
