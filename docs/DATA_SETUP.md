# 데이터 준비

`data/` 는 저장소에 포함되지 않는다(.gitignore). 시뮬레이션 데이터는 **재생성 가능**하고,
실데이터 두 종은 **각자의 라이선스로 직접 받아야** 한다. 아래 순서대로 하면 논문/문서의
모든 실험을 재현할 수 있다.

환경: python 3.9 + torch 2.2.2 (이 프로젝트는 `~/work/stock-v2/.venv-mps` 를 썼다).
아래 `python` 은 그 인터프리터를 가리킨다.

---

## 0. 공통 규칙 — 감사 게이트

**데이터를 새로 만들거나 변환할 때마다 반드시 무결성 감사를 통과시킨다.**

```bash
python scripts/audit_dataset.py --data <데이터경로>
# 마지막 줄이 "PASS: no integrity failures" 여야 한다
```

이 감사는 9종 검사(GT 타임라인 정형성, 관측-GT 일치, 질의 라벨 재계산, 질의 적격성,
모델 윈도우 인과성, JEPA 티처 범위, 앵커·피처 정합, 분할 격리, 시드 재생성 결정성)를 돌린다.
개발 중 치명적 버그 8건을 이 게이트가 잡았다 — 건너뛰지 말 것.

---

## 1. 시뮬레이션 데이터 (재생성, 약 10분)

가구 팩(`data/packs/pilot_packs.json`, 12가구 페르소나·시그니처 활동)은 저장소에 포함돼 있다.

```bash
python scripts/gen_dataset.py --out data/v5 --split train --homes 600 --packs data/packs/pilot_packs.json
python scripts/gen_dataset.py --out data/v5 --split val   --homes 36  --packs data/packs/pilot_packs.json
python scripts/gen_dataset.py --out data/v5 --split test  --homes 48  --packs data/packs/pilot_packs.json
python scripts/audit_dataset.py --data data/v5          # PASS 확인
python scripts/validate_dataset.py --dir data/v5/test   # moved 비율 등 통계
```

무라벨 사전학습 풀이 필요하면(JEPA 스케일링 실험):
```bash
python scripts/gen_dataset.py --out data/v5_pretrain --split pretrain --homes 5000 --packs data/packs/pilot_packs.json
```

베이스라인 통계 적합:
```bash
python - <<'EOF'
import glob, json, sys; sys.path.insert(0, ".")
from homejepa.baselines import fit_stats
json.dump(fit_stats(sorted(glob.glob("data/v5/train/ep_*.json"))),
          open("results/baseline_stats_v5.json", "w"))
EOF
```

---

## 2. HOMER+ — 실제 사람 루틴 (MIT, 자유 사용)

크라우드소싱된 생활 루틴을 VirtualHome 으로 실행한 3가구 데이터. 우리는 그 주민을
**동거인(물건을 옮기는 사람)** 으로 두고, 안경 착용자는 우리 페르소나·FOV 로 합성한다.

```bash
git clone --depth 1 https://github.com/Maithili/HOMER_PLUS.git data/homer_plus

# test 스플릿 -> 전이 평가용 (90 에피소드)
python scripts/convert_homer.py --out data/homer_v5
# train 스플릿 -> 무라벨 적응용 (195 에피소드)
python scripts/convert_homer.py --split train --user-seeds 1 --out data/homer_v5
python scripts/audit_dataset.py --data data/homer_v5 --train-sample 40 --regen-check 0
```

무라벨 유사라벨(현장 적응용) 생성과 품질 감사:
```bash
python scripts/pseudo_queries.py --src data/homer_v5/adapt --out data/homer_v5/pseudo
python scripts/verify_pseudo.py  --src data/homer_v5/adapt --pseudo data/homer_v5/pseudo
```
`verify_pseudo.py` 는 GT 와 대조해 유사라벨의 노이즈율을 **측정만** 한다(학습에는 미사용).
현재 값: 전체 일치 0.863 / 이동 케이스 0.607.

---

## 3. ADT (Aria Digital Twin) — 실센서 (Meta 연구 라이선스, 재배포 불가)

실제 AI 안경(Project Aria) + 모션캡처. **화면 밖에서도 물체 6DoF GT 가 존재하는 유일한
실데이터**이고 2인 세션이 있어 "남이 옮기는" 시나리오가 실제로 들어 있다.

### 3-1. 사용자 단계 (직접 해야 함, 5분)
1. https://www.projectaria.com/datasets/adt/ 접속 → **Access The Dataset**
2. 이메일 입력 / 라이선스 동의
3. 받은 링크 파일을 `data/adt/ADT_download_urls.json` 로 저장
   (서명 URL 이라 보통 14일 유효)

### 3-2. 선별 다운로드 + 변환 (자동)
```bash
python scripts/adt_download.py      # 2인 시퀀스 GT 만: 37개, ~3.8GB (전체는 3.5TB)
python scripts/convert_adt.py       # -> data/adt/eps/test (37 에피소드, 1,122 질의)
python scripts/audit_dataset.py --data data/adt/eps --train-sample 0 --regen-check 0
```

영상·depth·segmentation 은 받지 않는다. 물체 위치는 모캡 CSV, 관측 이벤트는
`2d_bounding_box.csv` 의 실제 프레임별 가시성에서 나오므로 **비전 모델이 필요 없다**.

한계: 시퀀스당 ~95초라 **분 단위 지평의 실센서 파이프라인 검증**용이지 장기 벤치가 아니다
(움직임 질의 27건). 시간 단위 지평은 HOMER+ 가 담당한다.

---

## 4. 전체 파이프라인 한 번에

각 단계별 체인 스크립트가 생성→감사→학습→평가를 담고 있다:

| 스크립트 | 내용 |
|---|---|
| `scripts/v1_chain.sh` | receptacle v1 데이터 + supervised 학습 |
| `scripts/v2_chain.sh` | 생활사 체인 동역학 |
| `scripts/p8b_chain.sh` | glance 수정판 재생성 + noid 학습 |
| `scripts/p10_prep.sh` | v5 생성 + 감사 + 유사라벨 (CPU 전용) |
| `scripts/p12_roomfeats.sh` | 조합 어트리뷰트 ON 3자 비교 |

## 5. 저장 용량

| 항목 | 크기 |
|---|---|
| 시뮬 v5 (684 에피소드) | ~330MB |
| HOMER+ 원본 클론 | 4.3GB |
| HOMER+ 변환본 | ~70MB |
| ADT GT (2인 37 시퀀스) | 13GB (압축 해제 후) |
| 학습 산출물 `results/` | ~220MB |
