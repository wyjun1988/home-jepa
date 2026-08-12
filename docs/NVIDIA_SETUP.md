# NVIDIA GPU(4090 / RunPod)에서 학습하기

코드는 디바이스-불가지론적 순수 torch 라 CUDA 에서 그대로 돈다. 모든 스크립트가 `--device` 를
받고, **기본값이 cuda > mps > cpu 자동 감지**이므로 NVIDIA 머신에서는 플래그를 생략해도 된다.
단, 지금까지의 모든 결과는 macOS MPS/CPU 에서 나왔다 — **CUDA 는 미검증 경로**이므로 §5 의
스모크를 첫 실행에 반드시 거친다.

## 1. 환경

```bash
# python >= 3.9. torch 는 2.2 이상이면 됨 (개발은 2.2.2 에서, 상위 버전 API 미사용)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install numpy
```

- 이 로컬 x86_64 맥의 `torch==2.2.2, numpy==1.26.4` 고정은 **맥 휠 한계였을 뿐** — Linux/CUDA
  에서는 최신 torch(2.4+)·numpy 를 쓰면 된다. torch<2.3 + numpy 2.x 조합만 피할 것.
- AMP(bf16/fp16)는 시도해 본 적 없음. 모델이 작아 fp32 로도 병목이 없으니 **fp32 유지 권장**.

## 2. 저장소 + 데이터

```bash
git clone git@github.com:wyjun1988/home-jepa.git && cd home-jepa

# 시뮬 데이터 재생성 (~10분, CPU) + 감사 게이트 — 데이터를 만든 머신에서 PASS 필수
python scripts/gen_dataset.py --out data/v5 --split train --homes 600 --packs data/packs/pilot_packs.json
python scripts/gen_dataset.py --out data/v5 --split val   --homes 36  --packs data/packs/pilot_packs.json
python scripts/gen_dataset.py --out data/v5 --split test  --homes 48  --packs data/packs/pilot_packs.json
python scripts/audit_dataset.py --data data/v5     # "PASS: no integrity failures"

# HOMER+ (전이 평가까지 할 때만)
git clone --depth 1 https://github.com/Maithili/HOMER_PLUS.git data/homer_plus
python scripts/convert_homer.py --out data/homer_v5
python scripts/convert_homer.py --split train --user-seeds 1 --out data/homer_v5
```

ADT 는 라이선스 파일이 필요하므로 로컬에서 쓰던 `data/adt/ADT_download_urls.json` 을 복사한 뒤
`python scripts/adt_download.py` (상세: DATA_SETUP §3).

## 3. 학습 커맨드 (로컬과 동일, --device 생략 가능)

```bash
# supervised 현행 최고 구성 재현
python scripts/train_supervised.py --data data/v5 --tag two_head_v5 --model two_head \
  --noid --gate-weight 0.5 --steps 5000 --val-every 500 --seed 0

# JEPA 2-stage (티처 정합판)
python scripts/train_jepa.py --data data/v5 --tag jepa_v6_align --noid \
  --event-horizons --aug-gap 0.3 --align-teacher --s1-steps 3000 --s2-steps 0 --seed 0
python scripts/train_jepa.py --data data/v5 --tag jepa_v6_align_ft --noid \
  --load-s1 results/jepa_jepa_v6_align_s1.pt --no-freeze --aux-weight 0.5 --gate-weight 0.5 \
  --s2-steps 5000 --seed 0
```

체인 스크립트(`scripts/p13_chain.sh` 등)는 macOS 전용 인터프리터 경로(`P=` 변수)만 자기
환경의 python 으로 바꾸면 그대로 쓸 수 있다.

## 4. 규모 감각과 4090 에 맡길 일

- 모델 ~2.5M 파라미터, bs 128 기준 VRAM 사용 < 2GB. **VRAM 은 전혀 병목이 아니다.**
- MPS 실측 1.3~1.5 s/step (bs 128, d128 L4) → 4090 은 대략 한 자릿수 배 빠를 것으로 기대
  (실측 후 이 문서에 기록할 것).
- 이 프로젝트에서 4090 이 가장 값진 일 세 가지:
  1. **시드 반복** — P12/P13 판정(JEPA vs scratch)의 시드 ±2 확정. 로컬에선 판당 3~5시간이라
     못 하던 것. `--seed 1 --seed 2` 로 태그만 바꿔 반복.
  2. **스케일 곡선** — 집 600→2,000·d 128→256 1회 측정 (스케일링 미측정 축).
  3. Qwen 팩 대량 생성 후 재학습 (LLM 추론도 GPU 로).
- 배치를 512+ 로 키우는 것은 가능하나 lr 재튜닝이 필요하므로, **비교 실험은 로컬과 같은
  bs 128 로 두고 병렬 시드로 처리량을 쓰는 편이 안전**하다.

## 5. 첫 실행 스모크 (CUDA 검증 절차)

```bash
# 1) 500 스텝만 돌려 loss 곡선이 로컬과 같은 궤적인지 (val_nll ~0.8 부근이면 정상)
python scripts/train_supervised.py --data data/v5 --tag cuda_smoke --model two_head \
  --noid --gate-weight 0.5 --steps 500 --val-every 250 --seed 0
# 2) 로컬에서 만든 체크포인트를 CUDA 로 평가 — 수치가 로컬 평가와 일치해야 함 (fp32 오차 범위)
python scripts/reeval.py --data data/v5
```

주의: 시드 재현성은 **같은 디바이스 안에서만** 성립한다(디바이스별 RNG 상이). 디바이스 간에는
수치가 아니라 결론이 재현되는지를 본다.

## 6. 결과 회수

`results/` 는 git 제외이므로 rsync 로 가져온다:

```bash
rsync -av 'REMOTE:~/home-jepa/results/*.pt' 'REMOTE:~/home-jepa/results/*_test.json' results/
python scripts/gen_manifest.py     # 세대 매니페스트 갱신 (compare.py 가 사용)
```

체크포인트에는 학습 args 가 저장되므로(noid 등 실행 조건 포함) 회수 후 로컬에서 `reeval.py`·
`eval_transfer.py`·`eval_refpast.py` 로 어떤 평가든 다시 돌릴 수 있다.

## 7. RunPod 메모

- 템플릿: 공식 PyTorch 이미지면 충분. 팟 준비 후 §1–2 를 그대로.
- 배포키: 이 저장소는 읽기 전용 클론이면 https 로 충분 (`git clone https://github.com/wyjun1988/home-jepa`).
  푸시가 필요하면 별도 배포키를 만들 것 — 로컬 키를 팟에 복사하지 말 것.
- 팟 반납 전에 §6 회수 확인. (stock-v2 관례와 동일: 결과는 tar 로도 백업)
