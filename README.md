# Evidence-response reviewer-requested experiments

리뷰어가 요청한 작은 추가 실험을 다른 GPU 서버에서 재현하기 위한 실행 패키지입니다. 논문 원본, 리뷰 내용, Author response 초안, 사진, 모델 가중치는 포함하지 않습니다. 저장된 checkpoint를 평가하는 실험부터 실행하고, 필요한 경우에만 단일 조건의 matched baseline 비교를 수행합니다.

**검증 상태:** CPU manifest/통계/CLI 검증 및 shell 구문 검사 통과. 실제 GPU smoke test 상태는 `VALIDATION.md`를 확인하세요. GPU 학습 전체가 검증됐다는 뜻은 아닙니다.

## 먼저 돌릴 실험

| 순서 | 명령 phase | 질문 | 계산 |
|---|---|---|---|
| 0 | `smoke` | 모델/이미지/adapter 로딩이 정상인가? | base+PEC 각 2개 질문 |
| 1 | `endpoints` | PEC가 gray에만 반응하는가? | 원 250질문, base+PEC seed0, gray/black/noise/다른 사진 |
| 2 | `random` | adversarial 질문에서만 보이는 현상인가? | 고정된 공식 POPE random 252질문, base+PEC |
| 3 | `matched` | IPO 및 동일 lr/steps의 DPO와 비교하면? | 한 모델, 한 seed, PEC/DPO/IPO 각 200 update |
| 4 | `endpoint_seeds` | endpoint 결과가 seed0에만 의존하는가? | 기존 PEC seed1/2 평가 |
| 5 | `pope_controls` | 같은 이미지에서 negative regime이 달라지면? | popular/adversarial 각 252질문 |

`endpoints`와 `random`부터 확보하세요. 25시간 동안 전 모델×전 seed×전 hyperparameter를 탐색할 계획이 아닙니다. 추가 실험은 reviewer 질문에 직접 대응하는 minor add-on 범위로 사용하고, 광범위한 새 연구 결과로 확대하지 마세요. 결과가 불리해도 모든 사전에 지정한 조건을 보고합니다.

## 1. 설치

권장: Linux, Python 3.10, CUDA 12.8와 호환되는 드라이버, 80GB GPU 1개 이상. 2개면 모델별로 분리 실행할 수 있습니다. 원 환경은 PyTorch 2.10.0+cu128, Transformers 5.5.0, Unsloth 2026.7.4였습니다.

```bash
git clone https://github.com/dlwlsrnjs/evidence.git
cd evidence
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
export USE_TF=0
export HF_HOME="$PWD/.cache/huggingface"
python prepare_data.py
python test_cpu.py
```

의존성 설치 실패를 무시한 채 진행하지 마세요. 설치 환경을 `pip freeze > results_environment.txt`로 기록하세요. 원 실험과 다른 library/kernel은 likelihood에 작은 차이를 만들 수 있어 base/PEC를 같은 환경에서 다시 평가해야 합니다. 원래 precision은 BF16이며 4bit/8bit로 조용히 바꾸지 않습니다.

## 2. 기존 사진과 checkpoint 준비 — 이것이 필수

GitHub에는 사진과 학습 adapter가 없습니다. 원 서버에서 필요한 파일을 별도로 옮기세요. 공개 base model은 실행 시 Hugging Face에서 다운로드합니다. adapter의 `adapter_model.safetensors`만 옮기지 말고 processor/tokenizer/config를 포함한 **폴더 전체**를 옮깁니다.

원 서버의 프로젝트 루트에서 최소 assets를 만들려면:

```bash
# 원 서버에서 실행. 읽기 권한이 있는 소유자 계정으로 수행합니다.
cd /home/ubuntu/342/jinkwon/orthocampus
tar -czf /tmp/evidence_assets.tar.gz \
  dual/hc_local/images \
  dual/pope_official/images \
  dual/out/adapter_pec_llava7b \
  dual/out/adapter_pec_q25_7b
```

기존 seed 확장까지 하려면 `dual/out/adapter_pec_llava_s1`, `adapter_pec_llava_s2`, `adapter_pec_q25_s1`, `adapter_pec_q25_s2` 폴더도 포함하세요. SCP/rsync 등으로 새 서버에 복사한 뒤 원하는 위치에 압축을 풉니다. 사진은 각 원 데이터셋의 이용 조건을 따르며 공개 저장소에 업로드하지 않습니다.

```bash
# 새 서버 예시. /data/evidence_assets 는 실제 압축 해제 경로로 바꿉니다.
export PEC_IMAGES=/data/evidence_assets/dual/hc_local/images
export POPE_IMAGES=/data/evidence_assets/dual/pope_official/images
export PEC_ADAPTERS=/data/evidence_assets/dual/out
python experiment.py check --data manifests/original_250.jsonl
python experiment.py check --data manifests/pope_random_252.jsonl
```

`missing_count: 0`이어야 합니다. 경로는 하드코딩하지 않았습니다. `PEC_IMAGES` 아래의 `amber/coco/nocaps/vizwiz` 하위폴더도 검색합니다.

## 3. 첫 GPU 검증

```bash
export CUDA_VISIBLE_DEVICES=0
export RESULTS_DIR=results_llava
bash run_requested.sh smoke llava
```

Qwen2.5는 `RESULTS_DIR=results_q25 bash run_requested.sh smoke q25`입니다. 최초 모델 다운로드/컴파일에 시간이 걸릴 수 있습니다. `smoke`는 작은 실행 검증이며 논문에 보고할 결과가 아닙니다.

GPU가 꽉 차면 HF CPU offload로 **추론만** 실행할 수 있습니다. 큰 host RAM이 필요하고 훨씬 느립니다. 이 프로젝트의 offload smoke에서도 질문당 시간이 크게 늘었으므로, 25시간 내 전체 실험 계획에는 여유 있는 GPU를 사용하세요. quantization은 적용하지 않습니다.

```bash
ENGINE=hf CPU_OFFLOAD=1 RESULTS_DIR=results_offload \
  bash run_requested.sh smoke llava
```

서로 다른 loader 결과를 한 비교에 섞지 마세요. 기본은 원 실험처럼 `ENGINE=unsloth`입니다. HF backend를 쓴다면 base와 모든 adapter를 같은 HF backend로 측정하고 결과에 표시합니다. `--cpu-offload`는 학습을 지원하지 않습니다.

## 4. 25시간 실행 순서

### GPU가 두 개일 때

터미널 A:

```bash
export CUDA_VISIBLE_DEVICES=0
export RESULTS_DIR=results_llava
bash run_requested.sh endpoints llava
bash run_requested.sh random llava
python report.py --results "$RESULTS_DIR"
```

터미널 B:

```bash
export CUDA_VISIBLE_DEVICES=1
export RESULTS_DIR=results_q25
bash run_requested.sh endpoints q25
bash run_requested.sh random q25
python report.py --results "$RESULTS_DIR"
```

**실제 처리시간으로 남은 계획을 결정하세요.** 논문에 기재된 25분/evaluation은 endpoint 수가 다른 이 실행의 시간 보장이 아닙니다. GPU 1개면 위 두 모델을 순차 실행하되 먼저 한 모델의 endpoints/random을 완성합니다. 결과 JSONL은 질문마다 저장됩니다.

### 이후 IPO + matched-budget control

```bash
export CUDA_VISIBLE_DEVICES=0
export RESULTS_DIR=results_matched_q25
bash run_requested.sh matched q25
python report.py --results "$RESULTS_DIR"
```

이 phase는 다음 조건을 모두 새로 맞춥니다:

- base: Qwen2.5-VL-7B; seed 0; LoRA r=16, alpha=32, dropout=.05; language attention만 학습.
- PEC/DPO/IPO: 각각 lr=3e-5, 200 updates, batch size 1, 동일 manifest 인덱스 sampling schedule.
- 원 평가 250질문의 이미지 filename을 학습 풀에서 제거한 2,365질문 사용.
- frozen reference는 dropout을 끄고 adapter를 비활성화한 base로 미리 계산.
- standard IPO: `(d_policy - d_reference - 1/(2*beta))²`, beta=.1. 앞의 전체 beta 상수를 생략한 squared-loss convention이며 loss scaling까지 보고해야 합니다.
- PEC: `softplus(.6 - (sigmoid(d_full)-sigmoid(d_gray))) + .1*abs(d_gray)`.
- 최종 step checkpoint만 평가; test 결과로 checkpoint/hyperparameter를 고르지 않음.

**기존 PEC를 이 새 matched PEC 대신 쓰면 안 됩니다.** 데이터가 달라졌기 때문입니다. 새 세 arm끼리만 동일조건 비교하세요. 한 seed이므로 multi-seed superiority를 주장할 수 없습니다. 동일 update/sample exposure이지 동일 FLOPs/wall-clock은 아닙니다. metadata에 reference 계산 시간, 학습 시간, total wall time과 peak CUDA memory가 저장됩니다.

시간이 남으면 기존 checkpoint의 `endpoint_seeds` 또는 `pope_controls`를 실행하세요. 전체 learning-rate grid나 대규모 신규 학습 캠페인은 이 패키지의 권장 실행 범위가 아닙니다.

## 5. 결과 집계와 paired 비교

```bash
python report.py --results results_q25
python report.py \
  --baseline results_q25/q25_base_endpoints.jsonl \
  --candidate results_q25/q25_pec0_endpoints.jsonl \
  > results_q25/paired_pec_vs_base.json
```

파일:

- `*.jsonl`: query별 gold-directed margins, endpoint, donor filename.
- `*.metadata.json`: 모델/manifest hash, engine, endpoint 설정.
- `*.complete.json`: 완료 표식과 결과 hash. 이 파일이 없으면 중간 결과입니다.
- `*.summary.json`, `report.csv`, `report.md`: split/class/endpoint별 DeltaP와 95% CI, 양의 response 방향 비율, accuracy 및 margin.
- paired JSON: 같은 query/image를 묶은 candidate-minus-baseline 변화와 CI.
- training 폴더: config, 정확한 query index trace, reference margins, loss log, adapter, complete.json.

95% CI는 5,000회 image-cluster percentile bootstrap입니다. 사진별 반복 질문을 독립 사진처럼 세지 않습니다. seed 간 SD와는 다른 불확실성이며 multiple-comparison correction은 없습니다. 여러 POPE split이 같은 positive 질문을 공유하므로 세 split을 독립적으로 합쳐 n을 부풀리지 않습니다.

중단된 파일을 자동으로 덮어쓰거나 이어 붙이지 않습니다. incomplete 결과가 있으면 다른 `RESULTS_DIR`로 재실행하거나 원 결과를 보존해 이름을 바꿉니다. 완료된 phase는 재실행 시 건너뜁니다.

## 6. 데이터/해석에서 지켜야 할 점

- `original_250`은 제출 실험과 같은 순서의 250질문입니다. historical rand/pop pool과 query/image 중복이 있습니다. 이 사실을 숨기지 마세요.
- `original_image_disjoint`는 historical pool의 모든 image filename을 제외한 61질문입니다. 작은 post-hoc sensitivity subset이며 independent replication이 아닙니다.
- 공식 POPE는 historical training pool filename을 제외한 공통 이미지 중 SHA256 순서 첫 42개를 선택했습니다. split마다 252질문, yes/no 각 126입니다. **이 선택은 새 모델 결과를 보기 전에 고정했습니다.** `data/manifest_metadata.json`에서 source/hash를 확인할 수 있습니다.
- unrelated donor는 동일 image filename을 제외하지만 대상 물체가 없다는 annotation은 확인하지 않습니다. 이 결과를 엄밀한 “무증거” 또는 semantic counterfactual로 부르지 말고 image-question alignment control로 보고하세요.
- gray는 원 사진 크기에 맞춥니다. 학습의 fixed 448x448 gray와 구분하려면 `--endpoints gray gray448`로 추가 평가할 수 있습니다.
- per-query response의 **부호**는 공통 bias/양의 rescaling에 불변입니다. **class-mean DeltaP의 부호**는 일반적으로 불변이 아닙니다.
- nats softplus ablation은 finite-target IPO가 아닙니다. `nats_sq` 옵션 역시 reference-free evidence contrast이므로 standard IPO로 보고하지 마세요.
- 일부 unseen endpoint에서 실패하거나 random split에서 효과가 없으면 그 결과에 맞춰 claim을 좁히세요. 기존 favorable seed만 선택하지 마세요.

## 출처와 라이선스

실행 코드 일부는 해당 연구 프로젝트의 MIT-licensed release에서 이식했습니다(`LICENSE`). 질문 manifest는 제출 프로젝트의 AMBER/nocaps/VizWiz/COCO 기반 presence 질문과 공식 POPE COCO 질문에서 가져왔습니다. model prediction 등 불필요한 열을 제거했고, POPE는 사전 고정 subset을 만들었습니다. 데이터는 원 출처의 이용 조건을 유지하며 이 저장소의 MIT 표기는 코드에 적용됩니다. 원 사진은 배포하지 않습니다.

POPE: Li et al., “Evaluating Object Hallucination in Large Vision-Language Models,” EMNLP 2023. IPO: Azar et al., “A General Theoretical Paradigm to Understand Learning from Human Preferences,” AISTATS 2024.

반박 제출 본문에는 이 GitHub 링크를 넣지 마세요. 해당 author-response 지침에 맞춰 검증된 숫자와 설명을 글자 제한 내에 직접 적어야 합니다.
