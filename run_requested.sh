#!/usr/bin/env bash
# Run one bounded reviewer-requested phase on one model. Existing completed runs are reused.
set -euo pipefail
cd "$(dirname "$0")"
PHASE=${1:-help}
FAMILY=${2:-llava}
PY=${PYTHON:-python}
OUT=${RESULTS_DIR:-results}
ADAPTERS=${PEC_ADAPTERS:-adapters}
ENGINE=${ENGINE:-unsloth}
mkdir -p "$OUT/logs"
case "$FAMILY" in
  llava) MODEL=unsloth/llava-1.5-7b-hf; PEC0=adapter_pec_llava7b; DPO0=adapter_dpo_llava7b; PEC1=adapter_pec_llava_s1; PEC2=adapter_pec_llava_s2;;
  q25) MODEL=unsloth/Qwen2.5-VL-7B-Instruct; PEC0=adapter_pec_q25_7b; DPO0=adapter_dpo_q25_7b; PEC1=adapter_pec_q25_s1; PEC2=adapter_pec_q25_s2;;
  *) echo 'model must be llava or q25' >&2; exit 2;;
esac
"$PY" prepare_data.py
EXTRA=()
if [[ ${CPU_OFFLOAD:-0} == 1 ]]; then
  if [[ $ENGINE != hf ]]; then echo 'CPU_OFFLOAD=1 requires ENGINE=hf' >&2; exit 2; fi
  EXTRA+=(--cpu-offload)
fi
run_eval() {
  local tag=$1 model=$2 data=$3
  shift 3
  local raw="$OUT/${FAMILY}_${tag}.jsonl"
  if [[ ! -f "$raw.complete.json" ]]; then
    if [[ -e "$raw" || -e "$raw.metadata.json" ]]; then
      echo "Partial run detected: $raw. Use a new RESULTS_DIR or preserve/rename the partial run before retry." >&2; exit 2
    fi
    "$PY" experiment.py evaluate --model "$model" --data "$data" --engine "$ENGINE" "${EXTRA[@]}" --out "$raw" "$@" 2>&1 | tee "$OUT/logs/${FAMILY}_${tag}.log"
  fi
  if [[ ! -f "$raw.summary.json" ]]; then
    "$PY" experiment.py summarize --data "$raw" --out "$raw.summary.json" > "$OUT/logs/${FAMILY}_${tag}_summary.log"
  fi
}
case "$PHASE" in
  smoke)
    "$PY" experiment.py check --data manifests/original_250.jsonl
    "$PY" experiment.py check --data manifests/pope_random_252.jsonl
    test -r "$ADAPTERS/$PEC0/adapter_model.safetensors"
    run_eval smoke_base "$MODEL" manifests/original_250.jsonl --endpoints gray --limit 2
    run_eval smoke_pec "$ADAPTERS/$PEC0" manifests/original_250.jsonl --endpoints gray --limit 2
    ;;
  endpoints)
    for condition in base pec0; do
      checkpoint=$MODEL; [[ $condition == pec0 ]] && checkpoint="$ADAPTERS/$PEC0"
      run_eval "${condition}_endpoints" "$checkpoint" manifests/original_250.jsonl --endpoints gray black noise other --other-repeats 1
    done
    ;;
  endpoint_seeds)
    run_eval pec1_endpoints "$ADAPTERS/$PEC1" manifests/original_250.jsonl --endpoints gray black noise other --other-repeats 1
    run_eval pec2_endpoints "$ADAPTERS/$PEC2" manifests/original_250.jsonl --endpoints gray black noise other --other-repeats 1
    ;;
  random)
    run_eval base_random "$MODEL" manifests/pope_random_252.jsonl --endpoints gray
    run_eval pec0_random "$ADAPTERS/$PEC0" manifests/pope_random_252.jsonl --endpoints gray
    ;;
  pope_controls)
    for split in popular adversarial; do
      run_eval "base_$split" "$MODEL" "manifests/pope_${split}_252.jsonl" --endpoints gray
      run_eval "pec0_$split" "$ADAPTERS/$PEC0" "manifests/pope_${split}_252.jsonl" --endpoints gray
    done
    ;;
  matched)
    # One model, one seed, one setting: direct IPO and matched-budget control.
    # All three must be trained on the same filtered pool; old PEC cannot stand in for this arm.
    for method in pec dpo ipo; do
      run="$OUT/${FAMILY}_matched_${method}_s0"
      if [[ ! -f "$run/complete.json" ]]; then
        if [[ -e "$run" ]]; then echo "Partial training run: $run; use a new RESULTS_DIR." >&2; exit 2; fi
        "$PY" experiment.py train --model "$MODEL" --method "$method" \
          --data manifests/matched_train.jsonl --held-out manifests/original_250.jsonl \
          --steps 200 --lr 3e-5 --seed 0 --beta 0.1 --delta 0.6 --lam 0.1 --out "$run" \
          2>&1 | tee "$OUT/logs/${FAMILY}_matched_${method}_s0.log"
      fi
      run_eval "matched_${method}_s0" "$run/adapter" manifests/original_250.jsonl --endpoints gray
    done
    ;;
  *) echo 'Usage: bash run_requested.sh {smoke|endpoints|endpoint_seeds|random|pope_controls|matched} {llava|q25}'; exit 2;;
esac
