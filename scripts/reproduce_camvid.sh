#!/usr/bin/env bash
# Train and evaluate the CamVid models (inside the container).
#
#   bash scripts/reproduce_camvid.sh                      # all four models
#   bash scripts/reproduce_camvid.sh unet_bnn unet_dnn    # a subset
#
# Each model is trained with --resume, so an interrupted run continues where it stopped.
# Logs go to runs/camvid_<model>/{train,eval}.log.
set -u

models=("$@")
[ ${#models[@]} -eq 0 ] && models=(unet_bnn unet_dnn segnet_bnn segnet_dnn)

status=0
for model in "${models[@]}"; do
    config="configs/camvid_${model}.yaml"
    out="runs/camvid_${model}"
    mkdir -p "$out"
    case "$model" in
        *_bnn) methods="dnn mc vq" ;;
        *) methods="dnn vq" ;;
    esac

    echo "=== $model: train ($(date))"
    if ! python scripts/train_seg.py --config "$config" --resume >>"$out/train.log" 2>&1; then
        echo "=== $model: training failed, see $out/train.log"
        status=1
        continue
    fi
    echo "=== $model: eval ($(date))"
    if ! python scripts/eval_seg.py --config "$config" --ckpt "$out/model.pt" --methods $methods >"$out/eval.log" 2>&1; then
        echo "=== $model: evaluation failed, see $out/eval.log"
        status=1
        continue
    fi
    tail -n $(( $(wc -w <<<"$methods") + 1 )) "$out/eval.log"
done
echo "=== done ($(date))"
exit $status
