#!/usr/bin/env bash
# Baseline spike: run stock `lerobot-eval` on the D1 fixture, with no Fathomline code in the path.
#
# Runs inside the repo's image, which supplies only the locked Python stack (lerobot 0.6.1,
# hf-libero, CPU torch) and the LIBERO meshes. Everything else is the stock path: LeRobot builds
# the env and the policy itself, and the checkpoint, the VLM and the tokenizer come from the Hub,
# not from the snapshots the image bakes in. Needs the network.
#
#   docker run --rm --cpus 12 --memory 24g \
#     -v "$PWD/docs/spike-baseline:/spike:ro" -v "$OUT:/out" \
#     --entrypoint bash fathomline /spike/run_stock_eval.sh /out/<name> [lerobot-eval flag ...]
#
# The defaults are the fixture's: `libero_object` task 0, 50 episodes from seed 1000. With
# `--eval.batch_size=B`, LeRobot gives sub-env j the init states j, j+B, j+2B, ..., so seed 1000+i
# meets init state i, as it does in the harness. Extra flags are appended and override the defaults.
# The spike's own runs overrode them (one sync env, 20 or 10 episodes, 4 CPUs each): see the
# Reproduce block in docs/SPIKE-baseline.md.
set -euo pipefail

out=${1:?usage: run_stock_eval.sh OUT_DIR [lerobot-eval flag ...]}
shift

# The image puts the harness on PYTHONPATH and switches the Hub off. Undo both.
unset PYTHONPATH
export HF_HUB_OFFLINE=0 HF_HOME=/tmp/hf
cd /tmp
if python -c "import fathomline" 2>/dev/null; then
  echo "fathomline is importable; this run must not have it on the path" >&2
  exit 1
fi

# D1's checkpoint revision, fetched from the Hub. lerobot-eval takes no revision flag.
revision=${POLICY_REVISION:-31d453f7edd78c839a8bbc39744a292686daf0de}
hf download lerobot/smolvla_libero --revision "$revision" --local-dir /tmp/policy >/dev/null
echo "policy lerobot/smolvla_libero@$revision"

# The camera mapping and the empty camera are the flags LeRobot's own CI passes for this
# checkpoint (.github/workflows/benchmark_tests.yml). Device and AMP are this host's.
exec lerobot-eval \
  --policy.path=/tmp/policy \
  --policy.device=cpu \
  --policy.use_amp=false \
  --policy.empty_cameras=1 \
  --env.type=libero \
  --env.task=libero_object \
  '--env.task_ids=[0]' \
  '--env.camera_name_mapping={"agentview_image": "camera1", "robot0_eye_in_hand_image": "camera2"}' \
  --eval.n_episodes=50 \
  --eval.batch_size=10 \
  --eval.use_async_envs=true \
  --seed=1000 \
  --output_dir="$out" \
  "$@"
