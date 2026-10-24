#!/usr/bin/env bash
# 2026-10-24 (post-reboot rerun): Nemotron 3.5 Bolt E4B + 74B, then Nemotron-3.5-Embed 1B/8B + Rerank-8B, card 2 :8020.
# ONE script owns the card-2 window: stops :8011 once, restores it once (trap). Replaces run-bolt-20261024.sh +
# 20261024-nemoembed/run-embed.sh, whose log-line chaining let :8011 restart onto card 2 under a bench -> DEVICE_LOST.
# Every launch goes through card2-empty.sh first; a non-empty card aborts the whole run.
set -uo pipefail
GUARD=/data/llm/launch/card2-empty.sh
IMG=llama.cpp:sycl-f16-9ae73e68f-patched-3bd0de68e
T0=$(date '+%F %T')
faults() { echo "-- kernel faults since start: $(journalctl -k --since "$T0" --no-pager | grep -cE 'Engine reset|CAT error|Timedout job|GuC')"; }
restore() {
  /data/llm/launch/gpu-teardown.sh llamacpp-bench 30 || true
  /data/llm/launch/gpu-teardown.sh llamacpp-embbench 30 || true
  echo "== $(date -u +%FT%TZ) restoring llamacpp-nemotron =="; docker start llamacpp-nemotron
}
trap restore EXIT
echo "== $(date -u +%FT%TZ) stopping llamacpp-nemotron for the card-2 window =="
# plain stop, NOT gpu-teardown.sh: that one ends in `docker rm -f` and would delete the prod container
docker stop -t 120 llamacpp-nemotron >/dev/null 2>&1 || true
$GUARD 120 || exit 1

# ---------- Bolt ----------
D=/data/llm/benchmarks/20261024-bolt; mkdir -p $D/void-r0; mv $D/*.json $D/*.serverlog $D/void-r0/ 2>/dev/null
E4=/data/llm/nemotron-3.5-bolt-e4b-GGUF; B74=/data/llm/nemotron-3.5-bolt-74b-a9b-GGUF
one() { local name=$1; shift
  echo "=== $name $(date -u +%FT%TZ)"
  $GUARD 90 || exit 1
  if ! env IMAGE=$IMG NAME=llamacpp-bench ALIAS=$name "$@" /data/llm/launch/bench-slot.sh; then echo "!! $name launch failed"; return 1; fi
  python3 /data/llm/benchmarks/bench-candidate.py --port 8020 --card 1 --name $name --out $D/$name.json 2>&1 | tail -30
  [[ ${PIPESTATUS[0]} -ne 0 ]] && echo "!! $name bench failed"
  docker logs llamacpp-bench > $D/$name.serverlog 2>&1
  /data/llm/launch/gpu-teardown.sh llamacpp-bench 30 || true
  faults
  # a hung card poisons every later arm: stop the run instead of piling resets on it
  grep -q DEVICE_LOST $D/$name.serverlog && { echo "!! DEVICE_LOST in $name - aborting run"; exit 1; }
}
for r in r1 r2; do
one bolt-e4b-solo-$r     MODEL_DIR=$E4 MODEL=Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf CTX=65536 MEM=16g
one bolt-e4b-dflash7-$r  MODEL_DIR=$E4 MODEL=Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf DRAFT=Nemotron-3.5-Bolt-E4B-DFlash-BF16.gguf SPEC_ARGS="--spec-type draft-dflash --spec-draft-n-max 7" CTX=65536 MEM=16g
one bolt-74b-solo-$r     MODEL_DIR=$B74 MODEL=NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf CTX=65536 MEM=28g
one bolt-74b-dflash6-$r  MODEL_DIR=$B74 MODEL=NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf DRAFT=NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr.gguf SPEC_ARGS="--spec-type draft-dflash --spec-draft-n-max 6" CTX=65536 MEM=28g
done
echo "BOLT DONE $(date -u +%FT%TZ)"

# ---------- Embed / rerank ----------
B=/data/llm/benchmarks/20261024-nemoembed; cd $B
G=/data/llm/nemotron-3.5-embed-GGUF; V=/data/llm/benchmarks/20261011-embgemma2/venv/bin/python
echo "=== quantize $(date -u +%FT%TZ)"
for m in 1b 8b; do [ -f $G/nemotron-3.5-embed-$m-Q8_0.gguf ] || docker run --rm -v $G:/m --entrypoint /app/llama $IMG quantize /m/nemotron-3.5-embed-$m-f16.gguf /m/nemotron-3.5-embed-$m-Q8_0.gguf Q8_0 2>&1 | tail -2; done
ls -la $G
echo "=== CPU refs $(date -u +%FT%TZ)"
$V ref.py /data/llm/hf-nemotron-3.5-embed-1b ref-1b.npy float32 2>&1 | tail -1
$V ref.py /data/llm/hf-nemotron-3.5-embed-8b ref-8b.npy bfloat16 2>&1 | tail -1
run() { # tag dir file style [extra...]
  local tag=$1 dir=$2 file=$3 style=$4; shift 4
  echo "=== $tag $(date -u +%FT%TZ)"
  $GUARD 90 || exit 1
  IMAGE=$IMG ./launch-embed.sh "$dir" "$file" "$@" || { echo "LAUNCH FAILED $tag"; return 1; }
  if [ "$style" = nemotron ]; then r=ref-1b.npy; [[ $tag == *8b* ]] && r=ref-8b.npy; $V parity.py http://localhost:8020 $r $tag; fi
  $V bench.py --url http://localhost:8020 --tag "$tag" --style $style || echo "BENCH FAILED $tag"
  docker logs llamacpp-embbench > results/$tag.serverlog 2>&1
  /data/llm/launch/gpu-teardown.sh llamacpp-embbench 30 || true
  faults
}
STAGE1_DUMP=$B/results/stage1 run embgemma-q8 /data/llm/embeddinggemma embeddinggemma-300m-qat-Q8_0.gguf gemma --pooling mean
run nemo-1b-q8  $G nemotron-3.5-embed-1b-Q8_0.gguf nemotron
run nemo-1b-f16 $G nemotron-3.5-embed-1b-f16.gguf  nemotron
MEM=12g run nemo-8b-q8 $G nemotron-3.5-embed-8b-Q8_0.gguf nemotron
echo "=== rerank $(date -u +%FT%TZ)"
$GUARD 90 || exit 1
export ZE_AFFINITY_MASK=1
venv-xpu/bin/python rerank.py /data/llm/bge-reranker-v2-m3-hf bge-v2-m3 --dtype fp16 2>&1 | grep -vi warn | tail -3
venv-xpu/bin/python rerank.py /data/llm/hf-nemotron-3.5-rerank-8b nemo-rerank-8b --dtype bf16 --bs 4 2>&1 | grep -vi warn | tail -3
faults
echo "ALLDONE $(date -u +%FT%TZ)"
