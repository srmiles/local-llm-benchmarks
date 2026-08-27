#!/usr/bin/env bash
/data/llm/launch/gpu-teardown.sh llamacpp-bench 30 >/dev/null 2>&1
/data/llm/launch/gpu-teardown.sh llamacpp-ppl 30 >/dev/null 2>&1
docker ps --format '{{.Names}}' | grep -qx llamacpp-nemotron && { echo "already up"; exit 0; }
bash /data/llm/launch/start-llamacpp-nemotron-agent.sh >/dev/null 2>&1
for i in $(seq 1 40); do curl -fsS -m 5 http://127.0.0.1:8011/health >/dev/null 2>&1 && { echo HEALTHY; exit 0; }; sleep 5; done
echo "DID NOT COME BACK"
