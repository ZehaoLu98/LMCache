#!/bin/bash
# Disaggregated prefill example with LMCache prefix caching (remote backend).
#
# This variant starts a LMCache remote server alongside the prefiller, decoder,
# and proxy.  The prefiller can retrieve KV caches from the remote backend on
# cache hits, and the decoder saves completed KV caches back to the remote
# backend after every request, so subsequent requests benefit from prefix
# reuse.
#
# Config files used:
#   configs/lmcache-prefiller-pd-with-remote-config.yaml
#   configs/lmcache-decoder-pd-with-remote-config.yaml
#
# Usage:
#   bash disagg_example_1p1d_with_lmcache.sh [options]
#
# Options:
#   --model MODEL                     HuggingFace model id (default: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)
#   --prefiller-device DEVICE_ID      CUDA device for the prefiller (default: 0)
#   --decoder-device DEVICE_ID        CUDA device for the decoder   (default: 1)
#   --gpu-memory-utilization FLOAT    GPU memory utilization fraction (default: 0.90)
#   --max-num-seqs INT                Max concurrent sequences        (default: 512)
#   --max-model-len INT               Max model context length        (default: 16384)
#   --lmcache-server-port PORT        Port for the LMCache remote server (default: 6800)

echo "Warning: LMCache disaggregated prefill support for vLLM v1 is experimental and subject to change."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIDS=()

# NOTE: For correct KV cache transfer, ensure all processes use the same
# PYTHONHASHSEED to keep the hash of the KV cache consistent across processes.
export PYTHONHASHSEED=0

# ── defaults ────────────────────────────────────────────────────────────────
MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
PREFILLER_DEVICE_ID=0
DECODER_DEVICE_ID=1
GPU_MEMORY_UTILIZATION=0.85
MAX_NUM_SEQS=2
MAX_MODEL_LEN=16384
LMCACHE_SERVER_PORT=6800

# ── argument parsing ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)                   MODEL="$2";                   shift 2 ;;
        --prefiller-device)        PREFILLER_DEVICE_ID="$2";     shift 2 ;;
        --decoder-device)          DECODER_DEVICE_ID="$2";       shift 2 ;;
        --gpu-memory-utilization)  GPU_MEMORY_UTILIZATION="$2";  shift 2 ;;
        --max-num-seqs)            MAX_NUM_SEQS="$2";            shift 2 ;;
        --max-model-len)           MAX_MODEL_LEN="$2";           shift 2 ;;
        --lmcache-server-port)     LMCACHE_SERVER_PORT="$2";     shift 2 ;;
        *)
            echo "Unknown option: $1"
            echo "Run with --help to see available options."
            exit 1
            ;;
    esac
done

# ── helpers ───────────────────────────────────────────────────────────────────
check_hf_token() {
    if [ -z "$HF_TOKEN" ]; then
        echo "HF_TOKEN is not set. Please set it to your Hugging Face token."
        exit 1
    fi
    if [[ "$HF_TOKEN" != hf_* ]]; then
        echo "HF_TOKEN is not a valid Hugging Face token. Please set it to your Hugging Face token."
        exit 1
    fi
    echo "HF_TOKEN is set and valid."
}

check_num_gpus() {
    num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    if [ "$num_gpus" -lt 2 ]; then
        echo "You need at least 2 GPUs to run disaggregated prefill."
        exit 1
    else
        echo "Found $num_gpus GPUs."
    fi
}

ensure_python_library_installed() {
    echo "Checking if $1 is installed..."
    python -c "import $1" > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        if [ "$1" == "nixl" ]; then
            echo "$1 is not installed. Please refer to https://github.com/ai-dynamo/nixl for installation."
        else
            echo "$1 is not installed. Please install it via pip install $1."
        fi
        exit 1
    else
        echo "$1 is installed."
    fi
}

cleanup() {
    echo "Stopping everything…"
    trap - INT TERM USR1   # prevent re-entrancy

    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Killing process $pid"
            kill "$pid" 2>/dev/null
        fi
    done

    sleep 2

    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Force killing process $pid"
            kill -9 "$pid" 2>/dev/null
        fi
    done

    kill -- -$$ 2>/dev/null

    echo "All processes stopped."
    exit 0
}

wait_for_server() {
    local port=$1
    local timeout_seconds=1200
    local start_time=$(date +%s)

    echo "Waiting for server on port $port..."

    while true; do
        if curl -s "localhost:${port}/v1/completions" > /dev/null; then
            return 0
        fi

        local now=$(date +%s)
        if (( now - start_time >= timeout_seconds )); then
            echo "Timeout waiting for server on port $port"
            return 1
        fi

        sleep 1
    done
}

# ── main ──────────────────────────────────────────────────────────────────────
main() {
    check_hf_token
    check_num_gpus
    ensure_python_library_installed lmcache
    ensure_python_library_installed nixl
    ensure_python_library_installed pandas
    ensure_python_library_installed datasets
    ensure_python_library_installed vllm

    trap cleanup INT
    trap cleanup USR1
    trap cleanup TERM

    echo "==================================================="
    echo "Model:                  $MODEL"
    echo "Prefiller device:       $PREFILLER_DEVICE_ID"
    echo "Decoder device:         $DECODER_DEVICE_ID"
    echo "GPU memory utilization: $GPU_MEMORY_UTILIZATION"
    echo "Max num seqs:           $MAX_NUM_SEQS"
    echo "Max model len:          $MAX_MODEL_LEN"
    echo "LMCache server port:    $LMCACHE_SERVER_PORT"
    echo "==================================================="
    echo "Logs: lmcache_server.log  prefiller.log  decoder.log  proxy.log"
    echo ""

    # ── 1. LMCache remote server ─────────────────────────────────────────────
    # The prefiller loads cached KV from this server on hits; the decoder
    # saves its KV back here after each request.
    echo "Launching LMCache remote server on port $LMCACHE_SERVER_PORT..."
    python -m lmcache.v1.server localhost $LMCACHE_SERVER_PORT \
        > >(tee lmcache_server.log) 2>&1 &
    lmcache_server_pid=$!
    PIDS+=($lmcache_server_pid)

    # Give the server a moment to bind before the vllm workers try to connect.
    sleep 2

    # ── 2. Disagg proxy ──────────────────────────────────────────────────────
    echo "Launching disagg proxy..."
    python3 "$SCRIPT_DIR/../disagg_proxy_server.py" \
        --host localhost \
        --port 9100 \
        --prefiller-host localhost \
        --prefiller-port 7100 \
        --num-prefillers 1 \
        --decoder-host localhost \
        --decoder-port 7200 \
        --decoder-init-port 7300 \
        --decoder-alloc-port 7400 \
        --proxy-host localhost \
        --proxy-port 7500 \
        --num-decoders 1 \
        > >(tee proxy.log) 2>&1 &
    proxy_pid=$!
    PIDS+=($proxy_pid)

    # ── 3. Decoder ───────────────────────────────────────────────────────────
    # Receives KV from the prefiller via NIXL, then saves it to the remote
    # backend so future prefill runs can skip the work.
    echo "Launching decoder..."
    PYTORCH_ALLOC_CONF=expandable_segments:True \
        UCX_TLS=cuda_ipc,cuda_copy,tcp \
        LMCACHE_CONFIG_FILE="$SCRIPT_DIR/configs/lmcache-decoder-pd-with-remote-config.yaml" \
        VLLM_ENABLE_V1_MULTIPROCESSING=1 \
        VLLM_WORKER_MULTIPROC_METHOD=spawn \
        CUDA_VISIBLE_DEVICES=$DECODER_DEVICE_ID \
        vllm serve $MODEL \
        --port 7200 \
        --enforce-eager \
        --no-enable-prefix-caching \
        --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
        --max-num-seqs $MAX_NUM_SEQS \
        --max-model-len $MAX_MODEL_LEN \
        --kv-transfer-config \
        '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_consumer","kv_connector_extra_config": {"discard_partial_chunks": false, "lmcache_rpc_port": "consumer1", "skip_last_n_tokens": 1}}' \
        > >(tee decoder.log) 2>&1 &
    decoder_pid=$!
    PIDS+=($decoder_pid)

    # ── 4. Prefiller ─────────────────────────────────────────────────────────
    # On a cache miss, runs prefill and NIXL-transfers KV to the decoder.
    # On a cache hit (from LocalCPU or remote backend), transfers the cached
    # KV directly — skipping GPU computation entirely.
    # --no-enable-prefix-caching is required: vLLM's own prefix cache would
    # return kv_transfer_params=null (no transfer), breaking the decoder.
    echo "Launching prefiller..."
    PYTORCH_ALLOC_CONF=expandable_segments:True \
        UCX_TLS=cuda_ipc,cuda_copy,tcp \
        LMCACHE_CONFIG_FILE="$SCRIPT_DIR/configs/lmcache-prefiller-pd-with-remote-config.yaml" \
        VLLM_ENABLE_V1_MULTIPROCESSING=1 \
        VLLM_WORKER_MULTIPROC_METHOD=spawn \
        CUDA_VISIBLE_DEVICES=$PREFILLER_DEVICE_ID \
        vllm serve $MODEL \
        --port 7100 \
        --enforce-eager \
        --no-enable-prefix-caching \
        --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
        --max-num-seqs $MAX_NUM_SEQS \
        --max-model-len $MAX_MODEL_LEN \
        --kv-transfer-config \
        '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_producer","kv_connector_extra_config": {"discard_partial_chunks": false, "lmcache_rpc_port": "producer1"}}' \
        > >(tee prefiller.log) 2>&1 &
    prefiller_pid=$!
    PIDS+=($prefiller_pid)

    wait_for_server 7200
    wait_for_server 7100
    wait_for_server 9100

    echo "==================================================="
    echo "All servers are up. Send requests to localhost:9100"
    echo "Press Ctrl-C to terminate all instances."
    echo "==================================================="

    while true; do
        sleep 1
    done
}

main
