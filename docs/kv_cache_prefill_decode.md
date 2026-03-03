# KV Cache in LMCache: Prefill, Decode, and Traffic Patterns

## Overview

LMCache reduces Time-To-First-Token (TTFT) and increases throughput by caching
and reusing KV caches across requests. It stores KV caches at multiple tiers
(GPU, CPU, Disk, Remote) and matches reused text patterns via prefix hashing
for cache hits.

---

## Prefill Phase

When a new request arrives, its prompt tokens go through the following flow:

### 1. Lookup (Prefix Matching)

The prompt token IDs are chunked into fixed-size chunks (default 256 tokens)
and hashed using incremental prefix hashing. LMCache queries its storage
backends to find matching cached KV chunks.

- **Entry point**: `get_num_new_matched_tokens()` in
  `lmcache/integration/vllm/vllm_v1_adapter.py`
- Uses `LookupClientInterface.lookup()` to check the token database
- Returns the number of tokens available in the external cache

### 2. Retrieve (Cache Hit)

If matching prefix chunks exist, the KV data is loaded from storage back into
GPU paged KV buffers:

```
Storage Backend (CPU/Disk/Remote)
        |
        v
  CPU MemoryObj (pinned tensor)
        |
        v
  GPUConnector.batched_to_gpu()
        |
        v
  vLLM Paged KV Buffer (GPU)
```

- **Entry point**: `LMCacheEngine.retrieve()` in `lmcache/v1/cache_engine.py`
- Only loads tokens beyond what vLLM already has locally computed
- Returns a boolean mask indicating which tokens were retrieved

### 3. Compute (Cache Miss)

Tokens without cached KV are computed normally by the model's forward pass.
This produces fresh KV cache entries on GPU.

### 4. Store (Save New KV)

After the forward pass, newly computed KV is saved to storage:

```
GPU Paged KV Buffer
        |
        v
  GPUConnector.batched_from_gpu()
        |
        v
  CPU MemoryObj (pinned tensor)
        |
        v
  StorageManager.batched_put()
        |
        +---> LocalCPUBackend (hot cache, in-memory)
        +---> LocalDiskBackend (NVMe/SSD)
        +---> RemoteBackend (Redis/S3/NIXL)
        +---> P2PBackend (other GPU nodes)
```

- **Entry point**: `wait_for_save()` in `lmcache/integration/vllm/vllm_v1_adapter.py`
- A `skip_leading_tokens` mask avoids re-storing chunks already in the cache
- Tokens are aligned to chunk boundaries before storing

---

## Decode Phase

Once prefill completes, the request enters decode mode (detected when
`len(new_token_ids) == 1`):

1. Each step generates **1 token**, extending the KV cache by one entry
2. The model attends over the **full KV cache** (cached prefix + all generated
   tokens so far)
3. **Saving is optional** — controlled by `save_decode_cache` (default `False`).
   Since decode tokens are unique to each request, caching them has lower reuse
   value
4. The KV stays in GPU VRAM; no storage traffic per decode step unless
   explicitly configured

---

## Traffic Patterns

### Summary Table

| Phase               | Direction                        | Data Size                             | Frequency             |
|---------------------|----------------------------------|---------------------------------------|-----------------------|
| **Prefill retrieve**| Storage -> CPU -> GPU            | Large burst (hundreds of MB)          | Once per request      |
| **Prefill store**   | GPU -> CPU -> Storage            | Same magnitude as retrieve            | Once per request      |
| **Decode**          | Stays in GPU                     | ~1 token x all layers (KBs)          | Every step (no I/O)   |
| **Disaggregated P/D** | Prefill node -> Decode node   | Entire prompt KV in one burst         | Once per handoff      |

### Traffic Characteristics

- **Prefill is bursty**: The big data movement happens during prefill — either
  loading cached KV or storing newly computed KV. This is the dominant traffic.

- **Decode is lightweight**: Only 1 token's KV per step. The overhead is
  negligible compared to prefill.

- **Chunking amortizes cost**: The 256-token chunk granularity means partial
  prefix matches still save significant compute and bandwidth. Only missing
  chunks are computed and stored.

- **Layerwise pipelining**: `store_layer()`/`retrieve_layer()` overlap
  GPU<->CPU transfer of layer N with storage I/O of layer N-1, hiding latency.

- **Multi-tier storage**: Hot cache (CPU RAM) serves most retrievals at memory
  bandwidth speeds. Disk and remote backends handle overflow with higher latency.

### Disaggregated Prefill/Decode

In disaggregated setups, prefill and decode run on separate instances:

```
Prefill Instance                     Decode Instance
      |                                    |
  Compute full KV                    Wait for KV
      |                                    |
  Store to cache ---------------------> Receive KV
      |       (NIXL / Socket / P2P)        |
  Done                               Generate tokens
```

This produces the single largest traffic burst: the full prompt KV is
transferred from the prefill instance to the decode instance over NIXL or
sockets, after which decode continues with minimal traffic.

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `lmcache/v1/cache_engine.py` | Core engine: `store()`, `retrieve()`, `store_layer()`, `retrieve_layer()` |
| `lmcache/integration/vllm/vllm_v1_adapter.py` | vLLM connector: `start_load_kv()`, `wait_for_save()`, `get_num_new_matched_tokens()` |
| `lmcache/v1/storage_backend/storage_manager.py` | Orchestrates multiple storage backends |
| `lmcache/v1/gpu_connector/gpu_connectors.py` | GPU <-> CPU data transfers |
| `lmcache/v1/token_database.py` | Token chunking and prefix hashing |
| `lmcache/v1/memory_management.py` | Memory allocation and reference counting |
