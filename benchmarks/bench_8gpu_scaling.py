import concurrent.futures
import json
import random
import time
import urllib.request
import uuid

BASE = "http://127.0.0.1:8000"
MODEL = "deepseek-v4.1-flash-1017"
HEADERS = {
    "Content-Type": "application/json",
    "anthropic-user-profile-id": "ds1017-bench-8gpu",
}
NOTHINK = {
    "reasoning_effort": "none",
    "chat_template_kwargs": {"thinking": False, "enable_thinking": False},
}
WORDS = (
    "alpha bravo cedar delta ember falcon granite harbor island juniper "
    "kestrel lantern meadow nickel orchid pepper quartz river saffron "
    "timber umber violet willow xenon yarrow zephyr"
).split()


def filler(target_tokens):
    rnd = random.Random(uuid.uuid4().int)
    lines = []
    n = 0
    while n < target_tokens:
        line = (
            f"{rnd.randrange(10**8):08d} "
            + " ".join(rnd.choice(WORDS) for _ in range(12))
            + ".\n"
        )
        lines.append(line)
        n += 17
    return "".join(lines)


def run_stream(body, timeout=1800):
    payload = {
        "model": MODEL,
        **NOTHINK,
        **body,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers=HEADERS,
    )
    start = time.monotonic()
    first = last = None
    usage = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            e = json.loads(line[5:])
            usage = e.get("usage") or usage
            for ch in e.get("choices") or []:
                if (ch.get("delta") or {}).get("content"):
                    now = time.monotonic()
                    first = first or now
                    last = now
    n = usage["completion_tokens"]
    pt = usage["prompt_tokens"]
    ttft = (first - start) if first else None
    gen_dur = (last - first) if (first and last and last > first) else None
    decode_tps = (n - 1) / gen_dur if (gen_dur and n > 1) else 0.0
    return {
        "prompt_tokens": pt,
        "completion_tokens": n,
        "ttft": ttft,
        "gen_dur": gen_dur,
        "decode_tps": decode_tps,
        "start": start,
        "first": first,
        "last": last,
    }


def bench_decode(concurrency, mode="code", max_tokens=512):
    def make_req(idx):
        uid = uuid.uuid4().hex
        if mode == "code":
            prompt = (
                f"Request {uid}-{idx}: Write a complete, dependency-free Python 3 "
                "library implementing a B+ tree, a skip list, and a thread-safe LRU "
                "cache with TTL expiration and comprehensive unit tests. Output only Python code."
            )
        else:
            prompt = (
                f"Request {uid}-{idx}: Write a detailed technical architectural treatise "
                "on GPU memory hierarchies, HBM2e bandwidth optimization, tensor cores, "
                "pipeline parallelism, and multi-tier NVMe KV caching. Use clear prose."
            )
        return {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "min_tokens": max_tokens,
            "temperature": 0.0,
        }

    with concurrent.futures.ThreadPoolExecutor(concurrency) as pool:
        rows = list(pool.map(lambda i: run_stream(make_req(i)), range(concurrency)))

    total_gen_tokens = sum(r["completion_tokens"] - 1 for r in rows)
    span_gen = max(r["last"] for r in rows) - min(r["first"] for r in rows)
    sum_stream_tps = sum(r["decode_tps"] for r in rows)
    avg_stream_tps = sum_stream_tps / len(rows)
    agg_tps = total_gen_tokens / span_gen if span_gen > 0 else sum_stream_tps
    return {
        "concurrency": concurrency,
        "mode": mode,
        "aggregate_decode_tps": round(agg_tps, 2),
        "sum_stream_tps": round(sum_stream_tps, 2),
        "per_stream_tps": round(avg_stream_tps, 2),
        "mean_ttft_s": round(sum(r["ttft"] for r in rows) / len(rows), 3),
    }


def bench_prefill(concurrency, target_tokens=16384):
    # Pre-generate distinct random prompts so client string generation is not timed
    bodies = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"Session {uuid.uuid4().hex}-{i}.\n"
                    + filler(target_tokens)
                    + "\nReply with OK.",
                }
            ],
            "max_tokens": 2,
            "temperature": 0.0,
        }
        for i in range(concurrency)
    ]
    t0 = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(concurrency) as pool:
        rows = list(pool.map(run_stream, bodies))
    span = max(r["first"] for r in rows) - min(r["start"] for r in rows)
    total_prompt = sum(r["prompt_tokens"] for r in rows)
    agg_prefill_tps = total_prompt / span
    return {
        "concurrency": concurrency,
        "prompt_tokens_per_req": rows[0]["prompt_tokens"],
        "total_prompt_tokens": total_prompt,
        "aggregate_prefill_tps": round(agg_prefill_tps, 1),
        "span_seconds": round(span, 2),
        "mean_ttft_s": round(sum(r["ttft"] for r in rows) / len(rows), 2),
        "max_ttft_s": round(max(r["ttft"] for r in rows), 2),
    }


results = {"decode_code": [], "decode_prose": [], "prefill_16k": []}

print("=== Benchmarking Decode (Code) at concurrency 1, 2, 4, 12 ===", flush=True)
for c in (1, 2, 4, 12):
    r = bench_decode(c, mode="code", max_tokens=512)
    results["decode_code"].append(r)
    print(json.dumps(r), flush=True)

print("=== Benchmarking Decode (Prose) at concurrency 1, 2, 4, 12 ===", flush=True)
for c in (1, 2, 4, 12):
    r = bench_decode(c, mode="prose", max_tokens=512)
    results["decode_prose"].append(r)
    print(json.dumps(r), flush=True)

print("=== Benchmarking Cold Prefill (16K per stream) at concurrency 1, 2, 4, 12 ===", flush=True)
for c in (1, 2, 4, 12):
    r = bench_prefill(c, target_tokens=14500)
    results["prefill_16k"].append(r)
    print(json.dumps(r), flush=True)

with open("/tmp/bench_8gpu_1_2_4_12.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved /tmp/bench_8gpu_1_2_4_12.json", flush=True)
