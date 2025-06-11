import argparse
import time

import aiohttp
import asyncio
import tqdm
import numpy as np

from typing import List
from collections.abc import Callable

from request import (generate_requests, generate_radom_requests, APIRequest,
                     APIResponse)

background_tasks = set()


async def send_request(url: str, request: APIRequest, pbar):
    timeout = aiohttp.ClientTimeout(total=20 * 60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            start_time = time.time()
            async with session.post(f"{url}/generate",
                                    json=request) as response:
                if response.status == 200 or response.status == 201:
                    data = await response.json()
                    end_time = time.time()
                    data['end_time'] = end_time
                    data['latency'] = (end_time - start_time)
                    return APIResponse(**data)
                else:
                    print(f"Failed: {response.status}")
        except aiohttp.ClientError as e:
            print(f"Request failed: {e}")
            return None
        except asyncio.TimeoutError:
            end_time = time.time()
            print("Timeout error: Server did not respons in time "
                  f"over {end_time - start_time:.2f}s")
            return None
        except Exception as e:
            print(f" Unexpected error: {e}")
            return None


async def run_client(
    url: str,
    requests: List[APIRequest],
    num_benchmark_reqs: int,
    intervals: List[float],
    pbar: tqdm.std.tqdm,
    cb: Callable[[asyncio.Future], None],
):
    for request, interval in zip(requests, intervals):
        task = asyncio.create_task(send_request(url, request, pbar))
        background_tasks.add(task)
        task.add_done_callback(cb)

        await asyncio.sleep(interval)

    done, pending = await asyncio.wait(background_tasks)


def main(args):
    print(args)

    num_requests = args.num_requests + args.num_padding_requests

    if args.dataset is not None:
        requests: List[APIRequest] = generate_requests(
            dataset_name=args.dataset,
            tokenizer_name=args.tokenizer,
            num_requests=num_requests,
            max_seq_len=args.max_seq_len,
            num_samples=args.num_samples,
            ignore_eos=not args.disable_ignore_eos,
        )
    elif args.input_len is not None and args.output_len is not None:
        requests: List[APIRequest] = generate_radom_requests(
            tokenizer_name=args.tokenizer,
            max_input_len=args.input_len,
            max_output_len=args.output_len,
            num_requests=num_requests,
            max_seq_len=args.max_seq_len,
            num_samples=args.num_samples,
        )
    else:
        raise RuntimeError(
            "Invalid configuration: you must either provide a '--dataset'"
            "or specify both '--input-len' and '--output-len' arguments.")

    num_benchmark_reqs = args.num_requests
    intervals = np.random.exponential(1.0 / args.rate, size=num_requests)

    url = f"http://{args.host}:{args.port}"

    outputs: APIResponse = []

    start_time = time.time()
    with tqdm.tqdm(total=len(requests)) as pbar:

        def callback(future: asyncio.Future):
            pbar.update(1)
            output = future.result()
            if pbar.n <= num_benchmark_reqs:
                if output is not None:
                    outputs.append(output)

        asyncio.run(
            run_client(
                url=url,
                requests=requests,
                num_benchmark_reqs=num_benchmark_reqs,
                intervals=intervals,
                pbar=pbar,
                cb=callback,
            ))

    end_time = outputs[-1]['end_time']
    elapsed_time = end_time - start_time

    print(f"\nBenchmark time: {elapsed_time:.2f} s")

    print(f"Throughput (request): {len(requests) / elapsed_time:.2f} reqs/s")

    total_num_tokens = sum(len(o['token_ids']) for o in outputs)
    print(
        f"Throughput (token): {total_num_tokens / elapsed_time:.2f} tokens/s")

    total_num_output_tokens = sum(o['output_len'] for o in outputs)
    print("Throughput (output token): "
          f"{total_num_output_tokens / elapsed_time:.2f} tokens/s")

    avg_req_latency = np.mean([o['latency'] for o in outputs])
    print(f"Avg request latency: {avg_req_latency:.2f} s")

    avg_norm_token_latency = np.mean(
        [o['latency'] / o['output_len'] for o in outputs])
    print("Avg normalized output token latency: "
          f"{avg_norm_token_latency:.4f} s")

    ttfts = [o['token_latencies'][0] for o in outputs]
    ttft_tails = np.percentile(
        ttfts,
        method="closest_observation",
        q=[50, 90, 99],
    )
    print("TTFT:")
    print("P50: {:.2f} ms, P90: {:.2f} ms, P99: {:.2f} ms".format(
        ttft_tails[0] * 1000, ttft_tails[1] * 1000, ttft_tails[2] * 1000))

    tpots = []
    for o in outputs:
        tpots += o['token_latencies'][1:]
    tpot_tails = np.percentile(
        tpots,
        method="closest_observation",
        q=[50, 90, 99],
    )
    print("TPOT:")
    print("P50: {:.2f} ms, P90: {:.2f} ms, P99: {:.2f} ms".format(
        tpot_tails[0] * 1000, tpot_tails[1] * 1000, tpot_tails[2] * 1000))

    if args.print_output_text:
        outputs.sort(key=lambda o: o['output_len'])
        print("Below are the generated text of the processed requests:")
        for i, output in enumerate(outputs):
            print("### Generated output for request {}: {}\n".format(
                i, output['output_text']))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--input-len", type=int)
    parser.add_argument("--output-len", type=int)
    parser.add_argument("--dataset", type=str, choices=["alpaca", "humaneval"])
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--num-requests", type=int, default=1000)
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--rate", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--disable-ignore-eos", action="store_true")
    parser.add_argument("--print-output-text", action="store_true")
    parser.add_argument("--num-padding-requests", type=int, default=32)
    args = parser.parse_args()

    np.random.seed(args.seed)

    main(args)
