# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Three ways to compute K3's latent MoE down-projection, measured head to head.

The op is ``hidden[T, 7168] @ W[3584, 7168].T``. Today ``W`` is replicated on
every rank, which costs 4.7GB per rank and reads all 49MB on every call.
The alternatives shard it:

* row shard  -- split the contraction dim; each rank contracts a 7168/TP slice
  into a full-width partial and the ranks all-reduce it.
* column shard -- split the output dim; each rank produces a 3584/TP block and
  the ranks all-gather on hidden.

Run per TP config (world = TP size), one process per rank, as a script the
way ``tune_route.py`` is::

    down_proj_shard_bench.py            # decode sweep, T = 9 * batch
    down_proj_shard_bench.py prefill    # prefill sweep

Three ways to measure nothing, all of which cost a day here:

1. The one-shot all-reduce lane must be armed with ``configure_group``.
   ``prepare_all_reduce_lane`` only widens an already-configured group and
   returns False otherwise, silently leaving the group on NCCL.
2. Time under CUDA-graph replay. Decode runs captured, and eager collective
   dispatch costs ~80us -- an order above the effect being measured.
3. The column-shard number is only meaningful when the group can reach the
   triton all-gather. A TP group spread over hosts is routed to NCCL by
   ``AutoBackend``'s topology test, which doubles that column here; a rack
   whose NVLink fabric spans those hosts wants the fabric probe instead.
4. Cycle weight copies past L2 on *every* path. One replicated weight is 49MB
   against a 129MB L2, so reusing a single tensor measures a fully resident
   weight; production streams a different layer's weight on every call. Warm,
   this sweep reports the row shard *losing* at TP16; cold it wins.
"""

from __future__ import annotations

import statistics
import sys
from types import SimpleNamespace

import torch
import torch.distributed as dist

HIDDEN, LATENT = 7168, 3584
DECODE_TS = tuple(9 * bs for bs in range(1, 33))  # DSPARK-8: T = 9 * batch
PREFILL_TS = (288, 576, 864, 1152, 1728, 2304, 4608, 8192)


def _server_args(world: int) -> None:
    from tokenspeed.runtime.utils.env import global_server_args_dict

    tp = SimpleNamespace(tp_size=world, tp_ep_size=world)
    global_server_args_dict["mapping"] = SimpleNamespace(
        nprocs_per_node=torch.cuda.device_count(), attn=tp, dense=tp, moe=tp
    )
    global_server_args_dict["chunked_prefill_size"] = 8192
    global_server_args_dict["max_prefill_tokens"] = 8192
    global_server_args_dict["max_model_len"] = 65536


def _backend(group, rank: int, max_tokens: int):
    from tokenspeed.runtime.distributed.comm_backend.auto import AutoBackend
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager,
    )

    process_group_manager.register_process_group("nccl", group, dist.group.WORLD)
    backend = AutoBackend()
    # Initial arming; prepare_all_reduce_lane only widens an armed group.
    armed = backend._trtllm_ar.configure_group(
        rank=rank, group=group, max_token_num=max_tokens, hidden_dim=LATENT
    )
    return backend, armed


def _weight_copies(rank: int, world: int):
    """Enough copies of each weight layout to overflow L2 on every path."""
    l2 = torch.cuda.get_device_properties(0).L2_cache_size
    col, row = LATENT // world, HIDDEN // world
    full = [
        torch.randn(LATENT, HIDDEN, device="cuda", dtype=torch.bfloat16)
        for _ in range(max(2, l2 // (LATENT * HIDDEN * 2) + 1))
    ]
    rows = [w[:, rank * row : (rank + 1) * row].contiguous() for w in full]
    cols = [w[rank * col : (rank + 1) * col, :].contiguous() for w in full]
    rows += [
        torch.randn(LATENT, row, device="cuda", dtype=torch.bfloat16)
        for _ in range(max(2, l2 // (LATENT * row * 2) + 1) - len(rows))
    ]
    cols += [
        torch.randn(col, HIDDEN, device="cuda", dtype=torch.bfloat16)
        for _ in range(max(2, l2 // (col * HIDDEN * 2) + 1) - len(cols))
    ]
    return full, rows, cols


def _bench(fn, per: int = 8, reps: int = 9) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(per):
            fn()
    torch.cuda.synchronize()
    dist.barrier()
    samples = []
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(reps):
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / per)
        dist.barrier()
    return statistics.median(samples)


def main() -> None:
    prefill = sys.argv[1:2] == ["prefill"]
    ts = PREFILL_TS if prefill else DECODE_TS
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    col, row = LATENT // world, HIDDEN // world
    _server_args(world)
    group = tuple(range(world))
    backend, armed = _backend(group, rank, max(ts))
    torch.manual_seed(11)
    full, rows, cols = _weight_copies(rank, world)

    if rank == 0:
        print(
            f"world={world} armed={armed} copies: repl {len(full)} row "
            f"{len(rows)} col {len(cols)}"
        )
        print(
            f"{'T':>6} {'replicated':>11} {'row+AR':>9} {'col+AG':>9} "
            f"{'AR net':>8} {'AG net':>8}  winner"
        )
    for tokens in ts:
        torch.manual_seed(500 + tokens % 997)
        hidden = torch.randn(tokens, HIDDEN, device="cuda", dtype=torch.bfloat16)
        shard_in = hidden[:, rank * row : (rank + 1) * row].contiguous()
        out = torch.empty(tokens, LATENT, device="cuda", dtype=torch.bfloat16)
        partial = torch.empty(tokens, LATENT, device="cuda", dtype=torch.bfloat16)
        block = torch.empty(tokens, col, device="cuda", dtype=torch.bfloat16)
        step = {"i": 0}

        def replicated():
            step["i"] += 1
            torch.mm(hidden, full[step["i"] % len(full)].t(), out=out)

        def row_all_reduce():
            step["i"] += 1
            torch.mm(shard_in, rows[step["i"] % len(rows)].t(), out=partial)
            backend.all_reduce(partial, group)

        def col_all_gather():
            step["i"] += 1
            torch.mm(hidden, cols[step["i"] % len(cols)].t(), out=block)
            backend.all_gather(block, group, dim=-1)

        base, ar, ag = (
            _bench(replicated),
            _bench(row_all_reduce),
            _bench(col_all_gather),
        )
        if rank == 0:
            winner = min((base, "replicated"), (ar, "row+AR"), (ag, "col+AG"))
            print(
                f"{tokens:6d} {base:10.2f}u {ar:8.2f}u {ag:8.2f}u "
                f"{base - ar:+7.2f}u {base - ag:+7.2f}u  {winner[1]}",
                flush=True,
            )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
