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

"""Vocab-sharded sonic sampling: each rank reduces its logits shard, the packed slabs
cross the TP group through the multicast exchange, and the merged draw must equal the
unsharded draw on the full logits, bitwise, on every rank (sample and chain verify)."""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytest.importorskip("sonic_sampler")

from tokenspeed_kernel.ops.sampling.sonic import (  # noqa: E402
    MAX_K,
    SamplingBuffers,
    ScopedIndicators,
    fused_multistep,
    fused_singular,
)
from tokenspeed_kernel.ops.sampling.sonic_exchange import SlabExchange  # noqa: E402

BLOCK_N = 4096
TRIALS = 8


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _buffers(rows: int, vocab: int, timesteps: int, device: torch.device):
    buffers = SamplingBuffers.default(
        size=rows, timesteps=timesteps, vocab_size=vocab, device=device
    )
    kinds = ["greedy", "topk", "topp", "topk_topp"]
    top_k = [
        1 if kinds[i % 4] == "greedy" else (8 if "topk" in kinds[i % 4] else MAX_K)
        for i in range(rows)
    ]
    top_p = [0.9 if "topp" in kinds[i % 4] else 1.0 for i in range(rows)]
    scoped = ScopedIndicators.from_params(
        size=rows,
        timesteps=timesteps,
        multiplicative=[1.0] * rows,
        frequency=[0.0] * rows,
        presence=[0.0] * rows,
        biases=[None] * rows,
        temperature=[1.0] * rows,
        top_k=top_k,
        top_p=top_p,
        min_p=[0.0] * rows,
        top_logprobs=[0] * rows,
        bitmasks=[None] * rows,
        greedy_drafts=True,
    )
    buffers.allocate(
        encodings=[torch.zeros(0, dtype=torch.int64)] * rows,
        multiplicative=[1.0] * rows,
        frequency=[0.0] * rows,
        presence=[0.0] * rows,
        biases=[None] * rows,
        temperature=[1.0] * rows,
        top_k=top_k,
        top_p=top_p,
        min_p=[0.0] * rows,
        top_logprobs=[0] * rows,
        positions=list(range(rows)),
        indicators=scoped,
    )
    torch.cuda.synchronize()
    counts = buffers.repetition.counts
    pen = buffers.repetition
    common = dict(
        indicators=buffers.flags.target,
        temperature=buffers.temperature,
        top_k=buffers.top_k,
        top_p=buffers.top_p,
        min_p=buffers.min_p,
        grammar=buffers.grammar,
        context_counts=counts.context,
        decode_counts=counts.decode,
        repetition_penalties=pen.multiplicative,
        frequency_penalties=pen.frequency,
        presence_penalties=pen.presence,
        logit_bias=buffers.bias,
        block_n=BLOCK_N,
        enable_pdl=False,
        update_counts=False,
    )
    return common, torch.tensor([k == 1 for k in top_k], device=device)


def _worker(rank: int, world: int, vocab: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    shard = vocab // world
    blocks = _cdiv(shard, BLOCK_N)
    gen = torch.Generator(device=device)
    lo, hi = rank * shard, (rank + 1) * shard
    for rows in (1, 8):
        common, greedy_rows = _buffers(rows, vocab, 1, device)
        seeds = torch.randint(
            0, 2**31, (rows,), device=device, generator=gen.manual_seed(7)
        ).to(torch.int64)
        offsets = torch.zeros(rows, dtype=torch.int32, device=device)
        exchange = SlabExchange(
            dist.group.WORLD, rank, world, 16, blocks, MAX_K, device
        )
        local = torch.empty(rows, blocks * MAX_K, dtype=torch.uint32, device=device)
        out_full = torch.empty(rows, 1, dtype=torch.int32, device=device)
        out_shard = torch.empty(rows, 1, dtype=torch.int32, device=device)
        shard_kwargs = dict(
            world_size=world, local_rank=rank, shard_size=shard, local_vocab=shard
        )
        for trial in range(TRIALS):
            gen.manual_seed(100 + trial)
            logits = (torch.randn(rows, vocab, device=device, generator=gen) * 3).to(
                torch.bfloat16
            )
            offsets.fill_(trial)
            ref = fused_singular(
                logits=logits,
                output_tokens=out_full,
                noise_seeds=seeds,
                noise_offsets=offsets,
                noise_steps=1,
                exchange=None,
                **common,
            ).tokens.clone()
            got = fused_singular(
                logits=logits[:, lo:hi].contiguous(),
                output_tokens=out_shard,
                noise_seeds=seeds,
                noise_offsets=offsets,
                noise_steps=1,
                scratchpad=local,
                exchange=exchange,
                **shard_kwargs,
                **common,
            ).tokens
            # greedy rows may break exact bf16 ties differently (ties excepted); stochastic rows are bitwise
            tie = logits.gather(1, got.long()).view(-1) == logits.gather(
                1, ref.long()
            ).view(-1)
            bad = (got.view(-1) != ref.view(-1)) & ~(greedy_rows & tie)
            assert not bool(
                bad.any()
            ), f"rank {rank} rows {rows} trial {trial}: {got.view(-1).tolist()} vs {ref.view(-1).tolist()}"
            peers = [torch.empty_like(got) for _ in range(world)]
            dist.all_gather(peers, got)
            assert all(
                torch.equal(p, got) for p in peers
            ), "ranks disagree on the sharded draw"
        # chain verify, gamma = 3
        gamma, n = 3, 4
        common_m, _ = _buffers(rows, vocab, n, device)
        exchange_m = SlabExchange(
            dist.group.WORLD, rank, world, 16 * n, blocks, MAX_K, device
        )
        local_m = torch.empty(
            rows * n, blocks * MAX_K, dtype=torch.uint32, device=device
        )
        drafted_full = torch.empty(rows, n, dtype=torch.int32, device=device)
        drafted_shard = torch.empty(rows, n, dtype=torch.int32, device=device)
        for trial in range(TRIALS):
            gen.manual_seed(500 + trial)
            logits = (
                torch.randn(rows * n, vocab, device=device, generator=gen) * 3
            ).to(torch.bfloat16)
            drafts = torch.randint(
                0, vocab, (rows, gamma), device=device, generator=gen
            ).to(torch.int32)
            offsets.fill_(trial * n)
            for buf in (drafted_full, drafted_shard):
                buf[:, :gamma].copy_(drafts)
                buf[:, gamma:].zero_()
            ref = fused_multistep(
                logits=logits,
                drafted_tokens=drafted_full,
                lookahead=gamma,
                output_tokens=drafted_full,
                noise_seeds=seeds,
                noise_offsets=offsets,
                noise_steps=n,
                exchange=None,
                **common_m,
            )
            ref_tokens, ref_offsets = ref.tokens.clone(), ref.offsets.clone()
            got = fused_multistep(
                logits=logits[:, lo:hi].contiguous(),
                drafted_tokens=drafted_shard,
                lookahead=gamma,
                output_tokens=drafted_shard,
                noise_seeds=seeds,
                noise_offsets=offsets,
                noise_steps=n,
                scratchpad=local_m,
                exchange=exchange_m,
                **shard_kwargs,
                **common_m,
            )
            assert torch.equal(
                got.offsets, ref_offsets
            ), f"rank {rank}: accept lengths differ"
            # compare the emitted prefix (accepted drafts + correction); later columns are unspecified
            for i in range(rows):
                k = int(ref_offsets[i].item()) + 1
                a, b = got.tokens[i, :k], ref_tokens[i, :k]
                if not torch.equal(a, b):
                    row_logits = logits[i * n : (i + 1) * n]
                    tied = torch.equal(
                        row_logits.gather(1, a.long()[:, None]),
                        row_logits.gather(1, b.long()[:, None]),
                    )
                    assert tied, f"rank {rank} row {i}: {a.tolist()} vs {b.tolist()}"
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs >= 2 GPUs")
@pytest.mark.parametrize("vocab", [151936, 163840])
def test_sharded_draws_equal_unsharded(vocab: int) -> None:
    world = min(4, torch.cuda.device_count())
    mp.spawn(
        _worker, args=(world, vocab, 29800 + vocab % 1000), nprocs=world, join=True
    )
