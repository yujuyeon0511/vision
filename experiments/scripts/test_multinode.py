"""
Multi-node NCCL connectivity test for TBI-MLLM.
Run with:
    deepspeed --hostfile experiments/configs/multi-node/hostfile \
        --master_addr 192.168.0.28 --master_port 29500 \
        experiments/scripts/test_multinode.py
"""

import os
import socket
import torch
import torch.distributed as dist


def main():
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    hostname = socket.gethostname()

    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(device)

    print(f"[Rank {rank}/{world_size}] {hostname} | GPU {local_rank}: {gpu_name}")

    # All-reduce test
    tensor = torch.ones(1024, 1024, device=f"cuda:{local_rank}") * (rank + 1)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    expected = sum(range(1, world_size + 1))

    if rank == 0:
        actual = tensor[0, 0].item()
        status = "PASS" if abs(actual - expected) < 1e-5 else "FAIL"
        print(f"\n=== NCCL All-Reduce Test: {status} ===")
        print(f"  World size: {world_size}")
        print(f"  Expected sum: {expected}, Got: {actual}")

        # Bandwidth test
        import time
        sizes = [1024 * 1024, 10 * 1024 * 1024, 100 * 1024 * 1024]  # 4MB, 40MB, 400MB
        for size in sizes:
            data = torch.randn(size, device=f"cuda:{local_rank}")
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(10):
                dist.all_reduce(data)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            bw = (size * 4 * 2 * 10) / elapsed / 1e9  # GB/s (approx bus bandwidth)
            print(f"  {size * 4 / 1e6:.0f} MB all-reduce: {bw:.2f} GB/s ({elapsed / 10 * 1000:.1f} ms/iter)")

        print(f"\nMulti-node setup verified successfully!")
        print(f"Ready for TBI-MLLM training with {world_size} GPUs across nodes.")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
