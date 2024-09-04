import triton
import triton.language as tl
import torch

@triton.autotune(
    configs = [
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages= 2, num_warps = 4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_stages= 2, num_warps = 4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_stages= 2, num_warps = 4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages= 2, num_warps = 4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages= 2, num_warps = 4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages= 2, num_warps = 4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_stages= 2, num_warps = 4),
    ],
    key = ['M', 'N', 'K'],
)

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M : tl.constexpr,
    BLOCK_N : tl.constexpr,
    BLOCK_K : tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid = N // BLOCK_N
    pid_m = pid // num_pid
    pid_n = pid % num_pid

    a_offset = a_ptr + (pid_m * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] * stride_am + tl.arange(0, BLOCK_K)[None, :] * stride_ak
    b_offset = b_ptr + (pid_n * BLOCK_N + tl.arange(0, BLOCK_N))[None, :] * stride_bn + tl.arange(0, BLOCK_K)[:, None] * stride_bk

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for _ in range(0, K, BLOCK_K):
        a = tl.load(a_offset)
        b = tl.load(b_offset)

        acc += tl.dot(a, b)

        a_offset += BLOCK_K * stride_ak
        b_offset += BLOCK_K * stride_bk
    
    acc = acc.to(tl.float16)
    c_offset = c_ptr + (BLOCK_M * pid_m + tl.arange(0, BLOCK_M))[:, None] * stride_cm + (BLOCK_N * pid_n + tl.arange(0, BLOCK_N))[None, :] * stride_cn
    tl.store(c_offset, acc)


def matmul(a, b):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64
    M, K = a.shape
    K, N = b.shape

    c = torch.empty((M, N), dtype=torch.float16, device=a.device)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]), )
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


if __name__ == "__main__":
    M, N, K = 128, 128, 128
    a = torch.ones(M, K, dtype=torch.float16, device="cuda")
    b = torch.ones(K, N, dtype=torch.float16, device="cuda")
    triton_output = matmul(a, b)
    torch_output = torch.matmul(a, b)

    print(f"triton_output_with_fp16_inputs={triton_output}")
    print(f"torch_output_with_fp16_inputs={torch_output}")
    if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K'],  # Argument names to use as an x-axis for the plot
        x_vals=[
            128 * i for i in range(2, 33)
        ],  # Different possible values for `x_name`
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        line_vals=['cublas', 'triton'],
        # Label name for the lines
        line_names=["cuBLAS", "Triton"],
        # Line styles
        styles=[('green', '-'), ('blue', '-')],
        ylabel="TFLOPS",  # Label name for the y-axis
        plot_name="matmul-performance",  # Name for the plot, used also as a file name for saving the plot.
        args={},
    )
)
def benchmark(M, N, K, provider):
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'cublas':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


benchmark.run(show_plots=True, print_data=True, save_path="./")