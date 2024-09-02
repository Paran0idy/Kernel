import triton
import triton.language as tl
import torch

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

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float16)
    for _ in range(0, K, BLOCK_K):
        a = tl.load(a_offset)
        b = tl.load(b_offset)

        acc += tl.dot(a, b, out_dtype=tl.float16)

        a_offset += BLOCK_K * stride_ak
        b_offset += BLOCK_K * stride_bk
    
    c_offset = c_ptr + (BLOCK_M * pid_m + tl.arange(0, BLOCK_M))[:, None] * stride_am + (BLOCK_N * pid_n + tl.arange(0, BLOCK_N))[None, :] * stride_bn
    tl.store(c_offset, acc)


def matmul(a, b,
           M, N, K,
):
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
        BLOCK_M, BLOCK_N, BLOCK_K,
    )
    return c


if __name__ == "__main__":
    M, N, K = 128, 128, 128
    a = torch.ones(M, K, dtype=torch.float16, device="cuda")
    b = torch.ones(K, N, dtype=torch.float16, device="cuda")
    triton_output = matmul(a, b, M, N, K)
    torch_output = torch.matmul(a, b)

    print(f"triton_output_with_fp16_inputs={triton_output}")
    print(f"torch_output_with_fp16_inputs={torch_output}")
    if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")