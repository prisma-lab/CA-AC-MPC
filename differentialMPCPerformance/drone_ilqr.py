from __future__ import annotations

from pathlib import Path
from typing import Optional
import os

import torch
from torch.utils.cpp_extension import load, CUDA_HOME

_EXT = None
_CUDA_KERNELS = None


def load_drone_cuda_kernels(
    *,
    build_directory: Optional[Path] = None,
    verbose: bool = False,
    force_rebuild: bool = False,
):
    """Load CUDA kernels for fused operations with autograd support."""
    global _CUDA_KERNELS
    if _CUDA_KERNELS is not None and not force_rebuild:
        return _CUDA_KERNELS

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available, cannot load CUDA kernels")

    root = Path(__file__).resolve().parent
    cpp_dir = root / "cpp"
    cu_path = cpp_dir / "drone_kernels.cu"
    wrapper_path = cpp_dir / "drone_cuda_wrapper.cpp"
    build_dir = build_directory or (root / "build")
    build_dir.mkdir(parents=True, exist_ok=True)

    cuda_home = CUDA_HOME or os.environ.get("CUDA_HOME")
    cuda_home = Path(cuda_home) if cuda_home else None

    lib_dirs = []
    if cuda_home is not None:
        for lib_name in ("lib64", "lib"):
            cand = cuda_home / lib_name
            if cand.exists():
                lib_dirs.append(cand)

    extra_ldflags = []
    for lib_dir in lib_dirs:
        extra_ldflags.append(f"-L{lib_dir}")
        extra_ldflags.append(f"-Wl,-rpath,{lib_dir}")

    extra_include = [str(cpp_dir)]  # Include cpp directory for header
    if cuda_home is not None:
        include_dir = cuda_home / "include"
        if include_dir.exists():
            extra_include.append(str(include_dir))

    _CUDA_KERNELS = load(
        name="drone_cuda_kernels",
        sources=[str(cu_path), str(wrapper_path)],
        extra_cflags=["-O3", "-ffast-math"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        extra_ldflags=extra_ldflags,
        extra_include_paths=extra_include,
        verbose=verbose,
        build_directory=str(build_dir),
    )
    return _CUDA_KERNELS


def load_drone_ilqr_ext(
    *,
    build_directory: Optional[Path] = None,
    verbose: bool = False,
    force_rebuild: bool = False,
    with_fused_kernels: bool = True,
):
    """Load the ILQR solver extension.

    Args:
        with_fused_kernels: If True and CUDA is available, compile with fused
            CUDA kernels for better performance.
    """
    global _EXT
    if _EXT is not None and not force_rebuild:
        return _EXT

    root = Path(__file__).resolve().parent
    cpp_dir = root / "cpp"
    cpp_path = cpp_dir / "drone_ilqr_ext.cpp"
    build_dir = build_directory or (root / "build")
    build_dir.mkdir(parents=True, exist_ok=True)

    cuda_home = CUDA_HOME or os.environ.get("CUDA_HOME")
    cuda_home = Path(cuda_home) if cuda_home else None

    lib_dirs = []
    if cuda_home is not None:
        for lib_name in ("lib64", "lib"):
            cand = cuda_home / lib_name
            if cand.exists():
                lib_dirs.append(cand)

    extra_ldflags = []
    for lib_dir in lib_dirs:
        extra_ldflags.append(f"-L{lib_dir}")
        extra_ldflags.append(f"-Wl,-rpath,{lib_dir}")

    extra_include = [str(cpp_dir)]  # Include cpp directory for headers
    if cuda_home is not None:
        include_dir = cuda_home / "include"
        if include_dir.exists():
            extra_include.append(str(include_dir))

    # Source files
    sources = [str(cpp_path)]

    # Add CUDA kernels if available and requested
    use_cuda = torch.cuda.is_available() and with_fused_kernels
    extra_cflags = ["-O3", "-ffast-math", "-march=native"]
    extra_cuda_cflags_list = [
        "-O3",
        "--use_fast_math",
        "-allow-unsupported-compiler",
        "--expt-relaxed-constexpr",
    ]

    # Check if NVCC is actually available before trying to use CUDA kernels
    can_compile_cuda = False
    if use_cuda and cuda_home is not None:
        nvcc_path = cuda_home / "bin" / "nvcc"
        if nvcc_path.exists():
            can_compile_cuda = True
        else:
            # Check system NVCC
            import shutil
            if shutil.which("nvcc") is not None:
                can_compile_cuda = True

    if use_cuda and can_compile_cuda:
        cu_path = cpp_dir / "drone_kernels.cu"
        wrapper_path = cpp_dir / "drone_cuda_wrapper.cpp"
        backward_kernel_path = cpp_dir / "backward_pass_kernel.cu"
        forward_kernel_path = cpp_dir / "forward_pass_kernel.cu"
        pnqp_kernel_path = cpp_dir / "pnqp_kernel.cu"
        if cu_path.exists() and wrapper_path.exists():
            sources.extend([str(cu_path), str(wrapper_path)])
            if backward_kernel_path.exists():
                sources.append(str(backward_kernel_path))
                extra_cflags.append("-DDRONE_ILQR_USE_FUSED_BACKWARD")
                extra_cuda_cflags_list.append("-DDRONE_ILQR_USE_FUSED_BACKWARD")
            if forward_kernel_path.exists():
                sources.append(str(forward_kernel_path))
                extra_cflags.append("-DDRONE_ILQR_USE_FUSED_FORWARD")
                extra_cuda_cflags_list.append("-DDRONE_ILQR_USE_FUSED_FORWARD")
            if pnqp_kernel_path.exists():
                sources.append(str(pnqp_kernel_path))
                extra_cflags.append("-DDRONE_ILQR_USE_PNQP")
                extra_cuda_cflags_list.append("-DDRONE_ILQR_USE_PNQP")
            extra_cflags.append("-DDRONE_ILQR_USE_CUDA_KERNELS")
            extra_cuda_cflags_list.append("-DDRONE_ILQR_USE_CUDA_KERNELS")
    else:
        # Fallback to ATen-only build (no fused CUDA kernels)
        use_cuda = False

    _EXT = load(
        name="drone_ilqr_ext_fused" if use_cuda else "drone_ilqr_ext",
        sources=sources,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags_list,
        extra_ldflags=extra_ldflags,
        extra_include_paths=extra_include,
        with_cuda=use_cuda,
        verbose=verbose,
        build_directory=str(build_dir),
    )

    # Enable TF32 for Ampere+ GPUs (RTX 30xx/40xx)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return _EXT
