"""SafetensorsヘッダーからSVDリサイズ × 量子化精度の掛け合わせサイズを事前推定するエンジン"""
import os
import math
from ..utils.safetensors_io import read_safetensors_header, is_vae_tensor, DTYPE_SIZES
from ..utils.gpu_info import PRECISION_OPTIONS

def estimate_quantized_size(filepath: str, svd_rank: str, precision_key: str, keep_vae_fp16: bool = True) -> tuple[int, int, float]:
    """
    Returns: (元サイズ bytes, 推定後サイズ bytes, 削減率 %)
    """
    if not os.path.exists(filepath):
        return 0, 0, 0.0

    original_size = os.path.getsize(filepath)

    if not filepath.lower().endswith(".safetensors"):
        return original_size, int(original_size * 0.5), 50.0

    try:
        header_len, metadata, tensors = read_safetensors_header(filepath)
    except Exception:
        return original_size, int(original_size * 0.5), 50.0

    p_info = PRECISION_OPTIONS.get(precision_key, {})
    target_bytes_per_elem = p_info.get("bytes_per_elem", 2.0)
    has_svd = (svd_rank and svd_rank != "none")
    target_rank_int = int(svd_rank) if has_svd else 999999

    estimated_tensor_data_size = 0

    for name, info in tensors.items():
        shape = info.get("shape", [])
        dtype = info.get("dtype", "F16")
        elem_orig_size = DTYPE_SIZES.get(dtype, 2)
        elem_target_size = target_bytes_per_elem if precision_key != "keep" else elem_orig_size

        if keep_vae_fp16 and is_vae_tensor(name):
            elem_target_size = 2 if dtype in ("F16", "BF16") else 4

        # LoRA SVD Rank リサイズ計算
        if has_svd and ("lora_down" in name or "lora_A" in name):
            orig_r = shape[0] if shape else 1
            new_r = min(orig_r, target_rank_int)
            numel = math.prod(shape) if shape else 0
            scaled_numel = int(numel * (new_r / max(1, orig_r)))
            estimated_tensor_data_size += int(scaled_numel * elem_target_size)
        elif has_svd and ("lora_up" in name or "lora_B" in name):
            orig_r = shape[1] if len(shape) > 1 else 1
            new_r = min(orig_r, target_rank_int)
            numel = math.prod(shape) if shape else 0
            scaled_numel = int(numel * (new_r / max(1, orig_r)))
            estimated_tensor_data_size += int(scaled_numel * elem_target_size)
        else:
            numel = math.prod(shape) if shape else 0
            estimated_tensor_data_size += int(numel * elem_target_size)

    estimated_total = 8 + header_len + estimated_tensor_data_size
    reduction_pct = (1.0 - (estimated_total / original_size)) * 100.0 if original_size > 0 else 0.0

    return original_size, estimated_total, max(0.0, reduction_pct)
