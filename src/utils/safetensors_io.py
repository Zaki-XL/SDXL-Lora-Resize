"""Safetensors ヘッダーの高速解析およびモデルメタデータインスペクター (Rank検出対応)"""
import json
import struct
import os
import math
from collections import Counter

DTYPE_SIZES = {
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
}

DTYPE_NAMES = {
    "F32": "Float32 (FP32)",
    "F16": "Float16 (FP16)",
    "BF16": "BFloat16 (BF16)",
    "F8_E4M3": "Float8 (e4m3fn)",
    "F8_E5M2": "Float8 (e5m2)",
    "I8": "Int8",
}

def read_safetensors_header(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"ファイルが見つかりません: {filepath}")

    with open(filepath, 'rb') as f:
        header_size_bytes = f.read(8)
        if len(header_size_bytes) < 8:
            raise ValueError("無効なSafetensorsファイルです (サイズ不足)")
        header_len = struct.unpack('<Q', header_size_bytes)[0]
        
        header_json_bytes = f.read(header_len)
        header = json.loads(header_json_bytes.decode('utf-8'))
        
    metadata = header.get("__metadata__", {})
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    return header_len, metadata, tensors

def is_vae_tensor(tensor_name: str) -> bool:
    name_lower = tensor_name.lower()
    return (
        "first_stage_model" in name_lower or
        "vae" in name_lower or
        name_lower.startswith("encoder.") or
        name_lower.startswith("decoder.")
    )

def inspect_model_metadata(filepath: str) -> dict:
    """
    Safetensorsヘッダーからモデル種別・元Rank・主要精度・パラメータ数・レイヤー構成を詳細解析します。
    """
    if not os.path.exists(filepath) or not filepath.lower().endswith(".safetensors"):
        return {
            "is_valid": False,
            "error_msg": "ファイルが存在しないか、.safetensors 形式ではありません。",
            "summary": "無効なファイル",
            "model_type": "非対応形式",
            "primary_dtype": "不明",
            "primary_dtype_code": "不明",
            "orig_rank": 0,
            "orig_rank_str": "-",
            "total_params": 0,
            "total_params_str": "不明",
            "tensor_count": 0,
            "unet_count": 0,
            "vae_count": 0,
            "vae_dtype": "なし"
        }

    try:
        header_len, metadata, tensors = read_safetensors_header(filepath)
    except Exception as e:
        return {
            "is_valid": False,
            "error_msg": f"Safetensorsヘッダー破損または解析失敗: {e}",
            "summary": "解析失敗",
            "model_type": "破損または非対応",
            "primary_dtype": "不明",
            "primary_dtype_code": "不明",
            "orig_rank": 0,
            "orig_rank_str": "-",
            "total_params": 0,
            "total_params_str": "不明",
            "tensor_count": 0,
            "unet_count": 0,
            "vae_count": 0,
            "vae_dtype": "なし"
        }

    if not tensors:
        return {
            "is_valid": False,
            "error_msg": "有効なテンソルデータが存在しません (中身が空のファイル)。",
            "summary": "データなし",
            "model_type": "空ファイル",
            "primary_dtype": "不明",
            "primary_dtype_code": "不明",
            "orig_rank": 0,
            "orig_rank_str": "-",
            "total_params": 0,
            "total_params_str": "0",
            "tensor_count": 0,
            "unet_count": 0,
            "vae_count": 0,
            "vae_dtype": "なし"
        }

    dtype_param_counter = Counter()
    total_params = 0
    unet_count = 0
    vae_count = 0
    vae_dtypes = set()
    lora_keys = 0
    detected_ranks = []

    has_sdxl_unet = False
    has_sd15_unet = False

    for name, info in tensors.items():
        shape = info.get("shape", [])
        dtype = info.get("dtype", "F32")
        numel = math.prod(shape) if shape else 0
        
        dtype_param_counter[dtype] += numel
        total_params += numel

        if "lora" in name.lower() or "lora_a" in name.lower() or "lora_down" in name.lower():
            lora_keys += 1
            if ("lora_down" in name or "lora_A" in name or "lora_down.weight" in name) and shape:
                detected_ranks.append(shape[0])

        if is_vae_tensor(name):
            vae_count += 1
            vae_dtypes.add(dtype)
        else:
            unet_count += 1

        if "model.diffusion_model.input_blocks" in name:
            if any(shape and s == 320 for s in shape) and len(tensors) > 400:
                has_sdxl_unet = True
            else:
                has_sd15_unet = True

    primary_dtype_code = dtype_param_counter.most_common(1)[0][0] if dtype_param_counter else "F32"
    primary_dtype_name = DTYPE_NAMES.get(primary_dtype_code, primary_dtype_code)

    # 元Rank (Dim) の特定
    if detected_ranks:
        most_common_rank = Counter(detected_ranks).most_common(1)[0][0]
        orig_rank = most_common_rank
        orig_rank_str = f"Rank {most_common_rank}"
    else:
        orig_rank = 0
        orig_rank_str = "-"

    # モデル種別の判定
    if lora_keys > 0 or detected_ranks:
        if has_sdxl_unet or "sdxl" in str(metadata).lower():
            model_type = "SDXL LoRA"
        else:
            model_type = "LoRA モデル"
    elif has_sdxl_unet or len(tensors) > 450:
        model_type = "SDXL Checkpoint"
    elif "flux" in str(metadata).lower() or any("double_blocks" in k for k in tensors):
        model_type = "FLUX.1 Model"
    elif has_sd15_unet or len(tensors) < 400:
        model_type = "SD 1.5 Checkpoint"
    else:
        model_type = "Safetensors モデル"

    if total_params >= 1_000_000_000:
        params_str = f"{total_params / 1_000_000_000:.2f} B"
    elif total_params >= 1_000_000:
        params_str = f"{total_params / 1_000_000:.1f} M"
    else:
        params_str = f"{total_params:,}"

    vae_dtype_str = "/".join([DTYPE_NAMES.get(d, d) for d in vae_dtypes]) if vae_dtypes else "なし"
    summary = f"{primary_dtype_code} ({model_type}, {params_str})"

    return {
        "is_valid": True,
        "error_msg": "",
        "summary": summary,
        "model_type": model_type,
        "primary_dtype": primary_dtype_name,
        "primary_dtype_code": primary_dtype_code,
        "orig_rank": orig_rank,
        "orig_rank_str": orig_rank_str,
        "total_params": total_params,
        "total_params_str": params_str,
        "tensor_count": len(tensors),
        "unet_count": unet_count,
        "vae_count": vae_count,
        "vae_dtype": vae_dtype_str
    }
