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

        nl = name.lower()
        if "lora" in nl or "hada" in nl or "lokr" in nl:
            lora_keys += 1
            if ("hada_w1_a" in nl or "hada_w2_a" in nl) and shape:
                detected_ranks.append(min(shape))
            elif ("lora_down" in nl or "lora.down" in nl or "lora_a" in nl or "lora.a" in nl or ".down.weight" in nl) and shape:
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

    # 1. メタデータからのアルゴリズム・Rank検出 (最優先)
    algo = ""
    meta_dim = None
    meta_conv_dim = None
    module_str = str(metadata.get("ss_network_module", "")).lower()

    args_raw = metadata.get("ss_network_args", "")
    if args_raw:
        try:
            if isinstance(args_raw, str):
                args_dict = json.loads(args_raw)
            else:
                args_dict = dict(args_raw)
            algo = str(args_dict.get("algo", "")).lower()
            if "conv_dim" in args_dict:
                meta_conv_dim = int(float(args_dict["conv_dim"]))
        except Exception:
            pass

    if "ss_network_dim" in metadata:
        try:
            meta_dim = int(float(metadata["ss_network_dim"]))
        except Exception:
            pass

    # 元Rank (Dim) の決定
    if meta_dim is not None and meta_dim > 0:
        orig_rank = meta_dim
        if meta_conv_dim is not None and meta_conv_dim > 0 and meta_conv_dim != meta_dim:
            orig_rank_str = f"Rank {meta_dim} (Conv {meta_conv_dim})"
        else:
            orig_rank_str = f"Rank {meta_dim}"
    elif detected_ranks:
        most_common_rank = Counter(detected_ranks).most_common(1)[0][0]
        orig_rank = most_common_rank
        orig_rank_str = f"Rank {most_common_rank}"
    else:
        orig_rank = 0
        orig_rank_str = "-"

    # 2. モデル種別の判定
    is_lycoris = ("lycoris" in module_str or "hada" in str(list(tensors.keys())).lower() or "lokr" in str(list(tensors.keys())).lower())
    is_sdxl = (has_sdxl_unet or "sdxl" in str(metadata).lower() or any("conditioner" in k for k in tensors))

    if is_lycoris or algo:
        if algo == "loha" or any("hada" in k.lower() for k in tensors):
            model_type = "LyCORIS (LoHa)"
        elif algo == "locon" or any("conv" in k.lower() for k in tensors):
            model_type = "LyCORIS (LoCon)"
        elif algo == "lokr" or any("lokr" in k.lower() for k in tensors):
            model_type = "LyCORIS (LoKr)"
        else:
            algo_name = algo.upper() if algo else "LoRA"
            model_type = f"LyCORIS ({algo_name})"
    elif lora_keys > 0 or detected_ranks:
        if is_sdxl:
            model_type = "SDXL LoRA"
        elif "flux" in str(metadata).lower() or any("double_blocks" in k for k in tensors):
            model_type = "FLUX.1 LoRA"
        elif has_sd15_unet or len(tensors) < 400:
            model_type = "SD 1.5 LoRA"
        else:
            model_type = "LoRA モデル"
    elif is_sdxl or len(tensors) > 450:
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
    summary = f"{primary_dtype_code} ({params_str})"

    # 3. 学習パラメータの解析 (画像枚数、リピート数、ステップ数など)
    train_img_count = None
    train_repeats = None
    train_steps = None
    train_epochs = None

    if "ss_num_train_images" in metadata:
        try:
            train_img_count = int(metadata["ss_num_train_images"])
        except Exception:
            pass

    if "ss_dataset_dirs" in metadata:
        try:
            ds_info = json.loads(metadata["ss_dataset_dirs"])
            if isinstance(ds_info, dict):
                img_sum = 0
                max_rep = 0
                for _, d_val in ds_info.items():
                    if isinstance(d_val, dict):
                        rep = int(d_val.get("n_repeats", 0))
                        cnt = int(d_val.get("img_count", 0))
                        if rep > max_rep:
                            max_rep = rep
                        img_sum += cnt
                if img_sum > 0:
                    train_img_count = img_sum
                if max_rep > 0:
                    train_repeats = max_rep
        except Exception:
            pass

    if "ss_max_train_steps" in metadata or "ss_steps" in metadata:
        try:
            train_steps = int(metadata.get("ss_max_train_steps") or metadata.get("ss_steps") or 0)
        except Exception:
            pass

    if "ss_num_epochs" in metadata or "ss_epochs" in metadata:
        try:
            train_epochs = int(metadata.get("ss_num_epochs") or metadata.get("ss_epochs") or 0)
        except Exception:
            pass

    info_parts = []
    if train_img_count is not None:
        info_parts.append(f"画像: {train_img_count}枚")
    if train_repeats is not None:
        info_parts.append(f"リピート: {train_repeats}回")
    if orig_rank > 0:
        info_parts.append(f"Rank: {orig_rank}")
    if train_steps is not None and train_steps > 0:
        info_parts.append(f"Steps: {train_steps:,}")
    train_info_str = " | ".join(info_parts) if info_parts else ""

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
        "vae_dtype": vae_dtype_str,
        "train_img_count": train_img_count,
        "train_repeats": train_repeats,
        "train_steps": train_steps,
        "train_epochs": train_epochs,
        "train_info_str": train_info_str
    }
