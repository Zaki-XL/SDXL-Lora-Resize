"""変換後モデルの整合性・健全性検証モジュール"""
import os
import torch
from safetensors.torch import load_file
from ..utils.safetensors_io import is_vae_tensor

def verify_quantized_model(
    orig_path_or_meta: str | dict,
    quant_path: str,
    format_key: str,
    keep_vae_fp16: bool = True,
    svd_rank: str = "none"
) -> tuple[bool, list[str]]:
    """
    変換後のモデルを検証し、(合格フラグ, 問題点リスト) を返します。
    orig_path_or_meta には元ファイルパス(str)または事前抽出メタデータ辞書(dict)を渡すことができます。
    メタデータ辞書を渡した場合は元ファイルの再読み込みI/Oを完全にスキップします。
    """
    issues = []
    
    if not os.path.exists(quant_path):
        return False, ["出力ファイルが存在しません。"]

    # 1. 元テンソル情報の取得 (辞書が渡された場合はディスクI/Oをスキップ)
    if isinstance(orig_path_or_meta, dict) and orig_path_or_meta:
        orig_meta = orig_path_or_meta
    else:
        try:
            orig_dict = load_file(orig_path_or_meta)
            orig_meta = {
                k: {
                    "shape": tuple(t.shape),
                    "dtype": t.dtype,
                    "dim": t.dim(),
                    "is_floating_point": t.is_floating_point()
                }
                for k, t in orig_dict.items()
            }
        except Exception as e:
            return False, [f"元ファイルの読み込みに失敗: {e}"]

    try:
        quant_dict = load_file(quant_path)
    except Exception as e:
        return False, [f"変換後ファイルの読み込みに失敗 (ファイル破損の可能性): {e}"]

    # 2. キーの完全一致確認 (スケール係数キーなどを除外して比較)
    orig_keys = set(orig_meta.keys())
    quant_keys = set(quant_dict.keys())
    
    # INT8/INT4/NVFP4のスケールキーを除外した実データキー
    base_quant_keys = {k for k in quant_keys if not (k.endswith(".scale") or k.endswith(".nvfp4_scale"))}

    missing_keys = orig_keys - base_quant_keys
    if missing_keys:
        sample_missing = list(missing_keys)[:3]
        issues.append(f"テンソルキーが欠落しています ({len(missing_keys)}個欠落, 例: {sample_missing})")

    # 3. テンソルごとの Shape & NaN / Inf / 型チェック
    nan_count = 0
    inf_count = 0
    shape_mismatch = 0
    vae_check_fail = 0
    has_svd = (svd_rank and svd_rank != "none")
    target_r = int(svd_rank) if has_svd else None

    with torch.inference_mode():
        for k in orig_keys:
            if k not in quant_dict:
                continue
            meta = orig_meta[k]
            quant_t = quant_dict[k]
            orig_shape = meta["shape"]
            orig_dim = meta["dim"]

            # Shapeチェック
            is_down = any(sub in k for sub in ("lora_down", "lora_A", "lora.down", "hada_w1_b", "hada_w2_b", ".down.weight"))
            is_up = any(sub in k for sub in ("lora_up", "lora_B", "lora.up", "hada_w1_a", "hada_w2_a", ".up.weight"))
            is_svd_weight = (is_down or is_up)

            if has_svd and is_svd_weight:
                # SVDリサイズ対象: 次元数が target_rank 以下になっているか、または元の次元の妥当性を確認
                if quant_t.dim() != orig_dim:
                    shape_mismatch += 1
                elif target_r is not None:
                    # down/up の Rank 次元が target_r 以下であることを確認
                    if is_down:
                        if quant_t.size(0) > max(target_r, orig_shape[0]):
                            shape_mismatch += 1
                    elif is_up:
                        if quant_t.dim() >= 2 and quant_t.size(1) > max(target_r, orig_shape[1]):
                            shape_mismatch += 1
            else:
                # SVDなし、または非LoRAテンソル: Shape完全一致を要求
                if tuple(quant_t.shape) != orig_shape:
                    shape_mismatch += 1

            # NaN / Inf チェック (ワンパス判定)
            if quant_t.is_floating_point():
                t_f32 = quant_t.to(torch.float32)
                if not torch.isfinite(t_f32).all():
                    if torch.isnan(t_f32).any():
                        nan_count += 1
                    if torch.isinf(t_f32).any():
                        inf_count += 1

            # VAE保護チェック
            if keep_vae_fp16 and is_vae_tensor(k):
                if quant_t.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                    vae_check_fail += 1

    if shape_mismatch > 0:
        issues.append(f"テンソルの形状(Shape)不一致が {shape_mismatch} 件検出されました。")
    if nan_count > 0:
        issues.append(f"非数(NaN)が {nan_count} 個のテンソルで検出されました（計算破綻）。")
    if inf_count > 0:
        issues.append(f"無限大(Inf)が {inf_count} 個のテンソルで検出されました。")
    if vae_check_fail > 0:
        issues.append(f"VAE保護対象テンソルが誤って低精度化されています ({vae_check_fail} 件)。")

    is_valid = (len(issues) == 0)
    return is_valid, issues
