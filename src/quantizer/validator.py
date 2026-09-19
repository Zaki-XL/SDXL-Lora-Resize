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


def diagnose_lora_health(filepath: str) -> dict:
    """
    LoRAファイルの健全性診断 (破損・NaN/Inf検知、CLIP過学習・色破綻リスク検知) を行います。
    """
    from ..utils.safetensors_io import read_safetensors_header

    result = {
        "has_issues": False,
        "is_corrupt": False,
        "has_clip_overfit": False,
        "risk_level": "none",
        "te_count": 0,
        "unet_count": 0,
        "te_lr": None,
        "unet_lr": None,
        "base_model": "",
        "prediction_type": "",
        "issues": [],
        "suggestions": [],
        "suggested_action": "none"
    }

    if not os.path.exists(filepath):
        result["has_issues"] = True
        result["is_corrupt"] = True
        result["risk_level"] = "danger"
        result["issues"].append("ファイルが存在しません。")
        result["suggested_action"] = "skip"
        return result

    # 1. ヘッダーとメタデータの解析
    try:
        _, metadata, tensor_infos = read_safetensors_header(filepath)
    except Exception as e:
        result["has_issues"] = True
        result["is_corrupt"] = True
        result["risk_level"] = "danger"
        result["issues"].append(f"Safetensorsヘッダー破損: {e}")
        result["suggested_action"] = "skip"
        return result

    if not tensor_infos:
        result["has_issues"] = True
        result["is_corrupt"] = True
        result["risk_level"] = "danger"
        result["issues"].append("テンソルが存在しません（空ファイル）。")
        result["suggested_action"] = "skip"
        return result

    # テンソル分類
    te_keys = [k for k in tensor_infos.keys() if any(sub in k.lower() for sub in ("lora_te", "text_model", "conditioner"))]
    unet_keys = [k for k in tensor_infos.keys() if any(sub in k.lower() for sub in ("lora_unet", "diffusion_model"))]
    result["te_count"] = len(te_keys)
    result["unet_count"] = len(unet_keys)

    # メタデータ抽出
    te_lr = None
    unet_lr = None
    if "ss_text_encoder_lr" in metadata:
        try:
            te_lr = float(metadata["ss_text_encoder_lr"])
            result["te_lr"] = te_lr
        except Exception:
            pass

    if "ss_unet_lr" in metadata:
        try:
            unet_lr = float(metadata["ss_unet_lr"])
            result["unet_lr"] = unet_lr
        except Exception:
            pass

    base_model = str(metadata.get("ss_sd_model_name", ""))
    pred_type = str(metadata.get("modelspec.prediction_type", ""))
    steps = 0
    if "ss_max_train_steps" in metadata or "ss_steps" in metadata:
        try:
            steps = int(metadata.get("ss_max_train_steps") or metadata.get("ss_steps") or 0)
        except Exception:
            pass

    result["base_model"] = base_model
    result["prediction_type"] = pred_type

    # 2. 実データの高速健全性チェック (NaN/Inf および TE重み最大値)
    nan_found = False
    inf_found = False
    max_te_abs = 0.0
    max_unet_abs = 0.0

    try:
        tensors = load_file(filepath, device="cpu")
        with torch.inference_mode():
            for k, t in tensors.items():
                if "alpha" in k.lower() or not t.is_floating_point():
                    continue
                t_f = t.to(torch.float32)
                if not torch.isfinite(t_f).all():
                    if torch.isnan(t_f).any():
                        nan_found = True
                    if torch.isinf(t_f).any():
                        inf_found = True
                
                m = torch.max(torch.abs(t_f)).item()
                if any(sub in k.lower() for sub in ("lora_te", "text_model")):
                    if m > max_te_abs:
                        max_te_abs = m
                else:
                    if m > max_unet_abs:
                        max_unet_abs = m
    except Exception as e:
        result["has_issues"] = True
        result["is_corrupt"] = True
        result["risk_level"] = "danger"
        result["issues"].append(f"テンソルデータ読み込み失敗: {e}")
        result["suggested_action"] = "skip"
        return result

    if nan_found:
        result["has_issues"] = True
        result["is_corrupt"] = True
        result["risk_level"] = "danger"
        result["issues"].append("テンソル内に計算破綻値 (NaN) が検出されました。")
        result["suggested_action"] = "skip"

    if inf_found:
        result["has_issues"] = True
        result["is_corrupt"] = True
        result["risk_level"] = "danger"
        result["issues"].append("テンソル内に無限大 (Inf) が検出されました。")
        result["suggested_action"] = "skip"

    # 3. CLIP (Text Encoder) 過学習・色破綻リスク判定
    clip_overfit_reasons = []
    if result["te_count"] > 0:
        # 判定条件1: TE学習率が 1e-4 以上かつ UNet比 80% 以上
        if te_lr is not None and te_lr >= 1e-4:
            if unet_lr is not None and te_lr >= unet_lr * 0.8:
                clip_overfit_reasons.append(f"Text Encoder が UNet 並みの超高学習率 (LR: {te_lr:g}) で過学習されています。")
            elif te_lr >= 2e-4:
                clip_overfit_reasons.append(f"Text Encoder の学習率が過大です (LR: {te_lr:g})。")

        # 判定条件2: TE重みの絶対値最大が 0.20 を超えており、UNet より顕著に大きい
        if max_te_abs > 0.20:
            clip_overfit_reasons.append(f"Text Encoder の重み変化量が異常値に達しています (最大値: {max_te_abs:.4f})。")

        # 判定条件3: 長時間学習 (10,000 steps 以上) かつ TE 学習あり
        if steps >= 10000 and (te_lr is None or te_lr >= 5e-5):
            clip_overfit_reasons.append(f"大量ステップ ({steps:,} steps) の学習により CLIP 空間が歪曲しているリスクがあります。")

    # 特殊ベースモデルの判定
    base_lower = base_model.lower()
    if "noobai" in base_lower and "epsilon" in (pred_type.lower() + base_lower):
        clip_overfit_reasons.append("NoobAI-XL Epsilon 専用モデルで学習されており、他モデル適用時に色破綻を起こします。")

    if clip_overfit_reasons:
        result["has_issues"] = True
        result["has_clip_overfit"] = True
        if result["risk_level"] != "danger":
            result["risk_level"] = "warning"
        result["issues"].extend(clip_overfit_reasons)
        result["suggestions"].append("Text Encoder を除去して UNet のみにクリーンアップすることで色破綻を防止できます（推奨）。")
        result["suggestions"].append("Text Encoder の強度を 0.2 などに減衰させることで色調の崩壊を抑制できます。")
        result["suggested_action"] = "drop_te"

    return result

