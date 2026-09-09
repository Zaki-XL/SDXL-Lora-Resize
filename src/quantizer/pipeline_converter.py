"""SVD Rank リサイズと量子化精度変換を掛け合わせて実行する複合パイプライン"""
import os
import time
import torch
from safetensors.torch import load_file, save_file
from .base import BaseQuantizer
from ..utils.safetensors_io import is_vae_tensor, read_safetensors_header

from .svd_resizer import low_rank_svd, find_lora_pairs

class CombinedPipelineConverter(BaseQuantizer):
    def __init__(self, svd_rank: str, precision_key: str, keep_vae_fp16: bool = True):
        super().__init__(keep_vae_fp16=keep_vae_fp16)
        self.svd_rank = svd_rank
        self.has_svd = (svd_rank and svd_rank != "none")
        self.target_rank = int(svd_rank) if self.has_svd else None
        self.precision_key = precision_key
        self.last_orig_meta = {}

    def process(self, input_path: str, output_path: str, progress_callback=None, log_callback=None) -> bool:
        if log_callback:
            svd_desc = f"Rank {self.target_rank}" if self.has_svd else "Rank維持"
            log_callback(f"[{self.device_name}] 複合処理開始 (SVD: {svd_desc} × 精度: {self.precision_key}): {os.path.basename(input_path)} ...")

        state_dict = load_file(input_path)
        total_keys = len(state_dict)
        final_dict = {}

        # Safetensors メタデータの取得と引き継ぎ準備
        try:
            _, orig_metadata, _ = read_safetensors_header(input_path)
        except Exception:
            orig_metadata = {}
        out_metadata = {str(mk): str(mv) for mk, mv in orig_metadata.items()}

        # 検証用の元テンソルメタデータ（Shape, dtype, dim）を軽量に記録
        self.last_orig_meta = {
            k: {
                "shape": tuple(t.shape),
                "dtype": t.dtype,
                "dim": t.dim(),
                "is_floating_point": t.is_floating_point()
            }
            for k, t in state_dict.items()
        }

        with torch.inference_mode():
            # ---------------- 1. SVD Rank Resize (LoRAの場合) ----------------
            if self.has_svd:
                pairs = find_lora_pairs(state_dict)
                total_pairs = len(pairs)
                dev = self.device if self.device.type == "cuda" else torch.device("cpu")
                processed_alphas = set()

                if log_callback and total_pairs > 0:
                    log_callback(f"[SVD開始] 対象LoRAペア: {total_pairs} 件の分解・特異値削減を計算中...")

                svd_start_time = time.time()
                log_interval = max(1, total_pairs // 5)  # 約20%刻みでログ通知

                for p_idx, (down_k, up_k, alpha_k) in enumerate(pairs):
                    if self.is_cancelled:
                        return False

                    down_t = state_dict[down_k]
                    up_t = state_dict[up_k]
                    orig_dtype = down_t.dtype

                    down_dev = down_t.to(device=dev, dtype=torch.float32)
                    up_dev = up_t.to(device=dev, dtype=torch.float32)
                    is_conv2d = (down_dev.dim() == 4 and up_dev.dim() == 4)

                    if is_conv2d:
                        r = down_dev.size(0)
                        in_dim = down_dev.size(1) * down_dev.size(2) * down_dev.size(3)
                        out_dim = up_dev.size(0)
                        down_mat = down_dev.flatten(start_dim=1)
                        up_mat = up_dev.squeeze(-1).squeeze(-1)
                    else:
                        r = down_dev.size(0)
                        in_dim = down_dev.size(1)
                        out_dim = up_dev.size(0)
                        down_mat = down_dev
                        up_mat = up_dev

                    if self.target_rank < r:
                        try:
                            new_up_mat, new_down_mat = low_rank_svd(up_mat, down_mat, self.target_rank)
                            k = self.target_rank

                            if is_conv2d:
                                new_down_t = new_down_mat.view(k, down_dev.size(1), down_dev.size(2), down_dev.size(3))
                                new_up_t = new_up_mat.view(out_dim, k, 1, 1)
                            else:
                                new_down_t = new_down_mat
                                new_up_t = new_up_mat

                            state_dict[down_k] = new_down_t.to(dtype=orig_dtype).cpu()
                            state_dict[up_k] = new_up_t.to(dtype=orig_dtype).cpu()
                            if alpha_k in state_dict and alpha_k not in processed_alphas:
                                orig_alpha = float(state_dict[alpha_k].item())
                                new_alpha = orig_alpha * (float(k) / max(1.0, float(r)))
                                state_dict[alpha_k] = torch.tensor(new_alpha, dtype=state_dict[alpha_k].dtype)
                                processed_alphas.add(alpha_k)
                        except Exception:
                            pass

                    if progress_callback:
                        elapsed = time.time() - svd_start_time
                        speed_pairs = (p_idx + 1) / max(0.001, elapsed)
                        # SVDフェーズは全体進捗の前半 (0%〜50%)
                        progress_callback(p_idx + 1, total_pairs * 2 if self.has_svd else total_keys, f"[SVD] {down_k}", speed_pairs)

                    if log_callback and ((p_idx + 1) % log_interval == 0 or p_idx + 1 == total_pairs):
                        pct = ((p_idx + 1) / total_pairs) * 100
                        log_callback(f"[SVD進捗] {p_idx + 1} / {total_pairs} ペア完了 ({pct:.1f}%) ...")

                if total_pairs > 0:
                    if log_callback:
                        log_callback(f"[OK] SVD Rank リサイズ計算完了 (Target Rank: {self.target_rank})")

                    # SVDリサイズに応じたメタデータ (Kohya形式等) の更新
                    out_metadata["ss_network_dim"] = str(self.target_rank)
                    if "ss_network_alpha" in out_metadata:
                        try:
                            orig_net_dim = float(orig_metadata.get("ss_network_dim", 128))
                            orig_net_alpha = float(orig_metadata.get("ss_network_alpha", 1))
                            new_net_alpha = orig_net_alpha * (float(self.target_rank) / max(1.0, orig_net_dim))
                            out_metadata["ss_network_alpha"] = f"{new_net_alpha:g}"
                        except Exception:
                            pass
                elif log_callback:
                    log_callback("[SVDスキップ] 対象LoRAペアが検出されなかったため（LoHa/LoKr等の特殊LyCORIS、または非LoRA）、SVDリサイズをスキップし元のRank構造を維持しました")

            # ---------------- 2. 量子化・精度変換 ----------------
            if log_callback:
                log_callback(f"[量子化開始] 精度: {self.precision_key} への変換処理中 (全 {total_keys} テンソル)...")

            start_time = time.time()
            quant_log_interval = max(1, total_keys // 4)  # 約25%刻みでログ通知

            for idx, (k, tensor) in enumerate(state_dict.items()):
                if self.is_cancelled:
                    return False

                if self.keep_vae_fp16 and is_vae_tensor(k):
                    if tensor.dtype in (torch.float32, torch.float64):
                        final_dict[k] = tensor.to(torch.float16)
                    else:
                        final_dict[k] = tensor
                elif self.precision_key == "keep":
                    final_dict[k] = tensor
                elif self.precision_key == "fp8_e4m3fn":
                    if tensor.is_floating_point():
                        if self.device.type == "cuda":
                            final_dict[k] = tensor.to(device=self.device).to(torch.float8_e4m3fn).cpu()
                        else:
                            final_dict[k] = tensor.to(torch.float8_e4m3fn)
                    else:
                        final_dict[k] = tensor
                elif self.precision_key == "fp8_e5m2":
                    if tensor.is_floating_point():
                        if self.device.type == "cuda":
                            final_dict[k] = tensor.to(device=self.device).to(torch.float8_e5m2).cpu()
                        else:
                            final_dict[k] = tensor.to(torch.float8_e5m2)
                    else:
                        final_dict[k] = tensor
                elif self.precision_key in ("int8", "int4"):
                    bits = 8 if self.precision_key == "int8" else 4
                    if tensor.is_floating_point() and tensor.numel() > 64:
                        t = tensor.to(self.device) if self.device.type == "cuda" else tensor
                        max_val = torch.max(torch.abs(t)).clamp(min=1e-5)
                        if bits == 8:
                            scale = 127.0 / max_val
                            final_dict[k] = (t * scale).round().clamp(-128, 127).to(torch.int8).cpu()
                            final_dict[f"{k}.scale"] = (max_val / 127.0).to(torch.float16).cpu()
                        else:
                            scale = 7.0 / max_val
                            final_dict[k] = (t * scale).round().clamp(-8, 7).to(torch.int8).cpu()
                            final_dict[f"{k}.scale"] = (max_val / 7.0).to(torch.float16).cpu()
                    else:
                        final_dict[k] = tensor
                elif self.precision_key == "nvfp4":
                    if tensor.is_floating_point() and tensor.numel() > 64:
                        t = tensor.to(self.device) if self.device.type == "cuda" else tensor
                        max_val = torch.max(torch.abs(t)).clamp(min=1e-5)
                        scale = 6.0 / max_val
                        scaled_t = (t * scale).clamp(-6.0, 6.0)
                        final_dict[k] = (scaled_t * 2.0).round().to(torch.int8).cpu()
                        final_dict[f"{k}.nvfp4_scale"] = (max_val / 6.0).to(torch.float16).cpu()
                    else:
                        final_dict[k] = tensor
                elif self.precision_key == "fp16":
                    final_dict[k] = tensor.to(torch.float16) if tensor.is_floating_point() else tensor
                else:
                    final_dict[k] = tensor

                elapsed = time.time() - start_time
                speed_mb = (os.path.getsize(input_path) * (idx + 1) / total_keys) / (1024 * 1024 * max(0.001, elapsed))

                if progress_callback:
                    if self.has_svd:
                        # SVDありの場合は後半50% (50%〜100%)
                        cur = total_keys + int((idx + 1) * (total_keys / total_keys))
                        progress_callback(cur, total_keys * 2, k, speed_mb)
                    else:
                        progress_callback(idx + 1, total_keys, k, speed_mb)

                if log_callback and ((idx + 1) % quant_log_interval == 0 or idx + 1 == total_keys):
                    pct = ((idx + 1) / total_keys) * 100
                    log_callback(f"[量子化進捗] {idx + 1} / {total_keys} テンソル完了 ({pct:.1f}%) [変換速度: {speed_mb:.1f} MB/s]")

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        if log_callback:
            log_callback(f"[保存中] ディスクへの書き込み中: {os.path.basename(output_path)} ...")

        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        save_file(final_dict, output_path, metadata=out_metadata if out_metadata else None)

        if log_callback:
            log_callback(f"[完了] ファイル出力完了: {output_path}")
        return True
