"""FP8 (e4m3fn, e5m2) 量子化コンバーター (GPUストリーミング高速化)"""
import os
import time
import torch
from safetensors.torch import load_file, save_file
from .base import BaseQuantizer
from ..utils.safetensors_io import is_vae_tensor, is_sensitive_precision_tensor

class FP8Quantizer(BaseQuantizer):
    def __init__(self, mode: str = "e4m3fn", keep_vae_fp16: bool = True):
        super().__init__(keep_vae_fp16=keep_vae_fp16)
        self.mode = mode
        if mode == "e4m3fn":
            self.target_dtype = torch.float8_e4m3fn
        elif mode == "e5m2":
            self.target_dtype = torch.float8_e5m2
        else:
            raise ValueError(f"不明なFP8モード: {mode}")

    def process(self, input_path: str, output_path: str, progress_callback=None, log_callback=None) -> bool:
        if log_callback:
            log_callback(f"[{self.device_name}] モデルをロード中: {os.path.basename(input_path)} ...")

        state_dict = load_file(input_path)
        total_tensors = len(state_dict)
        quantized_dict = {}
        
        start_time = time.time()
        converted_count = 0
        preserved_count = 0

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
            for idx, (k, tensor) in enumerate(state_dict.items()):
                if self.is_cancelled:
                    if log_callback:
                        log_callback("[中断] 処理が中止されました。")
                    return False

                is_protected = (self.keep_vae_fp16 and is_vae_tensor(k)) or is_sensitive_precision_tensor(k)
                if is_protected:
                    # VAEや高感度テンソル(DoRA/Norm/alpha)はFP16で保持
                    if tensor.dtype in (torch.float32, torch.float64):
                        quantized_dict[k] = tensor.to(dtype=torch.float16)
                    else:
                        quantized_dict[k] = tensor
                    preserved_count += 1
                else:
                    if tensor.is_floating_point():
                        # GPUが利用可能な場合はGPUで高速キャスト
                        if self.device.type == "cuda":
                            t_gpu = tensor.to(device=self.device, non_blocking=True)
                            q_t = t_gpu.to(self.target_dtype).cpu()
                            quantized_dict[k] = q_t
                        else:
                            quantized_dict[k] = tensor.to(self.target_dtype)
                        converted_count += 1
                    else:
                        quantized_dict[k] = tensor

                if progress_callback:
                    elapsed = time.time() - start_time
                    speed_mb = (os.path.getsize(input_path) * (idx + 1) / total_tensors) / (1024 * 1024 * max(0.001, elapsed))
                    progress_callback(idx + 1, total_tensors, k, speed_mb)

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        if log_callback:
            log_callback(f"変換完了 (量子化: {converted_count}個, VAE保護: {preserved_count}個)")
            log_callback(f"保存中: {output_path} ...")

        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        save_file(quantized_dict, output_path)

        if log_callback:
            log_callback(f"✓ 出力完了: {output_path}")
        return True
