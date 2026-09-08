"""INT8 / INT4 量子化コンバーター (GPUストリーミング高速化)"""
import os
import time
import torch
from safetensors.torch import load_file, save_file
from .base import BaseQuantizer
from ..utils.safetensors_io import is_vae_tensor

class IntQuantizer(BaseQuantizer):
    def __init__(self, bits: int = 8, keep_vae_fp16: bool = True):
        super().__init__(keep_vae_fp16=keep_vae_fp16)
        self.bits = bits

    def process(self, input_path: str, output_path: str, progress_callback=None, log_callback=None) -> bool:
        if log_callback:
            log_callback(f"[{self.device_name}] INT{self.bits} 量子化を開始: {os.path.basename(input_path)} ...")

        state_dict = load_file(input_path)
        total_tensors = len(state_dict)
        quantized_dict = {}
        start_time = time.time()

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
                    return False

                if self.keep_vae_fp16 and is_vae_tensor(k):
                    quantized_dict[k] = tensor.to(torch.float16) if tensor.is_floating_point() else tensor
                else:
                    if tensor.is_floating_point() and tensor.numel() > 64:
                        # GPU上で並列スケール計算・クランプ・丸め
                        t = tensor.to(self.device) if self.device.type == "cuda" else tensor
                        max_val = torch.max(torch.abs(t)).clamp(min=1e-5)
                        
                        if self.bits == 8:
                            scale = 127.0 / max_val
                            q_tensor = (t * scale).round().clamp(-128, 127).to(torch.int8).cpu()
                            quantized_dict[k] = q_tensor
                            quantized_dict[f"{k}.scale"] = (max_val / 127.0).to(torch.float16).cpu()
                        else: # 4bit
                            scale = 7.0 / max_val
                            q_tensor = (t * scale).round().clamp(-8, 7).to(torch.int8).cpu()
                            quantized_dict[k] = q_tensor
                            quantized_dict[f"{k}.scale"] = (max_val / 7.0).to(torch.float16).cpu()
                    else:
                        quantized_dict[k] = tensor

                if progress_callback:
                    elapsed = time.time() - start_time
                    speed_mb = (os.path.getsize(input_path) * (idx + 1) / total_tensors) / (1024 * 1024 * max(0.001, elapsed))
                    progress_callback(idx + 1, total_tensors, k, speed_mb)

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        save_file(quantized_dict, output_path)
        return True
