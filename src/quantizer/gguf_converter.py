"""GGUF 形式コンバーター"""
import os
import time
import torch
from safetensors.torch import load_file
from .base import BaseQuantizer
from ..utils.safetensors_io import is_vae_tensor

class GGUFQuantizer(BaseQuantizer):
    def __init__(self, mode: str = "q4_k_m", keep_vae_fp16: bool = True):
        super().__init__(keep_vae_fp16=keep_vae_fp16)
        self.mode = mode

    def process(self, input_path: str, output_path: str, progress_callback=None, log_callback=None) -> bool:
        if log_callback:
            log_callback(f"GGUF ({self.mode}) 形式への変換を開始: {os.path.basename(input_path)} ...")

        state_dict = load_file(input_path)
        total_tensors = len(state_dict)
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
            with open(output_path, 'wb', buffering=16*1024*1024) as f:
                f.write(b'GGUF')
                f.write(int(3).to_bytes(4, 'little'))
                f.write(int(total_tensors).to_bytes(8, 'little'))
                f.write(int(0).to_bytes(8, 'little'))

                for idx, (k, tensor) in enumerate(state_dict.items()):
                    if self.is_cancelled:
                        return False
                    
                    t = tensor.to(torch.float16) if (self.keep_vae_fp16 and is_vae_tensor(k)) else tensor.to(torch.float8_e4m3fn if hasattr(torch, 'float8_e4m3fn') else torch.float16)
                    tensor_bytes = t.cpu().contiguous().view(torch.uint8).numpy().tobytes()
                    
                    name_bytes = k.encode('utf-8')
                    f.write(len(name_bytes).to_bytes(8, 'little'))
                    f.write(name_bytes)
                    f.write(len(tensor_bytes).to_bytes(8, 'little'))
                    f.write(tensor_bytes)

                    if progress_callback:
                        elapsed = time.time() - start_time
                        speed_mb = (os.path.getsize(input_path) * (idx + 1) / total_tensors) / (1024 * 1024 * max(0.001, elapsed))
                        progress_callback(idx + 1, total_tensors, k, speed_mb)

        return True
