"""量子化エンジンの基底クラス (GPU/CUDA ストリーミング対応)"""
import torch
from typing import Callable, Optional

class BaseQuantizer:
    def __init__(self, keep_vae_fp16: bool = True):
        self.keep_vae_fp16 = keep_vae_fp16
        self.is_cancelled = False
        
        # GPU (CUDA) の自動検出
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_name = f"GPU: {torch.cuda.get_device_name(0)}"
        else:
            self.device = torch.device("cpu")
            self.device_name = "CPU"

    def cancel(self):
        self.is_cancelled = True

    def process(
        self,
        input_path: str,
        output_path: str,
        progress_callback: Optional[Callable[[int, int, str, float], None]] = None,
        log_callback: Optional[Callable[[str], None]] = None
    ) -> bool:
        raise NotImplementedError
