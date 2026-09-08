import os
import sys
import unittest
import torch
from safetensors.torch import save_file, load_file

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.utils.config_manager import ConfigManager
from src.utils.gpu_info import PRECISION_OPTIONS, SVD_RANK_OPTIONS
from src.utils.naming import generate_output_path, get_renamed_path
from src.utils.safetensors_io import inspect_model_metadata
from src.quantizer.estimator import estimate_quantized_size
from src.quantizer.pipeline_converter import CombinedPipelineConverter
from src.quantizer.fp8_converter import FP8Quantizer
from src.quantizer.int_converter import IntQuantizer

class TestSDXLQuantizerSuite(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.join(project_root, "tests", "test_data_main")
        os.makedirs(self.test_dir, exist_ok=True)
        
        self.dummy_model_path = os.path.join(self.test_dir, "test_sdxl_model.safetensors")
        dummy_tensors = {
            "model.diffusion_model.input_blocks.0.0.weight": torch.randn(320, 4, 3, 3, dtype=torch.float32),
            "first_stage_model.encoder.conv_in.weight": torch.randn(128, 3, 3, 3, dtype=torch.float32),
        }
        save_file(dummy_tensors, self.dummy_model_path)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            for root, dirs, files in os.walk(self.test_dir, topdown=False):
                for name in files:
                    try:
                        os.remove(os.path.join(root, name))
                    except:
                        pass
                for name in dirs:
                    try:
                        os.rmdir(os.path.join(root, name))
                    except:
                        pass
            try:
                os.rmdir(self.test_dir)
            except:
                pass

    def test_config_manager(self):
        cfg_path = os.path.join(self.test_dir, "test_config.ini")
        cfg = ConfigManager(cfg_path)
        cfg.set("precision_format", "fp8_e4m3fn")
        cfg.set("svd_rank", "32")
        
        cfg2 = ConfigManager(cfg_path)
        self.assertEqual(cfg2.get("precision_format"), "fp8_e4m3fn")
        self.assertEqual(cfg2.get("svd_rank"), "32")

    def test_naming(self):
        out_path = generate_output_path(self.dummy_model_path, "32", "fp8_e4m3fn")
        self.assertTrue(out_path.endswith("test_sdxl_model_rank32_fp8_e4m3fn.safetensors"))

    def test_estimator(self):
        orig_sz, est_sz, red_pct = estimate_quantized_size(self.dummy_model_path, "none", "fp8_e4m3fn", keep_vae_fp16=True)
        self.assertGreater(orig_sz, 0)
        self.assertGreater(est_sz, 0)
        self.assertLess(est_sz, orig_sz)

    def test_pipeline_converter(self):
        out_path = os.path.join(self.test_dir, "out_pipe.safetensors")
        converter = CombinedPipelineConverter(svd_rank="none", precision_key="fp8_e4m3fn", keep_vae_fp16=True)
        success = converter.process(self.dummy_model_path, out_path)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(out_path))

    def test_lora_svd_alpha_scaling(self):
        """LoRAのSVDリサイズ時にAlpha値が元Rankとの比率で正しく比例スケーリングされることを検証"""
        lora_dummy_path = os.path.join(self.test_dir, "test_lora_rank128.safetensors")
        lora_tensors = {
            "lora_unet_down_blocks_0_attentions_0_proj_in.lora_down.weight": torch.randn(128, 64, dtype=torch.float32),
            "lora_unet_down_blocks_0_attentions_0_proj_in.lora_up.weight": torch.randn(64, 128, dtype=torch.float32),
            "lora_unet_down_blocks_0_attentions_0_proj_in.alpha": torch.tensor(1.0, dtype=torch.float32),
        }
        meta_dict = {"ss_network_dim": "128", "ss_network_alpha": "1"}
        save_file(lora_tensors, lora_dummy_path, metadata=meta_dict)

        out_lora_path = os.path.join(self.test_dir, "test_lora_rank64.safetensors")
        converter = CombinedPipelineConverter(svd_rank="64", precision_key="keep", keep_vae_fp16=True)
        success = converter.process(lora_dummy_path, out_lora_path)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(out_lora_path))

        out_dict = load_file(out_lora_path)
        new_down = out_dict["lora_unet_down_blocks_0_attentions_0_proj_in.lora_down.weight"]
        new_up = out_dict["lora_unet_down_blocks_0_attentions_0_proj_in.lora_up.weight"]
        new_alpha = float(out_dict["lora_unet_down_blocks_0_attentions_0_proj_in.alpha"].item())

        self.assertEqual(new_down.shape[0], 64)
        self.assertEqual(new_up.shape[1], 64)
        # 元が Rank 128, Alpha 1.0 -> Rank 64 で Alpha は 0.5 になること
        self.assertAlmostEqual(new_alpha, 0.5, places=4)

        # 実効スケール比 (alpha / rank) が完全一致すること
        orig_scale = 1.0 / 128.0
        new_scale = new_alpha / 64.0
        self.assertAlmostEqual(orig_scale, new_scale, places=6)

        # メタデータの更新確認
        from src.utils.safetensors_io import read_safetensors_header
        _, out_meta, _ = read_safetensors_header(out_lora_path)
        self.assertEqual(out_meta.get("ss_network_dim"), "64")
        self.assertEqual(out_meta.get("ss_network_alpha"), "0.5")

if __name__ == '__main__':
    unittest.main()
