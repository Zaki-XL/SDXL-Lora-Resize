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

    def test_lycoris_loha_estimator_and_conversion(self):
        """LyCORIS (LoHa) の削減率推計が各ランク変更に正しく連動し、SVDリサイズ時も正常に圧縮されることを検証"""
        loha_dummy_path = os.path.join(self.test_dir, "test_lycoris_loha_r32.safetensors")
        loha_tensors = {
            "lora_unet_down_blocks_0_attentions_0_proj_in.hada_w1_a": torch.randn(1280, 32, dtype=torch.float16),
            "lora_unet_down_blocks_0_attentions_0_proj_in.hada_w1_b": torch.randn(32, 1280, dtype=torch.float16),
            "lora_unet_down_blocks_0_attentions_0_proj_in.hada_w2_a": torch.randn(1280, 32, dtype=torch.float16),
            "lora_unet_down_blocks_0_attentions_0_proj_in.hada_w2_b": torch.randn(32, 1280, dtype=torch.float16),
            "lora_unet_down_blocks_0_attentions_0_proj_in.alpha": torch.tensor(1.0, dtype=torch.float16),
        }
        meta_dict = {
            "ss_network_dim": "32",
            "ss_network_alpha": "1",
            "ss_network_module": "lycoris.kohya",
            "ss_network_args": '{"algo": "loha"}'
        }
        save_file(loha_tensors, loha_dummy_path, metadata=meta_dict)

        # 1. 削減率推計の検証 (選択ランクに応じて推定後サイズがリアルタイムに変化すること)
        # ① リサイズなし (none): 削減率 約0%
        _, _, red_none = estimate_quantized_size(loha_dummy_path, svd_rank="none", precision_key="keep", keep_vae_fp16=True)
        self.assertLess(red_none, 3.0)

        # ② 元Rank維持 (32): 削減率 約0%
        _, _, red_r32 = estimate_quantized_size(loha_dummy_path, svd_rank="32", precision_key="keep", keep_vae_fp16=True)
        self.assertLess(red_r32, 3.0)

        # ③ Rank 16 (半分に削減): 削減率 約 50%
        _, _, red_r16 = estimate_quantized_size(loha_dummy_path, svd_rank="16", precision_key="keep", keep_vae_fp16=True)
        self.assertGreater(red_r16, 45.0)
        self.assertLess(red_r16, 55.0)

        # ④ Rank 8 (1/4に削減): 削減率 約 75%
        _, _, red_r8 = estimate_quantized_size(loha_dummy_path, svd_rank="8", precision_key="keep", keep_vae_fp16=True)
        self.assertGreater(red_r8, 70.0)
        self.assertLess(red_r8, 80.0)

        # ⑤ 元Rankより大きい Rank 64: 拡大は行わず維持 (削減率 約0%)
        _, _, red_r64 = estimate_quantized_size(loha_dummy_path, svd_rank="64", precision_key="keep", keep_vae_fp16=True)
        self.assertLess(red_r64, 3.0)

        # ⑥ Rank 16 × FP8: 約 75% 削減 (Rank 50% × FP8 50% = 25% 残存)
        _, _, red_r16_fp8 = estimate_quantized_size(loha_dummy_path, svd_rank="16", precision_key="fp8_e4m3fn", keep_vae_fp16=True)
        self.assertGreater(red_r16_fp8, 70.0)
        self.assertLess(red_r16_fp8, 80.0)

        # 2. パイプライン変換の検証 (Rank 16 への SVD 圧縮が正しく実行され、Shape と Alpha が更新されること)
        out_loha_path = os.path.join(self.test_dir, "out_loha_rank16.safetensors")
        converter = CombinedPipelineConverter(svd_rank="16", precision_key="keep", keep_vae_fp16=True)
        success = converter.process(loha_dummy_path, out_loha_path)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(out_loha_path))

        out_dict = load_file(out_loha_path)
        w1_a = out_dict["lora_unet_down_blocks_0_attentions_0_proj_in.hada_w1_a"]
        w1_b = out_dict["lora_unet_down_blocks_0_attentions_0_proj_in.hada_w1_b"]
        w2_a = out_dict["lora_unet_down_blocks_0_attentions_0_proj_in.hada_w2_a"]
        w2_b = out_dict["lora_unet_down_blocks_0_attentions_0_proj_in.hada_w2_b"]
        new_alpha = float(out_dict["lora_unet_down_blocks_0_attentions_0_proj_in.alpha"].item())

        # リサイズ後の Shape 検証: (1280, 16) および (16, 1280)
        self.assertEqual(tuple(w1_a.shape), (1280, 16))
        self.assertEqual(tuple(w1_b.shape), (16, 1280))
        self.assertEqual(tuple(w2_a.shape), (1280, 16))
        self.assertEqual(tuple(w2_b.shape), (16, 1280))

        # Alpha の比例スケーリング検証: 元 1.0 -> 0.5
        self.assertAlmostEqual(new_alpha, 0.5, places=4)

        # メタデータの更新確認: ss_network_dim が 16 に更新されていること
        from src.utils.safetensors_io import read_safetensors_header
        _, out_meta, _ = read_safetensors_header(out_loha_path)
        self.assertEqual(out_meta.get("ss_network_dim"), "16")
        self.assertEqual(out_meta.get("ss_network_alpha"), "0.5")

    def test_dora_health_diagnosis_no_false_positive(self):
        """DoRAのdora_scale (ノルム3.5) をTE重み過学習と誤検知しないことの検証"""
        from src.quantizer.validator import diagnose_lora_health
        dora_dummy_path = os.path.join(self.test_dir, "test_dora_model.safetensors")
        tensors = {
            "lora_te1_text_model_encoder_layers_0.lora_down.weight": torch.full((16, 768), 0.02, dtype=torch.float16),
            "lora_te1_text_model_encoder_layers_0.lora_up.weight": torch.full((768, 16), 0.02, dtype=torch.float16),
            "lora_te1_text_model_encoder_layers_0.alpha": torch.tensor(16.0, dtype=torch.float16),
            "lora_te1_text_model_encoder_layers_0.dora_scale": torch.full((1, 768), 3.5, dtype=torch.float16),
            "lora_unet_down_blocks_0.lora_down.weight": torch.full((16, 320), 0.02, dtype=torch.float16),
            "lora_unet_down_blocks_0.lora_up.weight": torch.full((320, 16), 0.02, dtype=torch.float16),
        }
        metadata = {
            "ss_network_dim": "16",
            "ss_network_alpha": "16",
            "ss_text_encoder_lr": "5e-5",
            "ss_unet_lr": "1e-4",
            "ss_max_train_steps": "1000",
            "ss_num_train_images": "100"
        }
        save_file(tensors, dora_dummy_path, metadata=metadata)

        diag = diagnose_lora_health(dora_dummy_path)
        # dora_scale (3.5) に起因する「重み変化量が異常値」という誤検知が発生しないこと
        for issue in diag.get("issues", []):
            self.assertNotIn("重み変化量が異常値", issue)
        self.assertFalse(diag.get("has_clip_overfit", False))
        self.assertEqual(diag.get("suggested_action"), "none")

    def test_sensitive_tensor_quantization_protection(self):
        """FP8量子化時に dora_scale, alpha, b_norm, bias が FP16 で保護され、主重みのみFP8化されることの検証"""
        model_path = os.path.join(self.test_dir, "test_sensitive_model.safetensors")
        tensors = {
            "lora_unet_layer.lora_down.weight": torch.randn(16, 320, dtype=torch.float32),
            "lora_unet_layer.lora_up.weight": torch.randn(320, 16, dtype=torch.float32),
            "lora_unet_layer.alpha": torch.tensor(16.0, dtype=torch.float32),
            "lora_unet_layer.dora_scale": torch.full((1, 320), 1.25, dtype=torch.float32),
            "lora_unet_layer.b_norm": torch.full((320,), 0.002, dtype=torch.float32),
            "lora_unet_layer.bias": torch.full((320,), 0.01, dtype=torch.float32),
        }
        save_file(tensors, model_path)

        out_path = os.path.join(self.test_dir, "out_sensitive_fp8.safetensors")
        converter = CombinedPipelineConverter(svd_rank="none", precision_key="fp8_e4m3fn", keep_vae_fp16=True)
        success = converter.process(model_path, out_path)
        self.assertTrue(success)

        res = load_file(out_path)
        # 主重みは FP8 (float8_e4m3fn) に量子化
        self.assertEqual(res["lora_unet_layer.lora_down.weight"].dtype, torch.float8_e4m3fn)
        self.assertEqual(res["lora_unet_layer.lora_up.weight"].dtype, torch.float8_e4m3fn)

        # 高感度テンソルは FP16 で厳格に保護
        self.assertEqual(res["lora_unet_layer.alpha"].dtype, torch.float16)
        self.assertEqual(res["lora_unet_layer.dora_scale"].dtype, torch.float16)
        self.assertEqual(res["lora_unet_layer.b_norm"].dtype, torch.float16)
        self.assertEqual(res["lora_unet_layer.bias"].dtype, torch.float16)

        # 値の保持確認
        self.assertAlmostEqual(res["lora_unet_layer.alpha"].item(), 16.0, places=3)
        self.assertAlmostEqual(res["lora_unet_layer.dora_scale"][0, 0].item(), 1.25, places=3)
        self.assertAlmostEqual(res["lora_unet_layer.b_norm"][0].item(), 0.002, places=4)


if __name__ == '__main__':
    unittest.main()

