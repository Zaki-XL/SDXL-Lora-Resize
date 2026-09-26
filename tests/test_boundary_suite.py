# -*- coding: utf-8 -*-
"""過学習判定およびUIバインディングの境界値テストスイート (Boundary Value Analysis)"""
import os
import sys
import unittest
import torch
from safetensors.torch import save_file

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

from src.quantizer.validator import diagnose_lora_health

class TestBoundarySuite(unittest.TestCase):
    def setUp(self):
        self.test_dir = os.path.join(project_root, "tests", "test_data_boundary")
        os.makedirs(self.test_dir, exist_ok=True)

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

    def _create_mock_lora(self, filename, steps, img_count, max_repeats, network_dim=16, te_lr="5e-5", unet_lr="1e-4"):
        path = os.path.join(self.test_dir, filename)
        tensors = {
            "lora_unet_layer.lora_down.weight": torch.full((network_dim, 320), 0.02, dtype=torch.float16),
            "lora_unet_layer.lora_up.weight": torch.full((320, network_dim), 0.02, dtype=torch.float16),
            "lora_unet_layer.alpha": torch.tensor(16.0, dtype=torch.float16),
            "lora_te1_layer.lora_down.weight": torch.full((network_dim, 768), 0.02, dtype=torch.float16),
            "lora_te1_layer.lora_up.weight": torch.full((768, network_dim), 0.02, dtype=torch.float16),
            "lora_te1_layer.alpha": torch.tensor(16.0, dtype=torch.float16),
        }
        metadata = {
            "ss_network_dim": str(network_dim),
            "ss_network_alpha": "16",
            "ss_max_train_steps": str(steps),
            "ss_num_train_images": str(img_count),
            "ss_text_encoder_lr": te_lr,
            "ss_unet_lr": unet_lr,
            "ss_dataset_dirs": f'{{"dir1": {{"n_repeats": {max_repeats}, "img_count": {img_count}}}}}'
        }
        save_file(tensors, path, metadata=metadata)
        return path

    # ---------------- 1. 総ステップ数 (steps >= 2500) の境界値テスト ----------------
    def test_bva_01_total_steps_boundary_2499_vs_2500(self):
        """画像20枚・リピート50回の極小データセットにおいて、2499 steps は正常判定、2500 steps は過学習判定となる境界値検証"""
        # Case A: steps = 2499 (しきい値未満 -> 正常系)
        path_2499 = self._create_mock_lora("bva_steps_2499.safetensors", steps=2499, img_count=20, max_repeats=50)
        diag_2499 = diagnose_lora_health(path_2499)
        self.assertFalse(diag_2499["has_issues"], "steps=2499 は通常学習範囲として警告が出ないこと")
        self.assertFalse(diag_2499["has_clip_overfit"])
        self.assertEqual(diag_2499["suggested_action"], "none")

        # Case B: steps = 2500 (しきい値以上 -> 過学習検知)
        path_2500 = self._create_mock_lora("bva_steps_2500.safetensors", steps=2500, img_count=20, max_repeats=50)
        diag_2500 = diagnose_lora_health(path_2500)
        self.assertTrue(diag_2500["has_issues"], "steps=2500 は長時間繰り返しとして検知されること")
        self.assertTrue(diag_2500["has_clip_overfit"])
        self.assertEqual(diag_2500["suggested_action"], "keep", "TEは正常なためTE除去ではなくkeepが推奨されること")

    # ---------------- 2. ステップ/画像比率 (step_per_img >= 40) の境界値テスト ----------------
    def test_bva_02_step_per_img_boundary_39_vs_40(self):
        """steps=3000 において、1枚あたり39.47 steps は正常、40.0 steps は検知となる境界値検証"""
        # Case A: img_count = 76 -> 3000 / 76 = 39.47 steps/img (< 40)
        path_39 = self._create_mock_lora("bva_ratio_39.safetensors", steps=3000, img_count=76, max_repeats=10)
        diag_39 = diagnose_lora_health(path_39)
        self.assertFalse(diag_39["has_issues"])

        # Case B: img_count = 75 -> 3000 / 75 = 40.0 steps/img (>= 40)
        path_40 = self._create_mock_lora("bva_ratio_40.safetensors", steps=3000, img_count=75, max_repeats=10)
        diag_40 = diagnose_lora_health(path_40)
        self.assertTrue(diag_40["has_issues"])
        self.assertTrue(any("過剰な繰り返し学習" in iss for iss in diag_40["issues"]))

    # ---------------- 3. 画像枚数 (img_count <= 250) の境界値テスト ----------------
    def test_bva_03_image_count_boundary_250_vs_251(self):
        """steps=3000, max_repeats=35 において、250枚は少数画像検知、251枚は少数画像判定から外れる境界値検証"""
        # Case A: img_count = 250 (少数画像上限)
        path_250 = self._create_mock_lora("bva_img_250.safetensors", steps=3000, img_count=250, max_repeats=35)
        diag_250 = diagnose_lora_health(path_250)
        self.assertTrue(diag_250["has_issues"])

        # Case B: img_count = 251 (少数画像の定義を超えるため単独リピート警告から除外)
        path_251 = self._create_mock_lora("bva_img_251.safetensors", steps=3000, img_count=251, max_repeats=35)
        diag_251 = diagnose_lora_health(path_251)
        # 3000 / 251 = 11.95 steps/img (< 40) かつ img > 250
        self.assertFalse(diag_251["has_issues"])

    # ---------------- 4. リピート数 (max_repeats >= 30) の境界値テスト ----------------
    def test_bva_04_repeats_boundary_29_vs_30(self):
        """steps=3000, img_count=100 (30 steps/img < 40) において、リピート29回は正常、30回は検知となる境界値検証"""
        # Case A: max_repeats = 29
        path_29 = self._create_mock_lora("bva_rep_29.safetensors", steps=3000, img_count=100, max_repeats=29)
        diag_29 = diagnose_lora_health(path_29)
        self.assertFalse(diag_29["has_issues"])

        # Case B: max_repeats = 30
        path_30 = self._create_mock_lora("bva_rep_30.safetensors", steps=3000, img_count=100, max_repeats=30)
        diag_30 = diagnose_lora_health(path_30)
        self.assertTrue(diag_30["has_issues"])
        self.assertTrue(any("過剰な繰り返し学習" in iss for iss in diag_30["issues"]))

    # ---------------- 5. UIダイアログ初期選択バインディングの境界値テスト ----------------
    def test_bva_05_health_dialog_action_binding(self):
        """ダイアログの初期選択が suggested_action ('keep' vs 'drop_te' vs 'scale_te') に100%連動することの検証"""
        from PyQt6.QtWidgets import QApplication
        from src.gui.health_dialog import LoRAHealthDialog
        app = QApplication.instance() or QApplication(sys.argv)

        # Case A: suggested_action = "keep" -> rb_keep が初期チェックされ、【推奨】が付与されること
        health_keep = {
            "has_issues": True,
            "risk_level": "warning",
            "suggested_action": "keep",
            "issues": ["少数画像の繰り返し学習"],
            "suggestions": ["TE保持を推奨"]
        }
        dlg_keep = LoRAHealthDialog(None, "dummy.safetensors", health_keep)
        self.assertTrue(dlg_keep.rb_keep.isChecked(), "suggested=keep では rb_keep が初期選択されること")
        self.assertFalse(dlg_keep.rb_drop_te.isChecked())
        self.assertIn("【推奨】", dlg_keep.rb_keep.text())
        self.assertNotIn("【推奨】", dlg_keep.rb_drop_te.text())

        # Case B: suggested_action = "drop_te" -> rb_drop_te が初期チェックされ、【推奨】が付与されること
        health_drop = {
            "has_issues": True,
            "risk_level": "warning",
            "suggested_action": "drop_te",
            "issues": ["TE高学習率"],
            "suggestions": ["TE除去を推奨"]
        }
        dlg_drop = LoRAHealthDialog(None, "dummy.safetensors", health_drop)
        self.assertTrue(dlg_drop.rb_drop_te.isChecked(), "suggested=drop_te では rb_drop_te が初期選択されること")
        self.assertFalse(dlg_drop.rb_keep.isChecked())
        self.assertIn("【推奨】", dlg_drop.rb_drop_te.text())
        self.assertNotIn("【推奨】", dlg_drop.rb_keep.text())

        # Case C: suggested_action = "scale_te" -> rb_scale_te が初期チェックされること
        health_scale = {
            "has_issues": True,
            "risk_level": "warning",
            "suggested_action": "scale_te",
            "issues": ["TE中程度学習"],
            "suggestions": ["TE減衰を推奨"]
        }
        dlg_scale = LoRAHealthDialog(None, "dummy.safetensors", health_scale)
        self.assertTrue(dlg_scale.rb_scale_te.isChecked(), "suggested=scale_te では rb_scale_te が初期選択されること")

        # Case D: is_corrupt = True -> 破損ファイルは rb_keep 固定かつ他が無効化
        health_corrupt = {
            "has_issues": True,
            "is_corrupt": True,
            "risk_level": "danger",
            "suggested_action": "skip",
            "issues": ["NaN検出"]
        }
        dlg_corrupt = LoRAHealthDialog(None, "corrupt.safetensors", health_corrupt)
        self.assertTrue(dlg_corrupt.rb_keep.isChecked())
        self.assertFalse(dlg_corrupt.rb_drop_te.isEnabled())
        self.assertFalse(dlg_corrupt.rb_scale_te.isEnabled())


if __name__ == '__main__':
    unittest.main()
