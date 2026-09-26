"""出力ファイル名の生成と重複判定ユーティリティ (SVD + 量子化 複合命名対応)"""
import os
from .gpu_info import PRECISION_OPTIONS

def generate_output_path(input_path: str, svd_rank: str, precision_key: str, output_dir: str = "", te_action: str = "keep", orig_rank: int | None = None) -> str:
    """SVD Rank と 量子化精度、TE対策 を掛け合わせた出力先パスを生成"""
    base_dir = output_dir if (output_dir and os.path.isdir(output_dir)) else os.path.dirname(input_path)
    file_name = os.path.basename(input_path)
    stem, ext = os.path.splitext(file_name)

    suffixes = []
    
    # TE 対策サフィックス
    if te_action == "drop_te":
        suffixes.append("_clean_unet")
    elif te_action == "scale_te":
        suffixes.append("_te_scaled")

    # SVD Rank サフィックス (元Rankより大きいTarget Rankが指定された場合は拡大を行わず維持するためサフィックス付与をスキップ)
    if svd_rank and svd_rank != "none":
        try:
            target_r = int(svd_rank)
            if orig_rank is None or orig_rank > target_r:
                suffixes.append(f"_rank{svd_rank}")
        except (ValueError, TypeError):
            suffixes.append(f"_rank{svd_rank}")

    # 量子化サフィックス
    p_info = PRECISION_OPTIONS.get(precision_key, {})
    p_suffix = p_info.get("suffix", "")
    if p_suffix:
        if p_suffix.endswith(".gguf"):
            return os.path.join(base_dir, f"{stem}{''.join(suffixes)}{p_suffix}")
        suffixes.append(p_suffix)

    # 何も選択されていない場合のフォールバック
    if not suffixes:
        suffixes.append("_resized")

    new_name = f"{stem}{''.join(suffixes)}{ext}"
    return os.path.join(base_dir, new_name)

def get_renamed_path(target_path: str) -> str:
    base_dir = os.path.dirname(target_path)
    filename = os.path.basename(target_path)
    stem, ext = os.path.splitext(filename)

    idx = 1
    while True:
        candidate = os.path.join(base_dir, f"{stem}_{idx}{ext}")
        if not os.path.exists(candidate):
            return candidate
        idx += 1
