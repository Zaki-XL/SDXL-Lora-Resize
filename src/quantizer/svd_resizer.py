"""SVD (特異値分解) による LoRA / モデルの Rank リサイズエンジン"""
import os
import time
import math
import torch
from safetensors.torch import load_file, save_file
from .base import BaseQuantizer
from ..utils.safetensors_io import read_safetensors_header

def low_rank_svd(up_mat: torch.Tensor, down_mat: torch.Tensor, target_rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    up_mat: (out_dim, r), down_mat: (r, in_dim)
    QR分解を利用して O(r^2(M+N) + r^3) で超高速に SVD 分解と Rank 圧縮を実行します。
    """
    out_dim, r = up_mat.shape
    r_down, in_dim = down_mat.shape
    k = min(target_rank, r)

    if r < min(out_dim, in_dim):
        # QR-SVD 高速化パス
        Q_up, R_up = torch.linalg.qr(up_mat)
        Q_down, R_down = torch.linalg.qr(down_mat.T)

        M_mid = torch.mm(R_up, R_down.T)
        U_mid, S, Vh_mid = torch.linalg.svd(M_mid)

        U_k = torch.mm(Q_up, U_mid[:, :k])
        S_k = S[:k]
        Vh_k = torch.mm(Vh_mid[:k, :], Q_down.T)
    else:
        # フォールバック (フルSVD)
        W = torch.mm(up_mat, down_mat)
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        U_k = U[:, :k]
        S_k = S[:k]
        Vh_k = Vh[:k, :]

    sqrt_S = torch.diag(torch.sqrt(S_k))
    new_up_mat = torch.mm(U_k, sqrt_S)
    new_down_mat = torch.mm(sqrt_S, Vh_k)
    return new_up_mat, new_down_mat

def find_lora_pairs(state_dict: dict) -> list[tuple[str, str, str | None]]:
    """
    state_dict 内から LoRA の (down_key, up_key, alpha_key) のペア一覧を抽出します。
    Kohya (lora_down/up), Diffusers (lora_A/B, lora.down/up, lora_a/b, .down/.up), LyCORIS 等の多様な形式に対応。
    """
    key_lower_map = {k.lower(): k for k in state_dict.keys()}
    pairs = []
    
    patterns = [
        ("lora_down.weight", "lora_up.weight", "alpha"),
        ("lora.down.weight", "lora.up.weight", "alpha"),
        ("lora_a.weight", "lora_b.weight", "alpha"),
        ("lora.a.weight", "lora.b.weight", "alpha"),
        (".down.weight", ".up.weight", ".alpha"),
        ("hada_w1_b", "hada_w1_a", "alpha"),
        ("hada_w2_b", "hada_w2_a", "alpha"),
    ]
    
    for orig_k in state_dict.keys():
        kl = orig_k.lower()
        for down_pat, up_pat, alpha_pat in patterns:
            if down_pat in kl:
                target_up = kl.replace(down_pat, up_pat)
                up_k = key_lower_map.get(target_up)
                if up_k:
                    target_alpha = kl.replace(down_pat, alpha_pat)
                    alpha_k = key_lower_map.get(target_alpha)
                    pairs.append((orig_k, up_k, alpha_k))
                break

    return pairs

class SVDResizer(BaseQuantizer):
    """
    LoRAの重みペア (lora_down.weight, lora_up.weight) を SVD 分解し、
    不要な低エネルギーランクを切り捨てて指定の target_rank へ圧縮します。
    """
    def __init__(self, target_rank: int = 32, keep_vae_fp16: bool = True):
        super().__init__(keep_vae_fp16=keep_vae_fp16)
        self.target_rank = target_rank

    def process(self, input_path: str, output_path: str, progress_callback=None, log_callback=None) -> bool:
        if log_callback:
            log_callback(f"[{self.device_name}] SVD Rank Resize (Target Rank: {self.target_rank}) を開始: {os.path.basename(input_path)} ...")

        state_dict = load_file(input_path)
        total_keys = len(state_dict)
        resized_dict = {}

        # Safetensors メタデータの取得と引き継ぎ準備
        try:
            _, orig_metadata, _ = read_safetensors_header(input_path)
        except Exception:
            orig_metadata = {}
        out_metadata = {str(mk): str(mv) for mk, mv in orig_metadata.items()}
        
        # LoRA ペア (down / up) のマッピングを検出
        pairs = find_lora_pairs(state_dict)
        
        start_time = time.time()
        processed_pairs = 0
        processed_keys = set()

        with torch.inference_mode():
            for idx, (down_k, up_k, alpha_k) in enumerate(pairs):
                if self.is_cancelled:
                    return False

                down_t = state_dict[down_k]
                up_t = state_dict[up_k]
                orig_dtype = down_t.dtype

                # SVD 計算を GPU / CPU で実行
                dev = self.device if self.device.type == "cuda" else torch.device("cpu")
                down_dev = down_t.to(device=dev, dtype=torch.float32)
                up_dev = up_t.to(device=dev, dtype=torch.float32)

                is_conv2d = (down_dev.dim() == 4 and up_dev.dim() == 4)

                # Conv2d の場合は 2D 行列へ展開
                if is_conv2d:
                    # down: (r, in_c, k_h, k_w), up: (out_c, r, 1, 1)
                    r = down_dev.size(0)
                    in_dim = down_dev.size(1) * down_dev.size(2) * down_dev.size(3)
                    out_dim = up_dev.size(0)
                    
                    down_mat = down_dev.flatten(start_dim=1) # (r, in_dim)
                    up_mat = up_dev.squeeze(-1).squeeze(-1) # (out_dim, r)
                else:
                    # Linear: down: (r, in_dim), up: (out_dim, r)
                    r = down_dev.size(0)
                    in_dim = down_dev.size(1)
                    out_dim = up_dev.size(0)
                    down_mat = down_dev
                    up_mat = up_dev

                # ターゲットランクが元ランクより小さい場合のみ SVD 圧縮を実行
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

                        resized_dict[down_k] = new_down_t.to(dtype=orig_dtype).cpu()
                        resized_dict[up_k] = new_up_t.to(dtype=orig_dtype).cpu()

                        # alpha の調整 (存在する場合: 比例スケーリング)
                        if alpha_k in state_dict:
                            orig_alpha = float(state_dict[alpha_k].item())
                            new_alpha = orig_alpha * (float(k) / max(1.0, float(r)))
                            resized_dict[alpha_k] = torch.tensor(new_alpha, dtype=state_dict[alpha_k].dtype)
                            processed_keys.add(alpha_k)

                    except Exception as svd_err:
                        # SVD失敗時は元のテンソルを保持
                        resized_dict[down_k] = down_t
                        resized_dict[up_k] = up_t
                else:
                    # すでに target_rank 以下の場合はそのまま保持
                    resized_dict[down_k] = down_t
                    resized_dict[up_k] = up_t
                    if alpha_k in state_dict:
                        resized_dict[alpha_k] = state_dict[alpha_k]
                        processed_keys.add(alpha_k)

                processed_keys.add(down_k)
                processed_keys.add(up_k)
                processed_pairs += 1

                if progress_callback:
                    elapsed = time.time() - start_time
                    speed_mb = (os.path.getsize(input_path) * (idx + 1) / max(1, len(pairs))) / (1024 * 1024 * max(0.001, elapsed))
                    progress_callback(idx + 1, len(pairs), down_k, speed_mb)

        # その他の残りのテンソル（バイアス、メタデータ、その他の重み）をそのままコピー
        for k, v in state_dict.items():
            if k not in processed_keys:
                resized_dict[k] = v

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        if log_callback:
            if processed_pairs > 0:
                log_callback(f"SVD 圧縮完了: {processed_pairs} 組のLoRAレイヤーを Rank {self.target_rank} にリサイズ")
            else:
                log_callback(f"SVD 対象の標準LoRAレイヤーが検出されなかったため（LoHa/LoKr等の特殊LyCORIS、または非LoRA）、SVDリサイズをスキップし元のRank構造を維持しました")
            log_callback(f"保存中: {output_path} ...")

        # SVDリサイズに応じたメタデータの更新 (実際にLoRAレイヤーをリサイズした場合のみ)
        if processed_pairs > 0:
            out_metadata["ss_network_dim"] = str(self.target_rank)
            if "ss_network_alpha" in out_metadata:
                try:
                    orig_net_dim = float(orig_metadata.get("ss_network_dim", 128))
                    orig_net_alpha = float(orig_metadata.get("ss_network_alpha", 1))
                    new_net_alpha = orig_net_alpha * (float(self.target_rank) / max(1.0, orig_net_dim))
                    out_metadata["ss_network_alpha"] = f"{new_net_alpha:g}"
                except Exception:
                    pass

        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        save_file(resized_dict, output_path, metadata=out_metadata if out_metadata else None)

        if log_callback:
            log_callback(f"✓ 出力完了: {output_path}")
        return True
