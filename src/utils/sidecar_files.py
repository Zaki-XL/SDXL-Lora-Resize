"""モデルに付属するプレビュー画像 (webp/jpg/png) および metadata.json の検出・コピー・内容変換モジュール"""
import os
import json
import shutil
from typing import Optional

IMAGE_EXTENSIONS = [
    ".preview.png", ".preview.jpg", ".preview.webp",
    ".png", ".jpg", ".jpeg", ".webp"
]

METADATA_EXTENSIONS = [
    ".metadata.json", ".civitai.info", ".json"
]

def find_associated_files(model_path: str) -> dict:
    """
    モデルファイルと同名の画像ファイルおよびmetadata.jsonを検出します。
    """
    if not os.path.exists(model_path):
        return {"images": [], "metadata": None, "display_str": "-"}

    base_dir = os.path.dirname(model_path)
    filename = os.path.basename(model_path)
    stem, _ = os.path.splitext(filename)

    found_images = []
    found_metadata = None

    # 画像の検出 (優先度順)
    for ext in IMAGE_EXTENSIONS:
        candidate = os.path.join(base_dir, f"{stem}{ext}")
        if os.path.exists(candidate) and candidate not in found_images:
            found_images.append(candidate)

    # メタデータJSONの検出
    for ext in METADATA_EXTENSIONS:
        candidate = os.path.join(base_dir, f"{stem}{ext}")
        if os.path.exists(candidate):
            found_metadata = candidate
            break

    # 表示用文字列の生成
    has_img = len(found_images) > 0
    has_meta = found_metadata is not None

    if has_img and has_meta:
        img_ext = os.path.splitext(found_images[0])[1].replace(".", "").upper()
        display_str = f"JSON + {img_ext}"
    elif has_img:
        img_ext = os.path.splitext(found_images[0])[1].replace(".", "").upper()
        display_str = f"画像 ({img_ext})"
    elif has_meta:
        display_str = "JSON"
    else:
        display_str = "-"

    return {
        "images": found_images,
        "metadata": found_metadata,
        "display_str": display_str
    }

def process_and_copy_sidecar_files(
    orig_model_path: str,
    output_model_path: str,
    svd_rank: str,
    precision_key: str,
    log_callback=None
) -> list[str]:
    """
    付属する画像とメタデータJSONを変換後モデルに合わせてコピー＆JSON内容更新します。
    Returns: 生成された付属ファイルのパス一覧
    """
    sidecars = find_associated_files(orig_model_path)
    orig_stem, _ = os.path.splitext(os.path.basename(orig_model_path))
    out_dir = os.path.dirname(output_model_path)
    out_stem, _ = os.path.splitext(os.path.basename(output_model_path))

    created_files = []

    # 1. プレビュー画像のコピー & リネーム
    for img_path in sidecars["images"]:
        img_name = os.path.basename(img_path)
        # 拡張子部分（.preview.png などの二重拡張子にも対応）
        if img_name.startswith(orig_stem):
            ext_part = img_name[len(orig_stem):]
        else:
            _, ext_part = os.path.splitext(img_name)

        out_img_path = os.path.join(out_dir, f"{out_stem}{ext_part}")
        try:
            shutil.copy2(img_path, out_img_path)
            created_files.append(out_img_path)
            if log_callback:
                log_callback(f"[付属ファイル] プレビュー画像をコピー: {os.path.basename(out_img_path)}")
        except Exception as e:
            if log_callback:
                log_callback(f"[警告] 画像コピー失敗: {e}")

    # 2. metadata.json の内容変換 & 出力
    meta_path = sidecars["metadata"]
    if meta_path and os.path.exists(meta_path):
        meta_name = os.path.basename(meta_path)
        if meta_name.startswith(orig_stem):
            ext_part = meta_name[len(orig_stem):]
        else:
            _, ext_part = os.path.splitext(meta_name)

        out_meta_path = os.path.join(out_dir, f"{out_stem}{ext_part}")

        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # JSON内のファイル名・モデル名・Rank・精度の更新
            new_model_filename = os.path.basename(output_model_path)

            if isinstance(data, dict):
                # 0. 縮小後ファイルサイズの取得
                out_size_bytes = os.path.getsize(output_model_path) if os.path.exists(output_model_path) else 0
                out_size_kb = round(out_size_bytes / 1024.0, 2)

                # 1. バージョン追記タグの構築 (例: [Rank 32] [fp8_e4m3fn])
                tag_parts = []
                if svd_rank and svd_rank != "none":
                    tag_parts.append(f"[Rank {svd_rank}]")
                if precision_key and precision_key != "keep":
                    tag_parts.append(f"[{precision_key}]")
                tag_suffix = " ".join(tag_parts)

                # 2. ファイル名関連フィールドの更新
                for key in ["filename", "fileName", "file_name", "name", "model_name", "modelName"]:
                    if key in data and isinstance(data[key], str):
                        # 古いファイル名が含まれていれば新しいファイル名に置換
                        if orig_stem in data[key]:
                            data[key] = data[key].replace(orig_stem, out_stem)
                        elif key in ["filename", "fileName", "file_name"]:
                            data[key] = new_model_filename

                # 3. ファイルサイズフィールドの更新 (トップレベル)
                if "sizeKB" in data:
                    data["sizeKB"] = out_size_kb
                if "size" in data and isinstance(data["size"], (int, float)):
                    data["size"] = out_size_bytes
                if "fileSize" in data:
                    data["fileSize"] = out_size_bytes
                if "file_size" in data:
                    data["file_size"] = out_size_bytes

                # 4. バージョンフィールドの更新 (トップレベル)
                if tag_suffix:
                    for v_key in ["version", "versionName", "version_name"]:
                        if v_key in data and isinstance(data[v_key], str):
                            data[v_key] = f"{data[v_key]} {tag_suffix}".strip()
                    if not any(k in data for k in ["version", "versionName", "version_name"]):
                        data["version"] = tag_suffix

                # 4-1. model フィールドへの追記
                if "model" in data:
                    if isinstance(data["model"], dict):
                        if tag_suffix and "name" in data["model"] and isinstance(data["model"]["name"], str):
                            data["model"]["name"] = f"{data['model']['name']} {tag_suffix}".strip()
                        if tag_parts:
                            if "tags" in data["model"] and isinstance(data["model"]["tags"], list):
                                for tp in tag_parts:
                                    if len(data["model"]["tags"]) > 0 and isinstance(data["model"]["tags"][0], dict):
                                        if not any(item.get("name") == tp for item in data["model"]["tags"] if isinstance(item, dict)):
                                            data["model"]["tags"].append({"name": tp})
                                    else:
                                        if tp not in data["model"]["tags"]:
                                            data["model"]["tags"].append(tp)
                            else:
                                data["model"]["tags"] = list(tag_parts)
                    elif isinstance(data["model"], str) and tag_suffix:
                        data["model"] = f"{data['model']} {tag_suffix}".strip()

                # 4-2. tags フィールドへの追記 (トップレベル)
                if tag_parts:
                    if "tags" in data and isinstance(data["tags"], list):
                        for tp in tag_parts:
                            if len(data["tags"]) > 0 and isinstance(data["tags"][0], dict):
                                if not any(item.get("name") == tp for item in data["tags"] if isinstance(item, dict)):
                                    data["tags"].append({"name": tp})
                            else:
                                if tp not in data["tags"]:
                                    data["tags"].append(tp)
                    else:
                        data["tags"] = list(tag_parts)

                # 5. Civitai 形式の files / modelVersion(s) 内のサイズ・バージョン・タグ更新
                if "files" in data and isinstance(data["files"], list):
                    for item in data["files"]:
                        if isinstance(item, dict):
                            if "sizeKB" in item:
                                item["sizeKB"] = out_size_kb
                            if "size" in item:
                                item["size"] = out_size_bytes
                            if "name" in item and isinstance(item["name"], str):
                                if orig_stem in item["name"]:
                                    item["name"] = item["name"].replace(orig_stem, out_stem)
                                else:
                                    item["name"] = new_model_filename

                if "modelVersion" in data and isinstance(data["modelVersion"], dict):
                    mv = data["modelVersion"]
                    if tag_suffix and "name" in mv and isinstance(mv["name"], str):
                        mv["name"] = f"{mv['name']} {tag_suffix}".strip()
                    if "model" in mv and isinstance(mv["model"], dict):
                        if tag_suffix and "name" in mv["model"] and isinstance(mv["model"]["name"], str):
                            mv["model"]["name"] = f"{mv['model']['name']} {tag_suffix}".strip()
                    if tag_parts and "tags" in mv and isinstance(mv["tags"], list):
                        for tp in tag_parts:
                            if tp not in mv["tags"]:
                                mv["tags"].append(tp)
                    if "files" in mv and isinstance(mv["files"], list):
                        for item in mv["files"]:
                            if isinstance(item, dict):
                                if "sizeKB" in item:
                                    item["sizeKB"] = out_size_kb
                                if "size" in item:
                                    item["size"] = out_size_bytes
                                if "name" in item and isinstance(item["name"], str):
                                    if orig_stem in item["name"]:
                                        item["name"] = item["name"].replace(orig_stem, out_stem)
                                    else:
                                        item["name"] = new_model_filename

                if "modelVersions" in data and isinstance(data["modelVersions"], list):
                    for mv in data["modelVersions"]:
                        if isinstance(mv, dict):
                            if tag_suffix and "name" in mv and isinstance(mv["name"], str):
                                mv["name"] = f"{mv['name']} {tag_suffix}".strip()
                            if "model" in mv and isinstance(mv["model"], dict):
                                if tag_suffix and "name" in mv["model"] and isinstance(mv["model"]["name"], str):
                                    mv["model"]["name"] = f"{mv['model']['name']} {tag_suffix}".strip()
                            if "files" in mv and isinstance(mv["files"], list):
                                for item in mv["files"]:
                                    if isinstance(item, dict):
                                        if "sizeKB" in item:
                                            item["sizeKB"] = out_size_kb
                                        if "size" in item:
                                            item["size"] = out_size_bytes

                # 6. SVD Rank が指定されている場合の更新
                if svd_rank and svd_rank != "none":
                    target_rank_int = int(svd_rank)
                    if "ss_network_dim" in data:
                        data["ss_network_dim"] = str(target_rank_int)
                    if "ss_network_alpha" in data:
                        data["ss_network_alpha"] = str(float(target_rank_int))
                    if "dim" in data:
                        data["dim"] = target_rank_int
                    if "rank" in data:
                        data["rank"] = target_rank_int

                # 7. 量子化精度の記録
                if precision_key and precision_key != "keep":
                    data["quantization_precision"] = precision_key
                    if "ss_output_name" in data:
                        data["ss_output_name"] = out_stem

                # 8. 変換履歴メタデータの付与
                data["_resized_by"] = "SDXL Model & LoRA Quantizer"
                data["_resize_info"] = {
                    "original_file": os.path.basename(orig_model_path),
                    "svd_rank": svd_rank if svd_rank != "none" else "original",
                    "precision": precision_key,
                    "reduced_size_bytes": out_size_bytes,
                    "reduced_size_kb": out_size_kb
                }

            with open(out_meta_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            created_files.append(out_meta_path)
            if log_callback:
                log_callback(f"[付属ファイル] metadata.json を変換して出力: {os.path.basename(out_meta_path)}")

        except Exception as e:
            if log_callback:
                log_callback(f"[警告] メタデータJSONの変換失敗: {e}")

    return created_files
