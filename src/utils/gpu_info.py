"""GPUアーキテクチャ、量子化精度、SVD Rankオプションの定義"""

# SVD Rank リサイズオプション (key, label, description)
SVD_RANK_OPTIONS = [
    ("none", "リサイズなし (元のRankを維持)", "【現状維持】元のLoRAが持つ学習Rankと表現力（画風・ポーズ・構図等）をそのまま維持したい場合に選択"),
    ("128", "Target Rank 128 (緻密画風・超大容量)", "【緻密な画風・極大情報量向け】極めて複雑な画風全体の学習やダイナミックな構図など、膨大な情報量とディテールを極限まで保持したい場合に有効（※サイズ大・過学習に注意）"),
    ("64", "Target Rank 64 (大容量LoRAの標準化)", "【複雑なポーズ構図・緻密な画風・複雑な衣装向け】躍動感のあるポーズや複雑な構図、特定のイラストレーターの緻密な画風、複雑な衣装のキャラクターなどを正確に再現したい場合に有効"),
    ("32", "Target Rank 32 (高精度バランス・推奨)", "【キャラクター・標準ポーズ構図・衣装向け】特定のキャラクターや衣装、標準的なポーズや日常構図の再現に適しており、表現力とファイルサイズのバランスが最も優れた推奨設定"),
    ("16", "Target Rank 16 (軽量・高速)", "【シンプルキャラ・固定ポーズ・小物向け】シンプルな立ちポーズや単一キャラクター、特定アイテムの再現に適しており、ファイルサイズを抑えて軽量・高速化したい場合に有効"),
    ("8", "Target Rank 8 (極小・省VRAM)", "【全体の雰囲気・シンプルな物体向け】構図変化を伴わない全体の雰囲気変更や、シンプルな物体・概念の学習に適した極小・省VRAM設定"),
    ("4", "Target Rank 4 (最小)", "【ニュアンス微調整・最小容量向け】ポーズや構図の変化を伴わない最小限の色味・概念の微調整や、極限までファイル容量を削りたい場合に有効"),
]

# 量子化・精度オプション
PRECISION_OPTIONS = {
    "keep": {
        "name": "精度変換なし (元の精度を維持)",
        "suffix": "",
        "target_gpu": "全環境",
        "compatibility": "完全互換",
        "bytes_per_elem": 2.0,
        "description": "【完全互換・品質維持】元モデルのデータ形式（FP16/BF16など）を完全に維持し、互換性と画質を保ちます。"
    },
    "fp8_e4m3fn": {
        "name": "[RTX 40xx / 50xx] FP8 (e4m3fn) - 高精度 8bit Float (約50%縮小)",
        "suffix": "_fp8_e4m3fn",
        "target_gpu": "RTX 40xx (Ada Lovelace), RTX 50xx (Blackwell)",
        "compatibility": "ComfyUI / SD-WebUI Forge 標準対応",
        "bytes_per_elem": 1.0,
        "description": "【RTX 40xx/50xx推奨・高画質軽量化】画質劣化が極めて少なく、モデルサイズを約50%削減する高品質8bit浮動小数点形式。"
    },
    "fp8_e5m2": {
        "name": "[RTX 40xx / 50xx] FP8 (e5m2) - 広ダイナミックレンジ 8bit Float (約50%縮小)",
        "suffix": "_fp8_e5m2",
        "target_gpu": "RTX 40xx (Ada Lovelace), RTX 50xx (Blackwell)",
        "compatibility": "ComfyUI / SD-WebUI Forge 対応",
        "bytes_per_elem": 1.0,
        "description": "【広ダイナミックレンジ】指数部の範囲が広く数値のオーバーフローを防ぐ8bit浮動小数点形式。"
    },
    "int8": {
        "name": "[RTX 20xx / 30xx / 40xx / 50xx] INT8 - 汎用整数量子化 (約50%縮小)",
        "suffix": "_int8",
        "target_gpu": "RTX 20xx (Turing), RTX 30xx (Ampere), RTX 40xx, RTX 50xx (全RTX世代)",
        "compatibility": "PyTorch / bitsandbytes / ComfyUI 対応",
        "bytes_per_elem": 1.0,
        "description": "【全世代RTX対応・汎用軽量化】Turing/Ampere/Ada世代に広く対応し、サイズを約50%削減する汎用8bit整数量子化形式。"
    },
    "int4": {
        "name": "[RTX 20xx / 30xx / 40xx / 50xx] INT4 / NF4 - 汎用4bit量子化 (約75%縮小)",
        "suffix": "_int4",
        "target_gpu": "RTX 20xx (Turing), RTX 30xx (Ampere), RTX 40xx, RTX 50xx (全RTX世代)",
        "compatibility": "bitsandbytes / ComfyUI-NF4 対応",
        "bytes_per_elem": 0.5,
        "description": "【低VRAM環境・超軽量化】モデルサイズを約75%削減し、メモリ消費を最小限に抑えたい場合に特化した汎用4bit量子化形式。"
    },
    "nvfp4": {
        "name": "[RTX 50xx] NVFP4 / FP4 - Blackwell ネイティブ4bit Float (約75%縮小)",
        "suffix": "_nvfp4",
        "target_gpu": "RTX 50xx (Blackwell アーキテクチャ)",
        "compatibility": "次世代 PyTorch / TensorRT-LLM / ComfyUI 対応予定",
        "bytes_per_elem": 0.5,
        "description": "【RTX 50xx最適化・次世代高速化】Blackwell世代のハードウェアアクセラレーションに対応したネイティブ4bit浮動小数点形式。"
    },
    "fp16": {
        "name": "[全GPU / CPU] FP16 (Half Precision) - 標準16bit (約50%縮小 ※FP32時)",
        "suffix": "_fp16",
        "target_gpu": "全GPU (RTX / GTX / Radeon / Apple Silicon / CPU)",
        "compatibility": "完全互換",
        "bytes_per_elem": 2.0,
        "description": "【全GPU/CPU互換・標準形式】全環境で完全互換となる標準的な16bit浮動小数点形式（FP32の軽量化等に有効）。"
    },
    "gguf_q4": {
        "name": "[全GPU / CPU] GGUF Q4_K_M - ComfyUI-GGUF / sd.cpp 形式 (約70%縮小)",
        "suffix": "_q4_k_m.gguf",
        "target_gpu": "全GPU / CPU",
        "compatibility": "ComfyUI-GGUF, sd.cpp, llama.cpp",
        "bytes_per_elem": 0.6,
        "description": "【ComfyUI-GGUF / sd.cpp向け】GGUF対応ノードで直接読み込み可能な高効率量子化形式。"
    }
}

# 後方互換用エイリアス
QUANT_FORMATS = PRECISION_OPTIONS

def get_svd_rank_options() -> list:
    """現在の言語設定に応じたSVD Rankオプション一覧を返します"""
    from .i18n import t
    options = []
    for key, def_label, def_desc in SVD_RANK_OPTIONS:
        label = t(f"ranks.{key}.label")
        desc = t(f"ranks.{key}.desc")
        # 未翻訳の場合はデフォルト日本語
        if label.startswith("[MISSING:"):
            label = def_label
        if desc.startswith("[MISSING:"):
            desc = def_desc
        options.append((key, label, desc))
    return options

def get_precision_options() -> dict:
    """現在の言語設定に応じた量子化精度オプション辞書を返します"""
    from .i18n import t
    options = {}
    for key, info in PRECISION_OPTIONS.items():
        name = t(f"precisions.{key}.name")
        desc = t(f"precisions.{key}.desc")
        target_gpu = t(f"precisions.{key}.target_gpu")
        compatibility = t(f"precisions.{key}.compatibility")

        if name.startswith("[MISSING:"):
            name = info["name"]
        if desc.startswith("[MISSING:"):
            desc = info["description"]
        if target_gpu.startswith("[MISSING:"):
            target_gpu = info.get("target_gpu", "")
        if compatibility.startswith("[MISSING:"):
            compatibility = info.get("compatibility", "")

        item = dict(info)
        item["name"] = name
        item["description"] = desc
        item["target_gpu"] = target_gpu
        item["compatibility"] = compatibility
        options[key] = item
    return options

