"""多言語対応 (i18n) 管理モジュール

lang/ ディレクトリ内の *.json を動的に探索・読み込み、
キーに対応する翻訳文字列を提供します。
翻訳不足時は [MISSING: key] を返却してコンソールに警告を出力します。
"""
import os
import json
import glob
from typing import Dict, List, Any, Optional

class I18nManager:
    def __init__(self, lang_dir: Optional[str] = None):
        if lang_dir is None:
            # プロジェクトルートの lang/ ディレクトリをデフォルトとする
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.lang_dir = os.path.join(base_dir, "lang")
        else:
            self.lang_dir = lang_dir

        self.current_lang = "ja"
        self._translations: Dict[str, Any] = {}
        self.available_languages: List[Dict[str, str]] = []

        self.refresh_available_languages()
        self.set_language(self.current_lang)

    def refresh_available_languages(self) -> List[Dict[str, str]]:
        """lang/ ディレクトリ内の *.json をスキャンして利用可能な言語一覧を取得"""
        languages = []
        if os.path.isdir(self.lang_dir):
            json_files = glob.glob(os.path.join(self.lang_dir, "*.json"))
            for jf in json_files:
                code = os.path.splitext(os.path.basename(jf))[0]
                try:
                    with open(jf, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    name = data.get("language_name", code)
                except Exception as e:
                    name = code
                languages.append({"code": code, "name": name, "file": jf})

        # 言語コード順にソート (ja を先頭に優先表示)
        languages.sort(key=lambda x: (0 if x["code"] == "ja" else 1, x["code"]))
        self.available_languages = languages
        return languages

    def get_available_languages(self) -> Dict[str, str]:
        """利用可能な言語コードと表示名の辞書 {code: name} を返却"""
        return {item["code"]: item["name"] for item in self.available_languages}

    def set_language(self, lang_code: str) -> bool:
        """指定された言語コードの JSON ファイルをロードして切り替え"""
        target_file = os.path.join(self.lang_dir, f"{lang_code}.json")
        if not os.path.exists(target_file):
            print(f"[I18N ERROR] Language file not found for code '{lang_code}': {target_file}")
            # フォールバックとして ja を試みる
            target_file = os.path.join(self.lang_dir, "ja.json")
            if not os.path.exists(target_file):
                self._translations = {}
                self.current_lang = lang_code
                return False

        try:
            with open(target_file, "r", encoding="utf-8") as f:
                self._translations = json.load(f)
            self.current_lang = lang_code
            return True
        except Exception as e:
            print(f"[I18N ERROR] Failed to parse translation JSON '{target_file}': {e}")
            self._translations = {}
            return False

    def t(self, key: str, **kwargs) -> str:
        """
        翻訳文字列を取得します。
        ドット区切りの階層キー (例: 'main.btn_start') にも対応。
        翻訳キーが存在しない場合は [MISSING: {key}] を返却しエラーログを出力します。
        """
        keys = key.split(".")
        val = self._translations
        found = True

        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                found = False
                break

        if not found or val is None or not isinstance(val, (str, int, float)):
            err_str = f"[MISSING: {key}]"
            print(f"[I18N ERROR] Missing translation key '{key}' in language '{self.current_lang}'")
            return err_str

        text = str(val)
        if kwargs:
            try:
                text = text.format(**kwargs)
            except Exception as e:
                print(f"[I18N ERROR] Format error for key '{key}': {e}")
        return text

# シングルトンインスタンスとショートカット関数
i18n = I18nManager()

def t(key: str, **kwargs) -> str:
    return i18n.t(key, **kwargs)
