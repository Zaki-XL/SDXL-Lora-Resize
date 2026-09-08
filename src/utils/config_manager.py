"""設定ファイル (config.ini) の読み込み・保存管理モジュール"""
import os
import configparser

CONFIG_FILE = "config.ini"

DEFAULT_CONFIG = {
    "Settings": {
        "svd_rank": "none",
        "precision_format": "fp8_e4m3fn",
        "quant_format": "fp8_e4m3fn",
        "output_dir": "",
        "force_overwrite": "false",
        "keep_vae_fp16": "true",
        "language": "ja",
        "last_window_width": "1250",
        "last_window_height": "800"
    }
}

class ConfigManager:
    def __init__(self, config_path=CONFIG_FILE):
        self.config_path = config_path
        self.config = configparser.ConfigParser()
        self.load()

    def load(self):
        if not os.path.exists(self.config_path):
            self.config.read_dict(DEFAULT_CONFIG)
            self.save()
        else:
            self.config.read(self.config_path, encoding='utf-8')
            modified = False
            for section, options in DEFAULT_CONFIG.items():
                if not self.config.has_section(section):
                    self.config.add_section(section)
                    modified = True
                for key, val in options.items():
                    if not self.config.has_option(section, key):
                        self.config.set(section, key, val)
                        modified = True
            if modified:
                self.save()

    def save(self):
        with open(self.config_path, 'w', encoding='utf-8') as f:
            self.config.write(f)

    def get(self, key, fallback=None):
        return self.config.get("Settings", key, fallback=fallback)

    def getboolean(self, key, fallback=False):
        return self.config.getboolean("Settings", key, fallback=fallback)

    def set(self, key, value):
        if not self.config.has_section("Settings"):
            self.config.add_section("Settings")
        self.config.set("Settings", key, str(value))
        self.save()
