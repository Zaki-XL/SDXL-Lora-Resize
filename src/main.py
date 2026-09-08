"""SDXL Model Quantizer GUI アプリケーション エントリポイント"""
import sys
import os
import ctypes

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import QSharedMemory
from PyQt6.QtGui import QIcon
from src.gui.main_window import MainWindow

APP_UNIQUE_KEY = "SDXL_Model_Quantizer_SharedMemory_Lock_98a7b"

def main():
    # Windows タスクバーで独立したアプリアイコンとして認識させる
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("SDXLQuantizer.RankResizer.App.1.0")
    except Exception:
        pass

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # アプリアイコンの設定 (ウィンドウ左上 & タスクバー)
    base_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    icon_path = os.path.join(base_dir, "assets", "app_icon.png")
    if not os.path.exists(icon_path):
        icon_path = os.path.join(base_dir, "assets", "app_icon.ico")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    # QSharedMemory による二重起動ガード
    shared_memory = QSharedMemory(APP_UNIQUE_KEY)
    if not shared_memory.create(1):
        # 既にアタッチされているか、共有メモリが存在する
        QMessageBox.warning(
            None,
            "二重起動の防止",
            "SDXL Model Quantizer は既に起動しています。\n既存のウィンドウをご利用ください。"
        )
        sys.exit(0)

    window = MainWindow()
    window.show()
    
    ret = app.exec()
    shared_memory.detach()
    sys.exit(ret)

if __name__ == '__main__':
    main()
