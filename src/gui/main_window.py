"""メインウィンドウ GUI 実装 (選択削除・D&Dクリア・件数表示・シャドウ/グラデーションボタン対応)"""
import os
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QTableWidget, QTableWidgetItem, QHeaderView, QProgressBar,
    QTextEdit, QFileDialog, QCheckBox, QGroupBox, QMessageBox, QFrame,
    QGraphicsDropShadowEffect
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QColor, QKeyEvent

from ..utils.config_manager import ConfigManager
from ..utils.gpu_info import (
    SVD_RANK_OPTIONS, PRECISION_OPTIONS,
    get_svd_rank_options, get_precision_options
)
from ..utils.i18n import t, i18n
from ..utils.naming import generate_output_path
from ..utils.safetensors_io import inspect_model_metadata
from ..utils.sidecar_files import find_associated_files
from ..quantizer.estimator import estimate_quantized_size
from ..quantizer.validator import diagnose_lora_health
from .health_dialog import LoRAHealthDialog

class DropAreaWidget(QFrame):
    def __init__(self, parent_window):
        super().__init__()
        self.parent_window = parent_window
        self.setAcceptDrops(True)
        self.setStyleSheet("""
            QFrame {
                border: 2px dashed #3b82f6;
                border-radius: 8px;
                background-color: #f8fafc;
            }
            QFrame:hover {
                background-color: #eff6ff;
                border-color: #2563eb;
            }
        """)
        layout = QVBoxLayout(self)
        self.label = QLabel(t("main.drop_hint"))
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("color: #475569; font-size: 14px; font-weight: bold;")
        layout.addWidget(self.label)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        try:
            files = []
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                if os.path.isfile(file_path):
                    if file_path.lower().endswith(".safetensors"):
                        files.append(file_path)
                elif os.path.isdir(file_path):
                    for root, _, filenames in os.walk(file_path):
                        for fn in filenames:
                            if fn.lower().endswith(".safetensors"):
                                files.append(os.path.join(root, fn))
            if files:
                # 2. ドラッグ＆ドロップ時に既存の選択ファイルをクリアし、進捗バーを0にリセット
                self.parent_window.clear_files()
                self.parent_window.reset_progress_bars()
                self.parent_window.add_files(files)
        except Exception as e:
            self.parent_window.log(f"[エラー] D&Dファイル処理中に例外が発生しました: {e}")
        finally:
            event.acceptProposedAction()

class FileTableWidget(QTableWidget):
    """Deleteキーによる選択行削除およびD&D追加に対応したカスタムTableWidget"""
    def __init__(self, parent_window):
        super().__init__(0, 10)
        self.parent_window = parent_window
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            self.parent_window.drop_area.dropEvent(event)
        else:
            super().dropEvent(event)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.parent_window.remove_selected_files()
        else:
            super().keyPressEvent(event)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = ConfigManager()
        saved_lang = self.config.get("language", "ja")
        i18n.set_language(saved_lang)

        self.file_data = [] # list of dict
        self.worker = None
        self.loading_settings = True

        self.init_ui()
        self.load_saved_settings()
        self.loading_settings = False
        self.on_options_changed()

        # UI表示後にバックグラウンドで PyTorch/CUDA を非同期ウォームアップ
        QTimer.singleShot(150, self._start_backend_warmup)

    def _start_backend_warmup(self):
        """UI描画完了後にバックグラウンドで PyTorch を非同期ロードし、変換開始時のラグをゼロにする"""
        import threading

        def _warmup_task():
            try:
                import torch
                # バックグラウンドでCUDAが利用可能かチェック（ドライバ・DLLの事前展開）
                if torch.cuda.is_available():
                    _ = torch.cuda.is_initialized()
            except Exception:
                pass

        warmup_thread = threading.Thread(target=_warmup_task, daemon=True)
        warmup_thread.start()

    def init_ui(self):
        self.setWindowTitle(t("main.window_title"))
        self.resize(int(self.config.get("last_window_width", "1250")), int(self.config.get("last_window_height", "800")))

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # 1. ドラッグ＆ドロップ領域
        self.drop_area = DropAreaWidget(self)
        self.drop_area.setFixedHeight(70)
        main_layout.addWidget(self.drop_area)

        # 2. ファイル一覧テーブル (10カラム)
        self.table = FileTableWidget(self)
        self.table.setHorizontalHeaderLabels([
            t("main.table.col_orig_name"),
            t("main.table.col_model_type"),
            t("main.table.col_orig_rank"),
            t("main.table.col_sidecars"),
            t("main.table.col_format"),
            t("main.table.col_out_name"),
            t("main.table.col_orig_size"),
            t("main.table.col_est_size"),
            t("main.table.col_reduction"),
            t("main.table.col_status")
        ])
        
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: #ffffff;
                alternate-background-color: #f1f5f9;
                gridline-color: #e2e8f0;
                selection-background-color: #bfdbfe;
                selection-color: #000000;
            }
            QHeaderView::section {
                background-color: #e2e8f0;
                font-weight: bold;
                padding: 4px;
                border: 1px solid #cbd5e1;
            }
        """)

        header = self.table.horizontalHeader()
        for i in range(10):
            header.setSectionResizeMode(i, QHeaderView.ResizeMode.Interactive)
        
        self.table.setColumnWidth(0, 190) # 元ファイル名
        self.table.setColumnWidth(1, 130) # 判定形式
        self.table.setColumnWidth(2, 80)  # 元Rank
        self.table.setColumnWidth(3, 95)  # 付属ファイル
        self.table.setColumnWidth(4, 115) # データ精度
        self.table.setColumnWidth(5, 210) # 変換後ファイル名
        self.table.setColumnWidth(6, 80)  # 元サイズ
        self.table.setColumnWidth(7, 85)  # 推定後サイズ
        self.table.setColumnWidth(8, 70)  # 削減率
        self.table.setColumnWidth(9, 110) # ステータス

        self.table.itemSelectionChanged.connect(self.on_table_selection_changed)
        main_layout.addWidget(self.table)

        # 3. ファイル操作ボタン & 件数表示ラベル & 言語切り替え
        btn_layout = QHBoxLayout()
        
        self.btn_add_files = QPushButton(t("main.buttons.add_files"))
        self.btn_add_files.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:1 #f1f5f9);
                border: 1px solid #cbd5e1;
                border-radius: 5px;
                padding: 6px 14px;
                font-weight: bold;
                color: #334155;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f8fafc, stop:1 #e2e8f0);
                border-color: #94a3b8;
            }
        """)
        self.btn_add_files.clicked.connect(self.select_files)

        # 1. 選択削除ボタン
        self.btn_remove_selected = QPushButton(t("main.buttons.remove_selected"))
        self.btn_remove_selected.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:1 #fee2e2);
                border: 1px solid #fca5a5;
                border-radius: 5px;
                padding: 6px 14px;
                font-weight: bold;
                color: #b91c1c;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #fee2e2, stop:1 #fecaca);
                border-color: #f87171;
            }
        """)
        self.btn_remove_selected.clicked.connect(self.remove_selected_files)

        self.btn_clear_files = QPushButton(t("main.buttons.clear_files"))
        self.btn_clear_files.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffffff, stop:1 #f1f5f9);
                border: 1px solid #cbd5e1;
                border-radius: 5px;
                padding: 6px 14px;
                color: #64748b;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f1f5f9, stop:1 #e2e8f0);
            }
        """)
        self.btn_clear_files.clicked.connect(self.clear_files)

        # 3. 変換対象ファイル数・合計容量表示ラベル
        self.lbl_file_count = QLabel(t("main.labels.file_count_format", count=0, size="0.0 MB"))
        self.lbl_file_count.setStyleSheet("font-size: 13px; font-weight: bold; color: #1e293b; padding-left: 10px;")

        btn_layout.addWidget(self.btn_add_files)
        btn_layout.addWidget(self.btn_remove_selected)
        btn_layout.addWidget(self.btn_clear_files)
        btn_layout.addWidget(self.lbl_file_count)
        btn_layout.addStretch()

        # 言語切り替えUI
        self.lbl_lang = QLabel(t("main.language_label"))
        self.lbl_lang.setStyleSheet("font-size: 12px; font-weight: bold; color: #475569; margin-left: 10px;")
        self.combo_lang = QComboBox()
        self.combo_lang.setStyleSheet("""
            QComboBox {
                border: 1px solid #cbd5e1;
                border-radius: 5px;
                padding: 4px 10px;
                background-color: #ffffff;
                font-weight: 500;
                color: #1e293b;
            }
        """)
        available_langs = i18n.get_available_languages()
        for code, name in available_langs.items():
            self.combo_lang.addItem(name, userData=code)
        idx_cur_lang = self.combo_lang.findData(i18n.current_lang)
        if idx_cur_lang >= 0:
            self.combo_lang.setCurrentIndex(idx_cur_lang)
        self.combo_lang.currentIndexChanged.connect(self.on_language_changed)

        btn_layout.addWidget(self.lbl_lang)
        btn_layout.addWidget(self.combo_lang)

        main_layout.addLayout(btn_layout)

        # インスペクターカード
        self.lbl_inspector = QLabel(t("main.labels.inspector_default"))
        self.lbl_inspector.setStyleSheet("background-color: #f8fafc; border: 1px solid #e2e8f0; padding: 6px 10px; border-radius: 6px; color: #334155; font-size: 12px;")
        main_layout.addWidget(self.lbl_inspector)

        # 4. SVD Rank リサイズ & 量子化設定グループ
        self.settings_group = QGroupBox(t("main.labels.settings_group"))
        settings_layout = QVBoxLayout(self.settings_group)

        combo_row = QHBoxLayout()

        self.rank_label = QLabel(t("main.labels.rank_label"))
        self.rank_label.setFixedWidth(110)
        self.combo_rank = QComboBox()
        for idx, (key, text, desc) in enumerate(get_svd_rank_options()):
            self.combo_rank.addItem(text, userData=key)
            self.combo_rank.setItemData(idx, desc, Qt.ItemDataRole.ToolTipRole)
        self.combo_rank.view().setMouseTracking(True)
        self.combo_rank.currentIndexChanged.connect(self.on_options_changed)
        combo_row.addWidget(self.rank_label)
        combo_row.addWidget(self.combo_rank, 2)

        self.prec_label = QLabel(t("main.labels.prec_label"))
        self.prec_label.setFixedWidth(90)
        self.combo_precision = QComboBox()
        for idx, (key, info) in enumerate(get_precision_options().items()):
            self.combo_precision.addItem(info["name"], userData=key)
            self.combo_precision.setItemData(idx, info["description"], Qt.ItemDataRole.ToolTipRole)
        self.combo_precision.view().setMouseTracking(True)
        self.combo_precision.currentIndexChanged.connect(self.on_options_changed)
        combo_row.addWidget(self.prec_label)
        combo_row.addWidget(self.combo_precision, 3)

        settings_layout.addLayout(combo_row)

        self.info_card = QLabel()
        self.info_card.setWordWrap(True)
        self.info_card.setStyleSheet("background-color: #f1f5f9; padding: 8px; border-radius: 6px; color: #1e293b;")
        settings_layout.addWidget(self.info_card)

        opt_layout = QHBoxLayout()
        self.cb_keep_vae = QCheckBox(t("main.labels.keep_vae"))
        self.cb_keep_vae.setChecked(True)
        self.cb_keep_vae.stateChanged.connect(self.on_vae_option_changed)
        opt_layout.addWidget(self.cb_keep_vae)

        self.cb_force_overwrite = QCheckBox(t("main.labels.force_overwrite"))
        self.cb_force_overwrite.stateChanged.connect(self.on_force_overwrite_changed)
        opt_layout.addWidget(self.cb_force_overwrite)

        self.cb_drop_te = QCheckBox(t("main.labels.drop_te"))
        self.cb_drop_te.setToolTip(t("main.labels.drop_te_tooltip"))
        self.cb_drop_te.stateChanged.connect(self.on_drop_te_option_changed)
        opt_layout.addWidget(self.cb_drop_te)

        opt_layout.addStretch()

        self.out_label = QLabel(t("main.labels.output_dir"))
        self.edit_output_dir = QLabel(t("main.labels.output_dir_same"))
        self.edit_output_dir.setStyleSheet("color: #64748b;")
        self.btn_select_out = QPushButton(t("main.buttons.change_dir"))
        self.btn_select_out.clicked.connect(self.select_output_dir)
        self.btn_reset_out = QPushButton(t("main.buttons.reset_dir"))
        self.btn_reset_out.clicked.connect(self.reset_output_dir)

        opt_layout.addWidget(self.out_label)
        opt_layout.addWidget(self.edit_output_dir)
        opt_layout.addWidget(self.btn_select_out)
        opt_layout.addWidget(self.btn_reset_out)
        settings_layout.addLayout(opt_layout)

        main_layout.addWidget(self.settings_group)

        # 5. プログレスバー
        progress_layout = QVBoxLayout()
        self.lbl_progress = QLabel(t("main.status.waiting"))
        self.lbl_progress.setStyleSheet("font-weight: bold; color: #334155;")
        
        self.bar_file = QProgressBar()
        self.bar_file.setValue(0)
        self.bar_file.setFormat(t("main.status.progress_format_file"))

        self.bar_tensor = QProgressBar()
        self.bar_tensor.setValue(0)
        self.bar_tensor.setFormat(t("main.status.progress_format_tensor"))

        progress_layout.addWidget(self.lbl_progress)
        progress_layout.addWidget(self.bar_file)
        progress_layout.addWidget(self.bar_tensor)
        main_layout.addLayout(progress_layout)

        # 6. 実行 & 中断ボタン (4. ドロップシャドウ ＋ リッチグラデーション)
        exec_layout = QHBoxLayout()
        
        self.btn_start = QPushButton(t("main.buttons.start"))
        self.btn_start.setFixedHeight(46)
        self.btn_start.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_start.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3b82f6, stop:1 #1d4ed8);
                color: #ffffff;
                font-weight: bold;
                font-size: 15px;
                border: 1px solid #1e40af;
                border-radius: 8px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #60a5fa, stop:1 #2563eb);
                border-color: #1d4ed8;
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #1e40af, stop:1 #1e3a8a);
            }
            QPushButton:disabled {
                background: #94a3b8;
                border-color: #64748b;
                color: #e2e8f0;
            }
        """)

        # スタートボタンにドロップシャドウ効果を適用
        shadow_start = QGraphicsDropShadowEffect(self)
        shadow_start.setBlurRadius(14)
        shadow_start.setColor(QColor(37, 99, 235, 120))
        shadow_start.setOffset(0, 4)
        self.btn_start.setGraphicsEffect(shadow_start)
        self.btn_start.clicked.connect(self.start_quantization)

        self.btn_cancel = QPushButton(t("main.buttons.cancel"))
        self.btn_cancel.setFixedHeight(46)
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_cancel.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ef4444, stop:1 #b91c1c);
                color: #ffffff;
                font-weight: bold;
                font-size: 14px;
                border: 1px solid #991b1b;
                border-radius: 8px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f87171, stop:1 #dc2626);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #991b1b, stop:1 #7f1d1d);
            }
            QPushButton:disabled {
                background: #e2e8f0;
                border-color: #cbd5e1;
                color: #94a3b8;
            }
        """)

        shadow_cancel = QGraphicsDropShadowEffect(self)
        shadow_cancel.setBlurRadius(12)
        shadow_cancel.setColor(QColor(239, 68, 68, 90))
        shadow_cancel.setOffset(0, 3)
        self.btn_cancel.setGraphicsEffect(shadow_cancel)
        self.btn_cancel.clicked.connect(self.cancel_quantization)

        exec_layout.addWidget(self.btn_start, 4)
        exec_layout.addWidget(self.btn_cancel, 1)
        main_layout.addLayout(exec_layout)

        # 7. リアルタイムログウィンドウ
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFixedHeight(95)
        self.log_text.setStyleSheet("background-color: #0f172a; color: #38bdf8; font-family: Consolas, monospace; font-size: 12px;")
        main_layout.addWidget(self.log_text)

    def load_saved_settings(self):
        saved_lang = self.config.get("language", "ja")
        idx_l = self.combo_lang.findData(saved_lang)
        if idx_l >= 0:
            self.combo_lang.setCurrentIndex(idx_l)

        saved_rank = self.config.get("svd_rank", "none")
        idx_r = self.combo_rank.findData(saved_rank)
        if idx_r >= 0:
            self.combo_rank.setCurrentIndex(idx_r)

        saved_prec = self.config.get("precision_format", self.config.get("quant_format", "fp8_e4m3fn"))
        idx_p = self.combo_precision.findData(saved_prec)
        if idx_p >= 0:
            self.combo_precision.setCurrentIndex(idx_p)

        self.cb_keep_vae.setChecked(self.config.getboolean("keep_vae_fp16", True))
        self.cb_force_overwrite.setChecked(self.config.getboolean("force_overwrite", False))
        self.cb_drop_te.setChecked(self.config.getboolean("drop_te", False))

        saved_out = self.config.get("output_dir", "")
        if saved_out and os.path.isdir(saved_out):
            self.edit_output_dir.setText(saved_out)

    def on_language_changed(self, idx):
        lang_code = self.combo_lang.itemData(idx)
        if lang_code and lang_code != i18n.current_lang:
            i18n.set_language(lang_code)
            self.config.set("language", lang_code)
            self.retranslate_ui()

    def retranslate_ui(self):
        """現在の言語設定に合わせてUIの全テキストを動的に再描画"""
        self.setWindowTitle(t("main.window_title"))
        self.drop_area.label.setText(t("main.drop_hint"))

        # テーブルヘッダー
        self.table.setHorizontalHeaderLabels([
            t("main.table.col_orig_name"),
            t("main.table.col_model_type"),
            t("main.table.col_orig_rank"),
            t("main.table.col_sidecars"),
            t("main.table.col_format"),
            t("main.table.col_out_name"),
            t("main.table.col_orig_size"),
            t("main.table.col_est_size"),
            t("main.table.col_reduction"),
            t("main.table.col_status")
        ])

        # 操作ボタン
        self.btn_add_files.setText(t("main.buttons.add_files"))
        self.btn_remove_selected.setText(t("main.buttons.remove_selected"))
        self.btn_clear_files.setText(t("main.buttons.clear_files"))
        self.lbl_lang.setText(t("main.language_label"))
        self.btn_start.setText(t("main.buttons.start"))
        self.btn_cancel.setText(t("main.buttons.cancel"))
        self.btn_select_out.setText(t("main.buttons.change_dir"))
        self.btn_reset_out.setText(t("main.buttons.reset_dir"))

        # 設定グループ & ラベル
        self.settings_group.setTitle(t("main.labels.settings_group"))
        self.rank_label.setText(t("main.labels.rank_label"))
        self.prec_label.setText(t("main.labels.prec_label"))
        self.cb_keep_vae.setText(t("main.labels.keep_vae"))
        self.cb_force_overwrite.setText(t("main.labels.force_overwrite"))
        self.cb_drop_te.setText(t("main.labels.drop_te"))
        self.cb_drop_te.setToolTip(t("main.labels.drop_te_tooltip"))
        self.out_label.setText(t("main.labels.output_dir"))
        if not self.config.get("output_dir", ""):
            self.edit_output_dir.setText(t("main.labels.output_dir_same"))

        # プログレスバー
        if self.worker is None or not self.worker.isRunning():
            self.lbl_progress.setText(t("main.status.waiting"))
        self.bar_file.setFormat(t("main.status.progress_format_file"))
        self.bar_tensor.setFormat(t("main.status.progress_format_tensor"))

        # SVD Rank コンボボックスの更新（選択中アイテムを維持）
        cur_rank = self.combo_rank.currentData()
        self.combo_rank.blockSignals(True)
        self.combo_rank.clear()
        for idx, (key, text, desc) in enumerate(get_svd_rank_options()):
            self.combo_rank.addItem(text, userData=key)
            self.combo_rank.setItemData(idx, desc, Qt.ItemDataRole.ToolTipRole)
        idx_r = self.combo_rank.findData(cur_rank)
        if idx_r >= 0:
            self.combo_rank.setCurrentIndex(idx_r)
        self.combo_rank.blockSignals(False)

        # 量子化精度 コンボボックスの更新（選択中アイテムを維持）
        cur_prec = self.combo_precision.currentData()
        self.combo_precision.blockSignals(True)
        self.combo_precision.clear()
        for idx, (key, info) in enumerate(get_precision_options().items()):
            self.combo_precision.addItem(info["name"], userData=key)
            self.combo_precision.setItemData(idx, info["description"], Qt.ItemDataRole.ToolTipRole)
        idx_p = self.combo_precision.findData(cur_prec)
        if idx_p >= 0:
            self.combo_precision.setCurrentIndex(idx_p)
        self.combo_precision.blockSignals(False)

        # 設定情報カード & テーブル更新
        self.on_options_changed()
        self.refresh_table()
        if not self.table.selectedIndexes():
            self.lbl_inspector.setText(t("main.labels.inspector_default"))
        else:
            self.on_table_selection_changed()

    def closeEvent(self, event):
        self.config.set("last_window_width", str(self.width()))
        self.config.set("last_window_height", str(self.height()))
        event.accept()

    def select_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, t("main.dialogs.select_model_title"), "", "Safetensors (*.safetensors);;All Files (*.*)")
        if paths:
            self.add_files(paths)

    def select_output_dir(self):
        d = QFileDialog.getExistingDirectory(self, t("main.dialogs.select_dir_title"))
        if d:
            self.edit_output_dir.setText(d)
            self.config.set("output_dir", d)
            self.recalculate_estimates()

    def reset_output_dir(self):
        self.edit_output_dir.setText(t("main.labels.output_dir_same"))
        self.config.set("output_dir", "")
        self.recalculate_estimates()

    def add_files(self, filepaths: list[str]):
        svd_rank = self.combo_rank.currentData()
        prec_key = self.combo_precision.currentData()
        keep_vae = self.cb_keep_vae.isChecked()
        out_dir = self.config.get("output_dir", "")

        batch_te_action = None
        has_multiple = len(filepaths) > 1

        for fp in filepaths:
            if fp in [item["orig_path"] for item in self.file_data]:
                continue
            meta = inspect_model_metadata(fp)
            is_valid = meta.get("is_valid", True)
            err_msg = meta.get("error_msg", "")

            # 健全性・色破綻診断 (NaN/Inf破損, CLIP過学習検知)
            health = diagnose_lora_health(fp)
            te_action = "drop_te" if self.cb_drop_te.isChecked() else "keep"
            is_manual = False

            if health.get("has_issues") or health.get("has_clip_overfit"):
                if batch_te_action is not None:
                    if batch_te_action == "skip":
                        self.log(f"[スキップ] ユーザーの一括選択によりスキップ: {os.path.basename(fp)}")
                        continue
                    te_action = batch_te_action
                    is_manual = True
                else:
                    # 確認・選択ダイアログを表示
                    dialog = LoRAHealthDialog(self, fp, health, has_multiple=has_multiple)
                    if dialog.exec() == LoRAHealthDialog.DialogCode.Accepted:
                        chosen_action, apply_all = dialog.get_result()
                        te_action = chosen_action
                        is_manual = True
                        if apply_all:
                            batch_te_action = chosen_action
                    else:
                        # スキップ選択またはキャンセル
                        chosen_action, apply_all = dialog.get_result()
                        if apply_all:
                            batch_te_action = "skip"
                        self.log(f"[スキップ] 登録をスキップしました: {os.path.basename(fp)}")
                        continue

            # 異常検知時に画面下のログに出力
            if not is_valid or health.get("is_corrupt"):
                fn = os.path.basename(fp)
                detail = err_msg or (", ".join(health.get("issues", [])))
                self.log(f"[警告/異常検知] {fn}: {detail}")
            elif te_action != "keep":
                fn = os.path.basename(fp)
                self.log(f"[対策適用] {fn}: 対策={te_action} を設定しました")

            sidecars = find_associated_files(fp)
            out_fp = generate_output_path(fp, svd_rank, prec_key, out_dir, te_action=te_action)
            orig, est, _ = estimate_quantized_size(fp, svd_rank, prec_key, keep_vae, te_action=te_action)
            status_key = "corrupt" if (not is_valid or health.get("is_corrupt")) else "waiting"
            self.file_data.append({
                "orig_path": fp,
                "out_path": out_fp,
                "orig_size": orig,
                "est_size": est,
                "status_key": status_key,
                "status": t(f"main.status.{status_key}"),
                "meta": meta,
                "health": health,
                "te_action": te_action,
                "te_action_manual": is_manual,
                "sidecars": sidecars,
                "is_error": not is_valid or health.get("is_corrupt", False)
            })

        self.refresh_table()

    def remove_selected_files(self):
        """1. 選択行のファイルを一覧から削除"""
        selected_rows = sorted(set(index.row() for index in self.table.selectedIndexes()), reverse=True)
        if not selected_rows:
            return
        
        for row in selected_rows:
            if row < len(self.file_data):
                del self.file_data[row]

        self.refresh_table()
        self.lbl_inspector.setText(t("main.labels.inspector_removed"))

    def reset_progress_bars(self):
        """ファイル進捗・テンソル進捗・進捗ラベルを 0 (待機中) にリセット"""
        self.bar_file.setMaximum(100)
        self.bar_file.setValue(0)
        self.bar_tensor.setMaximum(100)
        self.bar_tensor.setValue(0)
        self.lbl_progress.setText(t("main.status.waiting"))

    def clear_files(self):
        self.file_data.clear()
        self.lbl_inspector.setText(t("main.labels.inspector_default"))
        self.reset_progress_bars()
        self.refresh_table()

    def on_options_changed(self):
        svd_rank = self.combo_rank.currentData()
        prec_key = self.combo_precision.currentData()
        if not prec_key:
            return
        
        if not self.loading_settings:
            self.config.set("svd_rank", svd_rank)
            self.config.set("precision_format", prec_key)
        
        precision_opts = get_precision_options()
        p_info = precision_opts.get(prec_key, {})
        rank_opts = get_svd_rank_options()
        rank_item = next((item for item in rank_opts if item[0] == svd_rank), None)
        rank_label = rank_item[1] if rank_item else (f"Target Rank {svd_rank}" if svd_rank != "none" else "None")
        rank_desc = rank_item[2] if rank_item else ""

        # コンボボックス本体のツールチップも更新
        self.combo_rank.setToolTip(f"<b>{rank_label}</b><br>{rank_desc}")
        self.combo_precision.setToolTip(f"<b>{p_info.get('name', '')}</b><br>{p_info.get('description', '')}")
        
        desc = (
            f"<b>{rank_label}</b> × <b>{p_info.get('name', '')}</b><br>"
            f"<b>{rank_desc}</b><br>"
            f"{p_info.get('description', '')}<br>"
            f"GPU: {p_info.get('target_gpu', '')} | {p_info.get('compatibility', '')}"
        )
        self.info_card.setText(desc)
        self.recalculate_estimates()

    def on_vae_option_changed(self):
        if not self.loading_settings:
            self.config.set("keep_vae_fp16", str(self.cb_keep_vae.isChecked()).lower())
        self.recalculate_estimates()

    def on_force_overwrite_changed(self):
        if not self.loading_settings:
            self.config.set("force_overwrite", str(self.cb_force_overwrite.isChecked()).lower())

    def on_drop_te_option_changed(self):
        if not self.loading_settings:
            self.config.set("drop_te", str(self.cb_drop_te.isChecked()).lower())
        self.recalculate_estimates()

    def recalculate_estimates(self):
        svd_rank = self.combo_rank.currentData()
        prec_key = self.combo_precision.currentData()
        keep_vae = self.cb_keep_vae.isChecked()
        out_dir = self.config.get("output_dir", "")
        global_drop_te = self.cb_drop_te.isChecked()

        for item in self.file_data:
            if not item.get("te_action_manual", False):
                item["te_action"] = "drop_te" if global_drop_te else "keep"
            te_act = item.get("te_action", "keep")
            item["out_path"] = generate_output_path(item["orig_path"], svd_rank, prec_key, out_dir, te_action=te_act)
            _, est, _ = estimate_quantized_size(item["orig_path"], svd_rank, prec_key, keep_vae, te_action=te_act)
            item["est_size"] = est
        self.refresh_table()

    def refresh_table(self):
        self.table.setRowCount(len(self.file_data))
        total_orig_bytes = 0

        for r, item in enumerate(self.file_data):
            orig = item["orig_size"]
            est = item["est_size"]
            total_orig_bytes += orig

            def _fmt_size(sz: int) -> str:
                if sz >= 1024**3:
                    return f"{sz / (1024**3):.2f} GB"
                elif sz >= 1024**2:
                    return f"{sz / (1024**2):.1f} MB"
                elif sz >= 1024:
                    return f"{sz / 1024:.1f} KB"
                else:
                    return f"{sz} B"

            orig_str = _fmt_size(orig)
            est_str = _fmt_size(est)
            red_pct = (1.0 - (est / orig)) * 100.0 if orig > 0 else 0.0

            is_error = item.get("is_error", False) or not item.get("meta", {}).get("is_valid", True)
            item_orig_name = QTableWidgetItem(os.path.basename(item["orig_path"]))
            if is_error:
                # ユーザー要求: エラーのあるファイルは画面上部にある対象ファイルを赤文字にする
                item_orig_name.setForeground(QColor("#dc2626"))
                font = item_orig_name.font()
                font.setBold(True)
                item_orig_name.setFont(font)
                err_text = item.get("meta", {}).get("error_msg", "破損または無効なファイル")
                item_orig_name.setToolTip(f"【⚠️ 異常・破損検知】\n{err_text}\n\nファイル: {item['orig_path']}")
            else:
                item_orig_name.setToolTip(item["orig_path"])

            # 列1: 判定形式
            model_type_str = item["meta"].get("model_type", "不明")
            te_action = item.get("te_action", "keep")
            if te_action == "drop_te":
                model_type_str += f" [{t('main.health.badge_drop_te')}]"
            elif te_action == "scale_te":
                model_type_str += f" [{t('main.health.badge_scale_te')}]"

            item_type = QTableWidgetItem(model_type_str)
            item_type.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if is_error:
                item_type.setForeground(QColor("#dc2626"))
            elif te_action == "drop_te":
                item_type.setForeground(QColor("#059669")) # 緑色でクリーンアップ表示
                font_type = item_type.font()
                font_type.setBold(True)
                item_type.setFont(font_type)
            elif "LyCORIS" in model_type_str:
                item_type.setForeground(QColor("#7c3aed"))
                font_type = item_type.font()
                font_type.setBold(True)
                item_type.setFont(font_type)
            elif "LoRA" in model_type_str:
                item_type.setForeground(QColor("#2563eb"))
                font_type = item_type.font()
                font_type.setBold(True)
                item_type.setFont(font_type)

            health = item.get("health", {})
            issues_tip = ""
            if health.get("issues"):
                issues_tip = f"\n\n【{t('main.health.issues_title')}】\n" + "\n".join(f"• {i}" for i in health["issues"])

            train_tip = ""
            if item["meta"].get("train_info_str"):
                train_tip = f"\n\n【🎯 学習設定】\n• {item['meta']['train_info_str']}"

            item_type.setToolTip(
                f"【判定形式】 {model_type_str}\n"
                f"元Rank: {item['meta'].get('orig_rank_str')}\n"
                f"主要精度: {item['meta'].get('primary_dtype')}\n"
                f"総パラメータ数: {item['meta'].get('total_params_str')}"
                f"{train_tip}"
                f"{issues_tip}"
            )

            # 列2: 元Rank
            item_rank = QTableWidgetItem(item["meta"].get("orig_rank_str", "-"))
            item_rank.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if is_error:
                item_rank.setForeground(QColor("#dc2626"))
            elif item["meta"].get("train_info_str"):
                item_rank.setToolTip(f"【🎯 学習設定】 {item['meta']['train_info_str']}")

            # 列3: 付属ファイル
            sidecar_str = item["sidecars"].get("display_str", "-")
            item_sidecar = QTableWidgetItem(sidecar_str)
            item_sidecar.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if sidecar_str != "-":
                item_sidecar.setForeground(QColor("#0284c7"))

            # 列4: データ精度
            item_format = QTableWidgetItem(item["meta"].get("summary", "不明"))
            if is_error:
                item_format.setForeground(QColor("#dc2626"))
            item_format.setToolTip(
                f"モデル種別: {item['meta'].get('model_type')}\n"
                f"元Rank: {item['meta'].get('orig_rank_str')}\n"
                f"主要精度: {item['meta'].get('primary_dtype')}\n"
                f"総パラメータ数: {item['meta'].get('total_params_str')}"
            )

            # 列5: 変換後ファイル名
            item_out_name = QTableWidgetItem(os.path.basename(item["out_path"]))
            item_out_name.setToolTip(item["out_path"])

            # 列6, 7: 元サイズ, 推定後サイズ
            item_orig = QTableWidgetItem(orig_str)
            item_est = QTableWidgetItem(est_str)
            item_orig.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            item_est.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

            # 列8: 削減率
            if red_pct >= 0:
                item_red = QTableWidgetItem(f"-{red_pct:.1f}%")
                if red_pct > 0:
                    item_red.setForeground(QColor("#16a34a"))
            else:
                item_red = QTableWidgetItem(f"+{abs(red_pct):.1f}%")
                item_red.setForeground(QColor("#dc2626"))
            item_red.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            # 列9: ステータス
            status_key = item.get("status_key")
            if is_error:
                status_text = t("main.status.corrupt")
            elif status_key:
                status_text = t(f"main.status.{status_key}")
            else:
                status_text = item.get("status", "")

            item_status = QTableWidgetItem(status_text)
            if is_error or status_key == "error" or "エラー" in status_text or "Error" in status_text or "错误" in status_text:
                item_status.setForeground(QColor("#dc2626"))
            elif status_key == "completed" or "完了" in status_text or "Completed" in status_text or "完成" in status_text:
                item_status.setForeground(QColor("#16a34a"))
            elif status_key == "skipped" or "パス" in status_text or "スキップ" in status_text or "Skipped" in status_text or "跳过" in status_text:
                item_status.setForeground(QColor("#d97706"))

            self.table.setItem(r, 0, item_orig_name)
            self.table.setItem(r, 1, item_type)
            self.table.setItem(r, 2, item_rank)
            self.table.setItem(r, 3, item_sidecar)
            self.table.setItem(r, 4, item_format)
            self.table.setItem(r, 5, item_out_name)
            self.table.setItem(r, 6, item_orig)
            self.table.setItem(r, 7, item_est)
            self.table.setItem(r, 8, item_red)
            self.table.setItem(r, 9, item_status)

        # 3. 変換対象件数 & 合計容量の更新
        count = len(self.file_data)
        if total_orig_bytes >= 1024**3:
            tot_str = f"{total_orig_bytes / (1024**3):.2f} GB"
        else:
            tot_str = f"{total_orig_bytes / (1024**2):.1f} MB"
        self.lbl_file_count.setText(t("main.labels.file_count_format", count=count, size=tot_str))

    def on_table_selection_changed(self):
        selected_rows = self.table.selectionModel().selectedRows()
        if not selected_rows:
            return
        row = selected_rows[0].row()
        if row < len(self.file_data):
            item = self.file_data[row]
            meta = item.get("meta", {})
            sidecars = item.get("sidecars", {})
            img_list = [os.path.basename(p) for p in sidecars.get("images", [])]
            meta_name = os.path.basename(sidecars.get("metadata", "")) if sidecars.get("metadata") else "-"
            img_str = ", ".join(img_list) if img_list else "-"

            if not meta.get("is_valid", True):
                err_text = meta.get("error_msg", t("main.status.corrupt"))
                desc = (
                    f"⚠️ <span style='color:#dc2626; font-weight:bold;'>[{t('main.status.corrupt')}]</span> "
                    f"<b>{os.path.basename(item['orig_path'])}</b>: <span style='color:#b91c1c;'>{err_text}</span>"
                )
            else:
                train_parts = []
                if meta.get("train_img_count"):
                    train_parts.append(f"学習画像枚数: <b>{meta['train_img_count']}枚</b>")
                if meta.get("train_repeats"):
                    train_parts.append(f"繰り返し回数 (n_repeats): <b>{meta['train_repeats']}回</b>")
                if meta.get("orig_rank") and meta.get("orig_rank") > 0:
                    train_parts.append(f"学習Rank: <b>{meta['orig_rank']}</b>")
                if meta.get("train_steps"):
                    train_parts.append(f"ステップ: <b>{meta['train_steps']:,}</b>")

                train_html = ""
                if train_parts:
                    train_html = f"<br><span style='color:#0369a1;'>🎯 {t('main.labels.train_info_label')} " + " | ".join(train_parts) + "</span>"

                desc = (
                    f"<b>{os.path.basename(item['orig_path'])}</b> | "
                    f"Type: <b>{meta.get('model_type', '-')}</b> | "
                    f"Rank: <span style='color:#2563eb; font-weight:bold;'>{meta.get('orig_rank_str', '-')}</span> | "
                    f"Sidecars: [Img: {img_str} / Meta: {meta_name}] | "
                    f"Dtype: <b>{meta.get('primary_dtype', '-')}</b>"
                    f"{train_html}"
                )
            self.lbl_inspector.setText(desc)

    def log(self, text: str):
        self.log_text.append(text)

    def start_quantization(self):
        if not self.file_data:
            QMessageBox.warning(self, t("main.dialogs.warn_no_files_title"), t("main.dialogs.warn_no_files_msg"))
            return

        # 異常ファイルを除外した有効リストの抽出
        valid_items = [item for item in self.file_data if item.get("meta", {}).get("is_valid", True)]
        invalid_items = [item for item in self.file_data if not item.get("meta", {}).get("is_valid", True)]

        if not valid_items:
            QMessageBox.critical(
                self,
                t("main.dialogs.error_all_invalid_title"),
                t("main.dialogs.error_all_invalid_msg", count=len(invalid_items))
            )
            return

        if invalid_items:
            self.log(f"[Info] {len(invalid_items)} corrupt/invalid file(s) excluded automatically.")

        file_list = [item["orig_path"] for item in valid_items]
        svd_rank = self.combo_rank.currentData()
        prec_key = self.combo_precision.currentData()
        out_dir = self.config.get("output_dir", "")
        keep_vae = self.cb_keep_vae.isChecked()
        force_overwrite = self.cb_force_overwrite.isChecked()

        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.bar_file.setMaximum(len(file_list))
        self.bar_file.setValue(0)

        from .worker_thread import QuantizeWorker
        te_actions = {item["orig_path"]: item.get("te_action", "keep") for item in valid_items}
        self.worker = QuantizeWorker(file_list, svd_rank, prec_key, out_dir, keep_vae, force_overwrite, te_actions=te_actions)
        self.worker.file_progress.connect(self.on_file_progress)
        self.worker.tensor_progress.connect(self.on_tensor_progress)
        self.worker.log_message.connect(self.log)
        self.worker.item_status_changed.connect(self.on_item_status_changed)
        self.worker.finished_all.connect(self.on_finished_all)
        self.worker.need_error_decision.connect(self.handle_error_decision)

        self.log("================ Start Processing ================")
        self.worker.start()

    def cancel_quantization(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.log("[Cancel Request] Cancellation requested...")

    def on_file_progress(self, cur, total):
        self.bar_file.setValue(cur)

    def on_tensor_progress(self, cur, total, name, speed):
        self.bar_tensor.setMaximum(total)
        self.bar_tensor.setValue(cur)
        self.lbl_progress.setText(t("main.status.progress_processing", name=name[:40], speed=speed))

    def on_item_status_changed(self, row, status, out_path):
        if row < len(self.file_data):
            self.file_data[row]["status"] = status
            if "完了" in status or "Completed" in status or "完成" in status:
                self.file_data[row]["status_key"] = "completed"
            elif "スキップ" in status or "パス" in status or "Skipped" in status or "跳过" in status:
                self.file_data[row]["status_key"] = "skipped"
            elif "エラー" in status or "Error" in status or "错误" in status:
                self.file_data[row]["status_key"] = "error"
            elif "中断" in status or "Cancelled" in status or "取消" in status:
                self.file_data[row]["status_key"] = "cancelled"

            if out_path:
                self.file_data[row]["out_path"] = out_path
            self.refresh_table()

    def handle_error_decision(self, filename: str, error_details: str, row_idx: int):
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle(t("main.dialogs.verify_error_title"))
        msg_box.setIcon(QMessageBox.Icon.Critical)
        msg_box.setText(t("main.dialogs.verify_error_msg", filename=filename, details=error_details))
        btn_continue = msg_box.addButton(t("main.dialogs.btn_continue"), QMessageBox.ButtonRole.AcceptRole)
        btn_stop = msg_box.addButton(t("main.dialogs.btn_stop"), QMessageBox.ButtonRole.RejectRole)

        msg_box.exec()
        if msg_box.clickedButton() == btn_continue:
            self.worker.set_error_decision("continue")
        else:
            self.worker.set_error_decision("stop")

    def on_finished_all(self, success, skip, fail):
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.lbl_progress.setText(t("main.status.completed"))
        self.log(f"================ Finished (Success: {success}, Skipped: {skip}, Failed: {fail}) ================")
        
        summary_msg = f"{t('main.status.completed')}\n\nSuccess: {success}\nSkipped: {skip}"
        if fail > 0:
            summary_msg += f"\nFailed: {fail}"
            QMessageBox.warning(self, "Finished (with errors)", summary_msg)
        else:
            QMessageBox.information(self, "Finished", summary_msg)
