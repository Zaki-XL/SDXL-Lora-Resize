"""LoRA 健全性・色破綻診断結果および対策選択ダイアログ"""
import os
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QRadioButton, QButtonGroup, QCheckBox, QFrame, QTextEdit,
    QScrollArea, QWidget
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor
from ..utils.i18n import t

class LoRAHealthDialog(QDialog):
    def __init__(self, parent, filepath: str, health_result: dict, has_multiple: bool = False):
        super().__init__(parent)
        self.filepath = filepath
        self.health = health_result
        self.has_multiple = has_multiple
        self.selected_action = "drop_te"
        self.apply_to_all = False

        self.setWindowTitle(t("main.health.dialog_title"))
        self.resize(680, 520)
        self.setModal(True)

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(14)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # 1. 警告ヘッダーバナー
        header_frame = QFrame()
        is_danger = self.health.get("risk_level") == "danger" or self.health.get("is_corrupt", False)
        bg_color = "#fef2f2" if is_danger else "#fffbeb"
        border_color = "#ef4444" if is_danger else "#f59e0b"
        text_color = "#991b1b" if is_danger else "#92400e"

        header_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {bg_color};
                border: 2px solid {border_color};
                border-radius: 8px;
                padding: 10px;
            }}
        """)
        header_layout = QVBoxLayout(header_frame)
        header_layout.setContentsMargins(10, 8, 10, 8)
        
        lbl_title = QLabel(t("main.health.warn_header"))
        lbl_title.setStyleSheet(f"font-size: 15px; font-weight: bold; color: {text_color}; border: none;")
        header_layout.addWidget(lbl_title)

        fn = os.path.basename(self.filepath)
        lbl_file = QLabel(f"{t('main.health.file_label')} <b>{fn}</b>")
        lbl_file.setStyleSheet(f"font-size: 13px; color: {text_color}; border: none;")
        header_layout.addWidget(lbl_file)
        main_layout.addWidget(header_frame)

        # 2. 検出された問題点一覧
        lbl_issues_head = QLabel(t("main.health.issues_title"))
        lbl_issues_head.setStyleSheet("font-size: 14px; font-weight: bold; color: #1e293b;")
        main_layout.addWidget(lbl_issues_head)

        issues_box = QTextEdit()
        issues_box.setReadOnly(True)
        issues_box.setStyleSheet("""
            QTextEdit {
                background-color: #f8fafc;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                padding: 8px;
                font-size: 13px;
                color: #334155;
            }
        """)
        issues_text = ""
        for iss in self.health.get("issues", []):
            issues_text += f"• {iss}\n"
        if not issues_text:
            issues_text = "• 潜在的なリスクが検出されました。"
        issues_box.setPlainText(issues_text.strip())
        issues_box.setFixedHeight(110)
        main_layout.addWidget(issues_box)

        # 3. 対策の選択肢
        lbl_opts_head = QLabel(t("main.health.options_title"))
        lbl_opts_head.setStyleSheet("font-size: 14px; font-weight: bold; color: #1e293b;")
        main_layout.addWidget(lbl_opts_head)

        opts_frame = QFrame()
        opts_frame.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 8px;
            }
            QRadioButton {
                font-size: 13px;
                color: #1e293b;
                padding: 6px;
            }
            QRadioButton:hover {
                background-color: #f1f5f9;
                border-radius: 4px;
            }
        """)
        opts_layout = QVBoxLayout(opts_frame)
        opts_layout.setSpacing(6)

        self.btn_group = QButtonGroup(self)

        is_corrupt = self.health.get("is_corrupt", False)

        # 選択肢1: TE除去
        self.rb_drop_te = QRadioButton(t("main.health.opt_drop_te"))
        self.rb_drop_te.setStyleSheet("font-weight: bold; color: #2563eb;")
        self.btn_group.addButton(self.rb_drop_te, 1)
        opts_layout.addWidget(self.rb_drop_te)

        # 選択肢2: TE減衰
        self.rb_scale_te = QRadioButton(t("main.health.opt_scale_te"))
        self.btn_group.addButton(self.rb_scale_te, 2)
        opts_layout.addWidget(self.rb_scale_te)

        # 選択肢3: そのまま保持
        self.rb_keep = QRadioButton(t("main.health.opt_keep"))
        self.btn_group.addButton(self.rb_keep, 3)
        opts_layout.addWidget(self.rb_keep)

        if is_corrupt:
            self.rb_drop_te.setEnabled(False)
            self.rb_scale_te.setEnabled(False)
            self.rb_keep.setChecked(True)
        else:
            self.rb_drop_te.setChecked(True)

        main_layout.addWidget(opts_frame)

        # 4. 「すべてに適用」チェックボックス
        if self.has_multiple:
            self.cb_apply_all = QCheckBox(t("main.health.apply_all"))
            self.cb_apply_all.setStyleSheet("font-size: 13px; font-weight: bold; color: #475569;")
            main_layout.addWidget(self.cb_apply_all)
        else:
            self.cb_apply_all = None

        # 5. アクションボタン
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.btn_skip = QPushButton(t("main.health.btn_skip"))
        self.btn_skip.setStyleSheet("""
            QPushButton {
                background-color: #f1f5f9;
                color: #475569;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #e2e8f0;
            }
        """)
        self.btn_skip.clicked.connect(self.on_skip)
        btn_layout.addWidget(self.btn_skip)

        self.btn_apply = QPushButton(t("main.health.btn_apply"))
        self.btn_apply.setStyleSheet("""
            QPushButton {
                background-color: #2563eb;
                color: #ffffff;
                border: none;
                border-radius: 6px;
                padding: 8px 20px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1d4ed8;
            }
        """)
        self.btn_apply.clicked.connect(self.on_apply)
        btn_layout.addWidget(self.btn_apply)

        main_layout.addLayout(btn_layout)

    def on_apply(self):
        checked_id = self.btn_group.checkedId()
        if checked_id == 1:
            self.selected_action = "drop_te"
        elif checked_id == 2:
            self.selected_action = "scale_te"
        else:
            self.selected_action = "keep"

        if self.cb_apply_all and self.cb_apply_all.isChecked():
            self.apply_to_all = True

        self.accept()

    def on_skip(self):
        self.selected_action = "skip"
        if self.cb_apply_all and self.cb_apply_all.isChecked():
            self.apply_to_all = True
        self.reject()

    def get_result(self) -> tuple[str, bool]:
        """(action: 'drop_te' | 'scale_te' | 'keep' | 'skip', apply_to_all: bool) を返却"""
        return self.selected_action, self.apply_to_all
