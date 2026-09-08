"""同名ファイル重複時の確認ダイアログ"""
from PyQt6.QtWidgets import QMessageBox, QCheckBox

class OverwriteDecision:
    OVERWRITE = "overwrite"
    SKIP = "skip"
    RENAME = "rename"
    CANCEL = "cancel"

def ask_overwrite_confirmation(parent, target_path: str, is_batch: bool = False) -> tuple[str, bool]:
    msg_box = QMessageBox(parent)
    msg_box.setWindowTitle("ファイル重複の確認")
    msg_box.setIcon(QMessageBox.Icon.Question)
    msg_box.setText(f"出力先に同名のファイルが既に存在します:\n\n{target_path}\n\nどのように処理しますか？")

    btn_overwrite = msg_box.addButton("上書き (&O)", QMessageBox.ButtonRole.AcceptRole)
    btn_rename = msg_box.addButton("別名で保存 (&R)", QMessageBox.ButtonRole.ActionRole)
    btn_skip = msg_box.addButton("スキップ (&S)", QMessageBox.ButtonRole.RejectRole)
    btn_cancel = msg_box.addButton("中止 (&C)", QMessageBox.ButtonRole.DestructiveRole)

    cb = None
    if is_batch:
        cb = QCheckBox("以降のファイルも同じ選択を適用する")
        msg_box.setCheckBox(cb)

    msg_box.exec()
    clicked = msg_box.clickedButton()
    apply_all = cb.isChecked() if cb else False

    if clicked == btn_overwrite:
        return OverwriteDecision.OVERWRITE, apply_all
    elif clicked == btn_rename:
        return OverwriteDecision.RENAME, apply_all
    elif clicked == btn_skip:
        return OverwriteDecision.SKIP, apply_all
    else:
        return OverwriteDecision.CANCEL, False
