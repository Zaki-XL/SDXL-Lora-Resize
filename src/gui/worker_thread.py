"""バックグラウンド処理を行うQThreadワーカー (付属ファイル連携・整合性検証対応)"""
import os
from PyQt6.QtCore import QThread, pyqtSignal
from ..utils.naming import generate_output_path
from ..utils.sidecar_files import process_and_copy_sidecar_files

class QuantizeWorker(QThread):
    file_progress = pyqtSignal(int, int)
    tensor_progress = pyqtSignal(int, int, str, float)
    log_message = pyqtSignal(str)
    item_status_changed = pyqtSignal(int, str, str)
    finished_all = pyqtSignal(int, int, int)
    need_error_decision = pyqtSignal(str, str, int)

    def __init__(self, file_list: list[str], svd_rank: str, precision_key: str, output_dir: str, keep_vae_fp16: bool, force_overwrite: bool, te_actions: dict = None):
        super().__init__()
        self.file_list = file_list
        self.svd_rank = svd_rank
        self.precision_key = precision_key
        self.output_dir = output_dir
        self.keep_vae_fp16 = keep_vae_fp16
        self.force_overwrite = force_overwrite
        self.te_actions = te_actions or {}
        self.is_cancelled = False
        self.current_converter = None
        self.error_decision = None

    def cancel(self):
        self.is_cancelled = True
        if self.current_converter:
            self.current_converter.cancel()

    def set_error_decision(self, decision: str):
        self.error_decision = decision

    def run(self):
        from ..quantizer.pipeline_converter import CombinedPipelineConverter
        from ..quantizer.validator import verify_quantized_model

        total_files = len(self.file_list)
        success_count = 0
        skip_count = 0
        fail_count = 0

        for idx, input_path in enumerate(self.file_list):
            if self.is_cancelled:
                self.log_message.emit("[中断] バッチ処理が中止されました。")
                break

            self.file_progress.emit(idx + 1, total_files)
            te_action = self.te_actions.get(input_path, "keep")
            target_path = generate_output_path(input_path, self.svd_rank, self.precision_key, self.output_dir, te_action=te_action)

            if os.path.exists(target_path):
                if not self.force_overwrite:
                    self.log_message.emit(f"[パス/スキップ] 既に存在するためパス: {os.path.basename(target_path)}")
                    self.item_status_changed.emit(idx, "パス (既存)", target_path)
                    skip_count += 1
                    continue
                else:
                    self.log_message.emit(f"[強制実行] 既存ファイルを上書きします: {os.path.basename(target_path)}")

            self.item_status_changed.emit(idx, "変換中...", target_path)

            converter = CombinedPipelineConverter(
                svd_rank=self.svd_rank,
                precision_key=self.precision_key,
                keep_vae_fp16=self.keep_vae_fp16,
                te_action=te_action
            )
            self.current_converter = converter

            try:
                def on_progress(cur, total, name, speed):
                    self.tensor_progress.emit(cur, total, name, speed)

                def on_log(msg):
                    self.log_message.emit(msg)

                success = converter.process(
                    input_path=input_path,
                    output_path=target_path,
                    progress_callback=on_progress,
                    log_callback=on_log
                )

                if not success:
                    self.item_status_changed.emit(idx, "中断", "")
                    continue

                # 整合性検証
                self.log_message.emit(f"[検証中] 変換後ファイルの整合性をチェックしています: {os.path.basename(target_path)} ...")
                if not target_path.lower().endswith(".gguf"):
                    orig_meta = getattr(converter, 'last_orig_meta', None) or input_path
                    is_valid, issues = verify_quantized_model(
                        orig_path_or_meta=orig_meta,
                        quant_path=target_path,
                        format_key=self.precision_key,
                        keep_vae_fp16=self.keep_vae_fp16,
                        svd_rank=self.svd_rank
                    )
                else:
                    is_valid, issues = True, []

                if is_valid:
                    self.log_message.emit("✓ 整合性検証合格: 全キー・Rank次元・NaN/Inf異常なし")

                    # 付属ファイル (画像 & metadata.json) のコピー & 変換実行
                    process_and_copy_sidecar_files(
                        orig_model_path=input_path,
                        output_model_path=target_path,
                        svd_rank=self.svd_rank,
                        precision_key=self.precision_key,
                        log_callback=on_log
                    )

                    success_count += 1
                    self.item_status_changed.emit(idx, "完了 (検証済)", target_path)
                else:
                    err_details = "\n".join(issues)
                    self.log_message.emit(f"[検証エラー] {err_details}")
                    
                    if os.path.exists(target_path):
                        try:
                            os.remove(target_path)
                            self.log_message.emit(f"[クリーンアップ] 破損ファイルを削除しました: {os.path.basename(target_path)}")
                        except Exception as rem_err:
                            self.log_message.emit(f"[警告] ファイル削除失敗: {rem_err}")

                    fail_count += 1
                    self.item_status_changed.emit(idx, "エラー (削除済)", "")

                    self.error_decision = None
                    self.need_error_decision.emit(os.path.basename(input_path), err_details, idx)

                    while self.error_decision is None:
                        if self.is_cancelled:
                            break
                        self.msleep(50)

                    if self.error_decision == "stop":
                        self.log_message.emit("[停止] ユーザーの指示により処理を停止しました。")
                        break
                    else:
                        self.log_message.emit("[継続] 次のファイルの変換へ進みます。")

            except Exception as e:
                self.log_message.emit(f"[エラー] 変換例外: {os.path.basename(input_path)} - {str(e)}")
                if os.path.exists(target_path):
                    try:
                        os.remove(target_path)
                    except:
                        pass
                fail_count += 1
                self.item_status_changed.emit(idx, f"エラー: {str(e)[:12]}", "")

        self.finished_all.emit(success_count, skip_count, fail_count)
