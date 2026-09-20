# タスク進捗管理: LoRA健全性診断・CLIP過学習検知および対策選択機能

## ステータス凡例
- [ ] 未着手 (Not Started)
- [/] 進行中 (In Progress)
- [x] 完了 (Completed)

---

## タスク一覧

### 1. 調査・設計フェーズ
- [x] `Uncensored_NoobaXLEpslon_v01.safetensors` の色破綻原因調査（CLIP過学習・TE超高学習率2e-4・Epsilon専用ベースモデル不一致を特定）
- [x] 健全性診断ロジックおよび対策選択ダイアログの仕様設計
- [x] ユーザー承認（Proceed）の獲得

### 2. コアロジック実装フェーズ
- [x] `src/quantizer/validator.py`: `diagnose_lora_health`（NaN/Inf破損、TE過学習、特殊ベースモデル検知）の実装
- [x] `src/quantizer/pipeline_converter.py`: TE除去 (`drop_te`) / TE減衰 (`scale_te`) 変換処理の実装
- [x] `src/quantizer/estimator.py`: TE除去時の推定サイズ連動計算
- [x] `src/utils/naming.py`: TE対策サフィックス対応

### 3. UI・国際化実装フェーズ
- [x] `lang/*.json`（ja, en, zh）: 診断結果・対策選択ダイアログ用多言語リソースの追加
- [x] `src/gui/health_dialog.py`: 健全性・色破綻診断結果ダイアログ `LoRAHealthDialog` の新設
- [x] `src/gui/main_window.py`: D&D時の診断実行、ダイアログ表示、テーブル表示・ツールチップ連携
- [x] `src/gui/worker_thread.py`: ワーカーへの `te_action` 伝達・実行

### 4. テスト・検証フェーズ
- [x] `scratch/test_suite.py`: 診断ロジック、TE除去/減衰変換、サイズ推計、ダイアログ連携の自動テスト追加
- [x] テスト実行（`scratch/validate_cmd.ps1` -> `scratch/temp_run.ps1` で全15件PASS、回帰テスト6件PASS、合計21件全PASS）
- [x] 実ファイル `Uncensored_NoobaXLEpslon_v01.safetensors` での診断・判定検証
### 5. 追加フェーズ: 過学習検知拡充 & GUI常設TE除去オプション
- [x] `src/quantizer/validator.py`: 過学習（少数画像×過多リピート・過多ステップ）、少数画像×高Rank、TE突出重みの自動診断ルール追加
- [x] `lang/*.json` (ja, en, zh): GUI常設TE除去オプション用テキスト追加
- [x] `src/gui/main_window.py`: 設定パネルに「Text Encoder を除去 (UNet のみにクリーンアップ)」チェックボックスを追加、動的推定再計算連動
- [x] `src/gui/main_window.py` & `src/gui/health_dialog.py`: 学習画像枚数、繰り返し回数 (n_repeats)、学習Rank、ステップ数のGUI表示対応
### 6. 整合性同期フェーズ: 案B (Tristate/PartiallyChecked 連動)
- [x] `lang/*.json` (ja, en, zh): 一部適用時ツールチップ用テキスト追加
- [x] `src/gui/main_window.py`: `cb_drop_te` を3状態（Tristate）化し、ファイル個別状態（全適用・一部適用・なし）と自動同期
- [x] `src/gui/main_window.py`: ユーザー操作時の一括適用・一括解除ロジック実装
- [x] テスト・検証（単体テスト追加 & テストスイート全17件PASS）
- [ ] ドキュメント更新 & Git Commit/Push
