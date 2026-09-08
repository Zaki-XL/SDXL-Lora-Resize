using System;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;
using System.Windows.Forms;

namespace SDXLQuantizerLauncher
{
    static class Program
    {
        private const string MutexName = "Global\\SDXL_Quantizer_SingleInstance_Mutex_98a7b";

        [DllImport("user32.dll")]
        private static extern bool SetForegroundWindow(IntPtr hWnd);

        [DllImport("user32.dll")]
        private static extern bool ShowWindowAsync(IntPtr hWnd, int nCmdShow);

        [DllImport("user32.dll", SetLastError = true)]
        private static extern IntPtr FindWindow(string lpClassName, string lpWindowName);

        private const int SW_RESTORE = 9;

        [STAThread]
        static void Main()
        {
            bool createdNew;
            using (Mutex mutex = new Mutex(true, MutexName, out createdNew))
            {
                if (!createdNew)
                {
                    // 既に起動中
                    IntPtr hWnd = FindWindow(null, "SDXL Model Quantizer & Resizer");
                    if (hWnd != IntPtr.Zero)
                    {
                        ShowWindowAsync(hWnd, SW_RESTORE);
                        SetForegroundWindow(hWnd);
                    }
                    else
                    {
                        MessageBox.Show(
                            "SDXL Model Quantizer は既に起動しています。",
                            "二重起動の防止",
                            MessageBoxButtons.OK,
                            MessageBoxIcon.Information);
                    }
                    return;
                }

                string baseDir = AppDomain.CurrentDomain.BaseDirectory;
                string venvPythonW = Path.Combine(baseDir, ".venv", "Scripts", "pythonw.exe");
                string mainScript = Path.Combine(baseDir, "src", "main.py");
                string setupBat = Path.Combine(baseDir, "setup_env.bat");

                if (!File.Exists(venvPythonW))
                {
                    DialogResult res = MessageBox.Show(
                        "仮想環境 (.venv) が見つかりませんでした。\n\n初期セットアップ (setup_env.bat) を実行して環境を構築しますか？",
                        "SDXL Quantizer - 初期セットアップ",
                        MessageBoxButtons.YesNo,
                        MessageBoxIcon.Question);

                    if (res == DialogResult.Yes)
                    {
                        if (File.Exists(setupBat))
                        {
                            ProcessStartInfo setupPsi = new ProcessStartInfo
                            {
                                FileName = "cmd.exe",
                                Arguments = "/c \"" + setupBat + "\"",
                                WorkingDirectory = baseDir,
                                UseShellExecute = true
                            };
                            Process.Start(setupPsi);
                        }
                    }
                    return;
                }

                if (!File.Exists(mainScript))
                {
                    MessageBox.Show("メインスクリプトが見つかりません: " + mainScript, "エラー", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    return;
                }

                try
                {
                    ProcessStartInfo psi = new ProcessStartInfo
                    {
                        FileName = venvPythonW,
                        Arguments = "\"" + mainScript + "\"",
                        WorkingDirectory = baseDir,
                        UseShellExecute = false,
                        CreateNoWindow = true
                    };
                    Process proc = Process.Start(psi);
                    // ランチャーがMutexを解放する前に少し待機するか、Python側でもガード
                }
                catch (Exception ex)
                {
                    MessageBox.Show("アプリケーションの起動に失敗しました:\n" + ex.Message, "起動エラー", MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
            }
        }
    }
}
