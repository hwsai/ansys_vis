# -*- coding: utf-8 -*-
"""
Step 2：極簡 GUI（Tkinter）
- 介面只有三件事：選檔、選結果類型、按「開始」
- 結果類型：Displacement / Von Mises stress / Predicted stress
- 不加其他功能（無色階鎖定、無 MSE、無邊界）

用法：
    python viewer_step2_gui_min.py
"""
import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import pyvista as pv
from vis_lab import Vis_tools

APP_TITLE = "最小可視化：選檔 + 結果 + 開始"
DEFAULT_DIR = "data"  # 檔案對話框預設目錄

RESULT_CHOICES = [
    ("Displacement", "dis"),
    ("Von Mises stress", "von"),
    ("Predicted stress", "pred"),
]


def visualize(file_path: str, result_key: str):
    """依選項執行最小可視化並顯示一個 PyVista 視窗。"""
    try:
        vis = Vis_tools(file_path)
    except Exception as e:
        messagebox.showerror("載入失敗", f"無法載入檔案：\n{e}")
        return

    # 依結果種類呼叫對應方法，保持最小邏輯
    if result_key == "dis":
        vis.dis_solution()
        title = "Displacement"
    elif result_key == "von":
        vis.stress_solution()
        title = "Von Mises stress"
    else:  # "pred"
        try:
            vis.stress_pr_solution()
            title = "Predicted stress"
        except KeyError as e:
            messagebox.showwarning("欄位缺失", f"找不到預測欄位：{e}\n改以 Von Mises 顯示。")
            vis.stress_solution()
            title = "Von Mises stress"

    grid = vis.subset
    p = pv.Plotter()
    p.add_text(title, font_size=12)
    p.add_mesh(grid, scalars="solution", cmap="jet")
    p.add_scalar_bar(title=title)
    p.show()


def build_ui(root):
    root.title(APP_TITLE)
    root.minsize(520, 160)

    # 置中
    root.update_idletasks()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    w, h = 560, 180
    x, y = (sw - w) // 2, (sh - h) // 3
    root.geometry(f"{w}x{h}+{x}+{y}")

    frm = ttk.Frame(root, padding=16)
    frm.pack(fill=tk.BOTH, expand=True)

    # --- 選檔 ---
    ttk.Label(frm, text="選擇檔案 (.rst/.vtu/.vtk)：").grid(row=0, column=0, sticky="w")
    file_var = tk.StringVar(value="")
    ent = ttk.Entry(frm, textvariable=file_var)
    ent.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(4, 10))
    frm.columnconfigure(0, weight=1)

    def browse():
        initdir = DEFAULT_DIR if os.path.isdir(DEFAULT_DIR) else os.getcwd()
        path = filedialog.askopenfilename(
            title="選擇結果檔",
            initialdir=initdir,
            filetypes=[
                ("ANSYS/VTK files", "*.rst *.vtu *.vtk"),
                ("All files", "*.*"),
            ],
        )
        if path:
            file_var.set(path)

    ttk.Button(frm, text="瀏覽…", command=browse).grid(row=1, column=3, sticky="w", padx=(8, 0))

    # --- 結果下拉 ---
    ttk.Label(frm, text="結果類型：").grid(row=2, column=0, sticky="w")
    choice_names = [name for name, _ in RESULT_CHOICES]
    combo = ttk.Combobox(frm, values=choice_names, state="readonly")
    combo.set(choice_names[1])  # 預設 Von Mises
    combo.grid(row=3, column=0, sticky="w", pady=(4, 10))

    # --- 開始 ---
    def on_start():
        path = file_var.get().strip()
        if not path:
            messagebox.showwarning("提醒", "請先選擇檔案。")
            return
        if not os.path.exists(path):
            messagebox.showerror("錯誤", f"找不到檔案：\n{path}")
            return
        sel_name = combo.get()
        # 找到對應 key
        key = next((k for name, k in RESULT_CHOICES if name == sel_name), "von")
        # 關閉 UI 再開渲染視窗（避免 Tk 被阻塞）
        root.destroy()
        visualize(path, key)

    ttk.Button(frm, text="開始", command=on_start).grid(row=3, column=3, sticky="e")


if __name__ == "__main__":
    # 在某些 IDE（如 Spyder）下，Tk/PyVista 視窗交互可能受 IDE 設定影響
    try:
        root = tk.Tk()
    except Exception as e:
        print("無法建立 Tk 視窗：", e)
        sys.exit(1)
    build_ui(root)
    root.mainloop()
