# -*- coding: utf-8 -*-
"""
Step 3：在極簡 GUI 版加入「勾選顯示邊界」
- 介面：選檔、結果下拉、[✓] 顯示邊界、開始
- 若勾選且資料含有 node_features（且欄位滿足需求），右側會多一格顯示邊界點（Load/Fixed）
- 仍維持最小化：不加 MSE、色階鎖定、快捷鍵等

用法：
    python viewer_step3_gui_bc.py
"""
import os
import sys
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import pyvista as pv
from vis_lab import Vis_tools

APP_TITLE = "最小可視化：選檔 + 結果 + 邊界"
DEFAULT_DIR = "data"

RESULT_CHOICES = [
    ("Displacement", "dis"),
    ("Von Mises stress", "von"),
    ("Predicted stress", "pred"),
]


def _add_boundary_subplot(plotter: pv.Plotter, grid: pv.PolyData, col: int) -> bool:
    """若 grid 具備 node_features（至少 9 欄），在第 col 欄畫出 Load/Fixed 點。
    回傳是否成功畫出。"""
    if 'node_features' not in grid.point_data:
        return False
    feats = grid.point_data['node_features']
    if feats.ndim != 2 or feats.shape[1] < 9:
        return False

    coords = grid.points
    loads = feats[:, 3:6]
    dofs  = feats[:, 6:9]

    tol = 1e-10
    is_load  = np.any(~np.isclose(loads, 0.0, atol=tol), axis=1)
    is_fixed = np.all(np.isclose(dofs,  0.0, atol=tol), axis=1)

    load_pts  = coords[is_load]
    fixed_pts = coords[is_fixed]
    if load_pts.size == 0 and fixed_pts.size == 0:
        return False

    # 點大小依模型尺度調整（對角線 1%）
    b = grid.bounds  # (xmin,xmax,ymin,ymax,zmin,zmax)
    diag = float(np.linalg.norm([b[1]-b[0], b[3]-b[2], b[5]-b[4]]))
    psize = max(diag * 0.01, 2.0)

    plotter.subplot(0, col)
    plotter.add_text('Boundary', font_size=12, viewport=True, position=(0.02, 0.96))
    plotter.add_mesh(grid, color='lightgray', opacity=1.0)
    if load_pts.size:
        plotter.add_mesh(pv.PolyData(load_pts), color='red', point_size=psize, render_points_as_spheres=True, label='Load')
    if fixed_pts.size:
        plotter.add_mesh(pv.PolyData(fixed_pts), color='blue', point_size=psize, render_points_as_spheres=True, label='Fixed')
    plotter.add_legend(); plotter.show_axes()
    return True


def visualize(file_path: str, result_key: str, show_bc: bool):
    try:
        vis = Vis_tools(file_path)
    except Exception as e:
        messagebox.showerror("載入失敗", f"無法載入檔案：\n{e}")
        return

    # 選擇要顯示的結果
    if result_key == "dis":
        vis.dis_solution(); title = "Displacement"
    elif result_key == "von":
        vis.stress_solution(); title = "Von Mises stress"
    else:
        try:
            vis.stress_pr_solution(); title = "Predicted stress"
        except KeyError as e:
            messagebox.showwarning("欄位缺失", f"找不到預測欄位：{e}\n改以 Von Mises 顯示。")
            vis.stress_solution(); title = "Von Mises stress"

    grid = vis.subset

    # 需要畫邊界嗎？
    cols = 2 if show_bc else 1
    p = pv.Plotter(shape=(1, cols))

    # 主結果
    p.subplot(0, 0)
    p.add_text(title, font_size=12, viewport=True, position=(0.02, 0.96))
    p.add_mesh(grid, scalars="solution", cmap="jet")
    p.add_scalar_bar(title=title)
    p.show_axes()

    # 邊界（若勾選 + 成功）
    if show_bc:
        ok = _add_boundary_subplot(p, grid, 1)
        if not ok:
            # 若無法畫（無欄位或無點），仍顯示空白子圖並提醒
            p.subplot(0, 1)
            p.add_text('Boundary (資料不足)', font_size=12, viewport=True, position=(0.02, 0.96))

    p.link_views()
    p.enable_parallel_projection()
    p.show()


def build_ui(root):
    root.title(APP_TITLE)
    root.minsize(560, 200)

    root.update_idletasks()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    w, h = 600, 220
    x, y = (sw - w) // 2, (sh - h) // 3
    root.geometry(f"{w}x{h}+{x}+{y}")

    frm = ttk.Frame(root, padding=16)
    frm.pack(fill=tk.BOTH, expand=True)
    frm.columnconfigure(0, weight=1)

    # --- 選檔 ---
    ttk.Label(frm, text="選擇檔案 (.rst/.vtu/.vtk)：").grid(row=0, column=0, sticky="w")
    file_var = tk.StringVar(value="")
    ent = ttk.Entry(frm, textvariable=file_var)
    ent.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(4, 10))

    def browse():
        initdir = DEFAULT_DIR if os.path.isdir(DEFAULT_DIR) else os.getcwd()
        path = filedialog.askopenfilename(
            title="選擇結果檔",
            initialdir=initdir,
            filetypes=[("ANSYS/VTK files", "*.rst *.vtu *.vtk"), ("All files", "*.*")],
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

    # --- 勾選：顯示邊界 ---
    show_bc_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(frm, text="顯示邊界條件（若有 node_features）", variable=show_bc_var).grid(row=3, column=1, sticky="w")

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
        key = next((k for name, k in RESULT_CHOICES if name == sel_name), "von")
        root.destroy()
        visualize(path, key, show_bc_var.get())

    ttk.Button(frm, text="開始", command=on_start).grid(row=3, column=3, sticky="e")


if __name__ == "__main__":
    try:
        root = tk.Tk()
    except Exception as e:
        print("無法建立 Tk 視窗：", e)
        sys.exit(1)
    build_ui(root)
    root.mainloop()
