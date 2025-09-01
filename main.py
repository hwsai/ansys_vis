# -*- coding: utf-8 -*-
"""
Step 4：在極簡 GUI 版加入「鎖定 2/3 同色階」
- 介面：選檔、結果下拉、[✓] 顯示邊界、[✓] 鎖定 2/3 同色階、開始
- 說明：
  * 仍然一次只顯示一種結果（Displacement 或 Von/Pred）。
  * 當勾選「鎖定 2/3 同色階」且選擇的是 Von 或 Pred 時：
      會同時計算真實 Von 與 Pred 的極值範圍（若 Pred 欄位存在），
      並用同一個色階範圍繪圖，方便你切換比較時色軸一致。
  * 若資料沒有 Pred 欄位，會退回僅以所選結果的色階繪圖。

用法：
    python viewer_step4_gui_bc_clim.py
"""
import os
import sys
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import pyvista as pv
from vis_lab import Vis_tools

APP_TITLE = "最小可視化：選檔 + 結果 + 邊界 + 同色階"
DEFAULT_DIR = "data"

RESULT_CHOICES = [
    ("Displacement", "dis"),
    ("Von Mises stress", "von"),
    ("Predicted stress", "pred"),
]


def _extract_solution_array(vis: Vis_tools, base_grid: pv.PolyData, kind: str):
    """回傳指定 kind 的 solution ndarray；kind in {'von','pred','dis'}。
    會複製 base_grid，避免原地覆寫衝突。失敗時回傳 (None, None)。
    """
    g = base_grid.copy()
    try:
        vis.subset = g
        if kind == 'von':
            vis.stress_solution(); title = 'Von Mises stress'
        elif kind == 'pred':
            vis.stress_pr_solution(); title = 'Predicted stress'
        else:
            vis.dis_solution(); title = 'Displacement'
        arr = g.point_data['solution']
        return g, arr, title
    except Exception as e:
        return None, None, None


def _add_boundary_subplot(plotter: pv.Plotter, grid: pv.PolyData, col: int) -> bool:
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

    b = grid.bounds
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


def visualize(file_path: str, result_key: str, show_bc: bool, lock_clim: bool):
    try:
        vis = Vis_tools(file_path)
    except Exception as e:
        messagebox.showerror("載入失敗", f"無法載入檔案：\n{e}")
        return

    base = vis.subset  # 原始子網格

    # 準備色階（僅在 von/pred 且勾選 lock_clim 時才試圖統一）
    clim = None
    if lock_clim and result_key in {"von", "pred"}:
        # 嘗試同時取 von 與 pred 的值域（pred 可能不存在）
        g_von, a_von, _ = _extract_solution_array(vis, base, 'von')
        g_pred, a_pred, _ = _extract_solution_array(vis, base, 'pred')
        vals = []
        if a_von is not None and np.size(a_von) > 0:
            vals.append((np.nanmin(a_von), np.nanmax(a_von)))
        if a_pred is not None and np.size(a_pred) > 0:
            vals.append((np.nanmin(a_pred), np.nanmax(a_pred)))
        if vals:
            vmin = float(np.nanmin([v[0] for v in vals]))
            vmax = float(np.nanmax([v[1] for v in vals]))
            if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
                clim = (vmin, vmax)
            else:
                clim = None  # 回退

    # 產出實際要顯示的結果
    g_show, _, title = _extract_solution_array(vis, base, result_key)
    if g_show is None:
        # 最後退回 Von Mises
        vis.subset = base.copy()
        try:
            vis.stress_solution(); g_show = vis.subset; title = 'Von Mises stress'
        except Exception as e:
            messagebox.showerror('錯誤', f'無法產生可視化：\n{e}')
            return

    cols = 2 if show_bc else 1
    p = pv.Plotter(shape=(1, cols))

    p.subplot(0, 0)
    p.add_text(title, font_size=12, viewport=True, position=(0.02, 0.96))
    p.add_mesh(g_show, scalars='solution', cmap='jet', clim=clim)
    p.add_scalar_bar(title=title)
    p.show_axes()

    if show_bc:
        ok = _add_boundary_subplot(p, base, 1)
        if not ok:
            p.subplot(0, 1)
            p.add_text('Boundary (資料不足)', font_size=12, viewport=True, position=(0.02, 0.96))

    p.link_views()
    p.enable_parallel_projection()
    p.show()


# ------------------- GUI -------------------

def build_ui(root):
    root.title(APP_TITLE)
    root.minsize(620, 240)

    root.update_idletasks()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    w, h = 660, 260
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

    # --- 勾選：顯示邊界 / 鎖定色階 ---
    show_bc_var = tk.BooleanVar(value=True)
    lock_clim_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(frm, text="顯示邊界條件（若有 node_features）", variable=show_bc_var).grid(row=3, column=1, sticky="w")
    ttk.Checkbutton(frm, text="鎖定 2/3 同色階（若存在 Pred）", variable=lock_clim_var).grid(row=3, column=2, sticky="w")

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
        visualize(path, key, show_bc_var.get(), lock_clim_var.get())

    ttk.Button(frm, text="開始", command=on_start).grid(row=3, column=3, sticky="e")


if __name__ == "__main__":
    try:
        import tkinter as tk
        from tkinter import ttk, filedialog, messagebox
    except Exception as e:
        print("無法載入 Tkinter：", e)
        sys.exit(1)
    root = tk.Tk()
    build_ui(root)
    root.mainloop()
