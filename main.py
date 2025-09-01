# -*- coding: utf-8 -*-
"""
Step 5：雙圖對照（Von vs Pred）+ 可選邊界
- 介面：選檔、[✓] 雙圖對照（Von 與 Pred 並排）、[✓] 顯示邊界、開始
- 行為：
  * 若勾選「雙圖對照」，主視區固定顯示左=Von、右=Pred；色階自動以兩者聯集統一。
  * 若資料缺少 Pred 欄位，會提示並退回單圖（Von）。
  * 勾選「顯示邊界」時，會在最右邊再加一格顯示 Boundary。
- 仍維持極簡：無 MSE、無快捷鍵、無其他選項。

用法：
    python viewer_step5_gui_dual.py
"""
import os
import sys
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import pyvista as pv
from vis_lab import Vis_tools

APP_TITLE = "最小可視化：雙圖對照（Von vs Pred）+ 邊界"
DEFAULT_DIR = "data"


def _extract_solution_array(vis: Vis_tools, base_grid: pv.PolyData, kind: str):
    """回傳 (grid_with_solution, ndarray, title)；失敗時 (None, None, None)。"""
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
    except Exception:
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

    # 點大小依模型尺度（對角線 1%）
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


def visualize(file_path: str, dual: bool, show_bc: bool):
    try:
        vis = Vis_tools(file_path)
    except Exception as e:
        messagebox.showerror("載入失敗", f"無法載入檔案：\n{e}")
        return

    base = vis.subset

    if dual:
        # 取 Von 與 Pred
        g_von, a_von, t_von = _extract_solution_array(vis, base, 'von')
        g_pred, a_pred, t_pred = _extract_solution_array(vis, base, 'pred')
        if g_von is None:
            messagebox.showerror('錯誤', '無法產生 Von Mises 視圖。')
            return
        if g_pred is None or a_pred is None or a_pred.size == 0:
            messagebox.showwarning('提示', '找不到 Pred 欄位，改為單圖（Von）。')
            # 單圖路徑
            cols = 2 if show_bc else 1
            p = pv.Plotter(shape=(1, cols))
            p.subplot(0, 0)
            p.add_text(t_von, font_size=12, viewport=True, position=(0.02, 0.96))
            p.add_mesh(g_von, scalars='solution', cmap='jet')
            p.add_scalar_bar(title=t_von)
            p.show_axes()
            if show_bc:
                ok = _add_boundary_subplot(p, base, 1)
                if not ok:
                    p.subplot(0, 1)
                    p.add_text('Boundary (資料不足)', font_size=12, viewport=True, position=(0.02, 0.96))
            p.link_views(); p.enable_parallel_projection(); p.show()
            return
        # 計算共用色階
        vmin = float(np.nanmin([np.nanmin(a_von), np.nanmin(a_pred)]))
        vmax = float(np.nanmax([np.nanmax(a_von), np.nanmax(a_pred)]))
        clim = (vmin, vmax) if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin else None

        cols = 3 if show_bc else 2
        p = pv.Plotter(shape=(1, cols))
        # 左：Von
        p.subplot(0, 0)
        p.add_text(t_von, font_size=12, viewport=True, position=(0.02, 0.96))
        p.add_mesh(g_von, scalars='solution', cmap='jet', clim=clim)
        p.add_scalar_bar(title=t_von)
        p.show_axes()
        # 右：Pred
        p.subplot(0, 1)
        p.add_text(t_pred, font_size=12, viewport=True, position=(0.02, 0.96))
        p.add_mesh(g_pred, scalars='solution', cmap='jet', clim=clim)
        p.add_scalar_bar(title=t_pred)
        p.show_axes()
        # 邊界
        if show_bc:
            ok = _add_boundary_subplot(p, base, 2)
            if not ok:
                p.subplot(0, 2)
                p.add_text('Boundary (資料不足)', font_size=12, viewport=True, position=(0.02, 0.96))
        p.link_views(); p.enable_parallel_projection(); p.show()
        return

    # 單圖（預設畫 Von）
    g_von, _, t_von = _extract_solution_array(vis, base, 'von')
    if g_von is None:
        messagebox.showerror('錯誤', '無法產生 Von Mises 視圖。')
        return

    cols = 2 if show_bc else 1
    p = pv.Plotter(shape=(1, cols))
    p.subplot(0, 0)
    p.add_text(t_von, font_size=12, viewport=True, position=(0.02, 0.96))
    p.add_mesh(g_von, scalars='solution', cmap='jet')
    p.add_scalar_bar(title=t_von)
    p.show_axes()
    if show_bc:
        ok = _add_boundary_subplot(p, base, 1)
        if not ok:
            p.subplot(0, 1)
            p.add_text('Boundary (資料不足)', font_size=12, viewport=True, position=(0.02, 0.96))
    p.link_views(); p.enable_parallel_projection(); p.show()


# ------------------- GUI -------------------

def build_ui(root):
    root.title(APP_TITLE)
    root.minsize(600, 220)

    root.update_idletasks()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    w, h = 640, 240
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

    # --- 勾選項 ---
    dual_var = tk.BooleanVar(value=True)
    show_bc_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(frm, text="雙圖對照（左：Von，右：Pred）", variable=dual_var).grid(row=2, column=0, sticky="w")
    ttk.Checkbutton(frm, text="顯示邊界條件（若有 node_features）", variable=show_bc_var).grid(row=2, column=1, sticky="w")

    # --- 開始 ---
    def on_start():
        path = file_var.get().strip()
        if not path:
            messagebox.showwarning("提醒", "請先選擇檔案。")
            return
        if not os.path.exists(path):
            messagebox.showerror("錯誤", f"找不到檔案：\n{path}")
            return
        root.destroy()
        visualize(path, dual_var.get(), show_bc_var.get())

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
