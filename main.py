# -*- coding: utf-8 -*-
"""
Step 6：左右可選下拉 + 雙圖開關 + 同色軸範圍（含可選邊界）
- 介面：
  • 檔案選擇
  • 左側結果下拉（Displacement / Von Mises / Predicted）
  • [✓] 雙圖對照 → 右側結果下拉（啟用/停用）
  • [✓] 同色軸範圍（當雙圖時，根據左右兩圖的值域統一）
  • [✓] 顯示邊界（若有 node_features）
  • 開始
- 注意：Displacement 在你的 Vis_tools 目前回傳的是向量場（3 分量）。
  為了能著色，我這裡會自動取其「向量幅值」做為標量來顯示（不會更動原資料）。

用法：
    python viewer_step6_gui_dual_custom.py
"""
import os
import sys
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import pyvista as pv
from vis_lab import Vis_tools

APP_TITLE = "可視化：左右可選 + 雙圖 + 同色軸 + 邊界"
DEFAULT_DIR = "data"

RESULT_CHOICES = [
    ("Displacement", "dis"),
    ("Von Mises stress", "von"),
    ("Predicted stress", "pred"),
]

# -------------------- 資料/視覺工具 --------------------

def _make_solution_grid(vis: Vis_tools, base_grid: pv.PolyData, kind: str):
    """回傳 (grid_with_scalar_solution, scalar_1d_array, title)
    - 會複製 base_grid，避免原地覆寫。
    - Displacement 會轉成幅值標量供著色。
    - 若 Pred 欄位不存在而選 pred，回傳 (None, None, None)。
    """
    g = base_grid.copy()
    try:
        vis.subset = g
        if kind == 'von':
            vis.stress_solution(); title = 'Von Mises stress'
        elif kind == 'pred':
            vis.stress_pr_solution(); title = 'Predicted stress'
        else:
            vis.dis_solution(); title = 'Displacement (magnitude)'
        arr = g.point_data['solution']
        # 若是向量場，轉成幅值
        if arr.ndim == 2 and arr.shape[1] > 1:
            mag = np.linalg.norm(arr, axis=1)
            g.point_data['solution_mag'] = mag
            scal_name = 'solution_mag'
            scal = mag
        else:
            scal_name = 'solution'
            scal = arr
        return g, scal, scal_name, title
    except KeyError:
        return None, None, None, None
    except Exception as e:
        print('生成 solution 失敗：', e)
        return None, None, None, None


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


def visualize(file_path: str, left_key: str, enable_dual: bool, right_key: str, lock_clim: bool, show_bc: bool, n_colors: int, show_edges: bool):
    try:
        vis = Vis_tools(file_path)
    except Exception as e:
        messagebox.showerror("載入失敗", f"無法載入檔案：\n{e}")
        return

    base = vis.subset

    # 準備左圖
    gL, sL, snameL, titleL = _make_solution_grid(vis, base, left_key)
    if gL is None:
        messagebox.showerror('錯誤', '無法產生左側視圖。')
        return

    # 決定欄數
    cols = 1
    gR = sR = snameR = titleR = None
    if enable_dual:
        gR, sR, snameR, titleR = _make_solution_grid(vis, base, right_key)
        if gR is None or sR is None:
            messagebox.showwarning('提示', '右側結果無法生成（欄位缺失或錯誤），改為單圖。')
        else:
            cols += 1
    if show_bc:
        cols += 1

    # 同色軸（僅在雙圖且兩側皆為標量時生效）
    clim = None
    if enable_dual and (sL is not None) and (sR is not None) and lock_clim:
        try:
            vmin = float(np.nanmin([np.nanmin(sL), np.nanmin(sR)]))
            vmax = float(np.nanmax([np.nanmax(sL), np.nanmax(sR)]))
            if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
                clim = (vmin, vmax)
        except Exception:
            clim = None

    # 建立 Plotter
    p = pv.Plotter(shape=(1, cols))

    # 左
    cur = 0
    p.subplot(0, cur)
    p.add_text(titleL, font_size=12, viewport=True, position=(0.02, 0.96))
    p.add_mesh(gL, scalars=snameL, cmap='jet', clim=clim, n_colors=n_colors, show_edges=show_edges)
    p.add_scalar_bar(title=titleL); p.show_axes()

    # 右（若有）
    if enable_dual and (gR is not None) and (sR is not None):
        cur += 1
        p.subplot(0, cur)
        p.add_text(titleR, font_size=12, viewport=True, position=(0.02, 0.96))
        p.add_mesh(gR, scalars=snameR, cmap='jet', clim=clim, n_colors=n_colors, show_edges=show_edges)
        p.add_scalar_bar(title=titleR); p.show_axes()

    # 邊界（若有）
    if show_bc:
        cur += 1
        ok = _add_boundary_subplot(p, base, cur)
        if not ok:
            p.subplot(0, cur)
            p.add_text('Boundary (資料不足)', font_size=12, viewport=True, position=(0.02, 0.96))

    p.link_views(); p.enable_parallel_projection(); p.show()


# -------------------- GUI --------------------

def build_ui(root):
    root.title(APP_TITLE)
    root.minsize(760, 260)
    root.update_idletasks()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    w, h = 800, 280
    x, y = (sw - w) // 2, (sh - h) // 3
    root.geometry(f"{w}x{h}+{x}+{y}")

    frm = ttk.Frame(root, padding=16)
    frm.pack(fill=tk.BOTH, expand=True)
    for c in range(4):
        frm.columnconfigure(c, weight=1 if c == 1 else 0)

    # 檔案
    ttk.Label(frm, text='選擇檔案 (.rst/.vtu/.vtk)：').grid(row=0, column=0, sticky='w')
    file_var = tk.StringVar(value='')
    ent = ttk.Entry(frm, textvariable=file_var)
    ent.grid(row=1, column=0, columnspan=3, sticky='ew', pady=(4, 10))

    def browse():
        initdir = DEFAULT_DIR if os.path.isdir(DEFAULT_DIR) else os.getcwd()
        path = filedialog.askopenfilename(
            title='選擇結果檔', initialdir=initdir,
            filetypes=[('ANSYS/VTK files', '*.rst *.vtu *.vtk'), ('All files', '*.*')],
        )
        if path:
            file_var.set(path)

    ttk.Button(frm, text='瀏覽…', command=browse).grid(row=1, column=3, sticky='w', padx=(8, 0))

    # 左側下拉
    ttk.Label(frm, text='左側結果：').grid(row=2, column=0, sticky='w')
    left_var = tk.StringVar(value=RESULT_CHOICES[1][0])  # 預設 Von Mises
    combo_left = ttk.Combobox(frm, values=[n for n, _ in RESULT_CHOICES], state='readonly', textvariable=left_var)
    combo_left.grid(row=3, column=0, sticky='w', pady=(4, 10))

    # 雙圖 + 右側下拉
    dual_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(frm, text='雙圖對照', variable=dual_var, command=lambda: combo_right.configure(state='readonly' if dual_var.get() else 'disabled')).grid(row=2, column=1, sticky='w')
    ttk.Label(frm, text='右側結果：').grid(row=2, column=2, sticky='w')
    right_var = tk.StringVar(value=RESULT_CHOICES[2][0])  # 預設 Predicted
    combo_right = ttk.Combobox(frm, values=[n for n, _ in RESULT_CHOICES], state='readonly', textvariable=right_var)
    combo_right.grid(row=3, column=2, sticky='w', pady=(4, 10))

    # 同色軸 / 邊界
    lock_clim_var = tk.BooleanVar(value=True)
    show_bc_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(frm, text='同一個色軸範圍（雙圖時）', variable=lock_clim_var).grid(row=3, column=1, sticky='w')
    ttk.Checkbutton(frm, text='顯示邊界條件（若有 node_features）', variable=show_bc_var).grid(row=3, column=3, sticky='w')

    # 色階數（n_colors）
    ncolors_var = tk.IntVar(value=10)
    ttk.Label(frm, text='色階數：').grid(row=4, column=0, sticky='w')
    tk.Spinbox(frm, from_=3, to=256, textvariable=ncolors_var, width=6).grid(row=4, column=1, sticky='w')

    # 顯示網格（Edges）
    edges_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(frm, text='顯示網格（Edges）', variable=edges_var).grid(row=4, column=2, sticky='w')

    # 初始根據勾選狀態設定右側下拉可用性
    combo_right.configure(state='readonly' if dual_var.get() else 'disabled')

    # 開始
    def on_start():
        path = file_var.get().strip()
        if not path:
            messagebox.showwarning('提醒', '請先選擇檔案。'); return
        if not os.path.exists(path):
            messagebox.showerror('錯誤', f'找不到檔案：\n{path}'); return

        # 轉成 key
        def name2key(name: str) -> str:
            for n, k in RESULT_CHOICES:
                if n == name: return k
            return 'von'
        left_key = name2key(left_var.get())
        right_key = name2key(right_var.get())

        root.destroy()
        visualize(
            path,
            left_key=left_key,
            enable_dual=dual_var.get(),
            right_key=right_key,
            lock_clim=lock_clim_var.get(),
            show_bc=show_bc_var.get(),
            n_colors=int(ncolors_var.get()),
            show_edges=edges_var.get(),
        )

    ttk.Button(frm, text='開始', command=on_start).grid(row=6, column=3, sticky='e')


if __name__ == '__main__':
    try:
        import tkinter as tk
        from tkinter import ttk, filedialog, messagebox
    except Exception as e:
        print('無法載入 Tkinter：', e)
        sys.exit(1)
    root = tk.Tk()
    build_ui(root)
    root.mainloop()
