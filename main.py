# -*- coding: utf-8 -*-
"""
可視化 GUI（模組化、可擴展架構）
- ResultRegistry：可插拔「結果種類」(dis/von/pred/...）
- MetricRegistry：可插拔「度量指標」（mse/mae/rmse/...）
- VisualBackend：可替換視覺化後端（目前 PyVista）
- UI：保持簡潔，只調用上述服務

用法：
    python viewer_modular.py
"""

import os
import sys
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# 視覺化後端 (當前為 PyVista)
import pyvista as pv
from vis_lab import Vis_tools


# -------------------------------
# Utilities
# -------------------------------

def _safe_norm_if_vector(arr: np.ndarray) -> np.ndarray:
    """若 arr 是 (N,3) 向量，轉為幅值；否則原樣返回。"""
    if arr is None:
        return None
    if arr.ndim == 2 and arr.shape[1] > 1:
        return np.linalg.norm(arr, axis=1)
    return arr


# -------------------------------
# Result Registry
# -------------------------------

class ResultRegistry:
    """
    管理「結果種類」的萃取與可視化設定：
    - 每個結果註冊兩個 handler：
      1) extract_scalar(vis, base_grid) -> (grid, array_1d, scalar_name, title)
         （用於畫圖：會回傳一個可直接 add_mesh 的 grid 與標量名）
      2) extract_scalar_only(vis, base_grid) -> array_1d
         （用於指標計算：不畫圖、效能更好）
    - 你可以很容易註冊新的結果種類（例如 'ux', 'uy', 'pmax' 等）
    """

    def __init__(self):
        self._handlers = {}

    def register(self, key: str, title: str,
                 extract_for_plot, extract_for_metrics):
        """
        key: 內部鍵（例如 'dis'、'von'、'pred'）
        title: 顯示用標題
        extract_for_plot: (vis, base_grid) -> (grid, scalars_name, title)
                          注意：此 handler 內部要把標量塞進 grid.point_data
        extract_for_metrics: (vis, base_grid) -> array_1d
        """
        self._handlers[key] = {
            "title": title,
            "plot": extract_for_plot,
            "metric": extract_for_metrics,
        }

    def title(self, key: str) -> str:
        return self._handlers[key]["title"]

    def extract_for_plot(self, key: str, vis: Vis_tools, base_grid: pv.PolyData):
        """回傳 (grid, scalars_name, title)；由視覺化後端使用。"""
        return self._handlers[key]["plot"](vis, base_grid)

    def extract_for_metrics(self, key: str, vis: Vis_tools, base_grid: pv.PolyData):
        """回傳 1D array；提供度量指標計算用。"""
        return self._handlers[key]["metric"](vis, base_grid)

    def keys(self):
        return list(self._handlers.keys())


ResultTypes = ResultRegistry()


# ---- 內建結果：Displacement（幅值） ----
def _plot_dis(vis: Vis_tools, base: pv.PolyData):
    g = base.copy()
    vis.subset = g
    vis.dis_solution()
    arr = g.point_data["solution"]
    arr = _safe_norm_if_vector(arr)
    g.point_data["dis_mag"] = arr
    return g, "dis_mag", "Displacement (magnitude)"

def _metric_dis(vis: Vis_tools, base: pv.PolyData):
    g = base.copy()
    vis.subset = g
    vis.dis_solution()
    arr = g.point_data["solution"]
    return _safe_norm_if_vector(arr)

ResultTypes.register(
    key="dis",
    title="Displacement",
    extract_for_plot=_plot_dis,
    extract_for_metrics=_metric_dis
)

# ---- 內建結果：Von Mises ----
def _plot_von(vis: Vis_tools, base: pv.PolyData):
    g = base.copy()
    vis.subset = g
    vis.stress_solution()
    arr = g.point_data["solution"]
    g.point_data["von"] = arr
    return g, "von", "Von Mises stress"

def _metric_von(vis: Vis_tools, base: pv.PolyData):
    g = base.copy()
    vis.subset = g
    vis.stress_solution()
    return g.point_data["solution"].ravel()

ResultTypes.register(
    key="von",
    title="Von Mises stress",
    extract_for_plot=_plot_von,
    extract_for_metrics=_metric_von
)

# ---- 內建結果：Predicted stress ----
def _plot_pred(vis: Vis_tools, base: pv.PolyData):
    g = base.copy()
    vis.subset = g
    vis.stress_pr_solution()
    arr = g.point_data["solution"]
    g.point_data["pred"] = arr
    return g, "pred", "Predicted stress"

def _metric_pred(vis: Vis_tools, base: pv.PolyData):
    g = base.copy()
    vis.subset = g
    vis.stress_pr_solution()
    return g.point_data["solution"].ravel()

ResultTypes.register(
    key="pred",
    title="Predicted stress",
    extract_for_plot=_plot_pred,
    extract_for_metrics=_metric_pred
)

# ⬇️（範例）要加新結果，只要這樣註冊
# def _plot_ux(vis, base):
#     g = base.copy()
#     vis.subset = g
#     vis.dis_solution()
#     ux = g.point_data["solution"][:, 0]  # 假設 dis 是 Nx3
#     g.point_data["ux"] = ux
#     return g, "ux", "Ux"
# def _metric_ux(vis, base):
#     g = base.copy()
#     vis.subset = g
#     vis.dis_solution()
#     return g.point_data["solution"][:, 0]
# ResultTypes.register("ux", "Ux", _plot_ux, _metric_ux)


# -------------------------------
# Metric Registry
# -------------------------------

class MetricRegistry:
    """
    管理「度量指標」計算：
    - handler: fn(a: 1d-array, b: 1d-array) -> float
    - 預設提供 mse，可輕鬆加 rmse/mae/r2 等
    """
    def __init__(self):
        self._handlers = {}

    def register(self, key: str, title: str, fn):
        self._handlers[key] = {"title": title, "fn": fn}

    def compute(self, key: str, a: np.ndarray, b: np.ndarray) -> float:
        return self._handlers[key]["fn"](a, b)

    def title(self, key: str) -> str:
        return self._handlers[key]["title"]

    def keys(self):
        return list(self._handlers.keys())


Metrics = MetricRegistry()

def _mse(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if not np.any(mask):
        raise ValueError("無有效資料可比較")
    a = a[mask]; b = b[mask]
    if a.shape != b.shape:
        raise ValueError(f"尺寸不一致: {a.shape} vs {b.shape}")
    return float(np.mean((a - b) ** 2))

Metrics.register("mse", "Mean Squared Error", _mse)

# ⬇️（範例）要加指標就註冊一下
# Metrics.register("mae", "Mean Absolute Error", lambda a,b: float(np.mean(np.abs(a-b))))
# Metrics.register("rmse", "Root MSE", lambda a,b: float(np.sqrt(np.mean((a-b)**2))))


# -------------------------------
# Visual Backend (PyVista)
# -------------------------------

class PyVistaBackend:
    """
    單一責任：接收「已萃取好的 grid/scalars」，把它畫出來。
    不知道怎麼算 MSE、不知道資料從哪來（UI/Registry 解決）。
    """
    def __init__(self):
        self.open_plotters = []

    def show(self, left_payload, right_payload, base_grid, show_bc: bool,
             lock_clim: bool, n_colors: int, show_edges: bool):
        """
        left_payload = (gridL, scalars_name_L, title_L)
        right_payload = (gridR, scalars_name_R, title_R) or None
        """
        cols = 1 + (1 if right_payload is not None else 0) + (1 if show_bc else 0)

        p = pv.Plotter(shape=(1, cols))

        # 同色軸
        clim = None
        if lock_clim and right_payload is not None:
            gL, sL, _ = left_payload
            gR, sR, _ = right_payload
            a = gL.point_data[sL]
            b = gR.point_data[sR]
            try:
                vmin = float(np.nanmin([np.nanmin(a), np.nanmin(b)]))
                vmax = float(np.nanmax([np.nanmax(a), np.nanmax(b)]))
                if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
                    clim = (vmin, vmax)
            except Exception:
                clim = None

        # 左圖
        cur = 0
        gL, scalL, titleL = left_payload
        p.subplot(0, cur)
        p.add_text(titleL, font_size=12, viewport=True, position=(0.02, 0.96))
        p.add_mesh(gL, scalars=scalL, cmap='jet', clim=clim, n_colors=n_colors, show_edges=show_edges, show_scalar_bar=False,)
        p.add_scalar_bar(title=titleL); p.show_axes()

        # 右圖
        if right_payload is not None:
            cur += 1
            gR, scalR, titleR = right_payload
            p.subplot(0, cur)
            p.add_text(titleR, font_size=12, viewport=True, position=(0.02, 0.96))
            p.add_mesh(gR, scalars=scalR, cmap='jet', clim=clim, n_colors=n_colors, show_edges=show_edges ,show_scalar_bar=False,)
            p.add_scalar_bar(title=titleR); p.show_axes()

        # 邊界條件
        if show_bc:
            cur += 1
            self._add_boundary_subplot(p, base_grid, cur)

        p.link_views(); p.enable_parallel_projection()
        p.show(auto_close=False, interactive_update=True)
        self.open_plotters.append(p)
        return p

    @staticmethod
    def _add_boundary_subplot(plotter: pv.Plotter, grid: pv.PolyData, col: int):
        if 'node_features' not in grid.point_data:
            plotter.subplot(0, col)
            plotter.add_text('Boundary (資料不足)', font_size=12, viewport=True, position=(0.02, 0.96))
            return

        feats = grid.point_data['node_features']
        if feats.ndim != 2 or feats.shape[1] < 9:
            plotter.subplot(0, col)
            plotter.add_text('Boundary (資料不足)', font_size=12, viewport=True, position=(0.02, 0.96))
            return

        coords = grid.points
        loads = feats[:, 3:6]
        dofs  = feats[:, 6:9]
        tol = 1e-10
        is_load  = np.any(~np.isclose(loads, 0.0, atol=tol), axis=1)
        is_fixed = np.all(np.isclose(dofs,  0.0, atol=tol), axis=1)
        load_pts  = coords[is_load]
        fixed_pts = coords[is_fixed]

        plotter.subplot(0, col)
        plotter.add_text('Boundary', font_size=12, viewport=True, position=(0.02, 0.96))
        plotter.add_mesh(grid, color='lightgray', opacity=1.0)

        b = grid.bounds
        diag = float(np.linalg.norm([b[1]-b[0], b[3]-b[2], b[5]-b[4]]))
        psize = max(diag * 0.01, 2.0)

        if load_pts.size:
            plotter.add_mesh(pv.PolyData(load_pts), color='red', point_size=psize,
                             render_points_as_spheres=True, label='Load')
        if fixed_pts.size:
            plotter.add_mesh(pv.PolyData(fixed_pts), color='blue', point_size=psize,
                             render_points_as_spheres=True, label='Fixed')
        plotter.add_legend(); plotter.show_axes()

    def close_all(self):
        while self.open_plotters:
            p = self.open_plotters.pop()
            try:
                p.close()
            except Exception:
                pass


# -------------------------------
# Application (Tk UI)
# -------------------------------

APP_TITLE = "可視化：模組化 / 可擴展 UI"
DEFAULT_DIR = "data"

# 對外顯示名稱與 key 的對照（UI 友善）
DISPLAY_CHOICES = [
    ("Displacement", "dis"),
    ("Von Mises stress", "von"),
    ("Predicted stress", "pred"),
]

METRIC_CHOICES = [
    ("MSE", "mse"),
    # ⬇️ 以後想加就列在這裡，例如：
    # ("MAE", "mae"),
    # ("RMSE", "rmse"),
]


class App:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.backend = PyVistaBackend()

        self._build_ui()

    # -------- UI 布局 --------
    def _build_ui(self):
        self.root.minsize(880, 380)
        self.root.update_idletasks()
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        w, h = 940, 380
        x, y = (sw - w) // 2, (sh - h) // 3
        self.root.geometry(f"{w}x{h}+{x}+{y}")

        frm = ttk.Frame(self.root, padding=16)
        frm.pack(fill=tk.BOTH, expand=True)
        for c in range(6):
            frm.columnconfigure(c, weight=1 if c in (1, 2, 3) else 0)

        # 檔案
        ttk.Label(frm, text='選擇檔案 (.rst/.vtu/.vtk)：').grid(row=0, column=0, sticky='w')
        self.file_var = tk.StringVar(value='')
        ent = ttk.Entry(frm, textvariable=self.file_var)
        ent.grid(row=1, column=0, columnspan=4, sticky='ew', pady=(4, 10))

        ttk.Button(frm, text='瀏覽…', command=self._browse).grid(row=1, column=4, sticky='w', padx=(8, 0))

        # 左右結果
        ttk.Label(frm, text='左側結果：').grid(row=2, column=0, sticky='w')
        self.left_var = tk.StringVar(value=DISPLAY_CHOICES[1][0])
        self.combo_left = ttk.Combobox(frm, values=[n for n, _ in DISPLAY_CHOICES],
                                       state='readonly', textvariable=self.left_var)
        self.combo_left.grid(row=3, column=0, sticky='w', pady=(4, 10))

        ttk.Label(frm, text='右側結果：').grid(row=2, column=2, sticky='w')
        self.right_var = tk.StringVar(value=DISPLAY_CHOICES[2][0])
        self.combo_right = ttk.Combobox(frm, values=[n for n, _ in DISPLAY_CHOICES],
                                        state='readonly', textvariable=self.right_var)
        self.combo_right.grid(row=3, column=2, sticky='w', pady=(4, 10))

        # 控制選項
        self.dual_var = tk.BooleanVar(value=True)
        self.lock_clim_var = tk.BooleanVar(value=True)
        self.show_bc_var = tk.BooleanVar(value=True)
        self.edges_var = tk.BooleanVar(value=True)
        self.ncolors_var = tk.IntVar(value=10)

        ttk.Checkbutton(frm, text='雙圖對照', variable=self.dual_var, command=self._update_dual_state).grid(row=2, column=1, sticky='w')
        ttk.Checkbutton(frm, text='同一個色軸範圍（雙圖時）', variable=self.lock_clim_var).grid(row=3, column=1, sticky='w')
        ttk.Checkbutton(frm, text='顯示邊界條件（若有 node_features）', variable=self.show_bc_var).grid(row=3, column=3, sticky='w')

        ttk.Label(frm, text='色階數：').grid(row=4, column=0, sticky='w')
        tk.Spinbox(frm, from_=3, to=256, textvariable=self.ncolors_var, width=6).grid(row=4, column=1, sticky='w')
        ttk.Checkbutton(frm, text='顯示網格（Edges）', variable=self.edges_var).grid(row=4, column=2, sticky='w')

        # 指標區塊
        metric_frame = ttk.Frame(frm)
        metric_frame.grid(row=5, column=0, columnspan=5, sticky='ew', pady=(8, 6))
        metric_frame.columnconfigure(3, weight=1)

        ttk.Label(metric_frame, text='度量：').grid(row=0, column=0, sticky='w')
        self.metric_var = tk.StringVar(value=METRIC_CHOICES[0][0])
        self.combo_metric = ttk.Combobox(metric_frame, values=[n for n, _ in METRIC_CHOICES],
                                         state='readonly', textvariable=self.metric_var, width=10)
        self.combo_metric.grid(row=0, column=1, sticky='w')

        ttk.Label(metric_frame, text='結果：').grid(row=0, column=2, sticky='w')
        self.mse_var = tk.StringVar(value='—')
        ttk.Label(metric_frame, textvariable=self.mse_var, width=24).grid(row=0, column=3, sticky='w')

        ttk.Button(metric_frame, text='計算', command=self._on_calc_metric).grid(row=0, column=4, sticky='e')

        # 按鈕列
        btns = ttk.Frame(frm)
        btns.grid(row=6, column=0, columnspan=5, sticky='ew', pady=(6, 0))
        ttk.Button(btns, text='開始', command=self._on_start).pack(side='right')
        ttk.Button(btns, text='關閉所有視窗', command=self.backend.close_all).pack(side='right', padx=8)

        self._update_dual_state()

    # -------- UI handlers --------
    def _browse(self):
        initdir = DEFAULT_DIR if os.path.isdir(DEFAULT_DIR) else os.getcwd()
        path = filedialog.askopenfilename(
            title='選擇結果檔', initialdir=initdir,
            filetypes=[('ANSYS/VTK files', '*.rst *.vtu *.vtk'), ('All files', '*.*')],
        )
        if path:
            self.file_var.set(path)

    def _update_dual_state(self):
        self.combo_right.configure(state='readonly' if self.dual_var.get() else 'disabled')
        if not self.dual_var.get():
            self.mse_var.set('—')

    @staticmethod
    def _display_to_key(name: str) -> str:
        for n, k in DISPLAY_CHOICES:
            if n == name:
                return k
        # fallback
        return DISPLAY_CHOICES[0][1]

    @staticmethod
    def _metric_display_to_key(name: str) -> str:
        for n, k in METRIC_CHOICES:
            if n == name:
                return k
        return METRIC_CHOICES[0][1]

    def _load_vis(self, path: str) -> Vis_tools:
        return Vis_tools(path)

    # ---- Compute metric (e.g. MSE) on current selections ----
    def _on_calc_metric(self):
        if not self.dual_var.get():
            messagebox.showinfo('提示', '請先勾選「雙圖對照」。')
            return

        path = self.file_var.get().strip()
        if not path:
            messagebox.showwarning('提醒', '請先選擇檔案。'); return
        if not os.path.exists(path):
            messagebox.showerror('錯誤', f'找不到檔案：\n{path}'); return

        left_key = self._display_to_key(self.left_var.get())
        right_key = self._display_to_key(self.right_var.get())
        metric_key = self._metric_display_to_key(self.metric_var.get())

        try:
            vis = self._load_vis(path)
            base = vis.subset

            a = ResultTypes.extract_for_metrics(left_key, vis, base)
            b = ResultTypes.extract_for_metrics(right_key, vis, base)

            if a is None or b is None:
                self.mse_var.set('資料不足或欄位缺失')
                return
            if a.shape != b.shape:
                self.mse_var.set(f'尺寸不一致 {a.shape} vs {b.shape}')
                return

            val = Metrics.compute(metric_key, a, b)
            self.mse_var.set(f'{val:.6f}')
        except Exception as e:
            self.mse_var.set('計算失敗')
            messagebox.showerror('度量計算失敗', f'原因：\n{e}')

    # ---- Visualize ----
    def _on_start(self):
        path = self.file_var.get().strip()
        if not path:
            messagebox.showwarning('提醒', '請先選擇檔案。'); return
        if not os.path.exists(path):
            messagebox.showerror('錯誤', f'找不到檔案：\n{path}'); return

        left_key = self._display_to_key(self.left_var.get())
        right_key = self._display_to_key(self.right_var.get())

        try:
            vis = self._load_vis(path)
        except Exception as e:
            messagebox.showerror("載入失敗", f"無法載入檔案：\n{e}")
            return

        base = vis.subset

        # 左側 payload
        try:
            gL, scalL, titleL = ResultTypes.extract_for_plot(left_key, vis, base)
        except Exception as e:
            messagebox.showerror('錯誤', f'無法產生左側視圖：\n{e}')
            return

        # 右側 payload（可選）
        right_payload = None
        if self.dual_var.get():
            try:
                gR, scalR, titleR = ResultTypes.extract_for_plot(right_key, vis, base)
                right_payload = (gR, scalR, titleR)
            except Exception:
                messagebox.showwarning('提示', '右側結果無法生成（欄位缺失或錯誤），改為單圖。')
                right_payload = None

        # 顯示
        self.backend.show(
            left_payload=(gL, scalL, titleL),
            right_payload=right_payload,
            base_grid=base,
            show_bc=self.show_bc_var.get(),
            lock_clim=self.lock_clim_var.get(),
            n_colors=int(self.ncolors_var.get()),
            show_edges=bool(self.edges_var.get())
        )


# -------------------------------
# main
# -------------------------------

if __name__ == '__main__':
    try:
        root = tk.Tk()
    except Exception as e:
        print('無法建立 Tk 視窗：', e)
        sys.exit(1)

    App(root)
    root.mainloop()
