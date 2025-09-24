import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
#import pyvista as pv
from vis_lab import Vis_tools

from result_registry import ResultTypes
from metric_registry import Metrics
from backend_pyvista import PyVistaBackend

APP_TITLE = "可視化：模組化 / 可擴展 UI"
DEFAULT_DIR = "data"

DISPLAY_CHOICES = [
    ("Displacement", "dis"),
    ("Von Mises stress", "von"),
    ("Predicted stress", "pred"),
]

METRIC_CHOICES = [
    ("MSE", "mse"),
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
