# main.py
import tkinter as tk
from ui_builder import UIBuilder

def run_visualization():
    print("🚀 可視化開始")
    print("file:", ui.vars["file_path"].get())
    print("左側:", ui.vars["left_result"].get())
    print("右側:", ui.vars["right_result"].get())
    print("顯示邊界條件:", ui.vars["show_bc"].get())

def calc_metric():
    print("📏 計算誤差（尚未實作）")

root = tk.Tk()
ui = UIBuilder(root, "ui_config.yaml", {
    "run_visualization": run_visualization,
    "calc_metric": calc_metric
})
root.mainloop()
