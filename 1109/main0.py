import sys
import tkinter as tk
from app_ui import App

if __name__ == '__main__':
    try:
        root = tk.Tk()
    except Exception as e:
        print('無法建立 Tk 視窗：', e)
        sys.exit(1)

    App(root)
    root.mainloop()
 