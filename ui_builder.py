# ui_builder.py
import tkinter as tk
from tkinter import ttk, filedialog
import yaml
import os

class UIBuilder:
    def __init__(self, root, config_path, callbacks=None):
        self.root = root
        self.callbacks = callbacks or {}
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.vars = {}      # 儲存所有變數（路徑、選項、旗標）
        self.widgets = {}   # 儲存所有元件（方便動態控制）
        self._build_ui()
        self._apply_dynamic_bindings()  # 支援 disable_if 條件

    # =====================================================
    # 🧩 主建構邏輯
    # =====================================================
    def _build_ui(self):
        self.root.title(self.config.get("title", "Config UI"))
        frm = ttk.Frame(self.root, padding=16)
        frm.pack(fill=tk.BOTH, expand=True)
        for item in self.config["ui"]:
            self._build_item(frm, item)

    def _build_item(self, parent, item):
        """遞迴建構：支援 row / section / 單一元件"""
        t = item["type"]

        if t == "row":
            row_frame = ttk.Frame(parent)
            row_frame.pack(fill="x", pady=4)
            for child in item["children"]:
                sub = ttk.Frame(row_frame)
                sub.pack(side="left", padx=6, expand=True, fill="x")
                self._build_item(sub, child)

        elif t == "section":
            sec = ttk.LabelFrame(parent, text=item.get("label", ""), padding=6)
            sec.pack(fill="x", pady=6)
            for child in item["children"]:
                self._build_item(sec, child)

        else:
            self._build_single_widget(parent, item)

    # =====================================================
    # 🧱 單一元件類別統一入口
    # =====================================================
    def _build_single_widget(self, parent, item):
        t = item["type"]
        if t == "label":
            w = ttk.Label(parent, text=item["label"])
            w.pack(anchor="w", pady=3)
        elif t == "file_input":
            w = self._build_file_input(parent, item)
        elif t == "combobox":
            w = self._build_combobox(parent, item)
        elif t == "checkbox":
            w = self._build_checkbox(parent, item)
        elif t == "spinbox":
            w = self._build_spinbox(parent, item)
        elif t == "entry":
            w = self._build_entry(parent, item)
        elif t == "button":
            w = self._build_button(parent, item)
        else:
            print(f"⚠️ 未支援的元件類型：{t}")
            return
        if "id" in item:
            self.widgets[item["id"]] = w

    # =====================================================
    # 🧩 各種元件建構
    # =====================================================
    def _build_file_input(self, parent, item):
        ttk.Label(parent, text=item["label"]).pack(anchor="w")

        var_path = tk.StringVar()
        ent = ttk.Entry(parent, textvariable=var_path)
        ent.pack(fill="x", pady=2)

        var_type = tk.StringVar(value="—")
        ttk.Label(parent, text="檔案類型：").pack(anchor="w")
        lbl_type = ttk.Label(parent, textvariable=var_type, foreground="blue")
        lbl_type.pack(anchor="w", pady=(0, 4))

        def _on_browse():
            filetypes = [("允許的檔案", " ".join(f"*{e}" for e in item["extensions"]))]
            path = filedialog.askopenfilename(filetypes=filetypes)
            if path:
                var_path.set(path)
                ext = os.path.splitext(path)[1].lower().replace('.', '')
                var_type.set(ext)

        ttk.Button(parent, text="瀏覽", command=_on_browse).pack()

        self.vars[item["id"]] = var_path
        type_id = item.get("type_id", f"{item['id']}_type")
        self.vars[type_id] = var_type
        return ent

    def _build_combobox(self, parent, item):
        ttk.Label(parent, text=item["label"]).pack(anchor="w")
        var = tk.StringVar(value=item["options"][0]["name"])
        combo = ttk.Combobox(parent, textvariable=var,
                             values=[opt["name"] for opt in item["options"]],
                             state="readonly")
        combo.pack(fill="x", pady=3)
        self.vars[item["id"]] = var
        return combo

    def _build_checkbox(self, parent, item):
        var = tk.BooleanVar(value=item.get("default", False))
        chk = ttk.Checkbutton(parent, text=item["label"], variable=var)
        chk.pack(anchor="w")
        self.vars[item["id"]] = var
        return chk

    def _build_spinbox(self, parent, item):
        ttk.Label(parent, text=item["label"]).pack(anchor="w")
        var = tk.IntVar(value=item.get("default", 0))
        spin = tk.Spinbox(parent, from_=item.get("from", 0), to=item.get("to", 100),
                          textvariable=var, width=6)
        spin.pack(anchor="w")
        self.vars[item["id"]] = var
        return spin

    def _build_entry(self, parent, item):
        """單行輸入欄"""
        ttk.Label(parent, text=item["label"]).pack(anchor="w")
        var = tk.StringVar(value=item.get("default", ""))
        ent = ttk.Entry(parent, textvariable=var)
        ent.pack(fill="x", pady=2)
        self.vars[item["id"]] = var
        return ent

    def _build_button(self, parent, item):
        label = item["label"]
        action = item.get("action")
        cmd = self.callbacks.get(action, lambda: print(f"⚠️ 未綁定動作：{action}"))
        btn = ttk.Button(parent, text=label, command=cmd)
        btn.pack(pady=4)
        return btn

    # =====================================================
    # 🔁 動態條件綁定（disable_if）
    # =====================================================
    def _apply_dynamic_bindings(self):
        for item in self.config["ui"]:
            self._bind_condition(item)

    def _bind_condition(self, item):
        if "children" in item:
            for c in item["children"]:
                self._bind_condition(c)

        cond = item.get("disable_if")
        if not cond:
            return

        target_id = item.get("id")
        if not target_id:
            return
        target_widget = self.widgets.get(target_id)
        if not target_widget:
            return

        negated = cond.strip().startswith("not ")
        var_name = cond.replace("not ", "").strip()
        control_var = self.vars.get(var_name)
        if not control_var:
            print(f"⚠️ 找不到變數：{var_name}")
            return

        def update_state(*_):
            val = control_var.get()
            disable = val if not negated else (not val)
            state = "disabled" if disable else "normal"
            # Combobox特殊處理
            if isinstance(target_widget, ttk.Combobox):
                state = "disabled" if disable else "readonly"
            target_widget.configure(state=state)

        control_var.trace_add("write", lambda *_: update_state())
        update_state()
