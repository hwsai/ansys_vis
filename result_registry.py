import numpy as np
import pyvista as pv
from vis_lab import Vis_tools


# 工具函數
def _safe_norm_if_vector(arr: np.ndarray) -> np.ndarray:
    if arr is None:
        return None
    if arr.ndim == 2 and arr.shape[1] > 1:
        return np.linalg.norm(arr, axis=1)
    return arr

class ResultRegistry:
    def __init__(self):
        self._handlers = {}

    def register(self, key: str, title: str, extract_for_plot, extract_for_metrics):
        self._handlers[key] = {
            "title": title,
            "plot": extract_for_plot,
            "metric": extract_for_metrics,
        }

    def title(self, key: str) -> str:
        return self._handlers[key]["title"]

    def extract_for_plot(self, key: str, vis: Vis_tools, base_grid: pv.PolyData):
        return self._handlers[key]["plot"](vis, base_grid)

    def extract_for_metrics(self, key: str, vis: Vis_tools, base_grid: pv.PolyData):
        return self._handlers[key]["metric"](vis, base_grid)

    def keys(self):
        return list(self._handlers.keys())


ResultTypes = ResultRegistry()

# ---- 內建結果：Displacement ----
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

ResultTypes.register("dis", "Displacement", _plot_dis, _metric_dis)

# ---- Von Mises ----
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

ResultTypes.register("von", "Von Mises stress", _plot_von, _metric_von)

# ---- Predicted stress ----
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

ResultTypes.register("pred", "Predicted stress", _plot_pred, _metric_pred)
