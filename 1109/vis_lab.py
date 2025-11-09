import os
import numpy as np
import pyvista as pv
from ansys.mapdl import reader as pymapdl_reader


class Vis_tools:
    """負責讀取 .rst / .vtu / .vtk 檔案，並提供結果解算與子集操作。"""

    def __init__(self, filepath: str):
        ext = os.path.splitext(filepath)[1].lower()

        if ext == ".rst":
            self._load_rst(filepath)
        elif ext in [".vtu", ".vtk"]:
            self.subset = pv.read(filepath)
        else:
            raise ValueError(f"不支援的檔案格式：{ext}，僅支援 .rst, .vtu, .vtk")

    # -------------------------------
    # 讀檔與初始化
    # -------------------------------
    def _load_rst(self, filepath: str):
        rst = pymapdl_reader.read_binary(filepath)

        # 讀取 nodal 解
        nnum, displacement = rst.nodal_displacement(0)
        nnum, stress = rst.principal_nodal_stress(0)

        grid = rst.grid
        grid.point_data['displacement'] = displacement
        grid.point_data['stress'] = stress

        # 移除非 solid 元素 (beam/shell/link...)
        grid = grid.extract_cells(grid.celltypes >= 10)
        self.subset = grid

    # -------------------------------
    # 子集擷取
    # -------------------------------
    def part_vis(self, num: int):
        """依照 cell component 遮罩擷取指定部分。"""
        component_keys = [k for k in self.subset.cell_data.keys()
                          if self.subset.cell_data[k].dtype == bool]

        if not component_keys:
            raise ValueError("目前網格沒有可用的 component 遮罩")

        if num < 1 or num > len(component_keys):
            raise ValueError(f"無效索引：{num}（有效範圍為 1 ~ {len(component_keys)}）")

        selected_key = component_keys[num - 1]
        mask = self.subset.cell_data[selected_key]
        self.subset = self.subset.extract_cells(mask)
        return self.subset

    def part_node_vis(self, num: int):
        """依照 node component 遮罩擷取部分網格。"""
        if not hasattr(self, "node_comp"):
            raise ValueError("此檔案未提供 node_comp 資訊")

        selected_component = list(self.node_comp.keys())[num - 1]
        node_ids = self.node_comp[selected_component]

        mesh_node_ids = self.subset.point_data['ansys_node_num']
        mask = np.isin(mesh_node_ids, node_ids)

        return self.subset.extract_points(mask, adjacent_cells=True, include_cells=True)

    # -------------------------------
    # 解的切換
    # -------------------------------
    def dis_solution(self):
        """將解設為位移向量 (Nx3)"""
        if 'displacement' not in self.subset.point_data:
            raise ValueError("檔案缺少 displacement 欄位")
        self.subset.point_data['solution'] = self.subset.point_data['displacement']
        return self.subset

    def stress_solution(self):
        """將解設為 Von Mises stress"""
        if 'stress' not in self.subset.point_data:
            raise ValueError("檔案缺少 stress 欄位")
        stress = self.subset.point_data['stress']
        von_mises_stress = stress[:, 4]  # 假設第 5 欄是 Von Mises
        self.subset.point_data['solution'] = von_mises_stress
        return self.subset

    def stress_pr_solution(self):
        """將解設為預測的 Von Mises stress"""
        if 'von_mises_stress_pred' not in self.subset.point_data:
            raise ValueError("檔案缺少 von_mises_stress_pred 欄位")
        self.subset.point_data['solution'] = self.subset.point_data['von_mises_stress_pred']
        return self.subset
