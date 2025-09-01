#import sys
from ansys.mapdl import reader as pymapdl_reader
#import pyvista as pv
import numpy as np
#from gnn_tl import gnn_tl
import pyvista as pv
import os

class Vis_tools:
    def __init__(self,rst_file):


        ext = os.path.splitext(rst_file)[1].lower()

        if ext == ".rst":
            rst = pymapdl_reader.read_binary(rst_file)

            # 擷取節點變位與主應力
            nnum, displacement = rst.nodal_displacement(0)
            nnum, stress = rst.principal_nodal_stress(0)

            # 取得網格並加上資料
            grid = rst.grid
            grid.point_data['displacement'] = displacement
            grid.point_data['stress'] = stress

            # 移除 beam/link/shell 等非 solid 元素
            grid = grid.extract_cells(grid.celltypes >= 10)

            self.subset = grid

        elif ext in [".vtu", ".vtk"]:
            self.subset = pv.read(rst_file)

        else:
            raise ValueError(f"不支援的檔案格式：{ext}，僅支援 .rst, .vtu, .vtk")
        
    def part_vis(self, num):
        
        """
        element_components_keys = list(self.elem_comp.keys())
        
        selected_component = element_components_keys[num - 1]
        element_ids = self.elem_comp[selected_component]
        mask = np.isin(self.subset.cell_data['ansys_elem_num'], element_ids)
        
        self.subset = self.subset.extract_cells(mask)
        """
        
        
        component_keys = list(self.subset.cell_data.keys())
    
        # 過濾出布林遮罩（避免選到其他非組件欄位）
        component_keys = [k for k in component_keys if self.subset.cell_data[k].dtype == bool]
    
        # 防呆：檢查 num 是否超出範圍
        if num < 1 or num > len(component_keys):
            raise ValueError(f"無效的組件索引：{num}（有效範圍為 1 ~ {len(component_keys)}）")
    
        # 取得選擇的欄位名稱
        selected_key = component_keys[num - 1]
    
        # 使用布林遮罩篩選單一組件
        mask = self.subset.cell_data[selected_key]
        self.subset = self.subset.extract_cells(mask)
            
        return self.subset
    def part_node_vis(self, num):
        # 選擇第 num 個 node component 名稱
        selected_component = list(self.node_comp.keys())[num - 1]
        node_ids = self.node_comp[selected_component]
    
        # 從點資料中找出符合的節點
        
    
        mesh_node_ids = self.subset.point_data['ansys_node_num']
        print(mesh_node_ids)
        mask = np.isin(mesh_node_ids, node_ids)
    
        # 根據符合的點建構子網格
        points_to_extract = self.subset.extract_points(mask, adjacent_cells=True, include_cells=True)
    
        return points_to_extract
        return self.subset
    
    def dis_solution(self):
        
        self.subset.point_data['solution'] = self.subset.point_data['displacement']
        
        return self.subset
    
    def stress_solution(self):
        stress = self.subset.point_data['stress']    
        
        #s1, s2, s3 = stress[:, 0], stress[:, 1], stress[:, 2]
        #von_mises_stress = np.sqrt(((s1 - s2)**2 + (s2 - s3)**2 + (s3 - s1)**2) / 2)
        von_mises_stress = stress[:, 4]
        self.subset.point_data['solution'] = von_mises_stress
        
        return self.subset
    def stress_pr_solution(self):
        self.subset.point_data['solution'] = self.subset.point_data['von_mises_stress_pred']
        
        return self.subset

        
    
    