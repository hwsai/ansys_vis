# -*- coding: utf-8 -*-
"""
Step 1：最小可執行可視化程式
- 只顯示 Von Mises 應力（實際解）
- 依賴你提供的 vis_lab.py（Vis_tools）
- 無任何額外介面/快捷鍵/色階/邊界條件

用法：
    python viewer_step1.py <path/to/file.rst|.vtu|.vtk>
"""
import argparse
import sys
import pyvista as pv
from vis_lab import Vis_tools


def main():
    parser = argparse.ArgumentParser(description='最小可視化：Von Mises stress')
    parser.add_argument('file', help='RST/VTU/VTK 檔案路徑')
    args = parser.parse_args()

    try:
        vis = Vis_tools(args.file)
    except Exception as e:
        print(f'❌ 載入失敗：{e}')
        sys.exit(1)

    # 只做一件事：把 Von Mises 應力算好塞到 point_data["solution"]
    vis.stress_solution()
    grid = vis.subset

    # 最小渲染：只畫一張圖，使用 "solution" 當作色標
    plotter = pv.Plotter()
    plotter.add_mesh(grid, scalars='solution', cmap='jet')
    plotter.add_scalar_bar(title='Von Mises stress')
    plotter.show()


if __name__ == '__main__':
    main()
