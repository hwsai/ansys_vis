import numpy as np
import pyvista as pv
import datetime  # 別忘了要 import




#鍵盤控制
def _toggle_projection(plotter):
    try:
        cam = plotter.camera
        current = getattr(cam, "parallel_projection", None)
        if current is None:
            if hasattr(cam, "GetParallelProjection") and hasattr(cam, "SetParallelProjection"):
                current = cam.GetParallelProjection()
                cam.SetParallelProjection(not current)
            else:
                print("⚠️ 相機物件不支援 parallel projection 切換")
                return
        else:
            cam.parallel_projection = not current

        plotter.render()
        is_parallel = getattr(cam, "parallel_projection", None)
        if is_parallel is None and hasattr(cam, "GetParallelProjection"):
            is_parallel = cam.GetParallelProjection()
        mode = "正交(Orthographic)" if is_parallel else "透視(Perspective)"
        print(f"🔁 投影模式切換為：{mode}")
    except Exception as e:
        print("⚠️ 切換投影失敗：", e)


def bind_keyboard_view_controls(plotter):
    def to_x_view(): plotter.view_vector((1, 0, 0))
    def to_y_view(): plotter.view_vector((0, 1, 0))
    def to_z_view(): plotter.view_vector((0, 0, 1))
    def to_iso_view(): plotter.camera_position = 'iso'

    def to_xy_diag(): plotter.view_vector((1, 1, 0), viewup=(0, 0, -1))
    def to_yz_diag(): plotter.view_vector((0, 1, 1))
    def to_xz_diag(): plotter.view_vector((1, 0, 1))

    def save_screenshot():
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.png"
        plotter.screenshot(filename)
        print(f"📸 畫面已儲存為 {filename}")

    plotter.add_key_event("x", to_x_view)
    plotter.add_key_event("y", to_y_view)
    plotter.add_key_event("z", to_z_view)
    plotter.add_key_event("i", to_iso_view)

    plotter.add_key_event("a", to_xy_diag)
    plotter.add_key_event("s", to_yz_diag)
    plotter.add_key_event("d", to_xz_diag)

    plotter.add_key_event("p", save_screenshot)
    plotter.add_key_event("k", lambda: _toggle_projection(plotter))

    print("✅ 鍵盤視角/截圖/投影控制綁定完成：")
    print("  x/y/z = 對正視角，i = 等角")
    print("  a/s/d = XY/YZ/XZ 斜視角，p = 儲存截圖")
    print("  k = 切換透視/正交")
    
    
    
    
class PyVistaBackend:
    def __init__(self):
        self.open_plotters = []

    def show(self, left_payload, right_payload, base_grid, show_bc: bool,
             lock_clim: bool, n_colors: int, show_edges: bool):
        cols = 1 + (1 if right_payload is not None else 0) + (1 if show_bc else 0)
        p = pv.Plotter(shape=(1, cols))

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

        cur = 0
        gL, scalL, titleL = left_payload
        p.subplot(0, cur)
        p.add_text(titleL, font_size=12, position=(0.02, 0.96))
        p.add_mesh(gL, scalars=scalL, cmap='jet', clim=clim,
                   n_colors=n_colors, show_edges=show_edges, show_scalar_bar=False)
        p.add_scalar_bar(title=titleL); p.show_axes()

        if right_payload is not None:
            cur += 1
            gR, scalR, titleR = right_payload
            p.subplot(0, cur)
            p.add_text(titleR, font_size=12, position=(0.02, 0.96))
            p.add_mesh(gR, scalars=scalR, cmap='jet', clim=clim,
                       n_colors=n_colors, show_edges=show_edges, show_scalar_bar=False)
            p.add_scalar_bar(title=titleR); p.show_axes()

        if show_bc:
            cur += 1
            self._add_boundary_subplot(p, base_grid, cur)

        p.link_views(); p.enable_parallel_projection()
        bind_keyboard_view_controls(p) 
        p.show(auto_close=False, interactive_update=True)
        self.open_plotters.append(p)
        return p

    @staticmethod
    def _add_boundary_subplot(plotter: pv.Plotter, grid: pv.PolyData, col: int):
        if 'node_features' not in grid.point_data:
            plotter.subplot(0, col)
            plotter.add_text('Boundary (資料不足)', font_size=12, position=(0.02, 0.96))
            return

        feats = grid.point_data['node_features']
        if feats.ndim != 2 or feats.shape[1] < 9:
            plotter.subplot(0, col)
            plotter.add_text('Boundary (資料不足)', font_size=12, position=(0.02, 0.96))
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
        plotter.add_text('Boundary', font_size=12, position=(0.02, 0.96))
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
