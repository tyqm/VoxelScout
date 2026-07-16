"""VoxelScout's integrated local spinal CT desktop viewer."""

from __future__ import annotations

import argparse
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib import colors as mpl_colors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from voxelscout.inspect_case import window_ct
from voxelscout.viewer_data import (
    REGION_COLOURS,
    SurfaceMesh,
    VolumeCase,
    build_surface_meshes,
    default_slice_indices,
    load_case,
    orthogonal_slice,
    vertebra_name,
    vertebra_region,
)


BACKGROUND = "#07111f"
PANEL = "#0d1b2a"
PANEL_ALT = "#10243a"
TEXT = "#e6edf7"
MUTED = "#91a4bb"
ACCENT = "#22d3ee"
SELECTED = "#facc15"
WARNING = "#f59e0b"


class ViewerApp:
    """Single-window CT, label, and 3D exploration application."""

    def __init__(
        self,
        root: tk.Tk,
        *,
        initial_image: Path | None = None,
        initial_label: Path | None = None,
    ) -> None:
        self.root = root
        self.case: VolumeCase | None = None
        self.meshes: tuple[SurfaceMesh, ...] = ()
        self.selected_label: int | None = None
        self.label_list: list[int] = []
        self.loading = False
        self._setting_scales = False
        self._worker_events: queue.Queue[tuple[object, ...]] = queue.Queue()

        root.title("VoxelScout · Spinal CT Explorer")
        root.geometry("1480x920")
        root.minsize(1120, 720)
        root.configure(bg=BACKGROUND)
        root.protocol("WM_DELETE_WINDOW", root.destroy)

        self._configure_styles()
        self._build_layout()
        self._build_figure()
        self._set_empty_state()

        if initial_image is not None:
            root.after(
                120,
                lambda: self.load_async(Path(initial_image), initial_label),
            )

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Sidebar.TFrame", background=PANEL)
        style.configure("Card.TFrame", background=PANEL_ALT)
        style.configure(
            "TLabel", background=PANEL, foreground=TEXT, font=("Segoe UI", 10)
        )
        style.configure(
            "Muted.TLabel", background=PANEL, foreground=MUTED, font=("Segoe UI", 9)
        )
        style.configure(
            "Section.TLabel",
            background=PANEL,
            foreground=ACCENT,
            font=("Segoe UI Semibold", 10),
        )
        style.configure(
            "TButton",
            background=PANEL_ALT,
            foreground=TEXT,
            padding=(12, 8),
            borderwidth=0,
            font=("Segoe UI Semibold", 9),
        )
        style.map(
            "TButton",
            background=[("active", "#183650"), ("disabled", "#172536")],
            foreground=[("disabled", "#5d7188")],
        )
        style.configure(
            "Accent.TButton", background="#0e7490", foreground="#ffffff"
        )
        style.map("Accent.TButton", background=[("active", "#0891b2")])
        style.configure(
            "Horizontal.TScale",
            background=PANEL,
            troughcolor="#20354a",
            sliderlength=18,
        )
        style.configure(
            "Vertical.TScrollbar",
            background="#1b3046",
            troughcolor=PANEL,
            arrowcolor=TEXT,
        )

    def _build_layout(self) -> None:
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        header = tk.Frame(self.root, bg="#091827", height=68)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_propagate(False)
        header.grid_columnconfigure(1, weight=1)

        brand = tk.Frame(header, bg="#091827")
        brand.grid(row=0, column=0, padx=(22, 26), pady=10, sticky="w")
        tk.Label(
            brand,
            text="VOXEL",
            bg="#091827",
            fg="#ffffff",
            font=("Segoe UI Semibold", 17),
        ).pack(side="left")
        tk.Label(
            brand,
            text="SCOUT",
            bg="#091827",
            fg=ACCENT,
            font=("Segoe UI Semibold", 17),
        ).pack(side="left")
        tk.Label(
            brand,
            text="  SPINAL CT EXPLORER",
            bg="#091827",
            fg=MUTED,
            font=("Segoe UI", 9),
        ).pack(side="left", pady=(6, 0))

        actions = tk.Frame(header, bg="#091827")
        actions.grid(row=0, column=1, padx=18, pady=10, sticky="e")
        self.open_button = ttk.Button(
            actions,
            text="打开 NIfTI",
            style="Accent.TButton",
            command=self.open_nifti,
        )
        self.open_button.pack(side="left", padx=4)
        self.dicom_button = ttk.Button(
            actions, text="打开 DICOM 文件夹", command=self.open_dicom
        )
        self.dicom_button.pack(side="left", padx=4)
        self.label_button = ttk.Button(
            actions, text="加载椎骨标签", command=self.open_label, state="disabled"
        )
        self.label_button.pack(side="left", padx=4)
        self.export_button = ttk.Button(
            actions, text="导出标注图", command=self.export_figure, state="disabled"
        )
        self.export_button.pack(side="left", padx=4)
        ttk.Button(actions, text="术语与帮助", command=self.show_help).pack(
            side="left", padx=4
        )

        body = tk.Frame(self.root, bg=BACKGROUND)
        body.grid(row=1, column=0, sticky="nsew")
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=1)

        self.sidebar = ttk.Frame(body, style="Sidebar.TFrame", width=292)
        self.sidebar.grid(row=0, column=0, sticky="nsw")
        self.sidebar.grid_propagate(False)
        self.sidebar.grid_columnconfigure(0, weight=1)

        self._build_sidebar()

        self.viewer_panel = tk.Frame(body, bg=BACKGROUND)
        self.viewer_panel.grid(row=0, column=1, padx=(10, 10), pady=10, sticky="nsew")
        self.viewer_panel.grid_rowconfigure(1, weight=1)
        self.viewer_panel.grid_columnconfigure(0, weight=1)

        status_bar = tk.Frame(self.viewer_panel, bg=PANEL_ALT, height=38)
        status_bar.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        status_bar.grid_propagate(False)
        status_bar.grid_columnconfigure(0, weight=1)
        self.status_var = tk.StringVar(value="等待导入影像")
        tk.Label(
            status_bar,
            textvariable=self.status_var,
            bg=PANEL_ALT,
            fg=TEXT,
            font=("Segoe UI", 9),
            anchor="w",
        ).grid(row=0, column=0, padx=14, pady=9, sticky="ew")
        self.local_badge = tk.Label(
            status_bar,
            text="● 本地处理",
            bg=PANEL_ALT,
            fg="#4ade80",
            font=("Segoe UI Semibold", 9),
        )
        self.local_badge.grid(row=0, column=1, padx=14)

        footer = tk.Frame(self.root, bg="#3a2507", height=42)
        footer.grid(row=2, column=0, sticky="ew")
        footer.grid_propagate(False)
        tk.Label(
            footer,
            text="⚠  本工具输出仅用于影像查看与解剖定位，不用于临床诊断或治疗决策。",
            bg="#3a2507",
            fg="#fcd34d",
            font=("Segoe UI Semibold", 10),
        ).pack(pady=10)

    def _build_sidebar(self) -> None:
        row = 0
        ttk.Label(self.sidebar, text="当前检查", style="Section.TLabel").grid(
            row=row, column=0, padx=18, pady=(18, 7), sticky="w"
        )
        row += 1
        self.source_var = tk.StringVar(value="尚未加载")
        self.geometry_var = tk.StringVar(value="—")
        self.labels_var = tk.StringVar(value="—")
        for variable in (self.source_var, self.geometry_var, self.labels_var):
            ttk.Label(
                self.sidebar,
                textvariable=variable,
                style="Muted.TLabel",
                justify="left",
                wraplength=252,
            ).grid(row=row, column=0, padx=18, pady=2, sticky="w")
            row += 1

        ttk.Separator(self.sidebar, orient="horizontal").grid(
            row=row, column=0, padx=18, pady=12, sticky="ew"
        )
        row += 1
        ttk.Label(self.sidebar, text="切片导航", style="Section.TLabel").grid(
            row=row, column=0, padx=18, pady=(0, 7), sticky="w"
        )
        row += 1

        self.slice_vars = [tk.IntVar(value=0) for _ in range(3)]
        self.slice_value_vars = [tk.StringVar(value="—") for _ in range(3)]
        self.scales: list[ttk.Scale] = []
        names = ("矢状位  X", "冠状位  Y", "轴位  Z")
        for axis, name in enumerate(names):
            line = tk.Frame(self.sidebar, bg=PANEL)
            line.grid(row=row, column=0, padx=18, sticky="ew")
            tk.Label(
                line, text=name, bg=PANEL, fg=TEXT, font=("Segoe UI", 9)
            ).pack(side="left")
            tk.Label(
                line,
                textvariable=self.slice_value_vars[axis],
                bg=PANEL,
                fg=ACCENT,
                font=("Consolas", 9),
            ).pack(side="right")
            row += 1
            scale = ttk.Scale(
                self.sidebar,
                from_=0,
                to=1,
                orient="horizontal",
                command=lambda value, a=axis: self.on_scale(a, value),
                state="disabled",
            )
            scale.grid(row=row, column=0, padx=18, pady=(1, 8), sticky="ew")
            self.scales.append(scale)
            row += 1

        ttk.Separator(self.sidebar, orient="horizontal").grid(
            row=row, column=0, padx=18, pady=(6, 12), sticky="ew"
        )
        row += 1
        labels_header = tk.Frame(self.sidebar, bg=PANEL)
        labels_header.grid(row=row, column=0, padx=18, sticky="ew")
        tk.Label(
            labels_header,
            text="可见椎体",
            bg=PANEL,
            fg=ACCENT,
            font=("Segoe UI Semibold", 10),
        ).pack(side="left")
        self.show_all_button = ttk.Button(
            labels_header,
            text="显示全部",
            command=self.show_all_labels,
            state="disabled",
        )
        self.show_all_button.pack(side="right")
        row += 1

        list_frame = tk.Frame(self.sidebar, bg=PANEL)
        list_frame.grid(row=row, column=0, padx=18, pady=(8, 8), sticky="nsew")
        list_frame.grid_rowconfigure(0, weight=1)
        list_frame.grid_columnconfigure(0, weight=1)
        self.sidebar.grid_rowconfigure(row, weight=1)
        self.vertebra_list = tk.Listbox(
            list_frame,
            bg="#091624",
            fg=TEXT,
            selectbackground="#0e7490",
            selectforeground="#ffffff",
            activestyle="none",
            borderwidth=0,
            highlightthickness=1,
            highlightbackground="#1d354c",
            font=("Segoe UI", 10),
            exportselection=False,
        )
        self.vertebra_list.grid(row=0, column=0, sticky="nsew")
        self.vertebra_list.bind("<<ListboxSelect>>", self.on_label_selected)
        scrollbar = ttk.Scrollbar(
            list_frame, orient="vertical", command=self.vertebra_list.yview
        )
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.vertebra_list.configure(yscrollcommand=scrollbar.set)
        row += 1

        self.selection_var = tk.StringVar(
            value="选择椎体后，三视图将自动定位并在 3D 中高亮。"
        )
        ttk.Label(
            self.sidebar,
            textvariable=self.selection_var,
            style="Muted.TLabel",
            justify="left",
            wraplength=252,
        ).grid(row=row, column=0, padx=18, pady=(0, 16), sticky="w")

    def _build_figure(self) -> None:
        self.figure = Figure(figsize=(12, 8), dpi=100, facecolor=BACKGROUND)
        grid = self.figure.add_gridspec(
            2, 2, left=0.035, right=0.985, bottom=0.07, top=0.94, wspace=0.08, hspace=0.17
        )
        self.axes_2d = [
            self.figure.add_subplot(grid[0, 0]),
            self.figure.add_subplot(grid[0, 1]),
            self.figure.add_subplot(grid[1, 0]),
        ]
        self.axis_3d = self.figure.add_subplot(grid[1, 1], projection="3d")
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.viewer_panel)
        self.canvas.get_tk_widget().configure(bg=BACKGROUND, highlightthickness=0)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
        self.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.canvas.mpl_connect("button_press_event", self.on_figure_click)

    def _set_empty_state(self) -> None:
        for axis, title in zip(
            self.axes_2d, ("矢状位", "冠状位", "轴位"), strict=True
        ):
            axis.clear()
            axis.set_facecolor("#08131f")
            axis.text(
                0.5,
                0.5,
                f"{title}\n等待影像",
                ha="center",
                va="center",
                color=MUTED,
                fontsize=13,
                transform=axis.transAxes,
            )
            axis.set_axis_off()
        self.axis_3d.clear()
        self.axis_3d.set_facecolor("#08131f")
        self.axis_3d.text2D(
            0.5,
            0.5,
            "3D 解剖视图\n将在加载后生成",
            ha="center",
            va="center",
            color=MUTED,
            fontsize=13,
            transform=self.axis_3d.transAxes,
        )
        self.axis_3d.set_axis_off()
        self.figure.suptitle(
            "导入受支持的脊柱 CT 开始浏览",
            color=TEXT,
            fontsize=13,
            y=0.985,
        )
        self.canvas.draw_idle()

    def _set_loading(self, loading: bool) -> None:
        self.loading = loading
        state = "disabled" if loading else "normal"
        self.open_button.configure(state=state)
        self.dicom_button.configure(state=state)
        if not loading and self.case is not None:
            self.label_button.configure(state="normal")
            self.export_button.configure(state="normal")
        elif loading:
            self.label_button.configure(state="disabled")
            self.export_button.configure(state="disabled")

    def _set_status_threadsafe(self, message: str) -> None:
        self._worker_events.put(("status", message))

    def open_nifti(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.root,
            title="选择 NIfTI CT",
            filetypes=(
                ("NIfTI image", "*.nii.gz *.nii"),
                ("All files", "*.*"),
            ),
        )
        if path:
            self.load_async(Path(path))

    def open_dicom(self) -> None:
        path = filedialog.askdirectory(parent=self.root, title="选择 DICOM 序列文件夹")
        if path:
            self.load_async(Path(path))

    def open_label(self) -> None:
        if self.case is None:
            return
        path = filedialog.askopenfilename(
            parent=self.root,
            title="选择椎骨分割标签",
            filetypes=(
                ("NIfTI label", "*.nii.gz *.nii"),
                ("All files", "*.*"),
            ),
        )
        if path:
            self.load_async(self.case.source_path, Path(path))

    def load_async(self, source_path: Path, label_path: Path | None = None) -> None:
        if self.loading:
            return
        self._set_loading(True)
        self.status_var.set("正在载入影像；较大的 DICOM/NIfTI 文件可能需要片刻…")
        while not self._worker_events.empty():
            try:
                self._worker_events.get_nowait()
            except queue.Empty:
                break

        def worker() -> None:
            try:
                case = load_case(
                    source_path,
                    label_path,
                    progress=self._set_status_threadsafe,
                )
                meshes = build_surface_meshes(
                    case, progress=self._set_status_threadsafe
                )
            except Exception as error:  # display a friendly GUI error boundary
                self._worker_events.put(("error", error))
                return
            self._worker_events.put(("done", case, meshes))

        threading.Thread(target=worker, daemon=True, name="voxelscout-loader").start()
        self.root.after(60, self._poll_worker_events)

    def _poll_worker_events(self) -> None:
        try:
            while True:
                event = self._worker_events.get_nowait()
                if event[0] == "status":
                    self.status_var.set(str(event[1]))
                elif event[0] == "error":
                    self._load_failed(event[1])
                    return
                elif event[0] == "done":
                    self._load_finished(event[1], event[2])
                    return
        except queue.Empty:
            pass
        if self.loading:
            self.root.after(60, self._poll_worker_events)

    def _load_failed(self, error: Exception) -> None:
        self._set_loading(False)
        self.status_var.set("载入失败")
        messagebox.showerror("无法载入影像", str(error), parent=self.root)

    def _load_finished(
        self, case: VolumeCase, meshes: tuple[SurfaceMesh, ...]
    ) -> None:
        self.case = case
        self.meshes = meshes
        self.selected_label = None
        self._set_loading(False)
        self._configure_case_controls()
        self.render_views(rebuild_3d=True)
        label_note = (
            f"已自动识别 {len(case.labels)} 个标签"
            if case.labels
            else "未发现椎骨标签；3D 显示为骨性结构预览"
        )
        self.status_var.set(f"载入完成 · {label_note}")

    def _configure_case_controls(self) -> None:
        assert self.case is not None
        case = self.case
        name = case.source_path.name if case.source_path.is_file() else case.source_path.name + " (DICOM)"
        self.source_var.set(f"影像：{name}")
        shape = " × ".join(str(value) for value in case.shape)
        spacing = " × ".join(f"{value:.2f}" for value in case.spacing_mm)
        self.geometry_var.set(
            f"尺寸：{shape} voxels\n体素：{spacing} mm · {case.orientation}"
        )
        if case.labels:
            self.labels_var.set(
                f"标签：{len(case.labels)} 个椎体 · 已与 CT 对齐"
            )
        else:
            self.labels_var.set("标签：未加载（可使用顶部按钮添加）")

        indices = default_slice_indices(case)
        self._setting_scales = True
        for axis, (scale, index, size) in enumerate(
            zip(self.scales, indices, case.shape, strict=True)
        ):
            scale.configure(from_=0, to=max(0, size - 1), state="normal")
            scale.set(index)
            self.slice_vars[axis].set(index)
            self.slice_value_vars[axis].set(f"{index + 1}/{size}")
        self._setting_scales = False

        self.vertebra_list.delete(0, tk.END)
        self.label_list = list(case.labels)
        for label in self.label_list:
            region = vertebra_region(label)
            self.vertebra_list.insert(
                tk.END, f"  {vertebra_name(label):<8}  {self._region_chinese(region)}"
            )
        if not self.label_list:
            self.vertebra_list.insert(tk.END, "  未加载椎骨标签")
        self.show_all_button.configure(
            state="normal" if self.label_list else "disabled"
        )
        self.selection_var.set(
            "选择椎体后，三视图将自动定位并在 3D 中高亮。"
            if self.label_list
            else "可浏览三视图和骨性 3D 预览；可靠的逐椎体名称需要分割标签。"
        )

    @staticmethod
    def _region_chinese(region: str) -> str:
        return {
            "Cervical": "颈椎",
            "Thoracic": "胸椎",
            "Lumbar": "腰椎",
            "Other": "其他",
        }[region]

    def current_indices(self) -> tuple[int, int, int]:
        return tuple(int(round(scale.get())) for scale in self.scales)

    def on_scale(self, axis: int, value: str) -> None:
        if self.case is None or self._setting_scales:
            return
        index = int(round(float(value)))
        self.slice_vars[axis].set(index)
        self.slice_value_vars[axis].set(f"{index + 1}/{self.case.shape[axis]}")
        self.render_views(rebuild_3d=False)

    def on_label_selected(self, _event: object = None) -> None:
        if self.case is None or not self.label_list:
            return
        selection = self.vertebra_list.curselection()
        if not selection:
            return
        label = self.label_list[selection[0]]
        self.selected_label = label
        centre = self.case.label_geometry[label].centre
        self._setting_scales = True
        for axis, value in enumerate(centre):
            index = int(round(value))
            self.scales[axis].set(index)
            self.slice_vars[axis].set(index)
            self.slice_value_vars[axis].set(
                f"{index + 1}/{self.case.shape[axis]}"
            )
        self._setting_scales = False
        region = vertebra_region(label)
        self.selection_var.set(
            f"已选择 {vertebra_name(label)} · {self._region_chinese(region)}\n"
            "黄色表示当前椎体；滚轮或滑块可继续浏览。"
        )
        self.status_var.set(f"已定位并高亮 {vertebra_name(label)}")
        self.render_views(rebuild_3d=True)

    def show_all_labels(self) -> None:
        if self.case is None:
            return
        self.selected_label = None
        self.vertebra_list.selection_clear(0, tk.END)
        self.selection_var.set("当前显示全部可见椎体；点击列表可单独高亮。")
        self.render_views(rebuild_3d=True)

    def _mask_overlay(self, mask_slice: np.ndarray) -> np.ndarray:
        rgba = np.zeros((*mask_slice.shape, 4), dtype=np.float32)
        for label in (int(value) for value in np.unique(mask_slice) if value != 0):
            colour = (
                SELECTED
                if label == self.selected_label
                else REGION_COLOURS[vertebra_region(label)]
            )
            alpha = 0.78 if label == self.selected_label else 0.38
            rgba[mask_slice == label, :3] = mpl_colors.to_rgb(colour)
            rgba[mask_slice == label, 3] = alpha
        return rgba

    def _draw_2d_axis(self, axis_index: int) -> None:
        assert self.case is not None
        axis = self.axes_2d[axis_index]
        index = self.current_indices()[axis_index]
        names = ("矢状位 · Sagittal", "冠状位 · Coronal", "轴位 · Axial")
        axis.clear()
        axis.set_facecolor("#050c14")
        image_slice = orthogonal_slice(self.case.image, axis_index, index)
        axis.imshow(
            window_ct(image_slice),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        if self.case.mask is not None:
            mask_slice = orthogonal_slice(self.case.mask, axis_index, index)
            axis.imshow(self._mask_overlay(mask_slice), interpolation="nearest")
            self._draw_slice_labels(axis, mask_slice)
        self._draw_crosshair(axis, axis_index)
        axis.set_title(
            f"{names[axis_index]}   {index + 1}/{self.case.shape[axis_index]}",
            color=TEXT,
            fontsize=10,
            pad=8,
        )
        axis.set_axis_off()

    def _draw_slice_labels(self, axis: object, mask_slice: np.ndarray) -> None:
        for label in (int(value) for value in np.unique(mask_slice) if value != 0):
            points = np.argwhere(mask_slice == label)
            if points.size == 0:
                continue
            y, x = points.mean(axis=0)
            axis.text(
                x,
                y,
                vertebra_name(label),
                color="#111827" if label == self.selected_label else "#ffffff",
                fontsize=8,
                fontweight="bold",
                ha="center",
                va="center",
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": SELECTED if label == self.selected_label else "#07111f",
                    "edgecolor": "none",
                    "alpha": 0.88,
                },
            )

    def _draw_crosshair(self, axis: object, axis_index: int) -> None:
        assert self.case is not None
        x, y, z = self.current_indices()
        if axis_index == 0:
            vertical, horizontal = y, self.case.shape[2] - 1 - z
        elif axis_index == 1:
            vertical, horizontal = x, self.case.shape[2] - 1 - z
        else:
            vertical, horizontal = x, self.case.shape[1] - 1 - y
        axis.axvline(vertical, color=ACCENT, alpha=0.5, linewidth=0.7)
        axis.axhline(horizontal, color=ACCENT, alpha=0.5, linewidth=0.7)

    def _draw_3d_axis(self) -> None:
        assert self.case is not None
        axis = self.axis_3d
        axis.clear()
        axis.set_facecolor("#08131f")
        if not self.meshes:
            axis.text2D(
                0.5,
                0.5,
                "当前数据无法生成 3D 表面",
                ha="center",
                va="center",
                color=MUTED,
                transform=axis.transAxes,
            )
        for mesh in self.meshes:
            if mesh.label is None:
                colour, alpha = "#dbeafe", 0.24
            else:
                selected = mesh.label == self.selected_label
                colour = (
                    SELECTED
                    if selected
                    else REGION_COLOURS[vertebra_region(mesh.label)]
                )
                alpha = 0.98 if selected else (0.52 if self.selected_label is None else 0.13)
            vertices = mesh.vertices_mm
            axis.plot_trisurf(
                vertices[:, 0],
                vertices[:, 1],
                vertices[:, 2],
                triangles=mesh.faces,
                color=colour,
                alpha=alpha,
                linewidth=0.0,
                antialiased=False,
                shade=True,
            )

        if self.case.label_geometry:
            spacing = np.asarray(self.case.spacing_mm)
            for label, geometry in self.case.label_geometry.items():
                if self.selected_label is not None and label != self.selected_label:
                    continue
                centre = np.asarray(geometry.centre) * spacing
                axis.text(
                    centre[0],
                    centre[1],
                    centre[2],
                    vertebra_name(label),
                    color=SELECTED if label == self.selected_label else "#ffffff",
                    fontsize=8 if self.selected_label is None else 11,
                    fontweight="bold",
                    ha="center",
                )

        extents = np.asarray(self.case.shape) * np.asarray(self.case.spacing_mm)
        axis.set_xlim(0, extents[0])
        axis.set_ylim(0, extents[1])
        axis.set_zlim(0, extents[2])
        axis.set_box_aspect(np.maximum(extents, 1.0))
        axis.view_init(elev=17, azim=-64)
        title = (
            f"3D 椎骨 · {vertebra_name(self.selected_label)} 已高亮"
            if self.selected_label is not None
            else ("3D 椎骨总览" if self.case.labels else "3D 骨性结构预览")
        )
        axis.set_title(title, color=TEXT, fontsize=10, pad=8)
        axis.set_axis_off()

    def render_views(self, *, rebuild_3d: bool) -> None:
        if self.case is None:
            return
        for axis_index in range(3):
            self._draw_2d_axis(axis_index)
        if rebuild_3d:
            self._draw_3d_axis()
        selected = (
            f" · 当前：{vertebra_name(self.selected_label)}"
            if self.selected_label is not None
            else ""
        )
        self.figure.suptitle(
            f"{self.case.source_path.name}{selected}",
            color=TEXT,
            fontsize=12,
            y=0.985,
        )
        for old_text in tuple(self.figure.texts):
            if old_text.get_gid() == "disclaimer":
                old_text.remove()
        note = self.figure.text(
            0.5,
            0.018,
            "VoxelScout · For viewing and anatomical orientation only · Not for clinical diagnosis",
            ha="center",
            color="#fbbf24",
            fontsize=8,
        )
        note.set_gid("disclaimer")
        self.canvas.draw_idle()

    def on_scroll(self, event: object) -> None:
        if self.case is None or event.inaxes not in self.axes_2d:
            return
        axis_index = self.axes_2d.index(event.inaxes)
        current = int(round(self.scales[axis_index].get()))
        delta = 1 if event.button == "up" else -1
        value = min(max(current + delta, 0), self.case.shape[axis_index] - 1)
        self.scales[axis_index].set(value)
        self.on_scale(axis_index, str(value))

    def on_figure_click(self, event: object) -> None:
        if (
            self.case is None
            or event.inaxes not in self.axes_2d
            or event.xdata is None
            or event.ydata is None
        ):
            return
        axis_index = self.axes_2d.index(event.inaxes)
        horizontal = int(round(event.xdata))
        vertical = int(round(event.ydata))
        x, y, z = self.current_indices()
        if axis_index == 0:
            y = horizontal
            z = self.case.shape[2] - 1 - vertical
        elif axis_index == 1:
            x = horizontal
            z = self.case.shape[2] - 1 - vertical
        else:
            x = horizontal
            y = self.case.shape[1] - 1 - vertical
        values = (x, y, z)
        self._setting_scales = True
        for axis, value in enumerate(values):
            value = min(max(value, 0), self.case.shape[axis] - 1)
            self.scales[axis].set(value)
            self.slice_value_vars[axis].set(
                f"{value + 1}/{self.case.shape[axis]}"
            )
        self._setting_scales = False
        self.render_views(rebuild_3d=False)

    def export_figure(self) -> None:
        if self.case is None:
            return
        initial = f"{self.case.source_path.name.split('.')[0]}_labelled.png"
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="导出标注图",
            defaultextension=".png",
            initialfile=initial,
            filetypes=(("PNG image", "*.png"), ("PDF document", "*.pdf")),
        )
        if not path:
            return
        try:
            self.figure.savefig(path, dpi=180, facecolor=self.figure.get_facecolor())
        except Exception as error:
            messagebox.showerror("导出失败", str(error), parent=self.root)
            return
        self.status_var.set(f"已导出标注图：{path}")
        messagebox.showinfo("导出完成", f"标注图已保存到：\n{path}", parent=self.root)

    def show_help(self) -> None:
        messagebox.showinfo(
            "术语与使用帮助",
            "快速使用\n"
            "1. 打开 NIfTI CT 文件或含一个 DICOM 序列的文件夹。\n"
            "2. VerSe 数据集中的配套标签会自动匹配；也可手动加载标签。\n"
            "3. 使用三个滑块、鼠标滚轮或点击切片移动交叉线。\n"
            "4. 点击椎体名称可定位并在三视图/3D 中高亮。\n\n"
            "常用术语\n"
            "CT — Computed Tomography，计算机断层扫描\n"
            "DICOM — 医学数字成像和通信格式\n"
            "NIfTI — 神经影像信息技术格式\n"
            "ROI — Region of Interest，感兴趣区域\n"
            "矢状位 / 冠状位 / 轴位 — 从侧面 / 正面 / 横断面查看\n\n"
            "隐私与用途\n"
            "影像只在本机读取和处理，不会上传。输出仅用于查看与解剖定位，"
            "不能替代专业医学诊断。",
            parent=self.root,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch the integrated VoxelScout spinal CT viewer."
    )
    parser.add_argument("--image", type=Path, help="Optional NIfTI file or DICOM directory")
    parser.add_argument("--label", type=Path, help="Optional aligned NIfTI vertebra mask")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = tk.Tk()
    ViewerApp(root, initial_image=args.image, initial_label=args.label)
    root.mainloop()


if __name__ == "__main__":
    main()
