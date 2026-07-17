"""Minimal PySide6 + VTK desktop interface centred on interactive 3D anatomy."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pyvista as pv
from PySide6.QtCore import QObject, QPoint, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QToolTip,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor
from vtkmodules.vtkRenderingCore import vtkCellPicker

from voxelscout.anatomy import vertebra_info
from voxelscout.desktop_data import (
    SegmentedCase,
    find_companion_mask,
    load_segmented_case,
)


BACKGROUND = "#09121d"
PANEL = "#111e2b"
PANEL_ALT = "#162738"
TEXT = "#edf4fa"
MUTED = "#91a4b7"
ACCENT = "#31c4b3"
GOLD = "#f4c95d"
HOVER = "#65d9ff"


class CaseLoader(QObject):
    progress = Signal(int, str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, ct_path: Path, mask_path: Path) -> None:
        super().__init__()
        self.ct_path = ct_path
        self.mask_path = mask_path

    @Slot()
    def run(self) -> None:
        try:
            case = load_segmented_case(
                self.ct_path,
                self.mask_path,
                progress=lambda value, message: self.progress.emit(value, message),
            )
        except Exception as error:
            self.failed.emit(str(error))
            return
        self.finished.emit(case)


class VoxelScoutWindow(QMainWindow):
    """One-window 3D vertebra explorer for education and communication."""

    def __init__(self) -> None:
        super().__init__()
        self.case: SegmentedCase | None = None
        self.actor_labels: dict[str, int] = {}
        self.label_actors: dict[int, object] = {}
        self.base_colours: dict[int, str] = {}
        self.hovered_label: int | None = None
        self.selected_label: int | None = None
        self._last_pick_at = 0.0
        self._load_thread: QThread | None = None
        self._load_worker: CaseLoader | None = None

        self.setWindowTitle("VoxelScout · Interactive 3D Spine")
        self.resize(1320, 820)
        self.setMinimumSize(980, 640)
        self._build_ui()
        self._configure_plotter()
        self._show_empty_scene()

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("root")
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        header = QFrame()
        header.setObjectName("header")
        header.setFixedHeight(72)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(24, 12, 22, 12)

        brand = QLabel("VOXEL<span style='color:#31c4b3'>SCOUT</span>")
        brand.setObjectName("brand")
        brand.setTextFormat(Qt.TextFormat.RichText)
        header_layout.addWidget(brand)
        subtitle = QLabel("INTERACTIVE 3D SPINE")
        subtitle.setObjectName("subtitle")
        header_layout.addWidget(subtitle)
        header_layout.addStretch(1)

        self.progress = QProgressBar()
        self.progress.setFixedWidth(230)
        self.progress.setRange(0, 100)
        self.progress.hide()
        header_layout.addWidget(self.progress)

        self.open_button = QPushButton("Open CT + segmentation")
        self.open_button.setObjectName("primaryButton")
        self.open_button.clicked.connect(self.open_case)
        header_layout.addWidget(self.open_button)

        self.reset_button = QPushButton("Reset")
        self.reset_button.setEnabled(False)
        self.reset_button.clicked.connect(self.reset_camera)
        header_layout.addWidget(self.reset_button)

        self.export_button = QPushButton("Export 3D image")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_image)
        header_layout.addWidget(self.export_button)
        outer.addWidget(header)

        content = QWidget()
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(14, 14, 14, 14)
        content_layout.setSpacing(14)

        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(305)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(20, 20, 20, 20)
        sidebar_layout.setSpacing(10)

        scan_heading = QLabel("SCAN DETAILS")
        scan_heading.setObjectName("sectionHeading")
        sidebar_layout.addWidget(scan_heading)
        self.scan_name = QLabel("No scan loaded")
        self.scan_name.setObjectName("scanName")
        self.scan_name.setWordWrap(True)
        sidebar_layout.addWidget(self.scan_name)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setObjectName("divider")
        sidebar_layout.addWidget(divider)

        selected_heading = QLabel("SELECTED VERTEBRA")
        selected_heading.setObjectName("sectionHeading")
        sidebar_layout.addWidget(selected_heading)
        self.selected_code = QLabel("—")
        self.selected_code.setObjectName("selectedCode")
        sidebar_layout.addWidget(self.selected_code)
        self.selected_name = QLabel("Hover over a vertebra")
        self.selected_name.setObjectName("selectedName")
        self.selected_name.setWordWrap(True)
        sidebar_layout.addWidget(self.selected_name)
        self.selected_region = QLabel(
            "Move the pointer over the 3D model to identify anatomy. Click to keep a vertebra selected."
        )
        self.selected_region.setObjectName("region")
        self.selected_region.setWordWrap(True)
        sidebar_layout.addWidget(self.selected_region)
        self.selected_explanation = QLabel("")
        self.selected_explanation.setObjectName("muted")
        self.selected_explanation.setWordWrap(True)
        sidebar_layout.addWidget(self.selected_explanation)
        sidebar_layout.addStretch(1)

        interaction_hint = QLabel(
            "Drag to rotate · Wheel to zoom\nMiddle-drag to pan · Click to select"
        )
        interaction_hint.setObjectName("hint")
        interaction_hint.setWordWrap(True)
        sidebar_layout.addWidget(interaction_hint)
        content_layout.addWidget(sidebar)

        viewer_frame = QFrame()
        viewer_frame.setObjectName("viewerFrame")
        viewer_layout = QVBoxLayout(viewer_frame)
        viewer_layout.setContentsMargins(1, 1, 1, 1)
        self.plotter = QtInteractor(viewer_frame)
        self.plotter.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        viewer_layout.addWidget(self.plotter.interactor)
        content_layout.addWidget(viewer_frame, 1)
        outer.addWidget(content, 1)

        footer = QFrame()
        footer.setObjectName("footer")
        footer.setFixedHeight(42)
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(20, 0, 20, 0)
        self.status = QLabel("Ready")
        self.status.setObjectName("status")
        footer_layout.addWidget(self.status)
        footer_layout.addStretch(1)
        outer.addWidget(footer)

        self.setCentralWidget(root)
        self.setStyleSheet(self._stylesheet())

    @staticmethod
    def _stylesheet() -> str:
        return f"""
        QWidget#root {{ background: {BACKGROUND}; color: {TEXT}; font-family: 'Segoe UI'; }}
        QFrame#header {{ background: #0d1926; border-bottom: 1px solid #203447; }}
        QLabel#brand {{ font-size: 24px; font-weight: 700; color: white; }}
        QLabel#subtitle {{ color: {MUTED}; font-size: 11px; margin-left: 10px; }}
        QPushButton {{ background: {PANEL_ALT}; color: {TEXT}; border: 1px solid #294159;
                       border-radius: 6px; padding: 9px 14px; font-weight: 600; }}
        QPushButton:hover {{ background: #213b51; border-color: #3f627f; }}
        QPushButton:disabled {{ color: #5e7285; background: #111e2a; border-color: #1d2b38; }}
        QPushButton#primaryButton {{ background: #167f76; border-color: #23a99d; color: white; }}
        QPushButton#primaryButton:hover {{ background: #1a998d; }}
        QProgressBar {{ color: {TEXT}; background: #0b1520; border: 1px solid #294159;
                        border-radius: 5px; text-align: center; height: 18px; }}
        QProgressBar::chunk {{ background: {ACCENT}; border-radius: 4px; }}
        QFrame#sidebar {{ background: {PANEL}; border: 1px solid #203447; border-radius: 8px; }}
        QFrame#viewerFrame {{ background: #050a10; border: 1px solid #203447; border-radius: 8px; }}
        QLabel#sectionHeading {{ color: {ACCENT}; font-size: 11px; font-weight: 700; }}
        QLabel#scanName {{ color: white; font-size: 17px; font-weight: 650; }}
        QLabel#selectedCode {{ color: {GOLD}; font-size: 42px; font-weight: 750; }}
        QLabel#selectedName {{ color: white; font-size: 17px; font-weight: 650; }}
        QLabel#region {{ color: {ACCENT}; font-size: 14px; font-weight: 600; }}
        QLabel#muted {{ color: {MUTED}; font-size: 12px; line-height: 1.35; }}
        QLabel#hint {{ color: #7990a5; background: #0b1722; border-radius: 6px; padding: 10px; }}
        QFrame#divider {{ color: #263b4e; background: #263b4e; max-height: 1px; }}
        QFrame#footer {{ background: #0d1926; border-top: 1px solid #203447; }}
        QLabel#status {{ color: {MUTED}; }}
        QLabel#notice {{ color: #f2bd58; font-weight: 600; }}
        """

    def _configure_plotter(self) -> None:
        self.plotter.set_background("#050a10")
        self.plotter.enable_anti_aliasing("fxaa")
        self._picker = vtkCellPicker()
        self._picker.SetTolerance(0.0008)
        self._vtk_interactor().AddObserver("MouseMoveEvent", self._on_mouse_move)
        self._vtk_interactor().AddObserver("LeftButtonPressEvent", self._on_left_click)

    def _vtk_interactor(self):
        wrapper = self.plotter.iren
        return getattr(wrapper, "interactor", wrapper)

    def _show_empty_scene(self) -> None:
        self.plotter.clear()
        self.plotter.set_background("#ffffff")
        self.plotter.render()

    @Slot()
    def open_case(self) -> None:
        ct_name, _ = QFileDialog.getOpenFileName(
            self,
            "Open spinal CT",
            str(Path.cwd() / "data"),
            "NIfTI volumes (*.nii *.nii.gz);;All files (*.*)",
        )
        if not ct_name:
            return
        ct_path = Path(ct_name)
        companion = find_companion_mask(ct_path)
        start = str(companion if companion is not None else ct_path.parent)
        mask_name, _ = QFileDialog.getOpenFileName(
            self,
            "Open matching vertebra segmentation",
            start,
            "NIfTI segmentation (*.nii *.nii.gz);;All files (*.*)",
        )
        if not mask_name:
            QMessageBox.information(
                self,
                "Segmentation required",
                "VoxelScout currently needs a trusted vertebra segmentation mask to build the 3D model.",
            )
            return
        self._start_loading(ct_path, Path(mask_name))

    def _start_loading(self, ct_path: Path, mask_path: Path) -> None:
        if self._load_thread is not None and self._load_thread.isRunning():
            return
        self.open_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.reset_button.setEnabled(False)
        self.progress.setValue(0)
        self.progress.show()
        self.status.setText("Preparing 3D spine…")

        thread = QThread(self)
        worker = CaseLoader(ct_path, mask_path)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_progress)
        worker.finished.connect(self._on_case_loaded)
        worker.finished.connect(thread.quit)
        worker.failed.connect(self._on_load_failed)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._clear_loader)
        thread.finished.connect(thread.deleteLater)
        self._load_thread = thread
        self._load_worker = worker
        thread.start()

    @Slot(int, str)
    def _on_progress(self, value: int, message: str) -> None:
        self.progress.setValue(value)
        self.status.setText(message)

    @Slot(object)
    def _on_case_loaded(self, case: SegmentedCase) -> None:
        self.case = case
        self.progress.hide()
        self.open_button.setEnabled(True)
        self.export_button.setEnabled(True)
        self.reset_button.setEnabled(True)
        self.status.setText(
            f"Ready · {len(case.meshes)} vertebrae · {case.mesh_memory_mib:.1f} MiB mesh data"
        )
        self._populate_scene(case)
        self._update_scan_details(case)
        self._set_selected(None)

    @Slot(str)
    def _on_load_failed(self, message: str) -> None:
        self.progress.hide()
        self.open_button.setEnabled(True)
        self.status.setText("Could not build the 3D spine")
        QMessageBox.critical(self, "Unable to open case", message)

    @Slot()
    def _clear_loader(self) -> None:
        self._load_thread = None
        self._load_worker = None

    def _populate_scene(self, case: SegmentedCase) -> None:
        self.plotter.clear()
        self.plotter.set_background("#050a10")
        self.actor_labels.clear()
        self.label_actors.clear()
        self.base_colours.clear()
        self.hovered_label = None
        self.selected_label = None

        for item in case.meshes:
            vtk_faces = np.column_stack(
                (np.full(len(item.faces), 3, dtype=np.int32), item.faces)
            ).ravel()
            mesh = pv.PolyData(item.vertices, vtk_faces)
            actor = self.plotter.add_mesh(
                mesh,
                color=item.colour,
                smooth_shading=True,
                specular=0.18,
                ambient=0.16,
                pickable=True,
                name=f"vertebra-{item.label}",
            )
            self.actor_labels[self._actor_key(actor)] = item.label
            self.label_actors[item.label] = actor
            self.base_colours[item.label] = item.colour

        self.plotter.add_axes(line_width=2, color="#91a4b7")
        self.plotter.add_text(
            "Hover to identify · Click to keep selected",
            position="upper_left",
            color="#91a4b7",
            font_size=10,
        )
        self.reset_camera()

    def _update_scan_details(self, case: SegmentedCase) -> None:
        self.scan_name.setText(case.name)
        shape = " × ".join(str(value) for value in case.shape)
        spacing = " × ".join(f"{value:.2f}" for value in case.spacing)
        self.scan_details.setText(
            f"Volume: {shape} voxels\n"
            f"Spacing: {spacing} mm\n"
            f"Orientation: {case.orientation}\n"
            f"Visible vertebrae: {len(case.labels)}\n"
            f"CT voxels loaded: no (header only)"
        )

    @staticmethod
    def _actor_key(actor: object) -> str:
        if actor is None:
            return ""
        getter = getattr(actor, "GetAddressAsString", None)
        return getter("") if getter is not None else str(id(actor))

    def _pick_label(self) -> int | None:
        x, y = self._vtk_interactor().GetEventPosition()
        if not self._picker.Pick(x, y, 0, self.plotter.renderer):
            return None
        return self.actor_labels.get(self._actor_key(self._picker.GetActor()))

    def _on_mouse_move(self, _caller: object, _event: str) -> None:
        now = time.monotonic()
        if now - self._last_pick_at < 0.04:
            return
        self._last_pick_at = now
        label = self._pick_label()
        if label == self.hovered_label:
            return
        previous = self.hovered_label
        self.hovered_label = label
        if previous is not None:
            self._refresh_actor(previous)
        if label is not None:
            self._refresh_actor(label)
            x, y = self._vtk_interactor().GetEventPosition()
            point = self.plotter.mapToGlobal(QPoint(int(x), self.plotter.height() - int(y)))
            QToolTip.showText(point, vertebra_info(label).tooltip, self.plotter)
            if self.selected_label is None:
                self._display_info(label, temporary=True)
        else:
            QToolTip.hideText()
            if self.selected_label is None:
                self._set_selected(None)
        self.plotter.render()

    def _on_left_click(self, _caller: object, _event: str) -> None:
        label = self._pick_label()
        if label is None:
            return
        previous = self.selected_label
        self.selected_label = label
        if previous is not None and previous != label:
            self._refresh_actor(previous)
        self._refresh_actor(label)
        self._display_info(label, temporary=False)
        self.status.setText(f"Selected {vertebra_info(label).code}")
        self.plotter.render()

    def _refresh_actor(self, label: int) -> None:
        actor = self.label_actors.get(label)
        if actor is None:
            return
        if label == self.selected_label:
            colour = GOLD
        elif label == self.hovered_label:
            colour = HOVER
        else:
            colour = self.base_colours[label]
        rgb = pv.Color(colour).float_rgb
        actor.GetProperty().SetColor(*rgb)

    def _display_info(self, label: int, *, temporary: bool) -> None:
        info = vertebra_info(label)
        self.selected_code.setText(info.code)
        self.selected_name.setText(info.anatomical_name)
        prefix = "Hovering · " if temporary else "Selected · "
        self.selected_region.setText(prefix + info.region_plain)
        self.selected_explanation.setText(info.explanation)

    def _set_selected(self, label: int | None) -> None:
        self.selected_label = label
        if label is not None:
            self._display_info(label, temporary=False)
            return
        self.selected_code.setText("—")
        self.selected_name.setText("Hover over a vertebra")
        self.selected_region.setText(
            "Move the pointer over the 3D model to identify anatomy. Click to keep a vertebra selected."
        )
        self.selected_explanation.setText("")

    @Slot()
    def reset_camera(self) -> None:
        if self.case is None:
            return
        self.plotter.view_isometric()
        self.plotter.reset_camera()
        self.plotter.render()

    @Slot()
    def export_image(self) -> None:
        if self.case is None:
            return
        default = f"{self.case.name}_3d_spine.png"
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export current 3D view",
            str(Path.cwd() / default),
            "PNG image (*.png)",
        )
        if not filename:
            return
        try:
            self.plotter.screenshot(filename, transparent_background=False)
        except Exception as error:
            QMessageBox.critical(self, "Export failed", str(error))
            return
        self.status.setText(f"Exported {Path(filename).name}")

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API
        if self._load_thread is not None and self._load_thread.isRunning():
            self._load_thread.requestInterruption()
            self._load_thread.quit()
            self._load_thread.wait(5000)
        self.plotter.close()
        event.accept()


def main() -> None:
    parser = argparse.ArgumentParser(description="VoxelScout desktop 3D spine viewer")
    parser.add_argument("--ct", type=Path, help="NIfTI CT volume to open")
    parser.add_argument("--mask", type=Path, help="Matching labelled segmentation")
    args, qt_args = parser.parse_known_args()
    if (args.ct is None) != (args.mask is None):
        parser.error("--ct and --mask must be supplied together")

    app = QApplication.instance() or QApplication([sys.argv[0], *qt_args])
    app.setApplicationName("VoxelScout")
    app.setStyle("Fusion")
    window = VoxelScoutWindow()
    window.show()
    if args.ct is not None and args.mask is not None:
        QTimer.singleShot(0, lambda: window._start_loading(args.ct, args.mask))
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
