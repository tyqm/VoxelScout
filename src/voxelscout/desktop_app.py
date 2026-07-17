"""Minimal PySide6 + VTK desktop interface centred on interactive 3D anatomy."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pyvista as pv
from PySide6.QtCore import QObject, QPoint, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QCloseEvent, QCursor
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


BACKGROUND = "#f9fafc"
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
        self._auto_rotation_paused = False

        self.setWindowTitle("Lenx")
        self.resize(960, 680)
        self.setMinimumSize(760, 520)
        self._build_ui()
        self._configure_plotter()
        self._show_empty_scene()

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("root")
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        content = QWidget()
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(10, 10, 10, 10)
        content_layout.setSpacing(10)

        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(220)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)

        controls_panel = QFrame()
        controls_panel.setObjectName("controlsPanel")
        controls_layout = QVBoxLayout(controls_panel)
        controls_layout.setContentsMargins(18, 18, 18, 18)
        controls_layout.setSpacing(8)

        self.open_button = QPushButton("Open Files")
        self.open_button.setObjectName("rainbowOpen")
        self.open_button.setFixedHeight(40)
        self.open_button.clicked.connect(self.open_case)
        controls_layout.addWidget(self.open_button)

        self.reset_button = QPushButton("Reset")
        self.reset_button.setObjectName("rainbowReset")
        self.reset_button.setFixedHeight(40)
        self.reset_button.setEnabled(False)
        self.reset_button.clicked.connect(self.reset_camera)
        controls_layout.addWidget(self.reset_button)

        self.export_button = QPushButton("Export")
        self.export_button.setObjectName("rainbowExport")
        self.export_button.setFixedHeight(40)
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_image)
        controls_layout.addWidget(self.export_button)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setFixedHeight(16)
        self.progress.hide()
        controls_layout.addWidget(self.progress)
        controls_layout.addStretch(1)
        sidebar_layout.addWidget(controls_panel, 1)

        divider = QFrame()
        divider.setObjectName("sidebarDivider")
        divider.setFixedHeight(1)
        sidebar_layout.addWidget(divider)

        info_panel = QFrame()
        info_panel.setObjectName("infoPanel")
        info_layout = QVBoxLayout(info_panel)
        info_layout.setContentsMargins(18, 18, 18, 18)
        info_layout.setSpacing(8)

        self.selected_code = QLabel("")
        self.selected_code.setObjectName("selectedCode")
        info_layout.addWidget(self.selected_code)
        self.selected_name = QLabel("")
        self.selected_name.setObjectName("selectedName")
        self.selected_name.setWordWrap(True)
        info_layout.addWidget(self.selected_name)
        self.selected_region = QLabel("")
        self.selected_region.setObjectName("region")
        self.selected_region.setWordWrap(True)
        info_layout.addWidget(self.selected_region)
        info_layout.addStretch(1)
        sidebar_layout.addWidget(info_panel, 1)
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

        self.setCentralWidget(root)
        self.setStyleSheet(self._stylesheet())

    @staticmethod
    def _stylesheet() -> str:
        return f"""
        QWidget#root {{ background: {BACKGROUND}; color: {TEXT}; font-family: 'Segoe UI'; }}
        QPushButton {{ background: {PANEL_ALT}; color: {TEXT}; border: 1px solid #294159;
                       border-radius: 6px; padding: 7px 12px; font-weight: 600; }}
        QPushButton#rainbowOpen {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 #ff5f6d, stop:1 #ff8a5b); border: none; color: white; }}
        QPushButton#rainbowReset {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 #ffd166, stop:1 #06d6a0); border: none; color: #0d1926; }}
        QPushButton#rainbowExport {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 #3a86ff, stop:1 #8338ec); border: none; color: white; }}
        QPushButton#rainbowOpen:hover, QPushButton#rainbowReset:hover,
        QPushButton#rainbowExport:hover {{ border: 2px solid rgba(255, 255, 255, 150); }}
        QPushButton#rainbowOpen:disabled, QPushButton#rainbowReset:disabled,
        QPushButton#rainbowExport:disabled {{ color: #68798a; background: #1b2a38; }}
        QProgressBar {{ color: {TEXT}; background: #0b1520; border: 1px solid #294159;
                        border-radius: 5px; text-align: center; height: 18px; }}
        QProgressBar::chunk {{ background: {ACCENT}; border-radius: 4px; }}
        QFrame#sidebar {{ background: {PANEL}; border: 1px solid #203447; border-radius: 8px; }}
        QFrame#controlsPanel, QFrame#infoPanel {{ background: transparent; border: none; }}
        QFrame#sidebarDivider {{ background: #294159; border: none; }}
        QFrame#viewerFrame {{ background: #050a10; border: 1px solid #203447; border-radius: 8px; }}
        QLabel#selectedCode {{ color: {GOLD}; font-size: 42px; font-weight: 750; }}
        QLabel#selectedName {{ color: white; font-size: 17px; font-weight: 650; }}
        QLabel#region {{ color: {ACCENT}; font-size: 14px; font-weight: 600; }}
        """

    def _configure_plotter(self) -> None:
        self.plotter.set_background("#050a10")
        self.plotter.enable_anti_aliasing("fxaa")
        self._picker = vtkCellPicker()
        self._picker.SetTolerance(0.0008)
        self._vtk_interactor().AddObserver("MouseMoveEvent", self._on_mouse_move)
        self._vtk_interactor().AddObserver("LeftButtonPressEvent", self._on_left_click)
        self._rotation_timer = QTimer(self)
        self._rotation_timer.setInterval(50)
        self._rotation_timer.timeout.connect(self._auto_rotate)
        self._rotation_timer.start()

    def _vtk_interactor(self):
        wrapper = self.plotter.iren
        return getattr(wrapper, "interactor", wrapper)

    def _show_empty_scene(self) -> None:
        self.plotter.clear()
        self.plotter.set_background("#050a10")
        self.plotter.render()

    @Slot()
    def _auto_rotate(self) -> None:
        if (
            self.case is None
            or self._auto_rotation_paused
            or not self.isVisible()
            or self.isMinimized()
        ):
            return
        viewport = self.plotter.interactor
        cursor_position = viewport.mapFromGlobal(QCursor.pos())
        if viewport.rect().contains(cursor_position):
            return
        self.plotter.camera.Azimuth(0.35)
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
    def _on_progress(self, value: int, _message: str) -> None:
        self.progress.setValue(value)

    @Slot(object)
    def _on_case_loaded(self, case: SegmentedCase) -> None:
        self.case = case
        self.progress.hide()
        self.open_button.setEnabled(True)
        self.export_button.setEnabled(True)
        self.reset_button.setEnabled(True)
        self._populate_scene(case)
        self._set_selected(None)

    @Slot(str)
    def _on_load_failed(self, message: str) -> None:
        self.progress.hide()
        self.open_button.setEnabled(True)
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
        self.reset_camera()

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
        self.selected_region.setText(info.region_plain)

    def _set_selected(self, label: int | None) -> None:
        self.selected_label = label
        if label is not None:
            self._display_info(label, temporary=False)
            return
        self.selected_code.setText("")
        self.selected_name.setText("")
        self.selected_region.setText("")

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
        self._auto_rotation_paused = True
        try:
            default = f"{self.case.name}_3d_spine.png"
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Export current 3D view",
                str(Path.cwd() / default),
                "PNG image (*.png)",
            )
            if not filename:
                return
            self.plotter.screenshot(filename, transparent_background=False)
        except Exception as error:
            QMessageBox.critical(self, "Export failed", str(error))
        finally:
            self._auto_rotation_paused = False

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API
        self._rotation_timer.stop()
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
