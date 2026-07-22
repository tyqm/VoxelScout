"""Minimal PySide6 + VTK desktop interface centred on interactive 3D anatomy."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pyvista as pv
from PySide6.QtCore import QObject, QPoint, QPointF, QRectF, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import (
    QColor,
    QCloseEvent,
    QCursor,
    QLinearGradient,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QRadialGradient,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QDial,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor
from vtkmodules.vtkRenderingCore import vtkCellPicker

from voxelscout.anatomy import vertebra_info
from voxelscout.appearance import AppearanceMode, colour_for_label
from voxelscout.desktop_data import (
    SegmentedCase,
)
from voxelscout.dicom_import import DicomSeries, convert_dicom_series, discover_ct_series
from voxelscout.ct_review import CTReviewDialog
from voxelscout.inference.backend import SegmentationBackend
from voxelscout.inference.workflow import load_case_for_ct
from voxelscout.spatial_guides import format_scale_length, nice_scale_length


BACKGROUND = "#f9fafc"
PANEL = "#111e2b"
PANEL_ALT = "#162738"
TEXT = "#edf4fa"
MUTED = "#91a4b7"
GOLD = "#f4c95d"
HOVER = "#65d9ff"


class CaseLoader(QObject):
    progress = Signal(int, str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        ct_path: Path,
        mask_path: Path | None = None,
        backend: SegmentationBackend | None = None,
        dicom_series: DicomSeries | None = None,
    ) -> None:
        super().__init__()
        self.ct_path = ct_path
        self.mask_path = mask_path
        self.backend = backend
        self.dicom_series = dicom_series

    @Slot()
    def run(self) -> None:
        try:
            ct_path = self.ct_path
            if self.dicom_series is not None:
                converted = convert_dicom_series(
                    self.dicom_series,
                    progress=lambda value, message: self.progress.emit(value, message),
                )
                ct_path = converted.nifti_path
            case = load_case_for_ct(
                ct_path,
                self.mask_path,
                backend=self.backend,
                progress=lambda value, message: self.progress.emit(value, message),
            )
        except Exception as error:
            self.failed.emit(str(error))
            return
        self.finished.emit(case)


class PillFrame(QFrame):
    """Paint a stable antialiased pill independent of native-window styling."""

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#e7e8ee"))
        painter.drawRoundedRect(QRectF(self.rect()), 21.0, 21.0)


class ScaleBarOverlay(QWidget):
    """Transparent physical scale drawn over the 3D viewport."""

    def __init__(
        self,
        parent: QWidget,
        flags: Qt.WindowType = Qt.WindowType.Widget,
    ) -> None:
        super().__init__(parent, flags)
        self._length_mm = 0.0
        self._pixels = 0.0
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setFixedSize(150, 44)

    def set_scale(self, length_mm: float, pixels: float) -> None:
        if np.isclose(self._length_mm, length_mm) and np.isclose(self._pixels, pixels):
            return
        self._length_mm = length_mm
        self._pixels = pixels
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt API
        if self._length_mm <= 0 or self._pixels <= 0:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor(238, 244, 249, 225), 2))
        right = self.width() - 8.0
        left = right - min(self._pixels, self.width() - 18.0)
        baseline = self.height() - 9.0
        painter.drawLine(QPointF(left, baseline), QPointF(right, baseline))
        painter.drawLine(QPointF(left, baseline - 4), QPointF(left, baseline + 4))
        painter.drawLine(QPointF(right, baseline - 4), QPointF(right, baseline + 4))
        text = format_scale_length(self._length_mm)
        painter.setPen(QColor(238, 244, 249, 235))
        painter.drawText(QPointF(right - 48, baseline - 8), text)


class AppearanceDial(QDial):
    """A tactile three-position dial constrained to a 180-degree arc."""

    def __init__(self) -> None:
        super().__init__()
        self.setRange(0, 2)
        self.setSingleStep(1)
        self.setPageStep(1)
        self.setWrapping(False)
        self.setFixedSize(112, 104)

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        centre = QPointF(self.width() / 2, 57)
        radius = 37.0

        arc_rect = QRectF(
            centre.x() - radius - 8,
            centre.y() - radius - 8,
            (radius + 8) * 2,
            (radius + 8) * 2,
        )
        painter.setPen(QPen(QColor("#4d91c2"), 2.0))
        painter.drawArc(arc_rect, 0, 180 * 16)

        for value in range(3):
            point = self._point_for_value(value, centre, radius + 10)
            inner = self._point_for_value(value, centre, radius + 3)
            painter.setPen(QPen(QColor("#8395a5"), 1.4))
            painter.drawLine(inner, point)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 95))
        painter.drawEllipse(QPointF(centre.x() + 2, centre.y() + 4), radius + 3, radius + 3)

        bezel = QLinearGradient(
            centre.x() - radius,
            centre.y() - radius,
            centre.x() + radius,
            centre.y() + radius,
        )
        bezel.setColorAt(0.0, QColor("#eef1f3"))
        bezel.setColorAt(0.48, QColor("#aab0b5"))
        bezel.setColorAt(1.0, QColor("#626b72"))
        painter.setBrush(bezel)
        painter.setPen(QPen(QColor("#dce3e8"), 1.2))
        painter.drawEllipse(centre, radius + 2, radius + 2)

        face = QRadialGradient(
            QPointF(centre.x() - 12, centre.y() - 14),
            radius * 1.35,
            centre,
        )
        face.setColorAt(0.0, QColor("#f1eee5"))
        face.setColorAt(0.52, QColor("#cbc8c0"))
        face.setColorAt(1.0, QColor("#8d9295"))
        painter.setBrush(face)
        painter.setPen(QPen(QColor("#67727b"), 1.2))
        painter.drawEllipse(centre, radius - 2, radius - 2)

        marker = self._point_for_value(self.value(), centre, radius * 0.67)
        painter.setBrush(QColor(0, 0, 0, 80))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(marker.x() + 1.5, marker.y() + 2), 8.0, 8.0)
        marker_fill = QRadialGradient(
            QPointF(marker.x() - 2, marker.y() - 3), 10.0, marker
        )
        marker_fill.setColorAt(0.0, QColor("#f5f3ee"))
        marker_fill.setColorAt(1.0, QColor("#8d9294"))
        painter.setBrush(marker_fill)
        painter.setPen(QPen(QColor("#62696e"), 1.0))
        painter.drawEllipse(marker, 7.5, 7.5)

    @staticmethod
    def _point_for_value(value: int, centre: QPointF, radius: float) -> QPointF:
        angle = np.pi + (int(value) / 2.0) * np.pi
        return QPointF(
            centre.x() + np.cos(angle) * radius,
            centre.y() + np.sin(angle) * radius,
        )

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API
        self._set_value_from_position(event.position())
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._set_value_from_position(event.position())
            event.accept()

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802 - Qt API
        step = 1 if event.angleDelta().y() > 0 else -1
        self.setValue(min(self.maximum(), max(self.minimum(), self.value() + step)))
        event.accept()

    def _set_value_from_position(self, position: QPointF) -> None:
        centre = QPointF(self.width() / 2, 57)
        dx = position.x() - centre.x()
        dy = position.y() - centre.y()
        angle = np.degrees(np.arctan2(-dy, dx))
        if angle < 0:
            value = 0 if dx < 0 else 2
        else:
            value = round((180.0 - min(angle, 180.0)) / 90.0)
        self.setValue(min(2, max(0, value)))


class LenxWindow(QMainWindow):
    """One-window 3D vertebra explorer for education and communication."""

    def __init__(self, backend: SegmentationBackend | None = None) -> None:
        super().__init__()
        self._backend = backend
        self.appearance_mode = AppearanceMode.LABELS
        self.case: SegmentedCase | None = None
        self.actor_labels: dict[str, int] = {}
        self.label_actors: dict[int, object] = {}
        self.base_colours: dict[int, str] = {}
        self.hovered_label: int | None = None
        self.selected_label: int | None = None
        self._last_pick_at = 0.0
        self._load_thread: QThread | None = None
        self._load_worker: CaseLoader | None = None
        self._zoom_limits: tuple[float, float] | None = None
        self._pan_limits: tuple[np.ndarray, np.ndarray] | None = None
        self._review_window: CTReviewDialog | None = None

        self.setWindowTitle("Lenx")
        self.setFixedSize(960, 680)
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

        sidebar = QWidget()
        sidebar.setFixedWidth(220)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(10)

        controls_panel = QFrame()
        controls_panel.setObjectName("controlsPanel")
        controls_layout = QVBoxLayout(controls_panel)
        controls_layout.setContentsMargins(18, 18, 18, 18)
        controls_layout.setSpacing(8)

        self.open_button = QPushButton("Open CT")
        self.open_button.setObjectName("redButton")
        self.open_button.setFixedHeight(40)
        open_menu = QMenu(self.open_button)
        open_menu.addAction("NIfTI file", self.open_nifti)
        open_menu.addAction("DICOM folder", self.open_dicom)
        self.open_button.setMenu(open_menu)
        controls_layout.addWidget(self.open_button)

        self.reset_button = QPushButton("Reset")
        self.reset_button.setObjectName("yellowButton")
        self.reset_button.setFixedHeight(40)
        self.reset_button.setEnabled(False)
        self.reset_button.clicked.connect(self.reset_camera)
        controls_layout.addWidget(self.reset_button)

        self.review_button = QPushButton("Review CT")
        self.review_button.setObjectName("purpleButton")
        self.review_button.setFixedHeight(40)
        self.review_button.setEnabled(False)
        self.review_button.clicked.connect(self.review_ct)
        controls_layout.addWidget(self.review_button)

        appearance_title = QLabel("Appearance")
        appearance_title.setObjectName("appearanceTitle")
        controls_layout.addWidget(appearance_title)
        self.appearance_dial = AppearanceDial()
        self.appearance_dial.setObjectName("appearanceDial")
        self.appearance_dial.setAccessibleName("Appearance preset")
        self.appearance_dial.setValue(tuple(AppearanceMode).index(self.appearance_mode))
        self.appearance_dial.valueChanged.connect(self._set_appearance_mode)
        controls_layout.addWidget(
            self.appearance_dial, 0, Qt.AlignmentFlag.AlignHCenter
        )
        dial_labels_widget = QWidget()
        dial_labels_widget.setFixedWidth(150)
        dial_labels = QHBoxLayout(dial_labels_widget)
        dial_labels.setContentsMargins(0, 0, 0, 0)

        controls_layout.addWidget(
            dial_labels_widget, 0, Qt.AlignmentFlag.AlignHCenter
        )

        controls_layout.addStretch(1)
        sidebar_layout.addWidget(controls_panel, 1)

        info_panel = QFrame()
        info_panel.setObjectName("infoPanel")
        info_layout = QVBoxLayout(info_panel)
        info_layout.setContentsMargins(18, 18, 18, 18)
        info_layout.setSpacing(8)
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
        self.scale_bar = ScaleBarOverlay(
            self,
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowDoesNotAcceptFocus,
        )
        self.scale_bar.hide()

        self.info_pill = PillFrame(
            self,
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowDoesNotAcceptFocus,
        )
        self.info_pill.setObjectName("infoPill")
        self.info_pill.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.info_pill.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.info_pill.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.info_pill.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.info_pill.setFixedHeight(42)
        self.info_pill.setMaximumWidth(580)
        pill_layout = QHBoxLayout(self.info_pill)
        pill_layout.setContentsMargins(18, 0, 18, 0)
        pill_layout.setSpacing(10)
        self.pill_code = QLabel("")
        self.pill_code.setObjectName("pillCode")
        pill_layout.addWidget(self.pill_code)
        self.pill_name = QLabel("")
        self.pill_name.setObjectName("pillName")
        pill_layout.addWidget(self.pill_name)
        self.pill_region = QLabel("")
        self.pill_region.setObjectName("pillRegion")
        pill_layout.addWidget(self.pill_region)
        self.info_pill.hide()
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
        QPushButton#redButton {{ background: #ef476f; border: none; color: white; }}
        QPushButton#yellowButton {{ background: #ffd166; border: none; color: #0d1926; }}
        QPushButton#purpleButton {{ background: #9b5de5; border: none; color: white; }}
        QPushButton#redButton:hover, QPushButton#yellowButton:hover,
        QPushButton#purpleButton:hover {{ border: 2px solid rgba(255, 255, 255, 150); }}
        QPushButton#redButton:disabled, QPushButton#yellowButton:disabled,
        QPushButton#purpleButton:disabled {{ color: #68798a; background: #1b2a38; }}
        QFrame#controlsPanel, QFrame#infoPanel {{ background: {PANEL};
            border: 1px solid #203447; border-radius: 8px; }}
        QFrame#viewerFrame {{ background: #050a10; border: 1px solid #203447; border-radius: 8px; }}
        QFrame#infoPill {{ background: #e7e8ee; border: none; border-radius: 21px; }}
        QLabel#pillCode {{ color: #252a34; font-size: 15px; font-weight: 700; }}
        QLabel#pillName {{ color: #3e4552; font-size: 13px; font-weight: 600; }}
        QLabel#pillRegion {{ color: #747d8c; font-size: 12px; font-weight: 500; }}
        QLabel#appearanceTitle {{ color: {MUTED}; font-size: 12px; font-weight: 600; }}
        QLabel#dialLabel {{ color: #93a4b5; font-size: 9px; }}
        """

    def _configure_plotter(self) -> None:
        self.plotter.set_background("#050a10")
        self.plotter.enable_anti_aliasing("fxaa")
        self._picker = vtkCellPicker()
        self._picker.SetTolerance(0.0008)
        self._vtk_interactor().AddObserver("MouseMoveEvent", self._on_mouse_move)
        self._vtk_interactor().AddObserver("LeftButtonPressEvent", self._on_left_click)
        self._vtk_interactor().AddObserver("InteractionEvent", self._on_interaction)
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
        self.scale_bar.hide()
        self.plotter.render()

    @Slot()
    def _auto_rotate(self) -> None:
        if (
            self.case is None
            or not self.isVisible()
            or self.isMinimized()
        ):
            return
        self._clamp_zoom()
        self._clamp_pan()
        self._update_3d_scale()
        viewport = self.plotter.interactor
        hovered_widget = QApplication.widgetAt(QCursor.pos())
        if hovered_widget is viewport or (
            hovered_widget is not None and viewport.isAncestorOf(hovered_widget)
        ):
            return
        self.plotter.camera.Azimuth(0.35)
        self.plotter.render()

    def _clamp_zoom(self) -> None:
        if self._zoom_limits is None:
            return
        camera = self.plotter.camera
        scale = float(camera.GetParallelScale())
        minimum, maximum = self._zoom_limits
        clamped = min(max(scale, minimum), maximum)
        if np.isclose(scale, clamped):
            return
        camera.SetParallelScale(clamped)
        self.plotter.reset_camera_clipping_range()
        self.plotter.render()

    def _update_3d_scale(self) -> None:
        if self.case is None or (
            self._review_window is not None and self._review_window.isVisible()
        ):
            self.scale_bar.hide()
            return
        viewport = self.plotter.interactor
        height = max(viewport.height(), 1)
        # Parallel projection gives one unambiguous world-mm scale across the view.
        mm_per_pixel = 2.0 * float(self.plotter.camera.GetParallelScale()) / height
        length, pixels = nice_scale_length(mm_per_pixel, 125.0)
        self.scale_bar.set_scale(length, pixels)
        origin = viewport.mapToGlobal(QPoint(0, 0))
        self.scale_bar.move(
            origin.x() + max(8, viewport.width() - self.scale_bar.width() - 10),
            origin.y() + max(8, viewport.height() - self.scale_bar.height() - 10),
        )
        self.scale_bar.show()
        self.scale_bar.raise_()

    def _clamp_pan(self) -> None:
        if self._pan_limits is None:
            return
        camera = self.plotter.camera
        focal_point = np.asarray(camera.GetFocalPoint(), dtype=float)
        lower, upper = self._pan_limits
        clamped = np.clip(focal_point, lower, upper)
        if np.allclose(focal_point, clamped):
            return
        position = np.asarray(camera.GetPosition(), dtype=float)
        shift = clamped - focal_point
        camera.SetFocalPoint(*clamped)
        camera.SetPosition(*(position + shift))
        self.plotter.reset_camera_clipping_range()

    def _on_interaction(self, _caller: object, _event: str) -> None:
        """Keep camera zoom and pan within useful model-relative bounds."""
        self._clamp_zoom()
        self._clamp_pan()
        self._update_3d_scale()

    @Slot()
    def open_case(self) -> None:
        """Compatibility entry point used by callers that trigger Open CT."""
        self.open_nifti()

    @Slot()
    def open_nifti(self) -> None:
        ct_name, _ = QFileDialog.getOpenFileName(
            self,
            "Open spinal CT",
            str(Path.cwd() / "data"),
            "NIfTI volumes (*.nii *.nii.gz);;All files (*.*)",
        )
        if not ct_name:
            return
        self._start_loading(Path(ct_name))

    @Slot()
    def open_dicom(self) -> None:
        folder_name = QFileDialog.getExistingDirectory(
            self,
            "Open DICOM CT folder",
            str(Path.cwd() / "data"),
        )
        if not folder_name:
            return
        self._show_loading_pill(0, "Reading DICOM series")
        QApplication.processEvents()
        try:
            candidates = discover_ct_series(Path(folder_name))
        except Exception as error:
            self.info_pill.hide()
            QMessageBox.critical(self, "Unable to open DICOM", str(error))
            return
        selected = candidates[0]
        if len(candidates) > 1:
            self.info_pill.hide()
            choices = [candidate.display_name for candidate in candidates]
            choice, accepted = QInputDialog.getItem(
                self,
                "Choose CT series",
                "Multiple CT series were found:",
                choices,
                0,
                False,
            )
            if not accepted:
                self.info_pill.hide()
                return
            selected = candidates[choices.index(choice)]
        self._start_loading(selected.directory, dicom_series=selected)

    def _start_loading(
        self,
        ct_path: Path,
        mask_path: Path | None = None,
        dicom_series: DicomSeries | None = None,
    ) -> None:
        if self._load_thread is not None and self._load_thread.isRunning():
            return
        if self._review_window is not None:
            self._review_window.close()
            self._review_window = None
        self.open_button.setEnabled(False)
        self.reset_button.setEnabled(False)
        self.review_button.setEnabled(False)
        self._show_loading_pill(0, "Reading CT")

        thread = QThread(self)
        worker = CaseLoader(ct_path, mask_path, self._backend, dicom_series)
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
        self._show_loading_pill(value, message)

    @Slot(object)
    def _on_case_loaded(self, case: SegmentedCase) -> None:
        if self._review_window is not None:
            self._review_window.close()
            self._review_window = None
        self.case = case
        self.info_pill.hide()
        self.open_button.setEnabled(True)
        self.reset_button.setEnabled(True)
        self.review_button.setEnabled(True)
        self._populate_scene(case)
        self._set_selected(None)

    @Slot(str)
    def _on_load_failed(self, message: str) -> None:
        self.info_pill.hide()
        self.open_button.setEnabled(True)
        self.reset_button.setEnabled(self.case is not None)
        self.review_button.setEnabled(self.case is not None)
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
            mesh_colour = colour_for_label(item.label, self.appearance_mode)
            vtk_faces = np.column_stack(
                (np.full(len(item.faces), 3, dtype=np.int32), item.faces)
            ).ravel()
            mesh = pv.PolyData(item.vertices, vtk_faces)
            actor = self.plotter.add_mesh(
                mesh,
                color=mesh_colour,
                smooth_shading=True,
                specular=0.18,
                ambient=0.16,
                pickable=True,
                name=f"vertebra-{item.label}",
            )
            self.actor_labels[self._actor_key(actor)] = item.label
            self.label_actors[item.label] = actor
            self.base_colours[item.label] = mesh_colour

        self.plotter.add_axes(
            line_width=2,
            color="#91a4b7",
            xlabel="R",
            ylabel="A",
            zlabel="S",
        )
        self.reset_camera()

    @Slot(int)
    def _set_appearance_mode(self, identifier: int) -> None:
        modes = tuple(AppearanceMode)
        if not 0 <= identifier < len(modes):
            return
        self.appearance_mode = modes[identifier]
        for label in self.label_actors:
            self.base_colours[label] = colour_for_label(label, self.appearance_mode)
            self._refresh_actor(label)
        if self.case is not None:
            self.plotter.render()

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
            if self.selected_label is None:
                self._display_info(label, temporary=True)
        else:
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
        self.pill_code.setText(info.code)
        self.pill_name.setText(info.anatomical_name)
        self.pill_region.setText(info.region_plain)
        self.info_pill.adjustSize()
        self._position_info_pill()
        if self._review_window is not None and self._review_window.isVisible():
            self.info_pill.hide()
            if not temporary:
                self._review_window.set_selected_label(
                    label, jump=True, notify=False
                )
            return
        self.info_pill.show()
        self.info_pill.raise_()

    def _show_loading_pill(self, value: int, message: str) -> None:
        self.pill_code.setText("Loading")
        self.pill_name.setText(message)
        self.pill_region.setText(f"{min(100, max(0, int(value)))}%")
        self.info_pill.adjustSize()
        self._position_info_pill()
        self.info_pill.show()
        self.info_pill.raise_()

    def _show_case_summary(self) -> None:
        """The reserved lower-left panel intentionally stays empty."""

    def _position_info_pill(self) -> None:
        viewport = self.plotter.interactor
        origin = viewport.mapToGlobal(QPoint(0, 0))
        pill_x = origin.x() + max(16, (viewport.width() - self.info_pill.width()) // 2)
        self.info_pill.move(pill_x, origin.y() + 16)

    def _set_selected(self, label: int | None) -> None:
        previous = self.selected_label
        self.selected_label = label
        if previous is not None and previous != label:
            self._refresh_actor(previous)
        if label is not None:
            self._refresh_actor(label)
            self._display_info(label, temporary=False)
            return
        self.info_pill.hide()
        self._show_case_summary()

    def _clear_selection(self) -> None:
        affected = {self.selected_label, self.hovered_label}
        self.selected_label = None
        self.hovered_label = None
        for label in affected:
            if label is not None:
                self._refresh_actor(label)
        self.info_pill.hide()
        self._show_case_summary()

    def moveEvent(self, event: object) -> None:  # noqa: N802 - Qt API
        super().moveEvent(event)
        if hasattr(self, "info_pill") and self.info_pill.isVisible():
            QTimer.singleShot(0, self._position_info_pill)
        if hasattr(self, "scale_bar") and self.scale_bar.isVisible():
            QTimer.singleShot(0, self._update_3d_scale)

    @Slot()
    def reset_camera(self) -> None:
        if self.case is None:
            return
        self._clear_selection()
        self.plotter.view_isometric()
        self.plotter.camera.SetParallelProjection(True)
        self.plotter.reset_camera()
        default_scale = float(self.plotter.camera.GetParallelScale())
        self._zoom_limits = (default_scale * 0.55, default_scale * 1.8)
        bounds = np.asarray(self.plotter.bounds, dtype=float).reshape(3, 2)
        centre = bounds.mean(axis=1)
        allowance = np.maximum(bounds[:, 1] - bounds[:, 0], 1.0) * 0.35
        self._pan_limits = (centre - allowance, centre + allowance)
        self._update_3d_scale()
        self.plotter.render()

    @Slot()
    def review_ct(self) -> None:
        if self.case is None:
            return
        try:
            self.scale_bar.hide()
            self.info_pill.hide()
            if self._review_window is not None:
                self._review_window.showNormal()
                self._review_window.raise_()
                self._review_window.activateWindow()
                return
            self._review_window = CTReviewDialog(
                self.case.ct_path,
                self,
                mask_path=self.case.mask_path,
                selected_label=self.selected_label,
                on_label_selected=self._select_from_review,
            )
            self._review_window.finished.connect(self._on_review_closed)
            self._review_window.show()
            self._review_window.raise_()
        except Exception as error:
            self._update_3d_scale()
            QMessageBox.critical(self, "Unable to review CT", str(error))

    @Slot(int)
    def _on_review_closed(self, _result: int) -> None:
        self._review_window = None
        self._update_3d_scale()
        if self.selected_label is not None:
            self._display_info(self.selected_label, temporary=False)

    def _select_from_review(self, label: int | None) -> None:
        if label is not None and label not in self.label_actors:
            return
        self._set_selected(label)
        self.plotter.render()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API
        self._rotation_timer.stop()
        self.info_pill.hide()
        self.scale_bar.hide()
        if self._load_thread is not None and self._load_thread.isRunning():
            self._load_thread.requestInterruption()
            self._load_thread.quit()
            self._load_thread.wait(5000)
        self.plotter.close()
        event.accept()


def main() -> None:
    parser = argparse.ArgumentParser(description="Lenx desktop 3D spine viewer")
    parser.add_argument("--ct", type=Path, help="NIfTI CT volume or DICOM folder")
    parser.add_argument("--mask", type=Path, help="Matching labelled segmentation")
    args, qt_args = parser.parse_known_args()
    if args.mask is not None and args.ct is None:
        parser.error("--mask requires --ct")

    app = QApplication.instance() or QApplication([sys.argv[0], *qt_args])
    app.setApplicationName("Lenx")
    app.setStyle("Fusion")
    window = LenxWindow()
    window.show()
    if args.ct is not None:
        def start_requested_case() -> None:
            if args.ct.is_dir():
                series = discover_ct_series(args.ct)[0]
                window._start_loading(series.directory, dicom_series=series)
            else:
                window._start_loading(args.ct, args.mask)

        QTimer.singleShot(0, start_requested_case)
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
