"""Small independent window for display-only CT review."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from voxelscout.ct_display import (
    DisplaySettings,
    automatic_window,
    inspect_hu,
    render_ct_slice,
)
from voxelscout.spatial_guides import (
    axial_edge_labels,
    format_scale_length,
    nice_scale_length,
)


class CTImageView(QLabel):
    """CT image label with orientation and physical-scale overlays."""

    def __init__(
        self,
        orientation: tuple[str, str, str],
        spacing: tuple[float, float, float],
    ) -> None:
        super().__init__()
        self._edges = axial_edge_labels(orientation)
        self._spacing = spacing
        self._source_width = 1
        self.setMinimumSize(320, 320)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background: black;")

    def set_ct_image(self, image: QImage, source_width: int) -> None:
        self._source_width = source_width
        self.setPixmap(
            QPixmap.fromImage(image).scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def paintEvent(self, event: object) -> None:  # noqa: N802 - Qt API
        super().paintEvent(event)
        pixmap = self.pixmap()
        if pixmap is None or pixmap.isNull():
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(
            (self.width() - pixmap.width()) / 2,
            (self.height() - pixmap.height()) / 2,
            pixmap.width(),
            pixmap.height(),
        )
        self._draw_text(painter, rect.left() + 10, rect.top() + 19, "Axial")
        self._draw_text(painter, rect.left() + 10, rect.center().y(), self._edges["left"])
        self._draw_text(painter, rect.right() - 17, rect.center().y(), self._edges["right"])
        self._draw_text(painter, rect.center().x() - 4, rect.top() + 19, self._edges["top"])
        self._draw_text(painter, rect.center().x() - 4, rect.bottom() - 8, self._edges["bottom"])

        screen_pixels_per_voxel = pixmap.width() / max(self._source_width, 1)
        mm_per_pixel = self._spacing[0] / screen_pixels_per_voxel
        length, pixels = nice_scale_length(mm_per_pixel, min(120.0, rect.width() * 0.35))
        right = rect.right() - 12
        baseline = rect.bottom() - 15
        left = right - pixels
        pen = QPen(QColor(242, 246, 250, 225), 2)
        painter.setPen(pen)
        painter.drawLine(QPointF(left, baseline), QPointF(right, baseline))
        painter.drawLine(QPointF(left, baseline - 4), QPointF(left, baseline + 4))
        painter.drawLine(QPointF(right, baseline - 4), QPointF(right, baseline + 4))
        self._draw_text(
            painter,
            right - 42,
            baseline - 7,
            format_scale_length(length),
        )

    @staticmethod
    def _draw_text(painter: QPainter, x: float, y: float, text: str) -> None:
        painter.setPen(QColor(0, 0, 0, 190))
        painter.drawText(QPointF(x + 1, y + 1), text)
        painter.setPen(QColor(242, 246, 250, 235))
        painter.drawText(QPointF(x, y), text)


class CTReviewDialog(QDialog):
    """Axial before/after viewer whose transforms never leave this window."""

    def __init__(self, ct_path: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Lenx — Review CT")
        image = nib.as_closest_canonical(nib.load(str(Path(ct_path).resolve())))
        if len(image.shape) != 3:
            raise ValueError(f"Expected a 3D CT volume, got shape {image.shape}")
        self._volume = np.asanyarray(image.dataobj)
        self._orientation = tuple(str(code) for code in nib.aff2axcodes(image.affine))
        self._spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
        self._report = inspect_hu(self._volume)
        center, width = automatic_window(self._volume)
        self._build_ui(center, width)
        self.setFixedSize(self.minimumSizeHint())
        self._update_controls()
        self._render()

    def _build_ui(self, center: float, width: float) -> None:
        layout = QVBoxLayout(self)
        images = QHBoxLayout()
        self.before = self._image_panel("Before")
        self.after = self._image_panel("After")
        images.addWidget(self.before[0], 1)
        images.addWidget(self.after[0], 1)
        layout.addLayout(images, 1)

        self.slice_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.setRange(0, self._volume.shape[2] - 1)
        self.slice_slider.setValue(self._volume.shape[2] // 2)
        self.slice_slider.valueChanged.connect(self._render)
        slice_row = QHBoxLayout()
        slice_row.addWidget(self.slice_slider, 1)
        self.slice_position = QLabel()
        slice_row.addWidget(self.slice_position)
        layout.addLayout(slice_row)

        controls = QHBoxLayout()
        window_form = QFormLayout()
        self.auto_window = QCheckBox("Auto window")
        self.auto_window.setChecked(True)
        self.auto_window.toggled.connect(self._update_controls)
        window_form.addRow(self.auto_window)
        self.center = self._spin(-10000, 10000, center, 10)
        self.width = self._spin(1, 20000, width, 10)
        window_form.addRow("Center", self.center)
        window_form.addRow("Width", self.width)
        self.clip = QCheckBox("Clip −1024…3071 HU")
        self.clip.setChecked(True)
        self.clip.toggled.connect(self._render)
        window_form.addRow(self.clip)
        controls.addLayout(window_form)

        transform_form = QFormLayout()
        self.transform = QComboBox()
        self.transform.addItems(("None", "Gamma", "Sigmoid"))
        self.transform.currentTextChanged.connect(self._update_controls)
        transform_form.addRow("Transform", self.transform)
        self.gamma = self._spin(0.1, 4.0, 0.8, 0.1)
        transform_form.addRow("Gamma", self.gamma)
        self.sigmoid_gain = self._spin(0.1, 30.0, 8.0, 0.5)
        transform_form.addRow("Sigmoid gain", self.sigmoid_gain)
        self.clahe = QCheckBox("CLAHE")
        self.clahe.toggled.connect(self._update_controls)
        transform_form.addRow(self.clahe)
        self.clahe_limit = self._spin(0.001, 0.2, 0.01, 0.005, decimals=3)
        transform_form.addRow("CLAHE limit", self.clahe_limit)
        controls.addLayout(transform_form)
        controls.addStretch(1)
        layout.addLayout(controls)

        report = self._report
        self.summary = QLabel(
            f"HU {report.finite_min:g}…{report.finite_max:g}   "
            f"non-finite {report.nonfinite_count}   "
            f"outside clip {report.below_clip_count + report.above_clip_count}"
        )
        self.summary.setStyleSheet("color: #68798a;")
        layout.addWidget(self.summary)

        for widget in (
            self.center,
            self.width,
            self.gamma,
            self.sigmoid_gain,
            self.clahe_limit,
        ):
            widget.valueChanged.connect(self._render)

    def _image_panel(self, title: str) -> tuple[QWidget, CTImageView]:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        heading = QLabel(title)
        heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        view = CTImageView(self._orientation, self._spacing)
        layout.addWidget(heading)
        layout.addWidget(view, 1)
        return panel, view

    @staticmethod
    def _spin(
        minimum: float,
        maximum: float,
        value: float,
        step: float,
        *,
        decimals: int = 1,
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.setValue(value)
        return spin

    def _settings(self) -> DisplaySettings:
        return DisplaySettings(
            window_center=self.center.value(),
            window_width=self.width.value(),
            clip_enabled=self.clip.isChecked(),
            transform=self.transform.currentText(),
            gamma=self.gamma.value(),
            sigmoid_gain=self.sigmoid_gain.value(),
            clahe_enabled=self.clahe.isChecked(),
            clahe_clip_limit=self.clahe_limit.value(),
        )

    def _update_controls(self) -> None:
        self.center.setEnabled(not self.auto_window.isChecked())
        self.width.setEnabled(not self.auto_window.isChecked())
        self.gamma.setEnabled(self.transform.currentText() == "Gamma")
        self.sigmoid_gain.setEnabled(self.transform.currentText() == "Sigmoid")
        self.clahe_limit.setEnabled(self.clahe.isChecked())
        self._render()

    def _render(self, *_args: object) -> None:
        if not hasattr(self, "before"):
            return
        index = self.slice_slider.value()
        self.slice_position.setText(f"Slice {index + 1} / {self._volume.shape[2]}")
        axial = self._volume[:, :, index]
        settings = self._settings()
        before = render_ct_slice(axial, settings.before_settings())
        after = render_ct_slice(axial, settings)
        self._set_image(self.before[1], before)
        self._set_image(self.after[1], after)

    @staticmethod
    def _set_image(label: CTImageView, pixels: np.ndarray) -> None:
        display = np.ascontiguousarray(np.flipud(pixels.T))
        image = QImage(
            display.data,
            display.shape[1],
            display.shape[0],
            display.strides[0],
            QImage.Format.Format_Grayscale8,
        ).copy()
        label.set_ct_image(image, pixels.shape[0])

    def resizeEvent(self, event: object) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self._render()
