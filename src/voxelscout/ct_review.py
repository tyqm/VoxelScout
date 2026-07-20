"""Small independent window for display-only CT review."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
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


class CTReviewDialog(QDialog):
    """Axial before/after viewer whose transforms never leave this window."""

    def __init__(self, ct_path: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Review CT")
        self.resize(900, 620)
        image = nib.as_closest_canonical(nib.load(str(Path(ct_path).resolve())))
        if len(image.shape) != 3:
            raise ValueError(f"Expected a 3D CT volume, got shape {image.shape}")
        self._volume = np.asanyarray(image.dataobj)
        self._report = inspect_hu(self._volume)
        center, width = automatic_window(self._volume)
        self._build_ui(center, width)
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
        layout.addWidget(self.slice_slider)

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

    @staticmethod
    def _image_panel(title: str) -> tuple[QWidget, QLabel]:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        heading = QLabel(title)
        heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        view = QLabel()
        view.setMinimumSize(320, 320)
        view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        view.setStyleSheet("background: black;")
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
        axial = self._volume[:, :, index]
        settings = self._settings()
        before = render_ct_slice(axial, settings.before_settings())
        after = render_ct_slice(axial, settings)
        self._set_image(self.before[1], before)
        self._set_image(self.after[1], after)

    @staticmethod
    def _set_image(label: QLabel, pixels: np.ndarray) -> None:
        display = np.ascontiguousarray(np.flipud(pixels.T))
        image = QImage(
            display.data,
            display.shape[1],
            display.shape[0],
            display.strides[0],
            QImage.Format.Format_Grayscale8,
        ).copy()
        label.setPixmap(
            QPixmap.fromImage(image).scaled(
                label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event: object) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self._render()
