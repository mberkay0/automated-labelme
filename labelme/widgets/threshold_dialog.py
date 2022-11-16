from qtpy.QtCore import Qt
from qtpy import QtWidgets
from labelme import __scale__


class ThresholdDialog(QtWidgets.QDialog):
    def __init__(self, confidence, mask_threshold, parent=None):
        super(ThresholdDialog, self).__init__(parent)
        self.setModal(True)
        self.setWindowTitle("Threshold Control")

        self.slider_confidence = self._create_slider(confidence)
        self.slider_mask_threshold = self._create_slider(mask_threshold)

        self.label_conf = QtWidgets.QLabel(str(confidence), self)
        self.label_maskth = QtWidgets.QLabel(str(mask_threshold), self)

        self.label_conf.move(12, -3)
        self.label_conf.setAlignment(Qt.AlignmentFlag.AlignCenter |
                                     Qt.AlignmentFlag.AlignVCenter)

        self.label_maskth.move(12, 25)
        self.label_maskth.setAlignment(Qt.AlignmentFlag.AlignCenter |
                                       Qt.AlignmentFlag.AlignVCenter)

        formLayout = QtWidgets.QFormLayout()
        formLayout.addRow(self.tr("Confidence"), self.slider_confidence)
        formLayout.addRow(self.tr("Mask Threshold"), self.slider_mask_threshold)
        self.setLayout(formLayout)

    def onNewValue(self, value):
        confidence = self.slider_confidence.value() / __scale__
        maskth = self.slider_mask_threshold.value() / __scale__

        self.label_conf.setText(str(confidence))
        self.label_maskth.setText(str(maskth))

    def _create_slider(self, set_value):
        slider = QtWidgets.QSlider(
            Qt.Horizontal, singleStep=1, 
            maximum=1 * __scale__, minimum=0 * __scale__
        )
        slider.setValue(set_value * __scale__)
        slider.valueChanged.connect(self.onNewValue)
        return slider
