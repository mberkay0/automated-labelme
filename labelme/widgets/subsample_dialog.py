from qtpy import QtWidgets
from qtpy.QtCore import Qt


class SubsamplingDialog(QtWidgets.QDialog):
    def __init__(self, value, parent=None):
        super(SubsamplingDialog, self).__init__(parent)
        self.setModal(True)
        self.setWindowTitle("Subsample Value")
        self.subsample_spin = QtWidgets.QSpinBox(self)
        self.subsample_spin.setRange(1, 120)
        self.subsample_spin.setGeometry(80, 80, 80, 30)
        self.subsample_spin.setAlignment(Qt.AlignmentFlag.AlignCenter |
                                         Qt.AlignmentFlag.AlignVCenter)
        self.subsample_spin.setValue(value)
        self.subsample_spin.setPrefix("Val: ")

  


