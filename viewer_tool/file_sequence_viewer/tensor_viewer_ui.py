# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'tensor_viewer.ui'
#
# Created by: PyQt5 UI code generator 5.15.0
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_TensorViewer(object):
    def setupUi(self, TensorViewer):
        TensorViewer.setObjectName("TensorViewer")
        TensorViewer.resize(1004, 878)
        self.centralwidget = QtWidgets.QWidget(TensorViewer)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.splitter_2 = QtWidgets.QSplitter(self.centralwidget)
        self.splitter_2.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_2.setObjectName("splitter_2")
        self.widget_2 = QtWidgets.QWidget(self.splitter_2)
        self.widget_2.setObjectName("widget_2")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.widget_2)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.slices = ImageDisplay(self.widget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.slices.sizePolicy().hasHeightForWidth())
        self.slices.setSizePolicy(sizePolicy)
        self.slices.setMinimumSize(QtCore.QSize(400, 0))
        self.slices.setText("")
        self.slices.setObjectName("slices")
        self.verticalLayout_3.addWidget(self.slices)
        self.page_controls = QtWidgets.QWidget(self.widget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.page_controls.sizePolicy().hasHeightForWidth())
        self.page_controls.setSizePolicy(sizePolicy)
        self.page_controls.setObjectName("page_controls")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout(self.page_controls)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.plot = ImageDisplay(self.page_controls)
        self.plot.setText("")
        self.plot.setObjectName("plot")
        self.horizontalLayout_7.addWidget(self.plot)
        self.verticalLayout_3.addWidget(self.page_controls)
        self.widget = QtWidgets.QWidget(self.splitter_2)
        self.widget.setMaximumSize(QtCore.QSize(300, 16777215))
        self.widget.setObjectName("widget")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.scrollArea = QtWidgets.QScrollArea(self.widget)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, -931, 317, 1651))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.verticalLayout_10 = QtWidgets.QVBoxLayout(self.scrollAreaWidgetContents)
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.groupBox_3 = QtWidgets.QGroupBox(self.scrollAreaWidgetContents)
        self.groupBox_3.setObjectName("groupBox_3")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.groupBox_3)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.label_6 = QtWidgets.QLabel(self.groupBox_3)
        self.label_6.setObjectName("label_6")
        self.verticalLayout_7.addWidget(self.label_6)
        self.normalization = QtWidgets.QComboBox(self.groupBox_3)
        self.normalization.setObjectName("normalization")
        self.normalization.addItem("")
        self.normalization.addItem("")
        self.verticalLayout_7.addWidget(self.normalization)
        self.label_7 = QtWidgets.QLabel(self.groupBox_3)
        self.label_7.setObjectName("label_7")
        self.verticalLayout_7.addWidget(self.label_7)
        self.colormap = QtWidgets.QComboBox(self.groupBox_3)
        self.colormap.setObjectName("colormap")
        self.colormap.addItem("")
        self.colormap.addItem("")
        self.colormap.addItem("")
        self.verticalLayout_7.addWidget(self.colormap)
        self.widget_3 = QtWidgets.QWidget(self.groupBox_3)
        self.widget_3.setObjectName("widget_3")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.widget_3)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_9 = QtWidgets.QLabel(self.widget_3)
        self.label_9.setObjectName("label_9")
        self.horizontalLayout_3.addWidget(self.label_9)
        self.label_8 = QtWidgets.QLabel(self.widget_3)
        self.label_8.setObjectName("label_8")
        self.horizontalLayout_3.addWidget(self.label_8)
        self.verticalLayout_8.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.v_min = QtWidgets.QDoubleSpinBox(self.widget_3)
        self.v_min.setMaximum(1.0)
        self.v_min.setSingleStep(0.01)
        self.v_min.setObjectName("v_min")
        self.horizontalLayout_4.addWidget(self.v_min)
        self.v_max = QtWidgets.QDoubleSpinBox(self.widget_3)
        self.v_max.setMaximum(1.0)
        self.v_max.setSingleStep(0.01)
        self.v_max.setProperty("value", 1.0)
        self.v_max.setObjectName("v_max")
        self.horizontalLayout_4.addWidget(self.v_max)
        self.verticalLayout_8.addLayout(self.horizontalLayout_4)
        self.downscale = QtWidgets.QCheckBox(self.widget_3)
        self.downscale.setObjectName("downscale")
        self.verticalLayout_8.addWidget(self.downscale)
        self.live_update_plot = QtWidgets.QCheckBox(self.widget_3)
        self.live_update_plot.setObjectName("live_update_plot")
        self.verticalLayout_8.addWidget(self.live_update_plot)
        self.verticalLayout_7.addWidget(self.widget_3)
        self.verticalLayout_10.addWidget(self.groupBox_3)
        self.groupBox_2 = QtWidgets.QGroupBox(self.scrollAreaWidgetContents)
        self.groupBox_2.setObjectName("groupBox_2")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.groupBox_2)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.label_4 = QtWidgets.QLabel(self.groupBox_2)
        self.label_4.setObjectName("label_4")
        self.verticalLayout_6.addWidget(self.label_4)
        self.tensor_min = QtWidgets.QLineEdit(self.groupBox_2)
        self.tensor_min.setObjectName("tensor_min")
        self.verticalLayout_6.addWidget(self.tensor_min)
        self.label_5 = QtWidgets.QLabel(self.groupBox_2)
        self.label_5.setObjectName("label_5")
        self.verticalLayout_6.addWidget(self.label_5)
        self.tensor_max = QtWidgets.QLineEdit(self.groupBox_2)
        self.tensor_max.setObjectName("tensor_max")
        self.verticalLayout_6.addWidget(self.tensor_max)
        self.verticalLayout_10.addWidget(self.groupBox_2)
        self.groupBox = QtWidgets.QGroupBox(self.scrollAreaWidgetContents)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox.sizePolicy().hasHeightForWidth())
        self.groupBox.setSizePolicy(sizePolicy)
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.label_frame_mean = QtWidgets.QLabel(self.groupBox)
        self.label_frame_mean.setObjectName("label_frame_mean")
        self.verticalLayout_5.addWidget(self.label_frame_mean)
        self.frame_mean = QtWidgets.QLineEdit(self.groupBox)
        self.frame_mean.setObjectName("frame_mean")
        self.verticalLayout_5.addWidget(self.frame_mean)
        self.label_min = QtWidgets.QLabel(self.groupBox)
        self.label_min.setObjectName("label_min")
        self.verticalLayout_5.addWidget(self.label_min)
        self.frame_min = QtWidgets.QLineEdit(self.groupBox)
        self.frame_min.setObjectName("frame_min")
        self.verticalLayout_5.addWidget(self.frame_min)
        self.label_max = QtWidgets.QLabel(self.groupBox)
        self.label_max.setObjectName("label_max")
        self.verticalLayout_5.addWidget(self.label_max)
        self.frame_max = QtWidgets.QLineEdit(self.groupBox)
        self.frame_max.setObjectName("frame_max")
        self.verticalLayout_5.addWidget(self.frame_max)
        self.verticalLayout_10.addWidget(self.groupBox)
        self.groupBox_4 = QtWidgets.QGroupBox(self.scrollAreaWidgetContents)
        self.groupBox_4.setObjectName("groupBox_4")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout(self.groupBox_4)
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.set_random_sequence = QtWidgets.QPushButton(self.groupBox_4)
        self.set_random_sequence.setObjectName("set_random_sequence")
        self.verticalLayout_9.addWidget(self.set_random_sequence)
        self.set_descending_sequence = QtWidgets.QPushButton(self.groupBox_4)
        self.set_descending_sequence.setObjectName("set_descending_sequence")
        self.verticalLayout_9.addWidget(self.set_descending_sequence)
        self.set_sequence_by_id = QtWidgets.QPushButton(self.groupBox_4)
        self.set_sequence_by_id.setObjectName("set_sequence_by_id")
        self.verticalLayout_9.addWidget(self.set_sequence_by_id)
        self.load_file_sequence = QtWidgets.QPushButton(self.groupBox_4)
        self.load_file_sequence.setObjectName("load_file_sequence")
        self.verticalLayout_9.addWidget(self.load_file_sequence)
        self.save_file_sequence = QtWidgets.QPushButton(self.groupBox_4)
        self.save_file_sequence.setObjectName("save_file_sequence")
        self.verticalLayout_9.addWidget(self.save_file_sequence)
        self.verticalLayout_10.addWidget(self.groupBox_4)
        self.groupBox_5 = QtWidgets.QGroupBox(self.scrollAreaWidgetContents)
        self.groupBox_5.setObjectName("groupBox_5")
        self.verticalLayout_11 = QtWidgets.QVBoxLayout(self.groupBox_5)
        self.verticalLayout_11.setObjectName("verticalLayout_11")
        self.groupBox_8 = QtWidgets.QGroupBox(self.groupBox_5)
        self.groupBox_8.setObjectName("groupBox_8")
        self.verticalLayout_12 = QtWidgets.QVBoxLayout(self.groupBox_8)
        self.verticalLayout_12.setObjectName("verticalLayout_12")
        self.widget_4 = QtWidgets.QWidget(self.groupBox_8)
        self.widget_4.setObjectName("widget_4")
        self.verticalLayout_16 = QtWidgets.QVBoxLayout(self.widget_4)
        self.verticalLayout_16.setObjectName("verticalLayout_16")
        self.label_multi_peak = QtWidgets.QRadioButton(self.widget_4)
        self.label_multi_peak.setObjectName("label_multi_peak")
        self.verticalLayout_16.addWidget(self.label_multi_peak)
        self.label_one_peak = QtWidgets.QRadioButton(self.widget_4)
        self.label_one_peak.setObjectName("label_one_peak")
        self.verticalLayout_16.addWidget(self.label_one_peak)
        self.label_double_peak_1 = QtWidgets.QRadioButton(self.widget_4)
        self.label_double_peak_1.setObjectName("label_double_peak_1")
        self.verticalLayout_16.addWidget(self.label_double_peak_1)
        self.label_double_peak_2 = QtWidgets.QRadioButton(self.widget_4)
        self.label_double_peak_2.setObjectName("label_double_peak_2")
        self.verticalLayout_16.addWidget(self.label_double_peak_2)
        self.label_three_peak = QtWidgets.QRadioButton(self.widget_4)
        self.label_three_peak.setObjectName("label_three_peak")
        self.verticalLayout_16.addWidget(self.label_three_peak)
        self.verticalLayout_12.addWidget(self.widget_4)
        self.label_contains_short_spike = QtWidgets.QCheckBox(self.groupBox_8)
        self.label_contains_short_spike.setObjectName("label_contains_short_spike")
        self.verticalLayout_12.addWidget(self.label_contains_short_spike)
        self.label_saddle = QtWidgets.QCheckBox(self.groupBox_8)
        self.label_saddle.setObjectName("label_saddle")
        self.verticalLayout_12.addWidget(self.label_saddle)
        self.verticalLayout_11.addWidget(self.groupBox_8)
        self.flow = QtWidgets.QGroupBox(self.groupBox_5)
        self.flow.setObjectName("flow")
        self.verticalLayout_13 = QtWidgets.QVBoxLayout(self.flow)
        self.verticalLayout_13.setObjectName("verticalLayout_13")
        self.label_bottom_up = QtWidgets.QRadioButton(self.flow)
        self.label_bottom_up.setObjectName("label_bottom_up")
        self.verticalLayout_13.addWidget(self.label_bottom_up)
        self.label_top_down = QtWidgets.QRadioButton(self.flow)
        self.label_top_down.setObjectName("label_top_down")
        self.verticalLayout_13.addWidget(self.label_top_down)
        self.label_lateral_medial = QtWidgets.QRadioButton(self.flow)
        self.label_lateral_medial.setObjectName("label_lateral_medial")
        self.verticalLayout_13.addWidget(self.label_lateral_medial)
        self.label_multiple = QtWidgets.QRadioButton(self.flow)
        self.label_multiple.setObjectName("label_multiple")
        self.verticalLayout_13.addWidget(self.label_multiple)
        self.label_no_flow = QtWidgets.QRadioButton(self.flow)
        self.label_no_flow.setObjectName("label_no_flow")
        self.verticalLayout_13.addWidget(self.label_no_flow)
        self.verticalLayout_11.addWidget(self.flow)
        self.stereotype = QtWidgets.QGroupBox(self.groupBox_5)
        self.stereotype.setObjectName("stereotype")
        self.verticalLayout_14 = QtWidgets.QVBoxLayout(self.stereotype)
        self.verticalLayout_14.setObjectName("verticalLayout_14")
        self.label_complex_dynamic = QtWidgets.QRadioButton(self.stereotype)
        self.label_complex_dynamic.setObjectName("label_complex_dynamic")
        self.verticalLayout_14.addWidget(self.label_complex_dynamic)
        self.label_lateral_static = QtWidgets.QRadioButton(self.stereotype)
        self.label_lateral_static.setObjectName("label_lateral_static")
        self.verticalLayout_14.addWidget(self.label_lateral_static)
        self.label_outer_rim = QtWidgets.QRadioButton(self.stereotype)
        self.label_outer_rim.setObjectName("label_outer_rim")
        self.verticalLayout_14.addWidget(self.label_outer_rim)
        self.label_medial_axis_event = QtWidgets.QRadioButton(self.stereotype)
        self.label_medial_axis_event.setObjectName("label_medial_axis_event")
        self.verticalLayout_14.addWidget(self.label_medial_axis_event)
        self.label_medial_axis_event_plus = QtWidgets.QRadioButton(self.stereotype)
        self.label_medial_axis_event_plus.setObjectName("label_medial_axis_event_plus")
        self.verticalLayout_14.addWidget(self.label_medial_axis_event_plus)
        self.label_other_type = QtWidgets.QRadioButton(self.stereotype)
        self.label_other_type.setObjectName("label_other_type")
        self.verticalLayout_14.addWidget(self.label_other_type)
        self.verticalLayout_11.addWidget(self.stereotype)
        self.additional = QtWidgets.QGroupBox(self.groupBox_5)
        self.additional.setObjectName("additional")
        self.verticalLayout_15 = QtWidgets.QVBoxLayout(self.additional)
        self.verticalLayout_15.setObjectName("verticalLayout_15")
        self.label_needs_trim = QtWidgets.QCheckBox(self.additional)
        self.label_needs_trim.setObjectName("label_needs_trim")
        self.verticalLayout_15.addWidget(self.label_needs_trim)
        self.label_flow_path_visible = QtWidgets.QCheckBox(self.additional)
        self.label_flow_path_visible.setObjectName("label_flow_path_visible")
        self.verticalLayout_15.addWidget(self.label_flow_path_visible)
        self.label_start_above_stop = QtWidgets.QCheckBox(self.additional)
        self.label_start_above_stop.setObjectName("label_start_above_stop")
        self.verticalLayout_15.addWidget(self.label_start_above_stop)
        self.label_very_noisy = QtWidgets.QCheckBox(self.additional)
        self.label_very_noisy.setObjectName("label_very_noisy")
        self.verticalLayout_15.addWidget(self.label_very_noisy)
        self.label_no_slow_wave = QtWidgets.QCheckBox(self.additional)
        self.label_no_slow_wave.setObjectName("label_no_slow_wave")
        self.verticalLayout_15.addWidget(self.label_no_slow_wave)
        self.verticalLayout_11.addWidget(self.additional)
        self.verticalLayout_10.addWidget(self.groupBox_5)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.verticalLayout_4.addWidget(self.scrollArea)
        self.verticalLayout_2.addWidget(self.splitter_2)
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout_2.addWidget(self.line)
        self.widget1 = QtWidgets.QWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget1.sizePolicy().hasHeightForWidth())
        self.widget1.setSizePolicy(sizePolicy)
        self.widget1.setObjectName("widget1")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget1)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.loading_info_stack = QtWidgets.QStackedWidget(self.widget1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.loading_info_stack.sizePolicy().hasHeightForWidth())
        self.loading_info_stack.setSizePolicy(sizePolicy)
        self.loading_info_stack.setMaximumSize(QtCore.QSize(1000, 16777215))
        self.loading_info_stack.setObjectName("loading_info_stack")
        self.page_info = QtWidgets.QWidget()
        self.page_info.setObjectName("page_info")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.page_info)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.file_info = QtWidgets.QLabel(self.page_info)
        self.file_info.setMinimumSize(QtCore.QSize(200, 0))
        self.file_info.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.file_info.setObjectName("file_info")
        self.horizontalLayout_2.addWidget(self.file_info)
        self.loading_info_stack.addWidget(self.page_info)
        self.page_progress = QtWidgets.QWidget()
        self.page_progress.setObjectName("page_progress")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.page_progress)
        self.verticalLayout.setObjectName("verticalLayout")
        self.progress = QtWidgets.QProgressBar(self.page_progress)
        self.progress.setMaximumSize(QtCore.QSize(180, 16777215))
        self.progress.setProperty("value", 0)
        self.progress.setObjectName("progress")
        self.verticalLayout.addWidget(self.progress)
        self.loading_info_stack.addWidget(self.page_progress)
        self.page_sws = QtWidgets.QWidget()
        self.page_sws.setObjectName("page_sws")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.page_sws)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_exp = QtWidgets.QLabel(self.page_sws)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_exp.sizePolicy().hasHeightForWidth())
        self.label_exp.setSizePolicy(sizePolicy)
        self.label_exp.setObjectName("label_exp")
        self.horizontalLayout_5.addWidget(self.label_exp)
        self.exp = QtWidgets.QSpinBox(self.page_sws)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.exp.sizePolicy().hasHeightForWidth())
        self.exp.setSizePolicy(sizePolicy)
        self.exp.setMinimumSize(QtCore.QSize(40, 0))
        self.exp.setMaximumSize(QtCore.QSize(999999, 16777215))
        self.exp.setObjectName("exp")
        self.horizontalLayout_5.addWidget(self.exp)
        self.label_run = QtWidgets.QLabel(self.page_sws)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_run.sizePolicy().hasHeightForWidth())
        self.label_run.setSizePolicy(sizePolicy)
        self.label_run.setObjectName("label_run")
        self.horizontalLayout_5.addWidget(self.label_run)
        self.run = QtWidgets.QSpinBox(self.page_sws)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.run.sizePolicy().hasHeightForWidth())
        self.run.setSizePolicy(sizePolicy)
        self.run.setMinimumSize(QtCore.QSize(40, 0))
        self.run.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.run.setObjectName("run")
        self.horizontalLayout_5.addWidget(self.run)
        self.label_sw = QtWidgets.QLabel(self.page_sws)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_sw.sizePolicy().hasHeightForWidth())
        self.label_sw.setSizePolicy(sizePolicy)
        self.label_sw.setObjectName("label_sw")
        self.horizontalLayout_5.addWidget(self.label_sw)
        self.sw = QtWidgets.QSpinBox(self.page_sws)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sw.sizePolicy().hasHeightForWidth())
        self.sw.setSizePolicy(sizePolicy)
        self.sw.setMinimumSize(QtCore.QSize(40, 0))
        self.sw.setMaximumSize(QtCore.QSize(999999, 16777215))
        self.sw.setMaximum(999)
        self.sw.setObjectName("sw")
        self.horizontalLayout_5.addWidget(self.sw)
        self.jump = QtWidgets.QPushButton(self.page_sws)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.jump.sizePolicy().hasHeightForWidth())
        self.jump.setSizePolicy(sizePolicy)
        self.jump.setMaximumSize(QtCore.QSize(50, 16777215))
        self.jump.setObjectName("jump")
        self.horizontalLayout_5.addWidget(self.jump)
        self.loading_info_stack.addWidget(self.page_sws)
        self.horizontalLayout.addWidget(self.loading_info_stack)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.start_play = QtWidgets.QPushButton(self.widget1)
        self.start_play.setObjectName("start_play")
        self.horizontalLayout.addWidget(self.start_play)
        self.stop_play = QtWidgets.QPushButton(self.widget1)
        self.stop_play.setObjectName("stop_play")
        self.horizontalLayout.addWidget(self.stop_play)
        self.previous = QtWidgets.QPushButton(self.widget1)
        self.previous.setMinimumSize(QtCore.QSize(80, 0))
        self.previous.setMaximumSize(QtCore.QSize(80, 16777215))
        self.previous.setObjectName("previous")
        self.horizontalLayout.addWidget(self.previous)
        self.next = QtWidgets.QPushButton(self.widget1)
        self.next.setMinimumSize(QtCore.QSize(80, 0))
        self.next.setMaximumSize(QtCore.QSize(80, 16777215))
        self.next.setObjectName("next")
        self.horizontalLayout.addWidget(self.next)
        self.slice = QtWidgets.QSpinBox(self.widget1)
        self.slice.setMaximum(999999999)
        self.slice.setObjectName("slice")
        self.horizontalLayout.addWidget(self.slice)
        self.verticalLayout_2.addWidget(self.widget1)
        TensorViewer.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(TensorViewer)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1004, 20))
        self.menubar.setObjectName("menubar")
        self.menuOpen = QtWidgets.QMenu(self.menubar)
        self.menuOpen.setObjectName("menuOpen")
        TensorViewer.setMenuBar(self.menubar)
        self.toolBar_2 = QtWidgets.QToolBar(TensorViewer)
        self.toolBar_2.setObjectName("toolBar_2")
        TensorViewer.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar_2)
        self.actionOpen = QtWidgets.QAction(TensorViewer)
        self.actionOpen.setObjectName("actionOpen")
        self.actionRotate = QtWidgets.QAction(TensorViewer)
        self.actionRotate.setObjectName("actionRotate")
        self.actionDetect = QtWidgets.QAction(TensorViewer)
        self.actionDetect.setObjectName("actionDetect")
        self.actionImbibition = QtWidgets.QAction(TensorViewer)
        self.actionImbibition.setObjectName("actionImbibition")
        self.actionSave = QtWidgets.QAction(TensorViewer)
        self.actionSave.setObjectName("actionSave")
        self.actionDisplay_xy_slices = QtWidgets.QAction(TensorViewer)
        self.actionDisplay_xy_slices.setObjectName("actionDisplay_xy_slices")
        self.actionDisplay_xz_slices = QtWidgets.QAction(TensorViewer)
        self.actionDisplay_xz_slices.setObjectName("actionDisplay_xz_slices")
        self.actionDisplay_yz_slices = QtWidgets.QAction(TensorViewer)
        self.actionDisplay_yz_slices.setObjectName("actionDisplay_yz_slices")
        self.actionShow_slices = QtWidgets.QAction(TensorViewer)
        self.actionShow_slices.setObjectName("actionShow_slices")
        self.actionShow_Detected_colum_view = QtWidgets.QAction(TensorViewer)
        self.actionShow_Detected_colum_view.setObjectName("actionShow_Detected_colum_view")
        self.actionShow_columnwise_imbibition_front = QtWidgets.QAction(TensorViewer)
        self.actionShow_columnwise_imbibition_front.setObjectName("actionShow_columnwise_imbibition_front")
        self.actionSlice_tensor = QtWidgets.QAction(TensorViewer)
        self.actionSlice_tensor.setObjectName("actionSlice_tensor")
        self.actionOpen_Tensor = QtWidgets.QAction(TensorViewer)
        self.actionOpen_Tensor.setObjectName("actionOpen_Tensor")
        self.close_all_tensors = QtWidgets.QAction(TensorViewer)
        self.close_all_tensors.setObjectName("close_all_tensors")
        self.load_dataset = QtWidgets.QAction(TensorViewer)
        self.load_dataset.setObjectName("load_dataset")
        self.menuOpen.addAction(self.load_dataset)
        self.menuOpen.addAction(self.actionOpen_Tensor)
        self.menuOpen.addSeparator()
        self.menuOpen.addAction(self.close_all_tensors)
        self.menubar.addAction(self.menuOpen.menuAction())

        self.retranslateUi(TensorViewer)
        self.loading_info_stack.setCurrentIndex(2)
        QtCore.QMetaObject.connectSlotsByName(TensorViewer)

    def retranslateUi(self, TensorViewer):
        _translate = QtCore.QCoreApplication.translate
        TensorViewer.setWindowTitle(_translate("TensorViewer", "Tensor Viewer"))
        self.groupBox_3.setTitle(_translate("TensorViewer", "Display Options"))
        self.label_6.setText(_translate("TensorViewer", "Normalization:"))
        self.normalization.setItemText(0, _translate("TensorViewer", "tensor normalization"))
        self.normalization.setItemText(1, _translate("TensorViewer", "frame normalization"))
        self.label_7.setText(_translate("TensorViewer", "Colormap:"))
        self.colormap.setItemText(0, _translate("TensorViewer", "grayscale"))
        self.colormap.setItemText(1, _translate("TensorViewer", "viridis"))
        self.colormap.setItemText(2, _translate("TensorViewer", "seismic"))
        self.label_9.setText(_translate("TensorViewer", "Vmin"))
        self.label_8.setText(_translate("TensorViewer", "Vmax"))
        self.downscale.setText(_translate("TensorViewer", "Downscale upon loading"))
        self.live_update_plot.setText(_translate("TensorViewer", "update plot"))
        self.groupBox_2.setTitle(_translate("TensorViewer", "Tensor meta"))
        self.label_4.setText(_translate("TensorViewer", "Tensor Min:"))
        self.label_5.setText(_translate("TensorViewer", "Tensor Max:"))
        self.groupBox.setTitle(_translate("TensorViewer", "Slice Meta"))
        self.label_frame_mean.setText(_translate("TensorViewer", "Frame Mean:"))
        self.label_min.setText(_translate("TensorViewer", "Frame Min:"))
        self.label_max.setText(_translate("TensorViewer", "Frame Max:"))
        self.groupBox_4.setTitle(_translate("TensorViewer", "File order"))
        self.set_random_sequence.setText(_translate("TensorViewer", "Set random sequence"))
        self.set_descending_sequence.setText(_translate("TensorViewer", "Set descending max sequence"))
        self.set_sequence_by_id.setText(_translate("TensorViewer", "Set order by ID"))
        self.load_file_sequence.setText(_translate("TensorViewer", "Load file sequence"))
        self.save_file_sequence.setText(_translate("TensorViewer", "Save order in dataset"))
        self.groupBox_5.setTitle(_translate("TensorViewer", "Label"))
        self.groupBox_8.setTitle(_translate("TensorViewer", "Shape (rather strict)"))
        self.label_multi_peak.setText(_translate("TensorViewer", "Multipeak"))
        self.label_one_peak.setText(_translate("TensorViewer", "One major peak"))
        self.label_double_peak_1.setText(_translate("TensorViewer", "Double peak (major/minor)"))
        self.label_double_peak_2.setText(_translate("TensorViewer", "Double peak (equal size)"))
        self.label_three_peak.setText(_translate("TensorViewer", "Three peak"))
        self.label_contains_short_spike.setText(_translate("TensorViewer", "contains short spike"))
        self.label_saddle.setText(_translate("TensorViewer", "saddle"))
        self.flow.setTitle(_translate("TensorViewer", "Major flow (rising phase, robust or None)"))
        self.label_bottom_up.setText(_translate("TensorViewer", "bottom up"))
        self.label_top_down.setText(_translate("TensorViewer", "top down"))
        self.label_lateral_medial.setText(_translate("TensorViewer", "lateral medial"))
        self.label_multiple.setText(_translate("TensorViewer", "multiple"))
        self.label_no_flow.setText(_translate("TensorViewer", "None / inapplicable"))
        self.stereotype.setTitle(_translate("TensorViewer", "Stereotype"))
        self.label_complex_dynamic.setText(_translate("TensorViewer", "complex dynamic (splittable)"))
        self.label_lateral_static.setText(_translate("TensorViewer", "lateral static / lateral-medial"))
        self.label_outer_rim.setText(_translate("TensorViewer", "outer rim"))
        self.label_medial_axis_event.setText(_translate("TensorViewer", "Medial axis event"))
        self.label_medial_axis_event_plus.setText(_translate("TensorViewer", "Medial axis event + focal areas"))
        self.label_other_type.setText(_translate("TensorViewer", "Other"))
        self.additional.setTitle(_translate("TensorViewer", "Additional"))
        self.label_needs_trim.setText(_translate("TensorViewer", "Needs trim"))
        self.label_flow_path_visible.setText(_translate("TensorViewer", "Flow path visible"))
        self.label_start_above_stop.setText(_translate("TensorViewer", "Start above stop"))
        self.label_very_noisy.setText(_translate("TensorViewer", "Very noisy"))
        self.label_no_slow_wave.setText(_translate("TensorViewer", "Quality too bad / no slow wave"))
        self.file_info.setText(_translate("TensorViewer", "No folder loaded ..."))
        self.label_exp.setText(_translate("TensorViewer", "exp"))
        self.label_run.setText(_translate("TensorViewer", "run"))
        self.label_sw.setText(_translate("TensorViewer", "sw"))
        self.jump.setText(_translate("TensorViewer", "jump"))
        self.start_play.setText(_translate("TensorViewer", " >"))
        self.stop_play.setText(_translate("TensorViewer", "||"))
        self.previous.setText(_translate("TensorViewer", "<<"))
        self.next.setText(_translate("TensorViewer", ">>"))
        self.menuOpen.setTitle(_translate("TensorViewer", "Start"))
        self.toolBar_2.setWindowTitle(_translate("TensorViewer", "toolBar_2"))
        self.actionOpen.setText(_translate("TensorViewer", "Open source files"))
        self.actionRotate.setText(_translate("TensorViewer", "Rotate tensor"))
        self.actionDetect.setText(_translate("TensorViewer", "Detect columns"))
        self.actionImbibition.setText(_translate("TensorViewer", "Find imbibition front"))
        self.actionSave.setText(_translate("TensorViewer", "Save current tensor"))
        self.actionDisplay_xy_slices.setText(_translate("TensorViewer", "Display xy slices"))
        self.actionDisplay_xz_slices.setText(_translate("TensorViewer", "Display xz slices"))
        self.actionDisplay_yz_slices.setText(_translate("TensorViewer", "Display yz slices"))
        self.actionShow_slices.setText(_translate("TensorViewer", "Show slices"))
        self.actionShow_Detected_colum_view.setText(_translate("TensorViewer", "Show detected colum view"))
        self.actionShow_columnwise_imbibition_front.setText(_translate("TensorViewer", "Show columnwise imbibition front"))
        self.actionSlice_tensor.setText(_translate("TensorViewer", "Slice tensor"))
        self.actionOpen_Tensor.setText(_translate("TensorViewer", "Open file directory"))
        self.close_all_tensors.setText(_translate("TensorViewer", "Close all tensors"))
        self.load_dataset.setText(_translate("TensorViewer", "Load dataset"))
from imagedisplay import ImageDisplay
