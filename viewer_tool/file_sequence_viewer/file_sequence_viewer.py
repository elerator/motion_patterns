

import matplotlib
from tensor_viewer_ui import Ui_TensorViewer

import breeze_resources#light layout
import numpy as np
import pandas as pd

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QWidget, QFileDialog, QMessageBox, QButtonGroup
from PyQt5.QtCore import QThread, Qt, pyqtSignal, QFile, QTextStream, QWaitCondition, QMutex

import os
from PIL import Image
import PIL

import sys
from time import sleep

import traceback
from imagedisplay import fig2rgb_array

import matplotlib.pyplot as plt
from skimage.draw import line_aa

import re
import pickle


from collections import defaultdict

class NestedDict(defaultdict):
    def __init__(self):
        super().__init__(self.__class__)
    def __reduce__(self):
        return (type(self), (), None, None, iter(self.items()))

class FileDialog(QWidget):
    outfilepath = pyqtSignal(str)
    folder = pyqtSignal(str)
    #filepath = pyqtSignal(str)

    def __init__(self, file_ending = ".csv"):
        """ A File dialog could either be used to show a dialog to create and output file (i.e. to get a nonexisting path),
            to open a file or to get the name of a folder. The respective member functions may be used in this respect. The filepath is returned and emitted as a pyqtSignal
        Args:
            file_ending: The file ending of the file to be selected (Use empty string for Folder selection)
        """
        self.file_ending = file_ending
        QWidget.__init__(self)

    def create_output_file(self):
        """ Opens dialog for non-existing files and returns path.
        Returns:
            Path to a non-existing file (str). If the specified filename does not end with self.filepath the respective ending is added.
        """
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getSaveFileName(None,"Select the output file", "", self.file_ending[:] +" (*."+self.file_ending[1:]+");;", options=options)

        if filename:
            if not filename.endswith(self.file_ending):
                filename += self.file_ending
            self.outfilepath.emit(filename)
        return filename

    def get_existing_file_path(self):
        """ Opens a file dialog for existing files. Emits path as signal outfilepath.
        Returns:
            Path to existing file. The ending specified in self.filepath is added if the file does not already end with named string (str)
        """
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getOpenFileName(None,"Select the output file", "", self.file_ending[1:] +" (*."+self.file_ending[1:]+");;", options=options)
        self.outfilepath.emit(filename)
        return filename


    def get_folder_path(self):
        """ Opens dialog for folder selection. Emits path as signal folder.
        Returns:
            Path to folder (str)
        """
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        path = QFileDialog.getExistingDirectory(None,"Select folder...",os.getcwd(),options=options)
        if path:
            self.folder.emit(path)
        return path

class TensorViewer(QWidget):
    def __init__(self, widget_handled, ui):
        """ Initializes Main App.
        args:
            widget_handled: For this widget events are handeled by MainApp
            ui: User interface
        """
        QWidget.__init__(self, widget_handled)

        self.ui = ui
        self.widget_handled = widget_handled #Widget for event Handling. All events are checked and if not arrow keys passed on. See eventFilter below
        self.tensor_loader = TensorViewer.TensorLoader()
        self.tensor = None
        self.current_slice = 0
        self.plot = None
        self.plot_n_samples = 1000
        self.frame_mean = []

        self.filenames = []
        self.file_indices = []#A sequence of indices 0..n or in random-mode shuffle(0..n). Used to determine next/previous file_idx
        self.file_idx = 0#current index of filenames
        self.idx_of_file_indices = 0#current index of file indices. Is incremented/decremented to determine nect/previous file_idx
        self.file_sequence_descending = TensorViewer.FileSequenceDescending(self)
        self.dataset = None

        self.subsampling_param = 1
        self.playback = TensorViewer.Playback()
        self.start_with_last_slice = False
        self.save_labels_for_idx = None
        self.centered_normalization = True
        self.label_file_path = None
        self.properties = ["double_peak_1", "double_peak_2",
"one_peak", "three_peak", "multi_peak", "bottom_up", "lateral_medial", "multiple", "no_flow", "top_down", "contains_short_spike", "saddle", "flow_path_visible", "needs_trim", "start_above_stop", "very_noisy", "no_slow_wave", "complex_dynamic", "lateral_static", "medial_axis_event", "medial_axis_event_plus", "outer_rim","other_type"]
        self.make_connections()

    def make_connections(self):
        """ Establishes connections between GUI and functionalities."""
        self.tensor_loader.tensor.connect(self.set_tensor)
        self.tensor_loader.tensor_min.connect(self.ui.slices.set_min)
        self.tensor_loader.tensor_max.connect(self.ui.slices.set_max)
        self.tensor_loader.tensor_min.connect(lambda x: self.ui.tensor_min.setText(str(np.round(x, 4))))
        self.tensor_loader.tensor_max.connect(lambda x: self.ui.tensor_max.setText(str(np.round(x, 4))))
        self.tensor_loader.frame_means.connect(self.set_frame_mean)

        self.ui.load_dataset.triggered.connect(self.load_label_file)
        self.ui.save_file_sequence.clicked.connect(self.save_file_sequence)

        self.ui.next.clicked.connect(lambda: self.display_slice(self.current_slice+1))
        self.ui.previous.clicked.connect(lambda: self.display_slice(self.current_slice-1))
        self.ui.next.clicked.connect(lambda: self.update_plot(self.current_slice+1))
        self.ui.previous.clicked.connect(lambda: self.update_plot(self.current_slice-1))
        self.ui.downscale.stateChanged.connect(self.toggle_subsampling)

        self.set_normalization(self.ui.normalization.currentText())
        self.ui.normalization.currentTextChanged.connect(self.set_normalization)
        self.ui.colormap.currentTextChanged.connect(self.set_colormap)
        self.ui.v_min.valueChanged.connect(self.ui.slices.set_lower_threshold)
        self.ui.v_max.valueChanged.connect(self.ui.slices.set_upper_threshold)

        self.ui.set_random_sequence.clicked.connect(lambda: [self.set_random_file_sequence(), self.set_file_idx(0),self.load_current_tensor()])
        self.ui.load_file_sequence.clicked.connect(self.load_file_sequence)
        self.ui.set_sequence_by_id.clicked.connect(lambda: [self.set_default_file_sequence(), self.set_file_idx(0),self.load_current_tensor()])
        self.ui.set_descending_sequence.clicked.connect(lambda: self.file_sequence_descending.set_sequence())

        self.tensor_loader.progress.connect(self.ui.progress.setValue)
        self.ui.start_play.clicked.connect(lambda: self.playback.start())
        self.ui.stop_play.clicked.connect(lambda: self.playback.stop())
        self.playback.next_frame.connect(lambda: self.next_slice())


    def set_file_idx(self, idx):
        self.file_idx = idx

    def set_normalization(self, normalization):
        if normalization == "frame normalization":
            self.ui.slices.set_framewise_normalization(True)
            self.display_slice(self.current_slice)
        elif normalization == "tensor normalization":
            self.ui.slices.set_framewise_normalization(False)
            self.display_slice(self.current_slice)

    """def normalize(self, tensor):
        if self.centered_normalization:
           factor = 1/ np.max([-np.nanmin(tensor), np.nanmax(tensor)])
           if factor != 0:
              tensor = tensor *factor
        else:
           tensor -= np.nanmin(tensor)
           max_val = np.nanmax(tensor)
           if max_val != 0:
              tensor /= np.nanmax(tensor)
           return tensor"""

    def toggle_subsampling(self):
        if self.ui.downscale.isChecked():
            self.subsampling_param = 4
        else:
            self.subsampling_param = 1

    def set_tensor(self, tensor):
        """ Sets 3D data tensor
        Args:
            tensor: 3D Numpy array
        """
        tensor = tensor[:,::self.subsampling_param,::self.subsampling_param]
        
        if not self.start_with_last_slice:
            self.current_slice = 0
        else:
            self.current_slice = len(tensor) -1
            self.clear_checkboxes()#clear checkboxes when we jumped back to previous tensor
            self.load_checkboxes()
            
        self.tensor = tensor
        self.ui.loading_info_stack.setCurrentIndex(0)
        self.ui.file_info.setText(self.latest_tensor_name)
        #print("Display slice " + str(self.current_slice))
        self.display_slice(self.current_slice)
        self.update_plot(self.current_slice)

        try:
           exp, run, sw = self.parse_filename()
           self.ui.exp.setValue(exp)
           self.ui.run.setValue(run)
           self.ui.sw.setValue(sw)
           self.ui.loading_info_stack.setCurrentIndex(2)
        except:
           QMessageBox.warning(self, "Warning", "The filepath could not be parsed. Ensure that the files meet the naming convention.")

    def parse_filename(self):
        print(re.match("exp_([0-9]+)_run_([0-9]+)_sw_([0-9]+)\.npy", self.latest_tensor_name).groups())
        exp, run, sw = re.match("exp_([0-9]+)_run_([0-9]+)_sw_([0-9]+).npy", self.latest_tensor_name).groups()
        return int(exp), int(run), int(sw)        


    def save_label(self, file_idx):
        """ Every time a new sample is loaded because the last frame is reached,
            we save the label for the last sample ID. Triggered by self.next_slice(). """
        if type(self.dataset) == type(None):
           return
        if len(self.filenames) == 0:
           QMessageBox.warning(self, "Warning", "No filenames.")
           return
        if not self.label_file_path:
           QMessageBox.warning(self, "Warning", "No label_file_path")
           return

        sws_id = os.path.basename(self.filenames[file_idx]).split(".")[0]
        print(sws_id)
        
        values = [getattr(self.ui, "label_" + p).isChecked() for p in self.properties]
        for p, v in zip(self.properties, values):
            self.dataset["sws"][sws_id]["labels"][p] = v
        self.dataset["sws"][sws_id]["labels"]["is_labeled"] = True

        try:                
           with open(self.label_file_path, "wb") as f:
                pickle.dump(self.dataset, f)
        except Exception as e:
           QMessageBox.warning(self, "Exception", "Label file writing error. " + str(e))

    def load_label_file(self):
        path = FileDialog(".pkl").get_existing_file_path()
        self.label_file_path = path
        try:
           with open(self.label_file_path, "rb") as f:
               self.dataset = pickle.load(f)
        except:
           return

    def load_file_sequence(self):
        if type(self.dataset) == type(None):
            QMessageBox.information(self, "Information", "Load label file first") 
        try:
            if type(self.dataset["label_sequence"]) == type([]):
                QMessageBox.information(self, "Information", "Loaded label sequence from dataset")
                self.file_indices = self.dataset["label_sequence"]
                self.set_file_idx(self.file_indices[0])
                self.load_current_tensor()
                self.load_checkboxes() 
            else:
                QMessageBox.warning(self, "Warning", "No labels in dataset")
        except:
            QMessageBox.warning(self, "Warning", "Loading labels from dataset failed"


)

    def set_default_file_sequence(self):
        self.file_indices = np.arange(len(self.filenames))
        
                

    def fast_plot(self, y, x, img = None, add=False):
        img_width = 1000
        img_height = 201

        if np.any(y>=200):
            #print("Y smaller zero occured")
            #print(y[y>=201])
            y[y>200] = 200
        if np.any(y<0):
            #print("Y smaller zero occured")
            #print(y[y<0])
            y[y<0] = 0

        if type(img) == type(None):
            img = np.ndarray((img_height, img_width, 3), dtype=np.uint8)
            img.fill(0)

        if not add:
            img.fill(0)

        rrs = []
        ccs = []
        vals = []
        for y1,y2,x1,x2 in zip(y[:-1],y[1:], x[:-1],x[1:]):
            rr, cc, val = line_aa(y1, x1, y2, x2)
            rrs.extend(rr)
            ccs.extend(cc)
            vals.extend(val)
        img[rrs, ccs, 1] = np.array(vals)  * 255

        img[:,499:501, 2] = 0
        img[:,499:501, 0] = 255
        img[:,499:501, 1] = 0
        return img

    def update_plot(self, frame=0):
        if not self.ui.live_update_plot.isChecked():
            return
        if type(self.plot) != type(None):
            self.plot.fill(0)
        frame += self.plot_n_samples//2
        x = np.arange(self.plot_n_samples)#vmax must be 1000

        for frame_mean in self.frame_mean:
            y = np.array(frame_mean[frame-self.plot_n_samples//2:frame+self.plot_n_samples//2], dtype=np.int)
            self.plot = self.fast_plot(y, x, self.plot, add=True)
            self.ui.plot.update(self.plot)

    def set_frame_mean(self, mean):
        mean *= -1
        mean += 1

        mean *= 200#for display purposes
        mean = np.pad(mean, self.plot_n_samples//2)
        assert np.min(mean) == 0.0
        assert np.max(mean) == 200.0

        self.frame_mean = [mean]
        self.update_plot()

    def set_colormap(self, colormap):
        if colormap == "grayscale":
            self.ui.slices.set_colormap("grayscale")
        elif colormap == "viridis":
            self.ui.slices.set_colormap("viridis")
        elif colormap == "seismic":
            self.ui.slices.set_colormap("seismic")
        self.display_slice(self.current_slice)

    def get_tensor(self):
        """ Returns current tensor """
        return self.tensor

    def get_current_slice(self):
        """ Getter for current slice"""
        slice = None
        idx = self.current_slice
        if type(self.tensor) != type(None):
            slice = self.tensor[idx,:,:].copy()
            slice[np.isnan(slice)] = 0
            self.ui.frame_mean.setText(str(np.round(np.mean(slice),4)))
            self.ui.frame_min.setText(str(np.round(np.min(slice),4)))
            self.ui.frame_max.setText(str(np.round(np.max(slice),4)))
            return slice
        return None

    def display_slice(self, idx):
        """ Displayes current slice"""
        old_idx = self.current_slice
        try:
            self.current_slice = idx
            self.ui.slices.update(self.get_current_slice())
            self.ui.slice.setValue(self.current_slice)
        except Exception as err:
            self.current_slice = old_idx
            print(err)
            traceback.print_tb(err.__traceback__)

    def set_next_idx(self):
        """ Sets next file index"""
        if self.idx_of_file_indices <= len(self.file_indices)-1:
            self.idx_of_file_indices += 1
        self.file_idx = self.file_indices[self.idx_of_file_indices]

    def set_previous_idx(self):
        if self.idx_of_file_indices > 0:
            self.idx_of_file_indices -= 1
        self.file_idx = self.file_indices[self.idx_of_file_indices]
   
    def clear_checkboxes(self):
        """ Uncheck all checkboxes"""
        for p in self.properties:
            name = "label_" + p
            try:
                getattr(self.ui, name).setChecked(False)
                self.ui.label_one_peak.setChecked(True)
                self.ui.label_no_flow.setChecked(True)
                self.ui.label_other_type.setChecked(True)
            except Exception as e:
                QMessage.warning(self,"error", "could not clear checkboxes" + str(e))

    def load_checkboxes(self):
        if type(self.dataset) == type(None):
           return
        id = self.latest_tensor_name.split(".")[0]
        if self.dataset["sws"][id]["labels"]["is_labeled"] == True:
           for prop in self.properties:
               if self.dataset["sws"][id]["labels"][prop] == True:
                  name = "label_" + prop
                  getattr(self.ui, name).setChecked(True)

    def next_slice(self):
        """ Shows next slice if possible. Triggers loading of next tensor when necessary"""
        if self.tensor_loader.isRunning():
           return#Do not update view, do not wake up running playback

        if type(self.tensor) != type(None) and self.current_slice >= len(self.tensor)-1:
           #Load new tensor file
           self.save_labels_for_idx = self.file_idx#file index to be labeled
           try:
           	self.set_next_idx()
           except:
              return
           self.start_with_last_slice = False
           self.load_current_tensor()
           return#Do not update view, do not wake up running playback

        if type(self.save_labels_for_idx) != type(None):#If a new file is loaded and the user skips to the second frame, save label for previous file
           self.save_label(self.save_labels_for_idx)
           self.clear_checkboxes()
           self.load_checkboxes()
           self.save_labels_for_idx = None#No need to save label for current file already

        self.display_slice(self.current_slice+1)
        self.update_plot(self.current_slice+1)
        self.playback.view_has_updated()#If playback is running but waiting (e.g. because new file was loaded) this wakes it up

    def previous_slice(self):
       """ Shows previous slice if possible """
       if self.tensor_loader.isRunning():
           return
       self.file_idx_to_label = False#If user skips back do not save label
       if self.current_slice < 0:
           try:
               self.set_previous_idx()
           except:
               return
           self.start_with_last_slice = True
           self.load_current_tensor()

           return
       self.display_slice(self.current_slice-1)
       self.update_plot(self.current_slice-1)
       self.playback.view_has_updated()

    def eventFilter(self, source, event):
        """ Filters key events such that arrow keys may be handled.
            Args:
                source: Source of event
                event: Event to be handled
        """
        if event.type() == QtCore.QEvent.KeyRelease:
            id_right = 16777236
            id_left = 16777234
            if event.key() == id_right:
               self.next_slice()

            elif event.key() == id_left:
               self.previous_slice()

        try:#When closing the app the widget handled might already have been destroyed
            return True#self.widget_handled.eventFilter(source, event)#Execute the default actions for the event
        except:
            return True#a true value prevents the event from being sent on to other objects

    def user_opens_folder(self):
        """ Get filenames and set index """
        try:
            dialog = FileDialog(".npy")
            self.path = dialog.get_folder_path()
            self.filenames = [os.path.join(self.path, f) for f in os.listdir(self.path)]
            self.filenames.sort()
            
            #Determine self.file_indices: default is by file id
            self.set_default_file_sequence()
            self.set_file_idx(self.file_indices[0])
            self.load_current_tensor()
        except:
            pass

    def set_random_file_sequence(self):
          np.random.seed(42)
          np.random.shuffle(self.file_indices)
          self.file_indices = list(self.file_indices)

    def save_file_sequence(self):
        #Save sequence to dataset
        if type(self.dataset) != type(None):
           self.dataset["label_sequence"] = self.file_indices
           try:
               with open(self.label_file_path, "wb") as f:
                    pickle.dump(self.dataset, f)
           except Exception as e:
               QMessageBox.warning(self,"Warning", "Could not save label sequence to dataset although path exists. " + str(e))
        else:
           QMessageBox.warning(self, "Warning", "Could not save label sequence because no dataset was loaded.")

    class FileSequenceDescending(QThread):
        sequence = pyqtSignal(list)
        progress = pyqtSignal(int)

        def __init__(self, outer):
            super(TensorViewer.FileSequenceDescending, self).__init__()
            self.outer = outer
            self.make_connections()

        def make_connections(self):
            self.sequence.connect(self._set_sequence)
            self.progress.connect(self.outer.ui.progress.setValue)

        def set_sequence(self):
            if len(self.outer.filenames) == 0:
               return 
            self.outer.ui.loading_info_stack.setCurrentIndex(1)
            self.start()

        def _set_sequence(self, sequence):
            """ Callback after parallel processing has finished. Sets file_indices in outer"""
            self.outer.ui.file_info.setText("Descending sequence ready")
            self.outer.ui.loading_info_stack.setCurrentIndex(0)
            self.outer.file_indices = sequence
            self.outer.idx_of_file_indices = 0
            self.outer.file_idx = self.outer.file_indices[self.outer.idx_of_file_indices]

            self.outer.load_current_tensor()

        def sort_x_by_y(self, X, Y):
            return [x for _, x in sorted(zip(Y, X))]

        def run(self):
            n_files = len(self.outer.filenames)
            max_vals = []
            for i, path in enumerate(self.outer.filenames):
                tensor = np.load(path)
                tensor -= np.nanmin(tensor)
                max_vals.append(np.nanmax(tensor))
                percentage = (i/n_files)*100
                self.progress.emit(int(percentage))
            idxs = np.arange(n_files)
            idxs = self.sort_x_by_y(idxs, -np.array(max_vals))

            for idx, idx1 in zip(idxs, idxs[1:]):
                assert max_vals[idx1] < max_vals[idx]
            self.sequence.emit(idxs)
            
    
    def load_current_tensor(self):
        """ Trigger tensor loader for current filename """
        try:
            path = self.filenames[self.file_idx]
            self.latest_tensor_name = os.path.basename(path) 
            self.tensor_loader.filepath = path
            self.ui.loading_info_stack.setCurrentIndex(0)
            self.tensor_loader.start()
            self.ui.loading_info_stack.setCurrentIndex(1)
        except:
            QMessageBox.warning(self, "Warning", "Could not load tensor. Make sure you selected a file directory")


    def close_all(self):
        self.tensor = None
        self.frame_mean = []
        self.ui.normalization.setEnabled(True)
        self.ui.normalization.blockSignals(False)
        self.ui.normalization.setCurrentIndex(0)

    class Playback(QThread):
        next_frame = pyqtSignal(bool)

        def __init__(self):
            super(TensorViewer.Playback, self).__init__()
            self.wait_condition = QWaitCondition()
            self.mutex = QMutex()
            self.is_running = False
            self.has_updated = True

        def stop(self):
            self.is_running = False
        
        def view_has_updated(self):
            self.wait_condition.wakeAll()

        def run(self):
            self.is_running = True
            self.mutex.unlock()
            while(self.is_running):
                self.has_updated = False
                self.next_frame.emit(True)
                self.mutex.lock()
                self.wait_condition.wait(self.mutex)
                self.mutex.unlock()
        

    class TensorLoader(QThread):
        tensor = pyqtSignal(np.ndarray)
        tensor_min = pyqtSignal(float)
        tensor_max = pyqtSignal(float)
        frame_means = pyqtSignal(np.ndarray)
        progress = pyqtSignal(int)

        def __init__(self):
            """ Thread for loading tensor from image files in parallel"""
            super(TensorViewer.TensorLoader, self).__init__()
            self.first_frames_only = None
            self.filepath = None
            self.load_blocksize = 128
            self.set_first_frames_only(2000)#TODO set via gui

        def set_first_frames_only(self, n_frames):
            self.first_frames_only = n_frames

        def load(self):
            blocksize = self.load_blocksize

            try:
                mmap = np.load(self.filepath, mmap_mode='r')
                y = np.empty_like(mmap)
                n_blocks = int(np.ceil(mmap.shape[0] / blocksize))
                for b in range(n_blocks):
                    self.progress.emit(int(100*(b/n_blocks)))# use any progress indicator
                    y[b*blocksize : (b+1) * blocksize] = mmap[b*blocksize : (b+1) * blocksize]
            finally:
                del mmap  # make sure file is closed again
            return y

        def run(self):
            """ Loads tensor. Emits data as self.tensor (PyQtSignal)."""
            try:
                tensor = self.load()
                #if self.first_frames_only:
                #   tensor = tensor[:self.first_frames_only]
                tensor -= np.nanmin(tensor)
                self.tensor.emit(tensor)

                min = np.nanmin(tensor)
                max = np.nanmax(tensor)
                self.tensor_min.emit(min)
                self.tensor_max.emit(max)

                mean = np.nanmean(tensor, axis=(1,2))
                mean -= np.nanmin(mean)
                max = np.nanmax(mean)
                if max != 0:
                   mean /= max
                self.frame_means.emit(mean)
            except Exception as e:
                print("No valid folder. Loading tensor failed")
                print(e)

class Main():
    def __init__(self):
        """ Initializes program. Starts app, creates window and implements functions accessible via action bar."""
        self.app = QtWidgets.QApplication(sys.argv)
        self.set_color_theme(self.app, "light")

        MainWindow = QtWidgets.QMainWindow()#Create a window
        self.main_ui = Ui_TensorViewer()#Instanciate our UI
        self.main_ui.setupUi(MainWindow)#Setup our UI as this MainWindow

        self.source_dir_opener = FileDialog()

        self.main_ui.centralwidget.setFocusPolicy(Qt.NoFocus)

        self.main_app = TensorViewer(self.main_ui.centralwidget, self.main_ui)#Install MainApp as event filter for handling of arrow keys

        self.main_ui.centralwidget.installEventFilter(self.main_app)

        self.make_connections()
        MainWindow.show()#and we show it directly

        self.app.exec_()
        sys.exit()

    def set_color_theme(self,app, color):
        """ Set ui color scheme to either dark or bright
        Args:
            app: PyQt App the color scheme is applied to
            color: String specifying the color scheme. Either "dark" or "bright".

        """
        path = ""
        if color == "dark":
            path += ":/dark.qss"
            self.use_light = False
        elif color == "light":
            path += ":/light.qss"
            self.use_light = True
        else:
            return
        file = QFile(path)
        file.open(QFile.ReadOnly | QFile.Text)
        stream = QTextStream(file)
        app.setStyleSheet(stream.readAll())

    def make_connections(self):
        """ Establishes connections between actions and GUI elements"""
        self.main_ui.actionOpen_Tensor.triggered.connect(self.main_app.user_opens_folder)
        self.main_ui.close_all_tensors.triggered.connect(self.main_app.close_all)


if __name__ == "__main__":
    m = Main()#start app
