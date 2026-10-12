# WingStrokeAnalyzer
# Copyright (c) 2026, Sung Soo Kim
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This software is distributed under the BSD 3-Clause License.
# See the LICENSE file in the project root for the full license text.


import sys
from pathlib import Path
import numpy as np
import scipy.io as sio
import time
import datetime
import threading
from dataclasses import dataclass, field
import pyqtgraph as pg
from PySide6 import QtCore
from PySide6.QtWidgets import (QApplication, QMainWindow, QSizePolicy, QWidget, QVBoxLayout,
                             QHBoxLayout, QGroupBox, QPushButton, QComboBox, QFileDialog,
                             QLabel, QSlider, QLineEdit, QCheckBox, QGridLayout, QFrame, QMessageBox)
from PySide6.QtCore import QThread, Signal, Slot, Qt, QTimer
from PySide6.QtGui import QIcon, QPen, QColor
import cv2

import nidaqmx
from nidaqmx.constants import LineGrouping
import nidaqmx.system
from collections import deque
import json



try:
    import PySpin
    PYSPIN_AVAILABLE = True
except ImportError:
    print("PySpin not found. Falling back to Simulation Mode.")
    PYSPIN_AVAILABLE = False



def get_daq_device_names():
    try:
        system = nidaqmx.system.System.local()
        # Returns a list of strings like ['Dev1', 'Dev2', 'cDAQ1Mod1']
        return [device.name for device in system.devices]
    except Exception as e:
        print(f"DAQ Error: {e}")
        return []
    

# ==========================================================
# Define a class for the graphics overlay and analysis
# ==========================================================
@dataclass
class WSASettings:
    wing_angles: dict = None # To be filled with 'left' and 'right' keys containing the respective angles
    
    # Threshold for image processing
    threshold: int = 2
    threshold_show_effect: bool = False

    # Baseline image to be subtracted for motion detection. Updated when "Not flying" is clicked.
    new_stationary_ref_frame_available: bool = False # Flag to indicate when a new reference frame is ready
    stationary_ref_frame: np.ndarray = field(
        default_factory=lambda: np.zeros((160, 128))
    )

    # masks
    new_masks_available: bool = False # Flag to indicate when new masks are ready
    masks: dict = None # To be filled with 'left' and 'right' keys containing the respective masks
    
    # Task control
    task_mode: str = None  # "preview", "run", "test_frame_rate", "test_wsa"

    # RUN mode settings
    run_options: dict = None # Placeholder for any future settings related to the "run" mode



# ==========================================================
# PART 1: Efficient Camera Thread (Multi-Threaded)
# ==========================================================
class CameraWorker(QThread):
    new_frame_ready = Signal() # image, timestamp

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False
        # Stub for PySpin objects
        self.system = None
        self.cam_list = None
        self.cam = None
        self.target_serial = None
        self.target_mode = None  # Store the mode string here
        
        
        # Simulation Data
        self.sim_frames = None
        self.sim_index = 0

        # Sending only the last frame
        self.latest_frame = None
        self.latest_timestamp = 0
        self._frame_lock = threading.Lock()

        self.test_frame_rate_frame_count = 0
        self.test_frame_rate_timestamp_first = 0
        self.test_frame_rate_timestamp_last = 0

        # Image processing - Settings container and lock for thread safety
        self._setting_lock = threading.Lock()
        self.settings = WSASettings()
        self.task_mode = None # Options: "preview", "run", "test_frame_rate", "test_wsa"
        self.settings.wing_angles = {'left': 5.0, 'right': 5.0}

        # Image processing - Persistent local variables
        self.stationary_ref_frame = np.zeros([160,128])
        self.thresh = 2
        self.thresh_show_effect = False
        self.masks = None

        # RUN mode settings
        self.settings.run_options = None
        self.video_writer = None
        self.nidaq_DIO_read_task = None
        self.nidaq_DIO_write_task = None
        self.nidaq_AO_task = None
        

    def load_simulation_data(self):
        """Internal helper to load the .mat file."""
        try:
            data = sio.loadmat('wingbeat.mat')
            self.sim_frames = data['wingbeat'][0]
            self.sim_index = 0
            return True
        except Exception as e:
            print(f"Simulation Error: {e}")
            return False            

    def update_settings(self, **kwargs):
        """
        Updates the settings container safely.
        Usage: worker.update_settings(left_hinge=(150, 200), threshold=140)
        """
        with self._setting_lock:
            for key, value in kwargs.items():
                if hasattr(self.settings, key):
                    setattr(self.settings, key, value)
            # Optional: print(f"Settings updated: {kwargs}")

    def get_settings(self, key):
        """
        Retrieves a setting value safely.
        Usage: worker.get_settings('threshold')
        """
        with self._setting_lock:
            if hasattr(self.settings, key):
                return getattr(self.settings, key)
        return None



    def get_cam_image(self):
        try:
            image_result = self.cam.GetNextImage(1000)
            if not image_result.IsIncomplete():
                # Get data directly as NumPy array (Low Resource)
                img_array = image_result.GetNDArray()
                timestamp = image_result.GetTimeStamp() / 1e9
                # Explicit copy required before emitting across thread boundaries
                #self.new_frame_ready.emit(img_array.copy(), timestamp)
                frame = img_array.copy()
        except PySpin.SpinnakerException as ex:
            print(f"PySpin Error: {ex}")
            frame = None
            timestamp = None
        image_result.Release()
        return frame.swapaxes(0, 1), timestamp

    def get_sim_image(self):
        frame = self.sim_frames[self.sim_index]['snapshot']
        timestamp = time.time()
        # Increment index
        self.sim_index = (self.sim_index + 1) % len(self.sim_frames)
        
        # Control frame rate (Approx 30 FPS)
        self.msleep(1)

        return frame.swapaxes(0, 1), timestamp

    def get_rand_image(self):
        img_shape = (360, 240)
        # Generate simple dummy image and text
        dummy_img = np.random.randint(0, 250, img_shape, dtype=np.uint8)
        # Simple visual to confirm animation
        timestamp = time.time()
        if (int(timestamp) % 2 == 0): dummy_img[10:100, 10:100] = 128
        
        # Control frame rate (Approx 30 FPS)
        self.msleep(33)

        return dummy_img.swapaxes(0, 1), timestamp


    def setup_daq_DIO(self):
        """Creates and starts the DAQ tasks."""

        try:
            # 1. Check if the device exists in the system
            system = nidaqmx.system.System.local()
            device_names = [d.name for d in system.devices]
            if self.settings.run_options["nidaq_dev_name_for_DIO"] not in device_names:
                print(f"NI-DAQmx Error: Device '{self.settings.run_options['nidaq_dev_name_for_DIO']}' not found. Available devices: {device_names}")
                self.cleanup_daq_DIO()
                return False
            
            # Create the tasks
            self.nidaq_DIO_read_task = nidaqmx.Task("ReadTask")
            self.nidaq_DIO_write_task = nidaqmx.Task("WriteTask")

            # Configure Channels (DATA/port0/line0 for Read, line1 for Write)
            n = f"{self.settings.run_options['nidaq_dev_name_for_DIO']}/port{self.settings.run_options['nidaq_DIO_port']}/line{self.settings.run_options['nidaq_DI_line']}"
            self.nidaq_DIO_read_task.di_channels.add_di_chan(
                n, 
                line_grouping=LineGrouping.CHAN_PER_LINE
            )
            
            n = f"{self.settings.run_options['nidaq_dev_name_for_DIO']}/port{self.settings.run_options['nidaq_DIO_port']}/line{self.settings.run_options['nidaq_DO_line']}"
            self.nidaq_DIO_write_task.do_channels.add_do_chan(
                n, 
                line_grouping=LineGrouping.CHAN_PER_LINE
            )
            
            # Pre-start tasks to reduce latency during the first loop iteration
            self.nidaq_DIO_read_task.start()
            self.nidaq_DIO_write_task.start()

            print(f"NI-DAQmx DIO tasks configured successfully on device '{self.settings.run_options['nidaq_dev_name_for_DIO']}'. Reading from line {self.settings.run_options['nidaq_DI_line']} and writing to line {self.settings.run_options['nidaq_DO_line']}.")
            return True

        except nidaqmx.DaqError as e:
            print(f"NI-DAQmx Setup Error: {e}")
            self.cleanup_daq_DIO() # Ensure any partially created tasks are cleaned up
            return False

    def cleanup_daq_DIO(self):
        """Stops and closes tasks to release the 'DATA' device."""
        # Stop and close the Read task
        if self.nidaq_DIO_read_task is not None:
            self.nidaq_DIO_read_task.stop()
            self.nidaq_DIO_read_task.close()
            self.nidaq_DIO_read_task = None
            
        # Stop and close the Write task
        if self.nidaq_DIO_write_task is not None:
            self.nidaq_DIO_write_task.stop()
            self.nidaq_DIO_write_task.close()
            self.nidaq_DIO_write_task = None
            
        print("NI-DAQmx DIO Tasks successfully cleared.")

    def setup_daq_AO(self):
        """Creates and starts the DAQ AO task."""
        try:
            # 1. Check if the device exists in the system
            system = nidaqmx.system.System.local()
            device_names = [d.name for d in system.devices]
            if self.settings.run_options["nidaq_dev_name_for_AO"] not in device_names:
                print(f"NI-DAQmx Error: Device '{self.settings.run_options['nidaq_dev_name_for_AO']}' not found. Available devices: {device_names}")
                self.cleanup_daq_AO()
                return False
            
            # Create the AO task
            self.nidaq_AO_task = nidaqmx.Task("AO_Task")

            # Configure AO Channel (AO0)
            n = f"{self.settings.run_options['nidaq_dev_name_for_AO']}/{self.settings.run_options['nidaq_AO_L_R']}"
            self.nidaq_AO_task.ao_channels.add_ao_voltage_chan(n, min_val=-10.0, max_val=10.0)
            n = f"{self.settings.run_options['nidaq_dev_name_for_AO']}/{self.settings.run_options['nidaq_AO_LWS']}"
            self.nidaq_AO_task.ao_channels.add_ao_voltage_chan(n, min_val=-10.0, max_val=10.0)

            # Pre-start task to reduce latency during the first loop iteration
            self.nidaq_AO_task.start()

            print(f"NI-DAQmx AO task configured successfully on device '{self.settings.run_options['nidaq_dev_name_for_AO']}' on channel {self.settings.run_options['nidaq_AO_L_R']} for L_R and {self.settings.run_options['nidaq_AO_LWS']} for LWS.")
            return True

        except nidaqmx.DaqError as e:
            print(f"NI-DAQmx AO Setup Error: {e}")
            self.cleanup_daq_AO() # Ensure any partially created tasks are cleaned up
            return False
        
    def cleanup_daq_AO(self):
        """Stops and closes the AO task."""
        if self.nidaq_AO_task is not None:
            self.nidaq_AO_task.stop()
            self.nidaq_AO_task.close()
            self.nidaq_AO_task = None
            print("NI-DAQmx AO Task successfully cleared.")

    def setup_video_writer(self, width, height, fps=30):
        # Use 'XVID' or 'MJPG' for good compatibility and speed
        try:
            # fourcc = cv2.VideoWriter_fourcc(*'XVID') # compression with loss
            # fourcc = cv2.VideoWriter_fourcc(*'MJPG') # less compression, larger file, but better for real-time writing
            fourcc = cv2.VideoWriter_fourcc(*'FFV1') # lossless compression, larger file, may be slower but preserves quality
            filename = f"movie_{self.settings.run_options['filename_postfix']}.avi"
            output_dir = Path("data") / "movies"
            output_dir.mkdir(parents=True, exist_ok=True)
            full_path = output_dir / filename
            save_path_str = str(full_path)            

            
            # width and height must match your camera frame exactly
            self.video_writer = cv2.VideoWriter(save_path_str, fourcc, fps, (width, height))
            if self.video_writer.isOpened():
                print(f"Video writer initialized successfully. Saving to {save_path_str}")
                return True
            else:
                return False
        except Exception as e:
            print(f"Error setting up video writer: {e}")
            self.video_writer = None
            return False
        
        # To read in, start with the following code snippet:
        # You might need to run: !pip install imageio[ffmpeg]
        # # or: conda install imageio-ffmpeg
        # import imageio
        # import matplotlib.pyplot as plt
        # video_path = "data\\movies\\movie_b3_145344.avi"
        # try:
        #     # 'ffmpeg' is the plugin that handles MKV/FFV1
        #     reader = imageio.get_reader(video_path,  format='ffmpeg')
        #     # Get the first frame (index 0)
        #     frame = reader.get_data(0)
        #     plt.figure(figsize=(8, 6))
        #     plt.imshow(frame) # imageio reads as RGB already!
        #     plt.title("Read via ImageIO")
        #     plt.show()
        #     reader.close()
        # except Exception as e:
        #     print(f"ImageIO also failed: {e}")
        
    def cleanup_video_writer(self):
        # ... your existing DAQ cleanup ...
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            print("Video file finalized.")
    

    def run(self):
        """The main high-speed loop. Uses PySpin if available."""
        self.running = True
        fps_timer = QTimer()
        fps_timer.start(1000) # Simple frame rate counter setup


        if PYSPIN_AVAILABLE and self.target_mode != "Simulation":
            print("Camera starts")
            # 1. Initialize PySpin
            self.system = PySpin.System.GetInstance()
            self.cam_list = self.system.GetCameras()
            if self.cam_list.GetSize() == 0:
                print("No PySpin camera detected.")
                self.cam_list.Clear()
                self.system.ReleaseInstance()
                return

            #self.cam = self.cam_list[0]
            if self.target_serial:
                self.cam = self.cam_list.GetBySerial(self.target_serial)
            else:
                self.cam = self.cam_list[0] # Fallback if no serial provided
                
            self.cam.Init()

            # Try to Set VideoMode (Chameleon 3)
            #self.set_node_value("VideoMode", self.target_mode)
            self.set_node_value("PixelFormat", self.target_mode)
                

            # To set the camera so that the buffer stores only the most recent image. That is, no buffering.
            # Get the stream node map
            s_node_map = self.cam.GetTLStreamNodeMap()
            # Access the buffer handling mode node
            handling_mode = PySpin.CEnumerationPtr(s_node_map.GetNode("StreamBufferHandlingMode"))
            if PySpin.IsAvailable(handling_mode) and PySpin.IsWritable(handling_mode):
                # Set to 'NewestOnly'
                handling_mode_entry = handling_mode.GetEntryByName("NewestOnly")
                handling_mode.SetIntValue(handling_mode_entry.GetValue())

            self.cam.BeginAcquisition()

            get_img_func = self.get_cam_image

        # --- Simulation images ---
        elif self.target_mode == "Simulation":
            print("Simulation starts")
            if not self.load_simulation_data():
                print("   Loading simulation data failed.")
                self.running = False
                return

            print("   Simulation data is loaded.")
            get_img_func = self.get_sim_image

        # --- When there is no simulation data ---
        else:
            # Simulation Mode for code development without a camera
            print(" No camera, no simulation data. Using random dots.")
            get_img_func = self.get_rand_image
        

        frame, timestamp = get_img_func() # test run to make sure camera is working and settings are correct. We can remove this later if we want to optimize a bit.
        print(f"   frame info: {frame.shape}, {frame.dtype}, timestamp: {timestamp:.3f}")
        # run_options = {
        #             "save_movie": self.chk_save.isChecked(),
        #             "filename_postfix": self.fly_ID.text(),
        #             "nidaq_dev_name_for_DIO": self.nidaq_dev_name_for_DIO.currentText(),
        #             "nidaq_DIO_port": self.nidaq_DIO_port.text(),
        #             "nidaq_DI_line": self.nidaq_DI_line.text(),
        #             "nidaq_DO_line": self.nidaq_DO_line.text(),
        #             "save_frame_skip": int(self.save_frame_skip.text()),
        #             "update_fly": self.chk_update_fly.isChecked(),
        #             "update_rate": int(self.update_rate.text()),
        #             "nidaq_dev_name_for_AO": self.nidaq_dev_name_for_AO.currentText(),
        #             "nidaq_AO_LWS": self.nidaq_AO_LWS.text(),
        #             "nidaq_AO_L_R": self.nidaq_AO_L_R.text()
        #         }

        
        rm = None
        with self._setting_lock:
            rm = self.settings.task_mode
        
        if rm in ["run", "test_frame_rate", "test_wsa"]:
            if not self.setup_daq_AO():
                print("   Failed to setup DAQ AO task.")
                return False
            
        if rm == "test_frame_rate":
            self.test_frame_rate_frame_count = 0
            self.test_frame_rate_timestamp_first = time.perf_counter()
            self.test_frame_rate_timestamp_last = 0

        if rm == "run":
            if self.settings.run_options is not None and self.settings.run_options["save_movie"] and self.setup_daq_DIO() and self.setup_video_writer(frame.shape[0], frame.shape[1]): 
                print("Movie saving is enabled. Frames will be saved every {} frames.".format(self.settings.run_options["save_frame_skip"]))
                saveImageCounter = 0 # Initialize the counter for saved frames
            else:
                print("Movie saving is disabled or setup failed. No frames will be saved.")
                saveImageCounter = None

            if self.settings.run_options is not None and self.settings.run_options["update_fly"]:
                print("Fly image display is enabled. Display will be updated every {} frames.".format(self.settings.run_options["update_rate"]))
                displayImageCounter = 0 # Initialize the counter for displayed frames
            else:
                print("Fly image display is disabled or setup failed. No frames will be displayed.")
                displayImageCounter = None



        run_frame_count = 0
        while self.running:
            with self._setting_lock:
                self.thresh_show_effect = self.settings.threshold_show_effect
                self.thresh  = self.settings.threshold
                current_mode = self.settings.task_mode
                new_masks_available = self.settings.new_masks_available
                new_stationary_ref_frame_available = self.settings.new_stationary_ref_frame_available
            if new_masks_available:
                with self._setting_lock:
                    self.masks = self.settings.masks
                    self.settings.new_masks_available = False # Reset the flag
            if new_stationary_ref_frame_available:
                with self._setting_lock:
                    self.stationary_ref_frame = self.settings.stationary_ref_frame
                    self.settings.new_stationary_ref_frame_available = False # Reset the flag
                
            frame, timestamp = get_img_func()
            if frame is None:
                break

            if current_mode == "run":
                ## Perform the WSA calculation and send it out via NIDAQ
                thresholded_img = self.compute_wing_angle(frame)
                self.wing_angle_nidaq_AO_out()

                ## Display every 10 frames?? Save every 7 frames??
                if displayImageCounter is not None:
                    if displayImageCounter == 0:
                        with self._frame_lock:
                            if self.thresh_show_effect:
                                self.latest_frame = thresholded_img * 255
                            else:
                                self.latest_frame = frame
                            self.latest_timestamp = timestamp
                        self.new_frame_ready.emit()
                    displayImageCounter = (displayImageCounter + 1) % self.settings.run_options["update_rate"]
                
                # Save movie
                if saveImageCounter is not None:
                    # print("saving active")
                    if saveImageCounter == 0 and self.nidaq_DIO_read_task.read(): # digital read from nidaq is boolean.
                        # print("Saving frame at timestamp {:.3f} with angles L:{:.1f}, R:{:.1f}, AO_L_R:{:.2f}, AO_LWS:{:.2f}".format(timestamp, l, r, a_l-a_r, a_l))
                        # # Save the current frame with a timestamped filename
                        # filename = f"frame_{timestamp:.3f}_{self.settings.run_options['filename_postfix']}.png"
                        # cv2.imwrite(filename, frame)
                        # # print(f"Saved frame to {filename}")
                        if len(frame.shape) == 2:
                            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                            self.video_writer.write(frame_bgr.swapaxes(0, 1))
                        else:
                            self.video_writer.write(frame.swapaxes(0, 1))
                        self.nidaq_DIO_write_task.write(True) # Send a digital pulse to indicate frame saved
                    else:
                        self.nidaq_DIO_write_task.write(False) # Ensure the digital line is low when not saving
                    saveImageCounter = (saveImageCounter + 1) % self.settings.run_options["save_frame_skip"]    
            
            elif current_mode ==  "test_frame_rate":
                # Perform the WSA calculation only
                thresholded_img = self.compute_wing_angle(frame)
                self.wing_angle_nidaq_AO_out()
                with self._frame_lock:
                    self.test_frame_rate_frame_count += 1
                    self.test_frame_rate_timestamp_last = time.perf_counter()
                
            
            elif current_mode == "test_wsa":
                thresholded_img = self.compute_wing_angle(frame)
                self.wing_angle_nidaq_AO_out()

                # self.new_frame_ready.emit(img_array.copy(), timestamp)
                with self._frame_lock:
                    # We overwrite the same memory slot. 
                    # If the GUI is slow, it simply misses this frame.
                    if self.thresh_show_effect:
                        self.latest_frame = thresholded_img * 255
                    else:
                        self.latest_frame = frame
                    self.latest_timestamp = timestamp
                self.new_frame_ready.emit()
                # Perform the WSA calculation and plot in the lower right image panel
                pass
            
            elif current_mode ==  "preview":
                # self.new_frame_ready.emit(img_array.copy(), timestamp)
                with self._frame_lock:
                    # We overwrite the same memory slot. 
                    # If the GUI is slow, it simply misses this frame.
                    self.latest_frame = frame
                    self.latest_timestamp = timestamp
                self.new_frame_ready.emit()


    def wing_angle_nidaq_AO_out(self):
        ## L_R and LWS AO output
        l = self.settings.wing_angles['left']
        r = self.settings.wing_angles['right']
        rectification_base = 30 # WSA considered to as not flying
        voltage_range = 8 # voltage corresponding to 180 degree.
        #  % something wrong in angle calculation
        # if isempty(angles) || ~isstruct(angles) || ~isfield(angles, 'left')...
        #         || isnan(angles.left) || isnan(angles.right)...
        #         || angles.left > 225 || angles.right > 225
        #     left = rectification_base;
        #     right = rectification_base;
        # end
        if np.isnan(l) or np.isnan(r) or l > 225 or r > 225:
            l = rectification_base
            r = rectification_base
        # % rectification and scaling: Each wing spans 0~8 Volt (or >8 for WBA>180)
        # a = ([left, right] - rectification_base) / (180 - rectification_base) * voltage_range;
        # a(a < 0) = 0;
        a_l = max(0, (l - rectification_base) / (180 - rectification_base) * voltage_range)
        a_r = max(0, (r - rectification_base) / (180 - rectification_base) * voltage_range)
        if a_l == 0 or a_r == 0:
            a_l = 0
            a_r = 0
        self.nidaq_AO_task.write([a_l-a_r, a_l])

    def compute_wing_angle(self, img):
        if self.masks is None: 
            pass
        if self.stationary_ref_frame is None or img.shape != self.stationary_ref_frame.shape:
            print("Reference frame not set or shape mismatch. Zeroing the reference frame.")
            print(img.shape)
            if self.stationary_ref_frame is not None:
                print(self.stationary_ref_frame.shape)
            self.stationary_ref_frame = np.zeros_like(img)

        thresholded_img = cv2.subtract(img, self.stationary_ref_frame)
        thresholded_img[thresholded_img < self.thresh] = 0
        thresholded_img[thresholded_img >= self.thresh] = 1

        for side, masks in self.masks.items():
            intensity = [
                np.mean(thresholded_img[y, x]) if x.size > 0 else 0 
                for x, y in masks
            ]
            intensity = np.array(intensity)
            if (np.mean(intensity)<0.1 and np.mean(intensity[0:3])>np.mean(intensity[6:])):
                intensity = np.zeros(len(masks))
            
            intensity = intensity[::-1]
            ang_val = np.arange(75, 186, 10) # The max angle is simply defined as 190d
            # 3. Concatenate [intensity 0] and extend ang_val
            # Note: intensity.flatten() ensures it's 1D before appending
            intensity = np.append(intensity.flatten(), 0)
            new_ang = ang_val[-1] + (ang_val[-1] - ang_val[-2])
            ang_val = np.append(ang_val.flatten(), new_ang)

            # 4. Calculate half_max_intensity
            # np.nanmax handles the ~isnan(intensity) check automatically
            half_max_intensity = np.nanmax(intensity)/2

            if half_max_intensity == 0:
                ang = 5 # Default angle when no wing is detected
            else:
                ang = 5 # Default if loop finishes without finding anything
                
                # 5. Reverse Loop (numel(ang_val)-1:-1:1)
                # In Python, we start at len-2 because intensity has been padded, 
                # and we want the index before the last '0' we added.
                for mi in range(len(ang_val) - 2, -1, -1):
                    if not np.isnan(intensity[mi]) and intensity[mi] >= half_max_intensity:
                        
                        # Check for leg vs wing (intensity(mi:-1:1))
                        # Slice mi:-1:-1 includes index 0
                        if np.mean(intensity[0:mi+1]) < 0.66:
                            ang = 5
                            break
                        
                        # 6. Linear Interpolation
                        # Note: intensity[mi+1] exists because we padded with 0
                        denom = (intensity[mi] - intensity[mi+1])
                        
                        if denom != 0:
                            ang = (ang_val[mi+1] * (intensity[mi] - half_max_intensity) +
                                ang_val[mi] * (half_max_intensity - intensity[mi+1])) / denom
                        else:
                            ang = ang_val[mi]
                            
                        break
                # 7. Final floor check
                if ang < 5:
                    ang = 5
            
            with self._setting_lock:
                self.settings.wing_angles[side] = ang
                # Optional: print(f"Computed {side} wing angle: {ang:.2f} degrees")

        return thresholded_img   

    def stop(self):
        """Safely shuts down the camera and PySpin instance."""
        self.running = False
        self.quit()
        self.wait() # Wait for the loop to finish its current iteration

        # Check if the attributes exist and are not None
        if PYSPIN_AVAILABLE and getattr(self, 'cam', None) is not None:
            try:
                if self.cam.IsStreaming():
                    self.cam.EndAcquisition()
                self.cam.DeInit()
                print("PySpin camera disconnected safely.")
            except Exception as e:
                print(f"Error during camera shutdown: {e}")
            finally:
                # Set to None instead of 'del' to avoid AttributeError next time
                self.cam = None 

        self.cleanup_daq_DIO() # Ensure DAQ tasks are cleaned up when stopping the camera thread
        self.cleanup_daq_AO()
        self.cleanup_video_writer()

        if PYSPIN_AVAILABLE and getattr(self, 'cam_list', None) is not None:
            self.cam_list.Clear()
            self.cam_list = None

        if PYSPIN_AVAILABLE and getattr(self, 'system', None) is not None:
            if self.system.IsInUse():
                self.system.ReleaseInstance()
            self.system = None

    def set_node_value(self, node_name, value_str):
        """Universally attempts to set an enumeration node by string name."""
        try:
            node_map = self.cam.GetNodeMap()
            node = PySpin.CEnumerationPtr(node_map.GetNode(node_name))
            if PySpin.IsAvailable(node) and PySpin.IsWritable(node):
                entry = node.GetEntryByName(value_str)
                if PySpin.IsAvailable(entry) and PySpin.IsReadable(entry):
                    node.SetIntValue(entry.GetValue())
                    print(f"Successfully set {node_name} to {value_str}")
                    return True
        except PySpin.SpinnakerException:
            print(f"Failed setting {node_name} to {value_str}. Restart the app.")
            pass

        print(f"Failed setting {node_name} to {value_str}")
        return False


# ==========================================================
# PART 2: The Main Application & UI Layout
# ==========================================================
class FlyExperimentApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wing Beat Analysis Port v2.0")
        self.resize(800, 450) # Approx size from MATLAB

        # Core components
        self.camera_worker = CameraWorker(self)
        
        # Create a menu bar
        self.create_menu_bar()
        self.status_bar = self.statusBar()
        

        # Setup the overall layout: Top, Left, Right panels
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.outer_layout = QVBoxLayout(self.main_widget)
        self.outer_layout.setContentsMargins(10, 10, 10, 10) # Remove padding around the edges

        # Create the Top Control Bar
        self.create_top_control_bar()
        self.outer_layout.addLayout(self.top_control_bar)

        # Create the Mid Layout (Left Controls + Right View/Plot)
        self.mid_layout = QHBoxLayout()
        self.outer_layout.addLayout(self.mid_layout)

        self.create_left_controls()
        self.create_right_view_panel()
        self.mid_layout.addLayout(self.left_control_layout, stretch=1)
        self.mid_layout.addWidget(self.right_view_panel, stretch=2)

        # Frame control
        self.camera_worker.new_frame_ready.connect(self.handle_new_frame_signal)
        self.first_frame_received = False
        self.last_captured_frame = np.zeros([160, 128])
        self.stationary_ref_frame = self.last_captured_frame.copy()

       # Inside FlyExperimentApp.__init__
        self.auto_stop_timer = QTimer(self)
        self.auto_stop_timer.setSingleShot(True)
        self.auto_stop_timer.timeout.connect(self.stop_camera_manually)


        # Triangle
        self.active_edit_mode = None  # None, "left", "right", "outer", "inner"
        self.points = {
            "left_hinge": (73.7, 64.3),
            "right_hinge": (89.7, 63.9),
            "outer_tip": (80.0, 20.0),
            "inner_tip": (80.0, 30.0),
            "inner_ratio": 5/6.0
        }

        self.init_overlays()

        self.active_run_mode = None # None, "preview", "test_wsa", "test_frame_rate", "run"

        # Load User Default settings
        self.load_setup_default()
        self.status_bar.showMessage("Ready")


    # ----------------------------------------
    # UI Layout Helpers (Big Picture Setup)
    # ----------------------------------------
    def create_top_control_bar(self):
        # Define the class attribute as a QGridLayout
        self.top_control_bar = QGridLayout() 

        # --- Adapter ---
        self.top_control_bar.addWidget(QLabel("Adapter"), 0, 0)
        self.combo_adapter = QComboBox()
        if PYSPIN_AVAILABLE:
            self.combo_adapter.addItems(["Spinnaker Cams", "Simulation"])
        else:
            self.combo_adapter.addItem("Simulation")
        self.top_control_bar.addWidget(self.combo_adapter, 0, 1)
        # to detect change

    
        # --- Camera ID & Mode ---
        self.top_control_bar.addWidget(QLabel("Camera ID"), 0, 2)
        self.combo_cam_id = QComboBox()
        self.top_control_bar.addWidget(self.combo_cam_id, 0, 3)
    
        self.top_control_bar.addWidget(QLabel("Camera mode"), 0, 4)
        self.combo_cam_mode = QComboBox()
        self.top_control_bar.addWidget(self.combo_cam_mode, 0, 5)
    
        # --- Preview/Stop Button ---
        self.btn_preview = QPushButton("Preview")
        self.top_control_bar.addWidget(self.btn_preview, 0, 6)
    
        # --- Duration ---
        self.preview_duration = QLineEdit("30")
        self.top_control_bar.addWidget(self.preview_duration, 0, 7)
        self.top_control_bar.addWidget(QLabel("s"), 0, 8)
    
        # Signal for Dynamic Repopulation
        self.combo_adapter.currentTextChanged.connect(self.on_adapter_changed)
        self.combo_cam_id.currentIndexChanged.connect(self.refresh_camera_modes)
        self.btn_preview.clicked.connect(self.toggle_run_mode)
    
        # Run initial population logic
        self.initialize_hardware_state()



    def create_left_controls(self):
        """Creates the 'Wing Beat Contour Extraction' and 'WB_interpreter' sections."""
        self.left_control_layout = QVBoxLayout()
        self.left_control_layout.setContentsMargins(0, 0, 0, 0)

        # Group 1: Contour Extraction
        group_contour = QGroupBox("Wing Stroke Contour Extraction Settings")
        layout_contour = QVBoxLayout(group_contour)

        # Buttons with coordinate readouts
        
        h = QHBoxLayout()
        self.btn_left_hinge = QPushButton("Left wing hinge")
        self.readout_left_hinge = QLabel("(XX.X, YY.Y)")
        self.readout_left_hinge.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        h.addWidget(self.btn_left_hinge, 2)
        h.addWidget(self.readout_left_hinge, 1)
        layout_contour.addLayout(h)

        h = QHBoxLayout()
        self.btn_right_hinge = QPushButton("Right wing hinge")
        self.readout_right_hinge = QLabel("(XX.X, YY.Y)")
        self.readout_right_hinge.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        h.addWidget(self.btn_right_hinge, 2)
        h.addWidget(self.readout_right_hinge, 1)
        layout_contour.addLayout(h)

        h = QHBoxLayout()
        self.btn_outer_tip = QPushButton("Outer front / mask boundary")
        self.readout_outer_tip = QLabel("(XX.X, YY.Y)")
        self.readout_outer_tip.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        h.addWidget(self.btn_outer_tip, 2)
        h.addWidget(self.readout_outer_tip, 1)
        layout_contour.addLayout(h)

        h = QHBoxLayout()
        self.btn_inner_boundary = QPushButton("Inner mask boundary")
        self.readout_inner_boundary = QLabel("(XX.X, YY.Y)")
        self.readout_inner_boundary.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        h.addWidget(self.btn_inner_boundary, 2)
        h.addWidget(self.readout_inner_boundary, 1)
        layout_contour.addLayout(h)

        # Threshold Slider and Text
        layout_contour.addWidget(QLabel("Body/wing outline threshold (for noise reduction)"))
        slider_layout = QHBoxLayout()
        slider_layout.setContentsMargins(10, 0, 10, 0)
        self.slider_thresh = QSlider(Qt.Horizontal)
        self.slider_thresh.setRange(0, 255)
        self.slider_thresh.setValue(10)
        self.thresh_val_label = QLabel("10")
        slider_layout.addWidget(self.slider_thresh)
        slider_layout.addWidget(self.thresh_val_label, stretch=0)
        layout_contour.addLayout(slider_layout)

        self.btn_show_thresh = QPushButton("Show threshold effect")
        self.btn_show_thresh.setCheckable(True)
        layout_contour.addWidget(self.btn_show_thresh)


        # For backgroiund subtraction
        h_layout = QHBoxLayout()
        self.btn_not_flying = QPushButton("Capture for background subtraction")
        h_layout.addWidget(self.btn_not_flying)
        layout_contour.addLayout(h_layout)


        # # Apply/Cancel
        # h_action = QHBoxLayout()
        # self.btn_cancel = QPushButton("Cancel")
        # self.btn_apply = QPushButton("Apply")
        # h_action.addWidget(self.btn_cancel)
        # h_action.addWidget(self.btn_apply)
        # layout_contour.addLayout(h_action)

        self.left_control_layout.addWidget(group_contour)

        # Group 3: WSA RUN (The smaller box from the bottom)
        group_test = QGroupBox("Test Wing Stroke Analyzer")
        layout_test = QGridLayout(group_test)


        test_btn_layout = QHBoxLayout()
        self.btn_test_wsa = QPushButton("Test WSA")
        test_btn_layout.addWidget(self.btn_test_wsa, 3)
        self.btn_test_frame_rate = QPushButton("Test frame rate")
        test_btn_layout.addWidget(self.btn_test_frame_rate, 3)
        self.lbl_hz = QLabel("0")
        self.lbl_hz.setAlignment(Qt.AlignCenter)
        test_btn_layout.addWidget(self.lbl_hz, 1)
        test_btn_layout.addWidget(QLabel("Hz"), 1)
        layout_test.addLayout(test_btn_layout, 0, 0, 1, 2)


        self.left_control_layout.addWidget(group_test)
        self.left_control_layout.addStretch() # Push everything up




        # Group 3: WSA RUN (The smaller box from the bottom)
        group_run = QGroupBox("RUN Wing Stroke Analyzer")
        layout_run = QGridLayout(group_run)


        h = QHBoxLayout()
        self.chk_save = QCheckBox("Save movie: ")
        self.chk_save.setChecked(False)
        h.addWidget(self.chk_save)
        self.fly_ID = QLineEdit("a")
        h.addWidget(self.fly_ID)
        h.addWidget(QLabel("(fly ID)"))
        layout_run.addLayout(h,1,0, 1, 2)
        
        h = QHBoxLayout()
        h.addWidget(QLabel("while NI Dev:"))
        self.nidaq_dev_name_for_DIO = QComboBox()
        self.populate_devices(self.nidaq_dev_name_for_DIO)
        h.addWidget(self.nidaq_dev_name_for_DIO)
        h.addWidget(QLabel("DIO Port"))
        self.nidaq_DIO_port = QLineEdit("0")
        h.addWidget(self.nidaq_DIO_port)
        h.addWidget(QLabel("Line"))
        self.nidaq_DI_line = QLineEdit("0")
        h.addWidget(self.nidaq_DI_line)
        h.addWidget(QLabel("is 1."))
        layout_run.addLayout(h,2,0, 1, 2)
        
        h = QHBoxLayout()
        h.addWidget(QLabel(" Sync TTL: DIO Port(same as above)/"))
        h.addWidget(QLabel("Output Line"))
        self.nidaq_DO_line = QLineEdit("1")
        h.addWidget(self.nidaq_DO_line)
        h.addWidget(QLabel("      "))
        layout_run.addLayout(h,3,0,1,2)
        
        h = QHBoxLayout()
        h.addWidget(QLabel("One out of "))
        self.save_frame_skip = QLineEdit("7")
        h.addWidget(self.save_frame_skip)
        h.addWidget(QLabel("frames will be saved (use 1 to save all)"))
        layout_run.addLayout(h,4,0, 1, 2)
        
        h = QHBoxLayout()
        self.chk_update_fly = QCheckBox("Display fly image every")
        self.chk_update_fly.setChecked(True)
        h.addWidget(self.chk_update_fly)
        self.update_rate = QLineEdit("10")
        self.update_rate.setAlignment(Qt.AlignRight)
        h.addWidget(self.update_rate)
        h.addWidget(QLabel(" frames                    "))
        layout_run.addLayout(h, 5, 0, 1, 2)


        h = QHBoxLayout()
        h.addWidget(QLabel("Run time (sec): "), 1)
        self.run_time = QLineEdit("3600")
        self.run_time.setAlignment(Qt.AlignRight)
        h.addWidget(self.run_time, 1)
        layout_run.addLayout(h, 6, 0, 1, 1)

        h = QHBoxLayout()
        h.addWidget(QLabel("NI AO Dev:"), 1)
        self.nidaq_dev_name_for_AO = QComboBox()
        self.populate_devices(self.nidaq_dev_name_for_AO)
        h.addWidget(self.nidaq_dev_name_for_AO, 1)
        layout_run.addLayout(h, 7, 0, 1, 1)

        h = QHBoxLayout()
        h.addWidget(QLabel("LWS AO:"), 1)
        self.nidaq_AO_LWS = QLineEdit("ao0")
        self.nidaq_AO_LWS.setAlignment(Qt.AlignRight)
        h.addWidget(self.nidaq_AO_LWS, 1)
        h.addWidget(QLabel("L-R AO:"), 1)
        self.nidaq_AO_L_R = QLineEdit("ao1")
        self.nidaq_AO_L_R.setAlignment(Qt.AlignRight)
        h.addWidget(self.nidaq_AO_L_R, 1)
        layout_run.addLayout(h, 8, 0, 1, 1)

        self.btn_run = QPushButton("Run")
        self.btn_run.setStyleSheet("background-color: #9ac2f4; min-width: 100px; min-height: 50px; color: black; font-weight: bold; font-size: 18px;")
        layout_run.addWidget(self.btn_run, 6, 1, 3, 1)
        


        self.left_control_layout.addWidget(group_run)
        self.left_control_layout.addStretch() # Push everything up


        # 1. Connect Buttons (Assuming these names match your UI)
        self.btn_not_flying.clicked.connect(self.on_click_not_flying)
        self.btn_left_hinge.clicked.connect(self.toggle_edit_mode)
        self.btn_right_hinge.clicked.connect(self.toggle_edit_mode)
        self.btn_outer_tip.clicked.connect(self.toggle_edit_mode)
        self.btn_inner_boundary.clicked.connect(self.toggle_edit_mode)
        self.slider_thresh.valueChanged.connect(self.on_threshold_changed)

        self.btn_show_thresh.clicked.connect(self.toggle_threshold_mode)
        
        self.btn_test_wsa.clicked.connect(self.toggle_run_mode)
        self.btn_test_frame_rate.clicked.connect(self.toggle_run_mode)
        self.btn_run.clicked.connect(self.toggle_run_mode)

        

    def create_right_view_panel(self):
        """Creates the main image panel and the plot below it."""
        self.right_view_panel = QSplitter(Qt.Vertical) # Resizable split

        # 1. High Performance Image View
        self.win_img = pg.GraphicsLayoutWidget()
        self.win_img.ci.setContentsMargins(0, 0, 0, 0) # Remove padding around the image
        self.win_img.ci.setSpacing(0) # Remove spacing between items in the GraphicsLayout (to remove gap between image and axes)
        self.view_img = self.win_img.addViewBox()
        self.view_img.setAspectLocked(True)
        self.view_img.setDefaultPadding(0.0) # Remove default padding of viewbox (view_img)
        self.view_img.autoRange(padding=0)
        self.view_img.invertY() # Invert Y to match screen coords (0,0 is top-left)
        
        # This handles the main image data
        self.img_item = pg.ImageItem()
        self.img_item.setImage(np.zeros((160, 128), dtype=np.uint8) ) # initialize the image
        self.view_img.addItem(self.img_item)
        
        # Setup the axes
        self.view_img.setRange(xRange=[0, 160], yRange=[0, 128], padding=0)
        #self.axis_x = pg.AxisItem(orientation='bottom')
        #self.axis_x.setParentItem(self.view_img)
        #self.axis_y = pg.AxisItem(orientation='left')
        #self.axis_y.setParentItem(self.view_img)

       
        # 2. Add processed data Plot (Currently blank placeholder)
        self.win_plot = pg.PlotWidget(title="Processed Real-time Data")
        # Blank placeholder style matching MATLAB 0 to 1 axes
        self.win_plot.getPlotItem().setRange(xRange=[0, 1], yRange=[0, 1])

        self.right_view_panel.addWidget(self.win_img)
        self.right_view_panel.addWidget(self.win_plot)
        self.right_view_panel.setSizes([400,200]) # Initial size ratio

        # Prep for plotting
        self.plot_max_points = 1500 
        self.plot_times = deque(maxlen=self.plot_max_points)
        self.plot_left_angles = deque(maxlen=self.plot_max_points)
        self.plot_right_angles = deque(maxlen=self.plot_max_points)
        self.plot_start_time = None

        self.left_curve = self.win_plot.plot(pen=pg.mkPen('r', width=2), name="Left")
        self.right_curve = self.win_plot.plot(pen=pg.mkPen('g', width=2), name="Right")

        self.win_plot.setLabel('bottom', 'Time', 's')
        self.win_plot.setLabel('left', 'Wing Angle', 'deg')

        # 2. Hook the Image Click
        self.img_item.scene().sigMouseClicked.connect(self.on_scene_clicked)
        


    def init_overlays(self):
        """Creates the persistent graphics items and does the first draw."""
        # 1. Create the persistent PlotDataItems
        # Using 'outer_line' and 'inner_line' to match your redraw_overlays function
        self.outer_line = pg.PlotDataItem(pen=pg.mkPen('b', width=2))
        self.inner_line = pg.PlotDataItem(pen=pg.mkPen('b', width=1, style=QtCore.Qt.DashLine))
        self.left_wing_front_edge = pg.PlotDataItem(pen=pg.mkPen('r', width=2))
        self.right_wing_front_edge = pg.PlotDataItem(pen=pg.mkPen('g', width=2))
        
        # 2. Add them to your ViewBox/PlotWidget
        # Make sure 'self.view_img' (or whatever you named your PlotWidget) is already initialized!
        self.view_img.addItem(self.outer_line)
        self.view_img.addItem(self.inner_line)
        self.view_img.addItem(self.left_wing_front_edge)
        self.view_img.addItem(self.right_wing_front_edge)
        
        # 3. Text Labels
        self.text_lw = pg.TextItem("LW", color=(255, 0, 0), anchor=(0, 1))
        self.text_lw.setPos(60, 60)
        self.view_img.addItem(self.text_lw)

        self.text_rw = pg.TextItem("RW", color=(0, 255, 0), anchor=(1, 1))
        self.text_rw.setPos(100, 60)
        self.view_img.addItem(self.text_rw)
        
        # 4. Apply the 5/6th default logic for the inner tip
        self.set_default_inner_median()
        
        # 5. Perform the first visual update
        self.redraw_overlays()


        # theta = np.linspace(np.pi/2, np.pi/4, 20)
        # self.curve_rw = pg.PlotCurveItem(100*np.cos(theta), 100*np.sin(theta), pen=pg.mkPen('g', width=2))
        # self.curve_rw.setPos(70, 0) # Pos from MATLAB
        # self.view_img.addItem(self.curve_rw)
        
        # self.curve_lw = pg.PlotCurveItem(100*np.cos(theta), 100*np.sin(theta), pen=pg.mkPen('r', width=2))
        # self.curve_lw.setPos(80, 0)
        # self.view_img.addItem(self.curve_lw)
        # # You'll need to define the 'grid segments' here too.

        

    # ==========================================
    # PART 3: Slot / Callback Function Stubs
    # ==========================================

    
    
    # ==========================================
    # Camera hardware (top bar) functions
    # ==========================================
    @Slot(str)
    def on_adapter_changed(self, text):
        self.initialize_hardware_state()

    def initialize_hardware_state(self):
        self.combo_cam_id.clear()
        self.combo_cam_mode.clear()
        
        if PYSPIN_AVAILABLE and self.combo_adapter.currentText() == "Spinnaker Cams":
            self.refresh_camera_ids() # Populates Cam ID and triggers Mode refresh
        else:
            self.combo_cam_id.addItem("Simulation")
            self.combo_cam_mode.addItem("Simulation")
            self.stop_camera_manually() # Force a disconnect reset

    def refresh_camera_ids(self):
        """Queries Spinnaker for connected cameras and updates the dropdown."""
        self.combo_cam_id.clear()

        # Check if we are in Simulation Mode
        if self.combo_adapter.currentText() == "Simulation":
            self.combo_cam_id.addItem("Simulation")
            return
        
        system = PySpin.System.GetInstance()
        cam_list = system.GetCameras()
        count = cam_list.GetSize()
        
        if count == 0:
            self.combo_cam_id.addItem("No cameras found")
        else:
            for i in range(count):
                cam = cam_list[i]  # This creates a reference
                
                nodemap_tldevice = cam.GetTLDeviceNodeMap()
                
                node_model = PySpin.CStringPtr(nodemap_tldevice.GetNode("DeviceModelName"))
                model = node_model.GetValue() if PySpin.IsReadable(node_model) else "Unknown"
                
                node_serial = PySpin.CStringPtr(nodemap_tldevice.GetNode("DeviceSerialNumber"))
                serial = node_serial.GetValue() if PySpin.IsReadable(node_serial) else f"ID_{i}"
                
                self.combo_cam_id.addItem(f"{model} ({serial})", userData=serial)
                
                # --- CRITICAL: Manually release the reference to THIS camera ---
                del cam 
                
            print(f"Discovered {count} camera(s).")

        # --- CRITICAL: Clear the list and delete the list reference ---
        cam_list.Clear()
        del cam_list 
        
        # Now the "door" is clear to be closed
        system.ReleaseInstance()

    @Slot()
    def refresh_camera_modes(self):
        """Replacement for FlyCapture F7 modes using Spinnaker PixelFormat."""
        self.combo_cam_mode.clear()

        # Check if we are in Simulation Mode
        if self.combo_cam_id.currentText() == "Simulation":
            self.combo_cam_mode.addItem("Simulation")
            return
        
        # Get the serial from the ID dropdown
        target_serial = self.combo_cam_id.currentData()
        if not target_serial:
            return
    
        system = PySpin.System.GetInstance()
        cam_list = system.GetCameras()
        
        try:
            # 1. Access the specific camera
            cam = cam_list.GetBySerial(target_serial)
            cam.Init() # Must Init to read nodes
            
            # 2. Get the PixelFormat Enumeration
            nodemap = cam.GetNodeMap()
            node_pixel_format = PySpin.CEnumerationPtr(nodemap.GetNode("PixelFormat"))
            
            if PySpin.IsReadable(node_pixel_format):
                # Get all supported modes (Mono8, Mono12, Mono16, etc.)
                entries = node_pixel_format.GetEntries()
                for entry in entries:
                    entry_ptr = PySpin.CEnumEntryPtr(entry)
                    if PySpin.IsAvailable(entry_ptr) and PySpin.IsReadable(entry_ptr):
                        self.combo_cam_mode.addItem(entry_ptr.GetSymbolic())
            
            # 3. Handle the "F7" Resolution aspect (Width/Height)
            # If you need to see current resolution in the dropdown label:
            width = PySpin.CIntegerPtr(nodemap.GetNode("Width")).GetValue()
            height = PySpin.CIntegerPtr(nodemap.GetNode("Height")).GetValue()
            print(f"Current Resolution: {width}x{height}")
    
            cam.DeInit()
            del cam # Crucial for -1004 error prevention
            
        except Exception as e:
            print(f"Could not query modes: {e}")
            self.combo_cam_mode.addItem("Default")
    
        # Clean up
        cam_list.Clear()
        del cam_list
        system.ReleaseInstance()


    # ==========================================
    # NIDAQ device
    # ==========================================

    def populate_devices(self, h):
        h.clear()
        devices = get_daq_device_names()
        
        if devices:
            h.addItems(devices)
        else:
            h.addItem("No Devices Found")



    # ==========================================
    # Experiment (image processing) functions
    # ==========================================
    @Slot() # The main slot for all 4 RUN-related buttons. It identifies which button was clicked and acts accordingly.
    def toggle_run_mode(self):
        # Identify which button sent the signal
        clicked_btn = self.sender()
        mode_map = {
            self.btn_preview: "preview",
            self.btn_test_wsa: "test_wsa",
            self.btn_test_frame_rate: "test_frame_rate",
            self.btn_run: "run"
        }
        requested_mode = mode_map.get(clicked_btn)

        
        # If already in this mode, turn it off and enable all
        if self.active_run_mode == requested_mode:
            self.active_run_mode = None
            self.stop_camera_manually()
            self.btn_run.setStyleSheet("background-color: #9ac2f4; min-width: 100px; min-height: 50px; color: black; font-weight: bold; font-size: 18px;")
            
            if requested_mode == "preview":
                pass
            elif requested_mode == "test_wsa":
                pass
            elif requested_mode == "test_frame_rate":
                pass
            elif requested_mode == "run":
                pass
                
        else:
            # Activate the specific mode, disable others, turn red
            self.active_run_mode = requested_mode
            for btn in mode_map.keys():
                if btn == clicked_btn:
                    btn.setEnabled(True)
                    btn.setStyleSheet("background-color: red; color: white; font-weight: bold;")
                else:
                    btn.setEnabled(False)
                    btn.setStyleSheet("")
                if requested_mode != "run":
                    self.btn_run.setStyleSheet("min-width: 100px; min-height: 50px;")

            self.first_frame_received = False    
            # 1. Pass the Serial Number
            self.camera_worker.target_serial = self.combo_cam_id.currentData()
            # 2. Pass the Video Mode string (e.g., "Mode 0")
            self.camera_worker.target_mode = self.combo_cam_mode.currentText()
            # 3. Set task mode
            self.camera_worker.update_settings(task_mode=requested_mode)
            # 4. Set the timer and other options based on the mode
            try:
                if requested_mode == "preview":
                    duration_ms = int(float(self.preview_duration.text()) * 1000)
                elif requested_mode == "test_wsa":
                    duration_ms = 12000
                elif requested_mode == "test_frame_rate":
                    duration_ms = 12000
                elif requested_mode == "run":
                    duration_ms = int(float(self.run_time.text()) * 1000)
                    self.btn_run.setStyleSheet("background-color: red; color: white; min-width: 100px; min-height: 50px; font-weight: bold; font-size: 18px;")
                
                if requested_mode in ["run", "test_wsa"]:
                    # Clear old data
                    self.plot_times.clear()
                    self.plot_left_angles.clear()
                    self.plot_right_angles.clear()
                    self.plot_start_time = None 
                    
                    # Set plot range for a clean start
                    self.win_plot.setXRange(0, 10)
                    self.win_plot.setYRange(0, 200) # Wing angles typically 0-190

                if requested_mode in ["run", "test_wsa", "test_frame_rate"]:
                    run_options = {
                        "save_movie": self.chk_save.isChecked(),
                        "filename_postfix": datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+"_"+self.fly_ID.text(),
                        "nidaq_dev_name_for_DIO": self.nidaq_dev_name_for_DIO.currentText(),
                        "nidaq_DIO_port": self.nidaq_DIO_port.text(),
                        "nidaq_DI_line": self.nidaq_DI_line.text(),
                        "nidaq_DO_line": self.nidaq_DO_line.text(),
                        "save_frame_skip": int(self.save_frame_skip.text()),
                        "update_fly": self.chk_update_fly.isChecked(),
                        "update_rate": int(self.update_rate.text()),
                        "nidaq_dev_name_for_AO": self.nidaq_dev_name_for_AO.currentText(),
                        "nidaq_AO_LWS": self.nidaq_AO_LWS.text(),
                        "nidaq_AO_L_R": self.nidaq_AO_L_R.text()
                    }
                    self.camera_worker.update_settings(run_options=run_options)
            except ValueError:
                duration_ms = 600000

            # 5. Start the thread
            self.auto_stop_timer.start(duration_ms)
            self.camera_worker.start()

    def stop_camera_manually(self):
        """Stops all active streams and resets UI."""
        self.active_run_mode = None
        self.auto_stop_timer.stop()

        if self.camera_worker.isRunning():
            self.camera_worker.stop()

        # Frame rate was tested
        if self.btn_test_frame_rate.isEnabled():
            with self.camera_worker._frame_lock:
                c = self.camera_worker.test_frame_rate_frame_count
                s = self.camera_worker.test_frame_rate_timestamp_first
                l = self.camera_worker.test_frame_rate_timestamp_last
            
            if s == 0 or l == 0 or l-s < 0.001:
                fr = 0
            else:
                fr = c/(l-s)
            self.lbl_hz.setText(f"{fr:.2f}")

        # Reset buttons
        self.btn_preview.setEnabled(True)
        self.btn_preview.setStyleSheet("")
        self.btn_test_wsa.setEnabled(True)
        self.btn_test_wsa.setStyleSheet("")
        self.btn_test_frame_rate.setEnabled(True)
        self.btn_test_frame_rate.setStyleSheet("")
        self.btn_run.setEnabled(True)
        self.btn_run.setStyleSheet("background-color: #9ac2f4; min-width: 100px; min-height: 50px; color: black; font-weight: bold; font-size: 18px;")
        
    @Slot()
    def on_click_not_flying(self):
        """Captures the current frame as the 'Stationary' reference."""
        self.stationary_ref_frame = self.last_captured_frame.copy()
        print(self.stationary_ref_frame.shape)
        self.camera_worker.update_settings(stationary_ref_frame=self.stationary_ref_frame)
        self.camera_worker.update_settings(new_stationary_ref_frame_available=True) # Trigger mask recalculation if needed
        print("Captured the current frame.")

    @Slot(int)
    def on_threshold_changed(self, value):
        """Updates the threshold value across the UI and the CameraWorker."""
        # 1. Update the text label next to the slider
        self.thresh_val_label.setText(str(value))
        
        # 2. Update your local points/settings dictionary if needed
        # self.points["threshold"] = value 

        # 3. Push to the CameraWorker safely
        # This uses the thread-safe update_settings method you already have
        self.camera_worker.update_settings(threshold=value)

    @Slot()
    def toggle_threshold_mode(self):
        if self.btn_show_thresh.isChecked():
            self.btn_show_thresh.setText("STOP showing threshold effect")
            self.btn_show_thresh.setStyleSheet("background-color: red; color: white;")
            with self.camera_worker._setting_lock:
                self.camera_worker.settings.threshold_show_effect = True
        else:
            self.btn_show_thresh.setText("Show threshold effect")
            self.btn_show_thresh.setStyleSheet("")
            with self.camera_worker._setting_lock:
                self.camera_worker.settings.threshold_show_effect = False
    
        
        



    # ==========================================
    # Receiving Frames and Display Logic
    # ==========================================
    @Slot()
    def handle_new_frame_signal(self):
        """
        Triggered by the signal, but pulls data manually.
        If 5 signals arrive but the GUI is busy, only the 
        latest one will be pulled when the GUI finally wakes up.
        """
        with self.camera_worker._frame_lock:
            fr = self.camera_worker.latest_frame # With buffer swap, this should be newest
            timestamp = self.camera_worker.latest_timestamp
            # IMPORTANT: Clear the reference if you want to ensure no double-processing
            self.camera_worker.latest_frame = None 

        if fr is None:
            return
            
        """The core display loop. Runs efficient image mapping."""
        # Ensure data is 2D before attempting to swap axes
        if fr.ndim == 2:
            # pyqtgraph prefers (Columns, Rows)
            frame = fr
        elif fr.ndim > 2:
            # If it's RGB or has extra dims, just take the first slice
            frame = fr[..., 0]
            
        self.img_item.setImage(frame, autoLevels=False)
        self.last_captured_frame = frame
            
        # If this is the first frame after clicking Preview, 
        # zoom out to show the full sensor (Fixes the cropping)
        if not self.first_frame_received:
            self.view_img.autoRange(padding=0) 
            self.first_frame_received = True
        
        # Redraw the wing edges
        self.redraw_overlays_wing_front_edge()


        if self.active_run_mode in ["test_wsa", "run"]:
            if self.plot_start_time is None:
                self.plot_start_time = time.perf_counter()
                
            current_time = time.perf_counter() - self.plot_start_time
            
            # Get angles from worker settings safely
            angles = self.camera_worker.get_settings('wing_angles')
            
            # Append data
            self.plot_times.append(current_time)
            self.plot_left_angles.append(angles['left'])
            self.plot_right_angles.append(angles['right'])
            
            # Update plot lines
            self.left_curve.setData(list(self.plot_times), list(self.plot_left_angles))
            self.right_curve.setData(list(self.plot_times), list(self.plot_right_angles))
            
            # Enforce the 10-second X-axis window (showing last 10s)
            self.win_plot.setXRange(current_time - 10, current_time, padding=0)        




    # ==========================================
    # Triangle Geometry
    # ==========================================
    @Slot()
    def toggle_edit_mode(self):
        # Identify which button sent the signal
        clicked_btn = self.sender()
        mode_map = {
            self.btn_left_hinge: "left",
            self.btn_right_hinge: "right",
            self.btn_outer_tip: "outer",
            self.btn_inner_boundary: "inner"
        }
        requested_mode = mode_map.get(clicked_btn)

        # If already in this mode, turn it off and enable all
        if self.active_edit_mode == requested_mode:
            self.active_edit_mode = None
            for btn in mode_map.keys():
                btn.setEnabled(True)
                btn.setStyleSheet("") 
        else:
            # Activate the specific mode, disable others, turn red
            self.active_edit_mode = requested_mode
            for btn in mode_map.keys():
                if btn == clicked_btn:
                    btn.setEnabled(True)
                    btn.setStyleSheet("background-color: red; color: white; font-weight: bold;")
                else:
                    btn.setEnabled(False)
                    btn.setStyleSheet("")

    def on_scene_clicked(self, event):
        print("inside on_scene_clicked")
        # Map the scene click back to the image coordinates
        pos = self.img_item.mapFromScene(event.scenePos())
        x, y = pos.x(), pos.y()
        print(f"  MouseClick: x={x}, y={y}")
        
        # Check if the click was actually inside the image bounds
        if 0 <= x <= self.img_item.width() and 0 <= y <= self.img_item.height():
            self.update_points_from_mode(x, y)
            self.redraw_overlays()   
            
    def update_inner_tip_from_click(self, click_x, click_y):
        # 1. Calculate Midpoint of Hinge Base
        lx, ly = self.points["left_hinge"]
        rx, ry = self.points["right_hinge"]
        mid_x, mid_y = (lx + rx) / 2, (ly + ry) / 2
        
        # 2. Outer Median Vector
        tx, ty = self.points["outer_tip"]
        median_vec = np.array([tx - mid_x, ty - mid_y])
        outer_median_len = np.linalg.norm(median_vec)
        
        if outer_median_len == 0: return
        
        unit_median = median_vec / outer_median_len
        
        # 3. Project click onto the median line to find the closest point
        click_vec = np.array([click_x - mid_x, click_y - mid_y])
        projection_len = np.dot(click_vec, unit_median)
        
        # 4. Apply Constraints
        # Min: Half of outer median
        # Max: Outer median minus 3 pixels
        min_len = outer_median_len * 0.5
        max_len = outer_median_len - 3.0
        
        final_len = np.clip(projection_len, min_len, max_len)
        
        # 5. Set Inner Tip coordinate
        new_inner_x = mid_x + unit_median[0] * final_len
        new_inner_y = mid_y + unit_median[1] * final_len
        self.points["inner_tip"] = (new_inner_x, new_inner_y)

    def update_points_from_mode(self, x, y):
        p = self.points
        
        # CASE 1: Moving either Hinge or the Outer Tip
        if self.active_edit_mode in ["left", "right", "outer"]:
            if self.active_edit_mode == "left":
                p["left_hinge"] = (x, y)
            elif self.active_edit_mode == "right":
                p["right_hinge"] = (x, y)
            elif self.active_edit_mode == "outer":
                p["outer_tip"] = (x, y)
    
            # Sync the Inner Tip to the new geometry using the stored ratio
            (mid_x, mid_y), median_vec, _ = self.get_outer_median_data()
            p["inner_tip"] = (
                mid_x + median_vec[0] * p["inner_ratio"],
                mid_y + median_vec[1] * p["inner_ratio"]
            )
    
        # CASE 2: Manually adjusting the Mask Depth
        elif self.active_edit_mode == "inner":
            # 1. Project click onto median (enforces the line & 3px constraints)
            self.update_inner_tip_from_click(x, y)
            
            # 2. Update the ratio based on this new position
            (mid_x, mid_y), median_vec, outer_len = self.get_outer_median_data()
            ix, iy = p["inner_tip"]
            inner_len = np.sqrt((ix - mid_x)**2 + (iy - mid_y)**2)
            
            if outer_len > 0:
                p["inner_ratio"] = inner_len / outer_len
        
    def get_outer_median_data(self):
        """
        Calculates the geometry of the outer triangle's median line.
        Returns:
            midpoint (tuple): (x, y) center of the hinge base
            median_vector (ndarray): The vector from midpoint to outer_tip
            length (float): The pixel length of the outer median
        """
        p = self.points
        lx, ly = p["left_hinge"]
        rx, ry = p["right_hinge"]
        tx, ty = p["outer_tip"]
        
        # 1. Calculate the midpoint of the base (between the two hinges)
        mid_x = (lx + rx) / 2.0
        mid_y = (ly + ry) / 2.0
        
        # 2. Create the vector from midpoint to the outer tip
        # We use numpy for easy vector math later
        median_vec = np.array([tx - mid_x, ty - mid_y])
        
        # 3. Calculate the total length of this median
        length = np.linalg.norm(median_vec)
        
        return (mid_x, mid_y), median_vec, length

    def set_default_inner_median(self):
        """Sets default to 1/6 of the outer median length."""
        lx, ly = self.points["left_hinge"]
        rx, ry = self.points["right_hinge"]
        tx, ty = self.points["outer_tip"]
        
        mid_x, mid_y = (lx + rx) / 2, (ly + ry) / 2
        vec = np.array([tx - mid_x, ty - mid_y])
        
        # Using 1/6th of total length from the base
        default_inner = (mid_x + vec[0]*5/6, mid_y + vec[1]*5/6)
        self.points["inner_tip"] = default_inner

    def redraw_overlays_wing_front_edge(self):
        p = self.points
        lx, ly = p["left_hinge"]
        rx, ry = p["right_hinge"]
        lwtx, lwty = self.calculate_wing_tip("left") # left wing tip
        rwtx, rwty = self.calculate_wing_tip("right") # right wing tip
        # Wing frong edges
        self.left_wing_front_edge.setData([lx,lwtx], [ly, lwty])
        self.right_wing_front_edge.setData([rx,rwtx], [ry, rwty])

    def redraw_overlays(self):
        p = self.points
        lx, ly = p["left_hinge"]
        rx, ry = p["right_hinge"]
        tx, ty = p["outer_tip"]
        ix, iy = p["inner_tip"]

        self.redraw_overlays_wing_front_edge()
        
        # Outer Triangle (Solid)
        # Sequence: Left -> Tip -> Right -> Left
        self.outer_line.setData([lx, tx, rx, lx], [ly, ty, ry, ly], pen='b')
        
        # Inner Triangle (Dashed)
        # Sequence: Left -> InnerTip -> Right
        # Note: It shares the base with the outer triangle
        dashed_pen = pg.mkPen('b', style=QtCore.Qt.DashLine)
        self.inner_line.setData([lx, ix, rx], [ly, iy, ry], pen=dashed_pen)
        
        # Update text labels next to buttons
        self.readout_left_hinge.setText(f"({lx:.1f}, {ly:.1f})")
        self.readout_right_hinge.setText(f"({rx:.1f}, {ry:.1f})")
        self.readout_outer_tip.setText(f"({tx:.1f}, {ty:.1f})")
        self.readout_inner_boundary.setText(f"({ix:.1f}, {iy:.1f})")

        self.draw_masks()

    def check_geometry_integrity(self):
        lx, ly = self.points["left_hinge"]
        rx, ry = self.points["right_hinge"]
        
        # Check if hinges are too close
        dist = np.sqrt((lx-rx)**2 + (ly-ry)**2)
        if dist < 5:
            print("Warning: Hinges are too close!")
            return False
            
        # Check if inner tip is still within bounds 
        # (Useful if outer tip was moved after inner tip was set)
        self.update_inner_tip_from_click(*self.points["inner_tip"])
        return True



        
    # ==========================================
    # Mask Calculation
    # ==========================================
    def calculate_wing_masks(self, side="left"):
        p = self.points
        # 1. Setup centers and tips based on side
        hinge = np.array(p[f"{side}_hinge"])
        o_tip = np.array(p["outer_tip"])
        i_tip = np.array(p["inner_tip"])
    
        # 2. Calculate Radii
        r_out = np.linalg.norm(o_tip - hinge)
        r_in = np.linalg.norm(i_tip - hinge)
    
        # 3. Calculate Base Angle (radians)
        # The line from hinge to outer_tip is our 0° reference
        diff = o_tip - hinge
        base_angle = np.arctan2(diff[1], diff[0])
    
        # 4. Determine "Outward" Direction
        # Vector from this hinge to the tip
        vec_to_tip = o_tip - hinge
        # Vector from this hinge to the OTHER hinge
        other_side = "right" if side == "left" else "left"
        other_hinge = np.array(p[f"{other_side}_hinge"])
        vec_to_other = other_hinge - hinge

        # 2D Cross Product: (x1*y2 - y1*x2)
        # This tells us if the 'other' hinge is clockwise or counter-clockwise
        cross_prod = vec_to_tip[0] * vec_to_other[1] - vec_to_tip[1] * vec_to_other[0]

        # If cross_prod > 0, the 'inside' is counter-clockwise (+). 
        # So 'outside' must be clockwise (-1).
        # If cross_prod < 0, the 'inside' is clockwise (-). 
        # So 'outside' must be counter-clockwise (+1).
        direction = -1 if cross_prod > 0 else 1
        
        mask_quads = []
        
        for i in range(12):
            # Each mask covers 10 degrees
            angle_start = base_angle + direction * np.radians(i * 10)
            angle_end = base_angle + direction * np.radians((i + 1) * 10)
            
            # Calculate 4 corners (Polar to Cartesian)
            # Order: Inner-Start, Inner-End, Outer-End, Outer-Start
            pts = [
                (hinge[0] + r_in * np.cos(angle_start), hinge[1] + r_in * np.sin(angle_start)),
                (hinge[0] + r_in * np.cos(angle_end),   hinge[1] + r_in * np.sin(angle_end)),
                (hinge[0] + r_out * np.cos(angle_end),  hinge[1] + r_out * np.sin(angle_end)),
                (hinge[0] + r_out * np.cos(angle_start), hinge[1] + r_out * np.sin(angle_start))
            ]
            mask_quads.append(pts)

            # Capture position for the Label at the 10th mask (index 9)
            if i == 9:
                # Midpoint angle of this mask
                mid_angle = (angle_start + angle_end) / 2.0
                # Position slightly inside the inner radius (e.g., 90% of r_in)
                # or exactly at r_in depending on your preference
                tx = hinge[0] + (r_in * 0.85) * np.cos(mid_angle)
                ty = hinge[1] + (r_in * 0.85) * np.sin(mid_angle)
                text_pos = (tx, ty)
            
        return mask_quads, self.precompute_masks(mask_quads), text_pos

    def precompute_masks(self, quad_corners_list):
        """
        quad_corners_list: List of quads, each with 4 (x,y) corners.
        returns: List of boolean masks
        """
        tmp = np.array(quad_corners_list)
        max_dim = int(np.ceil(np.max(tmp)))+1
        masks = []
        for corners in quad_corners_list:
            # Create a blank black canvas for this specific quad
            mask = np.zeros((max_dim, max_dim), dtype=np.uint8)
            
            # Convert corners to integer points for OpenCV
            pts = np.array(corners, dtype=np.int32)
            
            # Fill the quad area with white (255)
            cv2.fillPoly(mask, [pts], 255)
            
            # Store as a boolean array for faster indexing
            masks.append(np.where(mask))
        return masks

    def draw_masks(self):
        # 1. Clean up previous masks from the plot
        if not hasattr(self, 'mask_items'):
            self.mask_items = []
        for item in self.mask_items:
            self.view_img.removeItem(item) # Use view_box or plot_widget as per your code
        self.mask_items = []

        # 2. Draw Left (Red) and Right (Green)
        # Using QColor with transparency (Alpha=100) to match the "overlay" look
        settings = [
            ("left", QColor(255, 0, 0, 100), "RW"), 
            ("right", QColor(0, 255, 0, 100), "LW")
        ]

        masks = {}

        for side, color, label in settings:
            quad_coords, masks[side], text_pos = self.calculate_wing_masks(side)

            # Update text label position
            if side=="left":
                self.text_lw.setPos(text_pos[0], text_pos[1])
                self.text_lw.setAnchor((0.5, 0.5)) # Center the text on the point
            elif side == "right":
                self.text_rw.setPos(text_pos[0], text_pos[1])
                self.text_rw.setAnchor((0.5, 0.5)) # Center the text on the point
            
            # Draw each quad on the plot
            pen = QPen(color)
            pen.setWidth(1)
            for pts in quad_coords:
                # We use pg.PlotDataItem for the quads
                # We repeat the first point at the end to close the shape: [p1, p2, p3, p4, p1]
                xs = [pts[0][0], pts[1][0], pts[2][0], pts[3][0], pts[0][0]]
                ys = [pts[0][1], pts[1][1], pts[2][1], pts[3][1], pts[0][1]]
                
                # Create the quad
                # If you want filled shapes, use 'brush'
                quad_item = pg.PlotDataItem(xs, ys, pen=pen, brush=color)
                self.view_img.addItem(quad_item)
                self.mask_items.append(quad_item)

        self.camera_worker.update_settings(masks=masks)
        self.camera_worker.update_settings(new_masks_available=True) # Send the updated points as well (for logging or other logic)


    # ==========================================
    # Wing front edge calculation
    # ==========================================
    def calculate_wing_tip(self, side="left"):

        with self.camera_worker._setting_lock:
                angle = self.camera_worker.settings.wing_angles[side]


        p = self.points
        # 1. Setup centers and tips based on side
        hinge = np.array(p[f"{side}_hinge"])
        o_tip = np.array(p["outer_tip"])
        
        # 2. Calculate Radii
        r_out = np.linalg.norm(o_tip - hinge)
        
        # 3. Calculate Base Angle (radians)
        # The line from hinge to outer_tip is our 0° reference
        diff = o_tip - hinge
        base_angle = np.arctan2(diff[1], diff[0])
    
        # 4. Determine "Outward" Direction
        # Vector from this hinge to the tip
        vec_to_tip = o_tip - hinge
        # Vector from this hinge to the OTHER hinge
        other_side = "right" if side == "left" else "left"
        other_hinge = np.array(p[f"{other_side}_hinge"])
        vec_to_other = other_hinge - hinge

        # 2D Cross Product: (x1*y2 - y1*x2)
        # This tells us if the 'other' hinge is clockwise or counter-clockwise
        cross_prod = vec_to_tip[0] * vec_to_other[1] - vec_to_tip[1] * vec_to_other[0]

        # If cross_prod > 0, the 'inside' is counter-clockwise (+). 
        # So 'outside' must be clockwise (-1).
        # If cross_prod < 0, the 'inside' is clockwise (-). 
        # So 'outside' must be counter-clockwise (+1).
        direction = -1 if cross_prod > 0 else 1
        
        mask_quads = []

        tip_angle = base_angle + direction * np.radians(190-angle)
        tip_coord = (hinge[0] + r_out * 1.05 * np.cos(tip_angle), hinge[1] + r_out * 1.05 * np.sin(tip_angle))
            
        return tip_coord


    
    # ==========================================
    # Closing
    # ==========================================
    #     # --- Safety Cleanups ---
    def closeEvent(self, event):
        """Important: Safely shuts down the camera connection before closing window."""
        self.camera_worker.stop()
        event.accept()

    def create_menu_bar(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu('&File')

        # Save Action
        save_action = file_menu.addAction('Save Setup')
        #save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(self.save_setup)

        # Save as user default
        save_action = file_menu.addAction('Save Current Setup as User Default')
        #save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(self.save_setup_as_default)

        # Load Action
        load_action = file_menu.addAction('Load Setup')
        #load_action.setShortcut('Ctrl+L')
        load_action.triggered.connect(self.load_setup)

        # Load user default
        load_action = file_menu.addAction('Load User Default Setup')
        #load_action.setShortcut('Ctrl+L')
        load_action.triggered.connect(self.load_setup_default)


    def save_setup_as_default(self):
        # 1. Create the message box instance
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Overwriting the default setting")
        msg_box.setText("About to overwrite a default setting, if there is any. Do you want to continue?")
        msg_box.setIcon(QMessageBox.Question)

        # 2. Add buttons
        confirm_button = msg_box.addButton(QMessageBox.Yes) # or QMessageBox.Yes
        cancel_button = msg_box.addButton(QMessageBox.Cancel)

        # 3. SET CANCEL AS DEFAULT (This protects against accidental 'Enter' key presses)
        msg_box.setDefaultButton(cancel_button)

        # 4. Show dialog and capture result
        msg_box.exec()

        # 5. Check which button was clicked
        if msg_box.clickedButton() == confirm_button:
            self.save_setup(True)

        

    def save_setup(self, save_default = False):
        # Gather settings into a dictionary
        # Create a dictionary representing the current UI state
        config = {
            "combo_adapter": self.combo_adapter.currentText(),
            "combo_cam_id": self.combo_cam_id.currentText(),
            "combo_cam_mode": self.combo_cam_mode.currentText(),
            "preview_duration": self.preview_duration.text(),
            
            "triangle": self.points,
            "slider_thresh": self.slider_thresh.value(),
            
            "chk_save": self.chk_save.isChecked(),
            # "fly_ID": self.fly_ID.text(),
            "nidaq_dev_name_for_DIO": self.nidaq_dev_name_for_DIO.currentText(),
            "nidaq_DIO_port": self.nidaq_DIO_port.text(),
            "nidaq_DI_line": self.nidaq_DI_line.text(),
            "nidaq_DO_line": self.nidaq_DO_line.text(),
            "save_frame_skip": self.save_frame_skip.text(),
            
            "chk_update_fly": self.chk_update_fly.isChecked(),
            "update_rate": self.update_rate.text(),
            
            "run_time": self.run_time.text(),
            
            "nidaq_dev_name_for_AO": self.nidaq_dev_name_for_AO.currentText(),
            "nidaq_AO_LWS": self.nidaq_AO_LWS.text(),
            "nidaq_AO_L_R": self.nidaq_AO_L_R.text()
        }
        
        # Open a save file dialog
        if save_default:
            path = "./WingStrokeAnalyzer_user_default.json"
        else:
            path, _ = QFileDialog.getSaveFileName(self, "Save Setup", "", "JSON Files (*.json)")
        
        
        if path:
            with open(path, 'w') as f:
                json.dump(config, f, indent=4)
            self.status_bar.showMessage(f"Setup saved to {Path(path).name}", 3000)

    def load_setup_default(self):
        self.load_setup(True)

    def load_setup(self, load_default = False):
        # 1. Open the File Dialog
        if load_default:
            path = "./WingStrokeAnalyzer_user_default.json"
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Load Setup", "", "JSON Files (*.json)")
        
        if not path:
            return

        try:
            with open(path, 'r') as f:
                config = json.load(f)

            # 2. Iterate through the dictionary
            for key, value in config.items():
                # Special case: The 'triangle' points are an internal variable, not a widget
                if key == "triangle":
                    self.points = value
                    continue
                    
                # Check if the attribute exists in your class
                if not hasattr(self, key):
                    print(f"Warning: Widget '{key}' not found in UI.")
                    continue
                    
                widget = getattr(self, key)

                # 3. Update the widget based on its type
                if isinstance(widget, QLineEdit):
                    widget.setText(str(value))
                
                elif isinstance(widget, QComboBox):
                    # findText + setCurrentIndex is the safest way to set a ComboBox
                    index = widget.findText(str(value))
                    if index >= 0:
                        widget.setCurrentIndex(index)
                
                elif isinstance(widget, QCheckBox):
                    widget.setChecked(bool(value))
                
                elif isinstance(widget, QSlider):
                    widget.setValue(int(value))

            self.status_bar.showMessage(f"Successfully loaded: {path}", 3000)
            
            self.redraw_overlays()

        except Exception as e:
            print(f"Error loading config: {e}")
            self.status_bar.showMessage("Load failed! Check console.", 3000)

        


# Required QSplitter for resizable panels
from PySide6.QtWidgets import QSplitter

# ==========================================
# PART 4: Standard Python Execution
# ==========================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Using a fusion style helps bridge the look between different OS platforms
    app.setStyle("Fusion")
    
    window = FlyExperimentApp()
    window.show()
    sys.exit(app.exec())



    