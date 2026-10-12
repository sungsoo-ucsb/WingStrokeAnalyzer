# WingStrokeAnalyzer

**WingStrokeAnalyzer** is a Python application for monitoring the wing-stroke angles of tethered flies. It provides a live camera preview, configurable image-analysis masks, real-time estimates of left and right wing angles, NI-DAQ analog outputs, and optional video recording.

## Requirements

The setup described here is for **Windows (64-bit)** and uses:

- Python 3.10 in a Conda environment
- Qt 6 / PySide6 and PyQtGraph for the user interface
- FLIR/Teledyne Spinnaker and PySpin for camera acquisition
- NI-DAQmx drivers and the Python `nidaqmx` package for data acquisition
- NumPy, SciPy, and OpenCV for image processing

> **Installation order matters.** To reduce DLL and dependency conflicts, install **Qt/PySide6 → PySpin → NI-DAQmx Python support** in that order, after installing the system-level prerequisites below.
>
> **Version note:** These instructions describe the configuration used during development. In particular, **Spinnaker 4.2** is recommended here instead of 4.3 because of the reported Python 3.10 / NumPy compatibility in this setup. Check the compatibility of the exact Spinnaker/PySpin wheel, Python, and NumPy versions before upgrading.

## Installation

### 1. Install Microsoft Visual C++ Redistributable

Install the **x64 Visual C++ Redistributable (v14)** from [Microsoft's download page](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170).

### 2. Install Spinnaker 4.2

Install **Spinnaker 4.2** (not 4.3 for the configuration documented here). Obtain the installer and matching Python wheel from the camera manufacturer's distribution.

### 3. Install Anaconda and create an environment

Install [Anaconda](https://www.anaconda.com/download), then open **Anaconda Prompt**:

```powershell
conda create -n wingstrokeanalyzer python=3.10
conda activate wingstrokeanalyzer
```

### 4. Install Qt, PyQtGraph, and NumPy

```powershell
conda install -c conda-forge pyside6 pyqtgraph numpy
```

Verify that PySide6 imports successfully:

```powershell
python -c "from PySide6 import QtWidgets; print('Success!')"
```

### 5. Install PySpin

In Anaconda Prompt, navigate to the directory containing the **Spinnaker 4.2 wheel for Python 3.10 / Windows x64**, then install it:

```powershell
pip install spinnaker_python-4.2.0.88-cp310-cp310-win_amd64.whl
```

Verify the installation:

```powershell
python -c "import PySpin; print('PySpin Version:', PySpin.System.GetInstance().GetLibraryVersion())"
```

> The wheel filename must match the installer and Python version you actually have.

### 6. Install NI-DAQ support

Install the appropriate **NI-DAQmx system driver** for your NI hardware separately. Then install the Python bindings and TDMS support:

```powershell
conda install -c conda-forge nptdms nidaqmx-python
```

The PyPI package is named `nidaqmx` if installing via `pip` instead.

### 7. Install SciPy and OpenCV

```powershell
conda install -c conda-forge scipy opencv
```

**Optional development tools:**

```powershell
conda install jupyterlab
```

You may also use VS Code or Anaconda Navigator to manage the project. Neither is required to run the analyzer.

## Run WingStrokeAnalyzer

In Anaconda Prompt, change to the folder containing `WingStrokeAnalyzer.py`, then run:

```powershell
conda activate wingstrokeanalyzer
python WingStrokeAnalyzer.py
```

### Experimental workflow

1. **Preview:** Select the camera and click **Preview** to confirm that images are being acquired.
2. **Position the fly:** Center the tethered fly in the camera view.
3. **Define the analysis region:** Use the **left/right wing hinge** and **outer/inner mask boundary** controls to position the masks.
4. **Capture a stationary reference:** **While the fly is not flying**, click **Click when not flying with stationary leg**. This captures the reference image used for background subtraction.
5. **Adjust the threshold:** Use the threshold slider to isolate the wing shadows in the binarized image. **Show threshold effect** can help visualize the result.
6. **Configure NI-DAQ:** Select the DAQ device(s) and assign the analog output channels and, if using synchronized recording, digital I/O lines.
7. **Test and run:** Use **Test WSA** or **Test frame rate** to check the setup, then click **Run** to begin acquisition. Optionally enable video saving and configure the frame-saving interval.

## NI-DAQ analog outputs

The program provides two analog signals:

| Signal | Description |
| --- | --- |
| `LWS` | Scaled left wing-stroke angle |
| `L-R` | Difference between scaled left and right wing-stroke angles |

For each wing angle $\theta$ (degrees), the code scales the output as:

$$
V(\theta) = \max\left(0,\frac{\theta-30}{180-30}\times 8\right)\ \mathrm{V}.
$$

The **rectification baseline is 30°** and **180° corresponds to 8 V**. Angles below 30° yield 0 V; angles above 180° can produce values above 8 V (subject to DAQ output limits). The implemented code also sets **both** outputs to zero if either wing's scaled voltage is zero.

The output channels are:

$$
V_{LWS}=V_L, \qquad V_{L-R}=V_L-V_R.
$$

Recover the individual scaled voltages:

$$
V_L=V_{LWS}, \qquad V_R=V_{LWS}-V_{L-R}.
$$

For nonzero, non-rectified wing signals, convert voltage back to angle:

$$
\theta=30+\frac{150}{8}V.
$$

> A zero output does **not** uniquely identify an angle: values at or below the baseline are rectified, and the implementation may zero both channels together. See `wing_angle_nidaq_AO_out()` in the source for details.

## How the wing-angle analysis works

The core processing is implemented in `compute_wing_angle()`:

1. Subtract a reference image acquired while the fly is stationary.
2. Threshold the image to identify wing shadows.
3. Measure the fraction of wing-shadow pixels in successive angular masks.
4. Estimate the wing angle from the mask occupancy profile using half-maximum detection and interpolation.

The mask geometry can be adjusted interactively for different camera views and fly positions.

## Camera configuration

The application does not currently provide controls for all camera acquisition settings, such as cropping or the sensor's region of interest. Use **SpinView** to configure these parameters before running the application. **FlyCapture2** may be an alternative for supported older cameras.

Adjusting the acquisition region or other camera parameters may improve frame rate, depending on the camera and interface.

## Optional: JupyterLab configuration

If you installed JupyterLab, you can generate its server configuration:

```powershell
jupyter lab --generate-config
```

The configuration file is typically located at:

```text
C:\Users\<YourUser>\.jupyter\jupyter_server_config.py
```

Open it and set the desired working directory, for example:

```python
c.ServerApp.root_dir = 'C:/Experiment_Root'
```

Restart JupyterLab for changes to take effect:

```powershell
jupyter lab
```

This server configuration may apply to JupyterLab launches across Conda environments that use the same user configuration.

## Optional: Conda environment management

Remove a test environment:

```powershell
conda remove --name qt_test --all
```

Clone a working environment:

```powershell
conda create --name wingstrokeanalyzer --clone qt_test
```

> If `wingstrokeanalyzer` already exists, choose a different destination name or remove the old environment first.

## License

Released under the **BSD 3-Clause License**. See [`LICENSE`](LICENSE) for the complete terms. Add the `LICENSE` file to the repository before publication.
