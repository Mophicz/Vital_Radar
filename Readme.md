## Overview
- `app/` contains the GUI code and a `.spec` file to build an executable with `pyinstaller`

    ```
    pyinstaller "Vital Radar.spec" --noconfirm
    ```   
    
- `python/` and `matlab/` contain individual scripts used to test ideas separately and generate plots.

## The Vital Radar App
The code necessary for different parts of the app are separated into different folders. In the lowest level `vital_radar/` lies only the `main.py` script, where the `QApplication` is initiated, the stylesheet is loaded and the `MainWindow()` is called.

The rest of the code is separated into three categories: `gui/`, `processing/` and `walabot/`: 

- `gui/` contains the `main_window.py` file where the GUI layout, buttons and menus are defined and most of the code - from processing to visualization - comes together. `gui/recources/` is for additional files used by the GUI, like icons, graphics and the `style.qss`. `gui/widgets/` is for additional PyQt6 widget-objects, like the `ImageDisplayWidget` for handling the visualization of data.

- `processing/` contains all scripts for processing data, like filtering, spectrum estimation or adding utility functions.

- `walabot/` handles all direct interaction with the Walabot API and includes an object with the exact positions of the walabot radar's antennas in a 3D coordinate system with the origin placed as defined by the manufacturer.

## Adding new modes to the Vital Radar app
To add a new Displaymode the following steps are necessary:

- In `processing/display_modes.py` add a new Enum to the `DisplayMode` class
- In `computePlotData(signal_matrix, display_mode)` handle the new `DisplayMode` in the `match-case` block
    - Usually functions from another processing script are called here
    - The data that is returned here will be passed directly to `updateImage(data, display_mode)` in `gui/widgets/image_display.py`
- Lastly handle the plotting for the new `DisplayMode` in the `match-case` block in `updateImage()` like the examples
    - To keep this function readable create a new helper-functon `_plotDisplayModeName()`