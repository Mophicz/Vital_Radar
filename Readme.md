## Overview
- `app/` contains the GUI code and a `.spec` file to build an executable with `pyinstaller`

    ```
    pyinstaller "Vital Radar.spec" --noconfirm
    ```   
    
- `python/` and `matlab/` contain individual scripts used to test ideas separately and generate plots.

## Adding new modes to the Vital Radar app
To add a new Displaymode the following steps are necessary:

- In `processing/display_modes.py` add a new Enum to the `DisplayMode` class
- In `computePlotData(signal_matrix, display_mode)` handle the new `DisplayMode` in the `match-case` block
    - Usually functions from another processing script are called here
    - The data that is returned here will be passed directly to `updateImage(data, display_mode)` in `gui/widgets/image_display.py`
- Lastly handle the plotting for the new `DisplayMode` in the `match-case` block in `updateImage()` like the examples
    - To keep this function readable create a new helper-functon `_plotDisplayModeName()`