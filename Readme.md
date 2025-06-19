- `app/` contains the GUI code and a `.spec` file to build an executable with `pyinstaller`

    ```
    pyinstaller "Vital Radar.spec" --noconfirm
    ```   
    
- `python/` and `matlab/` contain individual scripts used to initially test ideas
- `data/` contains measured radar signals that can be used to test ideas without needing the radar to be connected