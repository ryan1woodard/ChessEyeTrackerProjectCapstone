# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build spec.

Build with::

    pyinstaller eye_tracker.spec

A plain ``pyinstaller app.py`` will produce an executable that starts and then
fails on the first frame, because MediaPipe ships its solution models and
binary graphs as package *data* which PyInstaller's dependency analysis does
not follow. The ``collect_data_files`` call below is what fixes that.
"""

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

datas = [
    ("config/default_config.json", "config"),
]
datas += collect_data_files("mediapipe")
binaries = collect_dynamic_libs("mediapipe")

hiddenimports = [
    "mediapipe.python.solutions.face_mesh",
    "matplotlib.backends.backend_qtagg",
]

# Qt modules and scientific stacks this application does not use; excluding them
# roughly halves the distribution size.
excludes = [
    "tkinter", "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets",
    "PySide6.Qt3DCore", "PySide6.QtMultimedia", "PySide6.QtQuick",
    "PySide6.QtQml", "PySide6.QtCharts", "PySide6.QtDataVisualization",
    "scipy", "pandas", "IPython", "notebook",
]

a = Analysis(
    ["app.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="EyeTracker",
    debug=False,
    strip=False,
    upx=False,
    console=False,          # no console window; diagnostics go to logs/
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="EyeTracker",
)
