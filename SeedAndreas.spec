from pathlib import Path


project_directory = Path(SPECPATH)

analysis = Analysis(
    [str(project_directory / "SeedAndreas.pyw")],
    pathex=[str(project_directory)],
    binaries=[],
    datas=[
        (str(project_directory / "README.md"), "."),
        (str(project_directory / "LICENSE.md"), "."),
        (str(project_directory / "CITY_GENERATION_METHOD.md"), "."),
    ],
    hiddenimports=[
        "sanandrais.cli",
        "sanandrais.contracts",
        "sanandrais.custom_world",
        "sanandrais.gta_integration",
        "sanandrais.generator",
        "sanandrais.los_santos",
        "sanandrais.street_graph",
        "sanandrais.textures",
        "sanandrais.neighborhoods",
        "sanandrais.path_nodes",
        "sanandrais.water",
        "sanandrais.gameplay",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

python_archive = PYZ(analysis.pure)

executable = EXE(
    python_archive,
    analysis.scripts,
    analysis.binaries,
    analysis.datas,
    [],
    name="SeedAndreas",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
    uac_admin=True,
)
