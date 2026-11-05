from __future__ import annotations

import ctypes
import importlib.util
import os
import sys
import traceback
from pathlib import Path


def show_startup_error(error_text: str) -> None:
    log_path = Path(__file__).resolve().with_name("SeedAndreas_startup_error.txt")
    try:
        log_path.write_text(error_text, encoding="utf-8")
    except OSError:
        pass

    message = (
        "Seed Andreas could not start.\n\n"
        + error_text
        + "\n\nA log was written to:\n"
        + str(log_path)
    )
    if os.name == "nt":
        ctypes.windll.user32.MessageBoxW(
            None,
            message,
            "Seed Andreas startup error",
            0x10,
        )
    else:
        print(message, file=sys.stderr)


def relaunch_elevated_without_console() -> bool:
    if os.name != "nt" or ctypes.windll.shell32.IsUserAnAdmin():
        return False

    executable = Path(sys.executable)
    if executable.name.lower() == "python.exe":
        pythonw = executable.with_name("pythonw.exe")
        if pythonw.is_file():
            executable = pythonw

    script = Path(__file__).resolve()
    result = ctypes.windll.shell32.ShellExecuteW(
        None,
        "runas",
        str(executable),
        f'"{script}"',
        str(script.parent),
        1,
    )
    if result <= 32:
        raise OSError(
            f"Windows could not start Seed Andreas as administrator "
            f"(ShellExecute error {result})."
        )
    return True


def run() -> None:
    if relaunch_elevated_without_console():
        return

    script_directory = Path(__file__).resolve().parent
    script_directory_text = str(script_directory)
    if not sys.path or sys.path[0] != script_directory_text:
        sys.path.insert(0, script_directory_text)

    launcher_path = script_directory / "launcher.py"
    module_name = "seed_andreas_local_launcher"
    module_spec = importlib.util.spec_from_file_location(module_name, launcher_path)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"Could not load the local launcher module from {launcher_path}.")
    launcher_module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = launcher_module
    module_spec.loader.exec_module(launcher_module)
    launcher_module.main()


if __name__ == "__main__":
    try:
        run()
    except Exception:
        show_startup_error(traceback.format_exc())
