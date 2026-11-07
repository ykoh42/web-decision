from __future__ import annotations

import os
import stat
from pathlib import Path


def assert_fresh_output_path(path: Path) -> Path:
    output = Path(os.path.abspath(Path(path).expanduser()))
    parent = Path(output.anchor)
    for part in output.parts[1:-1]:
        parent /= part
        try:
            info = parent.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"Output parent must not be a symlink: {parent}")
        if not stat.S_ISDIR(info.st_mode):
            raise NotADirectoryError(f"Output parent must be a directory: {parent}")

    try:
        output.lstat()
    except FileNotFoundError:
        return output
    else:
        raise FileExistsError(f"Refusing to overwrite output directory: {output}")


def create_fresh_output_directory(path: Path) -> Path:
    output = assert_fresh_output_path(path)
    output.mkdir(parents=True, exist_ok=False)
    return output


def write_new_text_file(path: Path, text: str) -> None:
    with Path(path).open("x", encoding="utf-8") as output:
        output.write(text)
