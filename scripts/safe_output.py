"""Create files below an already existing directory without following symlinks."""

import errno
import os
import secrets
import shutil
import stat
from pathlib import PurePosixPath


def open_directory(root, relative_path):
    """Open an existing directory under root without following path symlinks."""
    path = PurePosixPath(relative_path)
    parts = path.parts
    if path.is_absolute() or not parts or any(part in ("", ".", "..") for part in parts):
        raise ValueError("output directory must be a relative path without dot segments")

    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    fd = os.open(root, flags)
    try:
        for part in parts:
            next_fd = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        if not stat.S_ISDIR(os.fstat(fd).st_mode):
            raise NotADirectoryError(relative_path)
        return fd
    except BaseException:
        os.close(fd)
        raise


def atomic_write(directory_fd, name, write):
    """Atomically replace one file entry in an already-open directory."""
    if not name or name in (".", "..") or "/" in name or "\\" in name:
        raise ValueError("output name must be one path component")

    temporary = f".{name}.{secrets.token_hex(12)}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    fd = os.open(temporary, flags, 0o666, dir_fd=directory_fd)
    try:
        with os.fdopen(fd, "wb") as output:
            write(output)
        os.replace(temporary, name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
    except BaseException:
        try:
            os.unlink(temporary, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        raise


def require_absent(directory_fd, name):
    """Reject any existing entry without following a symlink."""
    if not name or name in (".", "..") or "/" in name or "\\" in name:
        raise ValueError("output name must be one path component")
    try:
        os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    raise FileExistsError(f"Refusing to overwrite existing output: {name}")


def create_new_file(directory_fd, name, write):
    """Create one file entry exclusively without following symlinks."""
    if not name or name in (".", "..") or "/" in name or "\\" in name:
        raise ValueError("output name must be one path component")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    fd = os.open(name, flags, 0o666, dir_fd=directory_fd)
    with os.fdopen(fd, "wb") as output:
        write(output)


def create_temporary_directory(parent_fd, prefix):
    """Create a private directory below an already-open parent."""
    if not prefix or prefix in (".", "..") or "/" in prefix or "\\" in prefix:
        raise ValueError("directory prefix must be one path component")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    for _ in range(10):
        name = f"{prefix}.{secrets.token_hex(12)}"
        try:
            os.mkdir(name, 0o700, dir_fd=parent_fd)
        except FileExistsError:
            continue
        try:
            fd = os.open(name, flags, dir_fd=parent_fd)
        except BaseException:
            os.rmdir(name, dir_fd=parent_fd)
            raise
        return name, fd
    raise FileExistsError("could not create a unique staging directory")


def remove_tree(parent_fd, name):
    """Remove a tree through directory descriptors without following links."""
    _validate_name(name)
    root_stat = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if not stat.S_ISDIR(root_stat.st_mode):
        raise NotADirectoryError(name)
    os.chmod(
        name,
        stat.S_IMODE(root_stat.st_mode) | 0o700,
        dir_fd=parent_fd,
        follow_symlinks=False,
    )
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory_fd = os.open(name, flags, dir_fd=parent_fd)
    try:
        for child in os.listdir(directory_fd):
            child_stat = os.stat(child, dir_fd=directory_fd, follow_symlinks=False)
            if stat.S_ISDIR(child_stat.st_mode):
                os.chmod(
                    child,
                    stat.S_IMODE(child_stat.st_mode) | 0o700,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
                remove_tree(directory_fd, child)
            else:
                os.unlink(child, dir_fd=directory_fd)
    finally:
        os.close(directory_fd)
    os.rmdir(name, dir_fd=parent_fd)


def promote_file(source_fd, source_name, target_fd, target_name):
    """Copy and publish a staged regular file without following symlinks."""
    _validate_name(source_name)
    _validate_name(target_name)
    source_flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    input_fd = os.open(source_name, source_flags, dir_fd=source_fd)
    source_stat = os.fstat(input_fd)
    if not stat.S_ISREG(source_stat.st_mode):
        os.close(input_fd)
        raise ValueError(f"staged output is not a regular file: {source_name}")
    source_mode = stat.S_IMODE(source_stat.st_mode)

    temporary = None
    output_fd = None
    try:
        for _ in range(10):
            temporary = f".{target_name}.{secrets.token_hex(12)}.tmp"
            try:
                output_fd = os.open(
                    temporary,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o666,
                    dir_fd=target_fd,
                )
                break
            except FileExistsError:
                temporary = None
        if output_fd is None or temporary is None:
            raise FileExistsError("could not create a unique publication file")
        os.fchmod(output_fd, source_mode)

        with os.fdopen(input_fd, "rb") as source:
            input_fd = -1
            with os.fdopen(output_fd, "wb") as target:
                output_fd = None
                shutil.copyfileobj(source, target, 1024 * 1024)
        os.link(
            temporary,
            target_name,
            src_dir_fd=target_fd,
            dst_dir_fd=target_fd,
            follow_symlinks=False,
        )
    finally:
        if input_fd >= 0:
            os.close(input_fd)
        if output_fd is not None:
            os.close(output_fd)
        if temporary is not None:
            try:
                os.unlink(temporary, dir_fd=target_fd)
            except FileNotFoundError:
                pass


def promote_directory(source_fd, source_name, target_fd, target_name):
    """Rename a staged directory into a newly reserved destination entry."""
    _validate_name(source_name)
    _validate_name(target_name)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    staged_fd = os.open(source_name, flags, dir_fd=source_fd)
    destination_created = False
    try:
        os.mkdir(target_name, 0o700, dir_fd=target_fd)
        destination_created = True
        try:
            os.rename(
                source_name,
                target_name,
                src_dir_fd=source_fd,
                dst_dir_fd=target_fd,
            )
        except OSError as error:
            if error.errno != errno.EXDEV:
                raise
            destination_fd = os.open(target_name, flags, dir_fd=target_fd)
            directory_modes = []
            try:
                _copy_tree(staged_fd, destination_fd, directory_modes)
                for directory_fd, mode in directory_modes:
                    os.fchmod(directory_fd, mode)
                os.fchmod(destination_fd, stat.S_IMODE(os.fstat(staged_fd).st_mode))
            finally:
                for directory_fd, _mode in directory_modes:
                    os.close(directory_fd)
                os.close(destination_fd)
    except BaseException as error:
        if destination_created:
            try:
                remove_tree(target_fd, target_name)
            except OSError as cleanup_error:
                raise cleanup_error from error
        raise
    finally:
        os.close(staged_fd)


def _copy_tree(source_fd, target_fd, directory_modes):
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    for name in os.listdir(source_fd):
        current = os.stat(name, dir_fd=source_fd, follow_symlinks=False)
        if stat.S_ISDIR(current.st_mode):
            os.mkdir(name, 0o700, dir_fd=target_fd)
            source_child = os.open(name, directory_flags, dir_fd=source_fd)
            target_child = os.open(name, directory_flags, dir_fd=target_fd)
            try:
                _copy_tree(source_child, target_child, directory_modes)
                directory_modes.append(
                    (os.dup(target_child), stat.S_IMODE(os.fstat(source_child).st_mode))
                )
            finally:
                os.close(source_child)
                os.close(target_child)
        elif stat.S_ISREG(current.st_mode):
            promote_file(source_fd, name, target_fd, name)
        else:
            raise ValueError(f"unsupported staged output type: {name}")


def _validate_name(name):
    if not name or name in (".", "..") or "/" in name or "\\" in name:
        raise ValueError("output name must be one path component")
