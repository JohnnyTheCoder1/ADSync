"""Render locally and publish complete media without exposing partial files."""

from __future__ import annotations

import errno
import hashlib
import logging
import os
import shutil
import tempfile
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

log = logging.getLogger("adsync")


def sha256_file(path: str | Path) -> str:
    """Hash bounded chunks and reject a file that changes during the read."""
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise OSError(f"File changed while hashing: {path}")
    return digest.hexdigest()


@contextmanager
def destination_publication_lock(destination: str | Path) -> Iterator[None]:
    """Serialize large writes and transfer reads per destination filesystem.

    The OS releases the advisory lock when a worker exits, including crashes.
    Persistent lock files live locally, so unavailable shares do not prevent
    creating the lock. Windows drive and UNC-share names identify volumes;
    POSIX uses the closest existing ancestor's device identifier.
    """
    destination = Path(destination).expanduser().resolve()
    if os.name == "nt":
        volume = destination.drive.casefold()
    else:
        ancestor = destination.parent
        while not ancestor.exists() and ancestor != ancestor.parent:
            ancestor = ancestor.parent
        volume = str(ancestor.stat().st_dev)
    local = os.environ.get("LOCALAPPDATA") if os.name == "nt" else None
    root = (Path(local) / "ADSync" if local else Path(tempfile.gettempdir()) / "adsync") / "volume-locks"
    root.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(volume.encode("utf-8")).hexdigest()
    with (root / f"{key}.lock").open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        if os.name == "nt":
            import msvcrt
            while True:
                handle.seek(0)
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as exc:
                    if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                        raise
                    time.sleep(0.1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class OutputPublicationError(OSError):
    """Rendering succeeded, but the completed local file could not be published."""

    def __init__(self, local_path: Path, destination: Path, cause: OSError):
        self.local_path = local_path
        self.destination = destination
        super().__init__(
            f"Could not publish output to {destination}: {cause}\n"
            f"Completed local output retained at: {local_path}"
        )


def resolve_output_path(
    video_path: str | Path,
    output: str | Path | None = None,
    output_dir: str | Path | None = None,
    *,
    suffix: str = ".synced.mkv",
) -> Path:
    """Explicit filename wins over directory option, environment, and input folder."""
    if output is not None:
        return Path(os.path.expandvars(output)).expanduser()
    default = Path(video_path).with_suffix(suffix)
    directory = output_dir if output_dir is not None else os.environ.get("ADSYNC_OUTPUT_DIR")
    if directory:
        return Path(os.path.expandvars(directory)).expanduser() / default.name
    return default


def validate_output_path(output_path: str | Path, inputs: Sequence[str | Path]) -> None:
    """Refuse input aliases, including existing hard links and symbolic links."""
    destination = Path(output_path)
    normalized = os.path.normcase(os.path.abspath(destination))
    for source in inputs:
        same = normalized == os.path.normcase(os.path.abspath(source))
        if not same:
            try:
                same = destination.samefile(source)
            except OSError:
                # An unavailable share is allowed here: render first and retain
                # the local file if publication still cannot reach it later.
                pass
        if same:
            raise ValueError(f"Output path must not overwrite an input file: {source}")


@contextmanager
def staged_output(
    output_path: str | Path,
    *,
    inputs: Sequence[str | Path] = (),
) -> Iterator[Path]:
    """Yield a local render path and publish it only after successful rendering.

    The caller must finish encoding and any media validation inside the context.
    Same-filesystem publication is a single atomic rename. Across filesystems,
    a bounded-memory copy goes to a unique nonmedia filename before final rename.
    A publication error leaves the completed local file available for recovery.
    """
    destination = Path(output_path).expanduser()
    validate_output_path(destination, inputs)
    # LOCALAPPDATA remains local even when a user runs from a UNC working folder
    # or configures a network TEMP directory on Windows.
    local_base = None
    if os.environ.get("ADSYNC_STAGING_DIR"):
        local_base = Path(os.environ["ADSYNC_STAGING_DIR"])
        local_base.mkdir(parents=True, exist_ok=True)
    elif os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        local_base = Path(os.environ["LOCALAPPDATA"]) / "ADSync" / "staging"
        local_base.mkdir(parents=True, exist_ok=True)
    stage_dir = Path(tempfile.mkdtemp(prefix="adsync-output-", dir=local_base))
    local_path = stage_dir / destination.name
    preserve = False
    try:
        yield local_path
        if not local_path.is_file() or local_path.stat().st_size == 0:
            raise ValueError(f"Rendered output is missing or empty: {local_path}")
        preserve = True
        try:
            validate_output_path(destination, inputs)
            _publish(local_path, destination)
        except OSError as exc:
            raise OutputPublicationError(local_path, destination, exc) from exc
        except BaseException:
            log.warning("Completed local output retained at: %s", local_path)
            raise
        preserve = False
    finally:
        if not preserve:
            shutil.rmtree(stage_dir, ignore_errors=True)


def _publish(local_path: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        # The ready local artifact can become visible immediately on the same
        # filesystem; avoid copying large movies a second time unnecessarily.
        os.replace(local_path, destination)
        return
    except OSError as exc:
        if exc.errno != errno.EXDEV and getattr(exc, "winerror", None) != 17:
            raise
    _copy_publish(local_path, destination)


def publish_completed_file(
    local_path: str | Path,
    destination: str | Path,
    *,
    expected_sha256: str | None = None,
    overwrite: bool = True,
) -> Path:
    """Publish an already-complete file atomically, keeping its original copy."""
    local_path, destination = Path(local_path), Path(destination)
    validate_output_path(destination, inputs=[local_path])
    if not local_path.is_file() or local_path.stat().st_size == 0:
        raise ValueError(f"Completed output is missing or empty: {local_path}")
    try:
        _copy_publish(local_path, destination, expected_sha256=expected_sha256, overwrite=overwrite)
    except OSError as exc:
        raise OutputPublicationError(local_path, destination, exc) from exc
    return destination


def _copy_publish(local_path: Path, destination: Path, *, expected_sha256: str | None = None,
                  overwrite: bool = True) -> None:
    # Hash locally before waiting for the storage lane. Only the sequential
    # destination copy and its integrity read occupy that lane.
    expected = expected_sha256 or sha256_file(local_path)
    with destination_publication_lock(destination):
        _copy_publish_locked(local_path, destination, expected, overwrite=overwrite)


def _copy_publish_locked(local_path: Path, destination: Path, expected_sha256: str,
                         *, overwrite: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {destination}")
    log.info("Publishing completed output to %s", destination)
    partial: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=".adsync-", suffix=".adsync-part",
            dir=destination.parent, delete=False,
        ) as target:
            partial = Path(target.name)
            with local_path.open("rb") as source:
                shutil.copyfileobj(source, target, length=4 * 1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
        if sha256_file(partial) != expected_sha256:
            raise OSError(f"Transfer integrity hash mismatch for {destination}")
        if overwrite:
            os.replace(partial, destination)
        elif os.name == "nt":
            # Windows rename fails if another process creates the destination.
            os.rename(partial, destination)
        else:
            # A no-clobber publication without a check/rename race on POSIX.
            os.link(partial, destination)
            partial.unlink()
    finally:
        if partial is not None:
            try:
                partial.unlink(missing_ok=True)
            except OSError:
                # A disconnected share can also prevent cleanup. The leftover
                # extension is deliberately not recognized as a media file.
                log.warning("Could not remove unpublished partial file: %s", partial)
