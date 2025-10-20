import contextlib
import tempfile
from pathlib import Path
from typing import NamedTuple


class CacheDirs(NamedTuple):
    """
    Named tuple for cache directories.
    """

    transformation: str
    augmentation: str
    validation: str
    test: str


@contextlib.contextmanager
def tf_cache(cache_dir: str = ".tf-cache/"):
    """
    Context manager for caching with multiple temporary directories.

    Args:
        cache_dir: Base directory to store cached images.

    Yields:
        CacheDirs: Named tuple with temporary directory paths
    """
    path = Path(cache_dir)
    path.mkdir(parents=True, exist_ok=True)

    try:
        with contextlib.ExitStack() as stack:
            transformation_dir = stack.enter_context(
                tempfile.TemporaryDirectory(dir=path, prefix="transformation")
            )
            augmentation_dir = stack.enter_context(
                tempfile.TemporaryDirectory(dir=path, prefix="augmentation")
            )
            validation_dir = stack.enter_context(
                tempfile.TemporaryDirectory(dir=path, prefix="validation")
            )
            test_dir = stack.enter_context(
                tempfile.TemporaryDirectory(dir=path, prefix="test")
            )

            transformation_path = Path(transformation_dir).resolve()
            augmentation_path = Path(augmentation_dir).resolve()
            validation_path = Path(validation_dir).resolve()
            test_path = Path(test_dir).resolve()

            yield CacheDirs(
                transformation=transformation_path.as_posix() + "/",
                augmentation=augmentation_path.as_posix() + "/",
                validation=validation_path.as_posix() + "/",
                test=test_path.as_posix() + "/",
            )
    finally:
        path.rmdir()
