"""Cleaning of files ZPix downloads or compiles."""

import os
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from shutil import rmtree

import gradio as gr
from gradio_cleaning import CleaningItem, CleaningValue
from huggingface_hub import scan_cache_dir

from source.py.blocking_task import BlockingTask
from source.py.image_pipe import ImagePipeline
from source.py.user_data_dir import get_triton_user_cache_dir, get_user_cache_dir


@dataclass
class Cleanable:
    """Category of files user may clean."""

    label: str
    get_size: Callable[[], int]
    """Get bytes freed by cleaning it, which may take a while."""

    delete: Callable[[], None]


def get_dir_size(path: str | os.PathLike[str]) -> int:
    """Get total size of a directory, in bytes.

    Uses os.scandir, much faster than Path.rglob with many files, as it gets file
    sizes while listing them on Windows.
    """
    with os.scandir(path) as entries:
        return sum(
            get_dir_size(entry)
            if entry.is_dir(follow_symlinks=False)
            else entry.stat(follow_symlinks=False).st_size
            for entry in entries
        )


def remove_dirs(dirs: list[Path]) -> None:
    """Remove directories, skipping files in use, e.g. loaded Triton kernels."""
    for d in dirs:
        rmtree(d, ignore_errors=True)


def find_cleanables(
    models_list: Path, loaded_repo_id: str | None, t: Callable[[str], str]
) -> dict[str, Cleanable]:
    """Find categories of files user may clean, by ID, skipping missing ones."""
    # Hugging Face cache is shared with other apps, so we only target
    # models ZPix has curated since its beginning.
    curated_ids = set(models_list.read_text(encoding="utf-8").split())

    cache_info = scan_cache_dir()
    models_strategy = cache_info.delete_revisions(
        *(
            revision.commit_hash
            for repo in cache_info.repos
            if repo.repo_id in curated_ids
            for revision in repo.revisions
            # Loaded model is in use: removing it would break the app. Not its
            # old revisions though, left by updates, as ZPix loads "main" ref.
            if repo.repo_id != loaded_repo_id or not revision.refs
        )
    )

    # Target only these subfolders: on Linux and macOS, ZPix cache folder also
    # holds the venv, uv and Python bytecode, used by the running app.
    compiled_dirs = [get_user_cache_dir() / "Inductor", get_triton_user_cache_dir()]
    compiled_dirs = [d for d in compiled_dirs if d.exists()]

    cleanables: dict[str, Cleanable] = {}
    if models_strategy.expected_freed_size:
        cleanables["curated_models"] = Cleanable(
            t("Models downloaded by ZPix, except the loaded one"),
            lambda: models_strategy.expected_freed_size,
            models_strategy.execute,
        )
    if compiled_dirs:
        cleanables["compiled_shapes"] = Cleanable(
            t("Shapes compiled by PyTorch and Triton"),
            lambda: sum(get_dir_size(d) for d in compiled_dirs),
            partial(remove_dirs, compiled_dirs),
        )

    return cleanables


def clean(
    ids: list[str],
    models_list: Path,
    loaded_repo_id: str | None,
    t: Callable[[str], str],
) -> None:
    """Clean categories of files by ID, blocking other critical tasks."""
    with BlockingTask.run(t("Please wait, a cleaning is in progress.")):
        # Found again, so only files found by ZPix can be removed.
        cleanables = find_cleanables(models_list, loaded_repo_id, t)

        for cleanable_id in ids:
            if cleanable_id in cleanables:
                cleanables[cleanable_id].delete()


def validate_cleaning() -> None:
    """Validate that a cleaning can start, blocking otherwise.

    Raises:
        gr.Error: If a blocking task (e.g. a model download) is running.
    """
    if BlockingTask.is_running:
        raise gr.Error(str(BlockingTask.message), duration=6)


def open_cleaning_modal(
    models_list: Path, image_pipe: ImagePipeline, t: Callable[[str], str]
):
    """Show cleaning modal at once, unless there is nothing to clean.

    Sizes are measured later, as it may take a while.
    """
    cleanables = find_cleanables(models_list, image_pipe.repo_id, t)
    if not cleanables:
        gr.Info(t("Nothing to clean."))

    return gr.update(
        visible=bool(cleanables),
        value=CleaningValue(
            items=[CleaningItem(id=k, label=c.label) for k, c in cleanables.items()],
            selected=[],
        ),
    )


def measure_cleanables(
    models_list: Path, image_pipe: ImagePipeline, t: Callable[[str], str]
) -> CleaningValue:
    """Show sizes in cleaning modal, once measured."""
    cleanables = find_cleanables(models_list, image_pipe.repo_id, t)

    return CleaningValue(
        items=[
            CleaningItem(id=k, label=c.label, size=c.get_size())
            for k, c in cleanables.items()
        ],
        selected=[],
    )


def clean_selected(
    value: CleaningValue,
    models_list: Path,
    image_pipe: ImagePipeline,
    t: Callable[[str], str],
) -> None:
    """Clean files user selected in cleaning modal."""
    # As it can take a while with many files.
    gr.Info(t("Cleaning started."), duration=2)
    clean(value.selected, models_list, image_pipe.repo_id, t)

    if "compiled_shapes" in value.selected:
        image_pipe.forget_compiled()
