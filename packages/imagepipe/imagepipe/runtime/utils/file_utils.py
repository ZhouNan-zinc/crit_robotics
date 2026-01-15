import json
import os
from os import PathLike

from .constants import (
    SAFE_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    PACKAGE_DIRECTORY
)
from .logging import getLogger

logger = getLogger(__name__)

def resolve_files(
    path_or_repo_id: str|PathLike,
    filenames: list[str],
    *,
    subfolder: str = "",
    _raise_exceptions_for_missing_entries: bool = True,
) -> list[str]|None:
    if subfolder is None:
        subfolder = ""

    path_or_repo_id = str(path_or_repo_id)

    if not os.path.isdir(path_or_repo_id):
        raise OSError(
            f"path_or_repo_id must be a local directory, got: {path_or_repo_id}"
        )

    full_filenames = [os.path.join(subfolder, f) for f in filenames]

    resolved_files: list[str] = []
    for rel in full_filenames:
        abs_path = os.path.join(path_or_repo_id, rel)

        if os.path.isfile(abs_path):
            resolved_files.append(abs_path)
        else:
            if not _raise_exceptions_for_missing_entries:
                continue

            if rel == os.path.join(subfolder, "config.json"):
                return None

            raise OSError(
                f"Missing required file: {rel}\n"
                f"Looked in: {path_or_repo_id}"
            )

    return resolved_files if resolved_files else None

def resolve_file(
    path_or_repo_id: PathLike,
    filename: str,
    *,
    subfolder: str = "",
    _raise_exceptions_for_missing_entries: bool = True,
) -> str|None:
    files = resolve_files(
        path_or_repo_id=path_or_repo_id,
        filenames=[filename],
        subfolder=subfolder,
        _raise_exceptions_for_missing_entries=_raise_exceptions_for_missing_entries,
    )
    if files is None:
        return None
    return files[0]


def get_checkpoint_shard_files(
    pretrained_model_name_or_path: str|PathLike,
    resolved_archive_file: str|PathLike,
) -> tuple[list[str], dict]:
    """
    Offline-only shard file resolver.

    Args:
        pretrained_model_name_or_path:
            Local folder that contains shard files (recommended),
            OR local path of index json file.

        resolved_archive_file:
            Local path of the index json file (e.g. pytorch_model.bin.index.json
            or model.safetensors.index.json)

    Returns:
        checkpoint_files:
            List of absolute local shard paths.

        sharded_metadata:
            Parsed json metadata dict, includes "weight_map" and optional metadata.

    Raises:
        FileNotFoundError:
            If index file or shard file is missing.
        ValueError:
            If index json is invalid or contains no weight_map.
    """
    index_path = os.fspath(resolved_archive_file)
    if not os.path.isfile(index_path):
        raise FileNotFoundError(
            f"Index file not found (offline mode requires local file): {index_path}"
        )

    # Load index json
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to parse checkpoint index json: {index_path}") from e

    if not isinstance(index, dict) or "weight_map" not in index:
        raise ValueError(
            f"Invalid checkpoint index json: missing 'weight_map'. file={index_path}"
        )

    weight_map = index["weight_map"]
    if not isinstance(weight_map, dict) or len(weight_map) == 0:
        raise ValueError(f"Invalid checkpoint index json: empty 'weight_map'. file={index_path}")

    # Determine base folder where shard files should live
    # Priority:
    # 1) if pretrained_model_name_or_path is a dir -> use it
    # 2) else -> use index file's directory
    pm_path = os.fspath(pretrained_model_name_or_path)
    if os.path.isdir(pm_path):
        base_dir = pm_path
    else:
        # pretrained_model_name_or_path might be a file
        # fallback to index directory
        base_dir = os.path.dirname(index_path)

    # Collect unique shard filenames in index
    shard_filenames = sorted(set(weight_map.values()))

    checkpoint_files: list[str] = []
    missing_files: list[str] = []

    for fname in shard_filenames:
        # Weight map values are relative filenames (usually).
        # We'll resolve them under base_dir.
        local_path = os.path.join(base_dir, fname)
        local_path = os.path.abspath(local_path)

        if not os.path.isfile(local_path):
            missing_files.append(local_path)
        else:
            checkpoint_files.append(local_path)

    if missing_files:
        # Offline-only: do NOT attempt any download
        missing_str = "\n".join(missing_files[:50])
        suffix = "" if len(missing_files) <= 50 else f"\n... and {len(missing_files) - 50} more"
        raise FileNotFoundError(
            "Offline mode: some shard checkpoint files are missing locally.\n"
            f"Index: {index_path}\n"
            f"Base dir: {base_dir}\n"
            "Missing shard files:\n"
            f"{missing_str}{suffix}"
        )

    sharded_metadata = index
    return checkpoint_files, sharded_metadata


def _add_variant(weights_name: str, variant: str | None = None) -> str:
    if variant is not None:
        path, name = weights_name.rsplit(".", 1)
        weights_name = f"{path}.{variant}.{name}"
    return weights_name


def get_resolved_checkpoint_files(
    pretrained_model_name_or_path: str | os.PathLike | None,
    variant: str | None,
    use_safetensors: bool | None,
    transformers_explicit_filename: str | None = None,
) -> tuple[list[str] | None, dict | None]:
    """Get all the checkpoint filenames based on `pretrained_model_name_or_path`, and optional metadata if the
    checkpoints are sharded.
    This function will download the data if necessary.
    """
    if transformers_explicit_filename is not None:
        if not transformers_explicit_filename.endswith(".safetensors") and not transformers_explicit_filename.endswith(
            ".safetensors.index.json"
        ):
            raise ValueError(
                "The transformers file in the config seems to be incorrect: it is neither a safetensors file "
                "(*.safetensors) nor a safetensors index file (*.safetensors.index.json): "
                f"{transformers_explicit_filename}"
            )

    is_sharded = False

    if pretrained_model_name_or_path is not None:
        if os.path.isdir(os.path.join(PACKAGE_DIRECTORY, pretrained_model_name_or_path)) and not os.path.isdir(pretrained_model_name_or_path):
            pretrained_model_path = os.path.join(PACKAGE_DIRECTORY, pretrained_model_name_or_path)
        else:
            pretrained_model_path = pretrained_model_name_or_path

        if os.path.isdir(pretrained_model_path):
            if transformers_explicit_filename is not None:
                # If the filename is explicitly defined, load this by default.
                archive_file = os.path.join(pretrained_model_name_or_path, transformers_explicit_filename)
                is_sharded = transformers_explicit_filename.endswith(".safetensors.index.json")
            elif use_safetensors is not False and os.path.isfile(
                os.path.join(pretrained_model_name_or_path, _add_variant(SAFE_WEIGHTS_NAME, variant))
            ):
                # Load from a safetensors checkpoint
                archive_file = os.path.join(
                    pretrained_model_name_or_path, _add_variant(SAFE_WEIGHTS_NAME, variant)
                )
            elif use_safetensors is not False and os.path.isfile(
                os.path.join(pretrained_model_name_or_path, _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant))
            ):
                # Load from a sharded safetensors checkpoint
                archive_file = os.path.join(
                    pretrained_model_name_or_path, _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant)
                )
                is_sharded = True
            elif not use_safetensors and os.path.isfile(
                os.path.join(pretrained_model_name_or_path, _add_variant(WEIGHTS_NAME, variant))
            ):
                # Load from a PyTorch checkpoint
                archive_file = os.path.join(
                    pretrained_model_name_or_path, _add_variant(WEIGHTS_NAME, variant)
                )
            elif not use_safetensors and os.path.isfile(
                os.path.join(pretrained_model_name_or_path, _add_variant(WEIGHTS_INDEX_NAME, variant))
            ):
                # Load from a sharded PyTorch checkpoint
                archive_file = os.path.join(
                    pretrained_model_name_or_path, _add_variant(WEIGHTS_INDEX_NAME, variant)
                )
                is_sharded = True
            elif use_safetensors:
                raise OSError(
                    f"Error no file named {_add_variant(SAFE_WEIGHTS_NAME, variant)} found in directory"
                    f" {pretrained_model_name_or_path}."
                )
            else:
                raise OSError(
                    f"Error no file named {_add_variant(SAFE_WEIGHTS_NAME, variant)}, or {_add_variant(WEIGHTS_NAME, variant)},"
                    f" found in directory {pretrained_model_name_or_path}."
                )
        else:
            # set correct filename
            if transformers_explicit_filename is not None:
                filename = transformers_explicit_filename
                is_sharded = transformers_explicit_filename.endswith(".safetensors.index.json")
            elif use_safetensors is not False:
                filename = _add_variant(SAFE_WEIGHTS_NAME, variant)
            else:
                filename = _add_variant(WEIGHTS_NAME, variant)

            try:
                # Load from URL or cache if already cached
                resolved_archive_file = resolve_file(pretrained_model_name_or_path, filename)

                # Since we set _raise_exceptions_for_missing_entries=False, we don't get an exception but a None
                # result when internet is up, the repo and revision exist, but the file does not.
                if resolved_archive_file is None and filename == _add_variant(SAFE_WEIGHTS_NAME, variant):
                    # Maybe the checkpoint is sharded, we try to grab the index name in this case.
                    resolved_archive_file = resolve_file(
                        pretrained_model_name_or_path,
                        _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant),
                    )
                    if resolved_archive_file is not None:
                        is_sharded = True
                    elif use_safetensors:
                        if resolved_archive_file is None:
                            raise OSError(
                                f"{pretrained_model_name_or_path} does not appear to have a file named"
                                f" {_add_variant(SAFE_WEIGHTS_NAME, variant)} or {_add_variant(SAFE_WEIGHTS_INDEX_NAME, variant)} "
                                "and thus cannot be loaded with `safetensors`. Please do not set `use_safetensors=True`."
                            )
                    else:
                        # This repo has no safetensors file of any kind, we switch to PyTorch.
                        filename = _add_variant(WEIGHTS_NAME, variant)
                        resolved_archive_file = resolve_file(
                            pretrained_model_name_or_path, filename
                        )
                if resolved_archive_file is None and filename == _add_variant(WEIGHTS_NAME, variant):
                    # Maybe the checkpoint is sharded, we try to grab the index name in this case.
                    resolved_archive_file = resolve_file(
                        pretrained_model_name_or_path,
                        _add_variant(WEIGHTS_INDEX_NAME, variant)
                    )
                    if resolved_archive_file is not None:
                        is_sharded = True

            except OSError:
                # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted
                # to the original exception.
                raise
            except Exception as e:
                # For any other exception, we throw a generic error.
                raise OSError(
                    f"Can't load the model for '{pretrained_model_name_or_path}'. "
                    f"Make sure '{pretrained_model_name_or_path}' is the correct path to a directory "
                    f"containing a file named {_add_variant(WEIGHTS_NAME, variant)}. "
                ) from e
        logger.info(f"loading weights file {archive_file}")
        resolved_archive_file = archive_file

    sharded_metadata = None
    if is_sharded:
        checkpoint_files, sharded_metadata = get_checkpoint_shard_files(
            pretrained_model_name_or_path,
            resolved_archive_file
        )
    else:
        checkpoint_files = [resolved_archive_file] if pretrained_model_name_or_path is not None else None

    return checkpoint_files, sharded_metadata

