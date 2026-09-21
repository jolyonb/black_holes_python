"""The one typed shim over h5py: arrays, scalars and JSON mappings in and out of a group.

h5py ships no type stubs, so every read comes back untyped. Rather than scatter casts through the code, the few
operations the run files need are written once here with the types the rest of the code expects; `records.py` and
the output module use these and never touch h5py directly.
"""

import json
from pathlib import Path
from typing import Any, Literal, cast

import h5py
import numpy as np

from pbh.types import FloatArray

type Group = h5py.Group
type File = h5py.File


def write_array(group: Group, name: str, values: FloatArray) -> None:
    """Store a float array as a dataset."""
    group.create_dataset(name, data=np.asarray(values, dtype=np.float64))  # type: ignore[reportUnknownMemberType]


def read_array(group: Group, name: str) -> FloatArray:
    """Read a float array dataset."""
    dataset = cast(h5py.Dataset, group[name])
    return np.asarray(cast(Any, dataset[()]), dtype=np.float64)


def write_float(group: Group, name: str, value: float) -> None:
    """Store a float as an attribute of the group."""
    group.attrs[name] = float(value)


def read_float(group: Group, name: str) -> float:
    """Read a float attribute."""
    return float(cast(Any, group.attrs[name]))


def write_int(group: Group, name: str, value: int) -> None:
    """Store an integer as an attribute of the group."""
    group.attrs[name] = int(value)


def read_int(group: Group, name: str) -> int:
    """Read an integer attribute."""
    return int(cast(Any, group.attrs[name]))


def write_text(group: Group, name: str, value: str) -> None:
    """Store a string as an attribute of the group."""
    group.attrs[name] = value


def read_text(group: Group, name: str) -> str:
    """Read a string attribute."""
    return str(cast(Any, group.attrs[name]))


def write_mapping(group: Group, name: str, mapping: dict[str, Any]) -> None:
    """Store a JSON-serialisable mapping as one string attribute; numpy scalars and arrays are converted."""
    group.attrs[name] = json.dumps(mapping, default=_as_builtin, sort_keys=True)


def read_mapping(group: Group, name: str) -> dict[str, Any]:
    """Read a mapping stored by `write_mapping`."""
    return cast(dict[str, Any], json.loads(read_text(group, name)))


def _as_builtin(value: object) -> object:
    """For `json.dumps`, a value it does not know: numpy scalars and arrays become Python numbers and lists."""
    if isinstance(value, np.generic):
        return cast(object, value.item())
    if isinstance(value, np.ndarray):
        return cast(list[Any], value.tolist())
    raise TypeError(f"{type(value).__name__} is not JSON-serialisable")


def create_group(group: Group, name: str) -> Group:
    """Create a subgroup."""
    return group.create_group(name)  # type: ignore[reportUnknownMemberType]


def subgroup(group: Group, name: str) -> Group:
    """An existing subgroup."""
    found = group[name]
    if not isinstance(found, h5py.Group):
        raise KeyError(f"{name!r} is not a group")
    return found


# --- files and columns: what the evolution file's tables are made of ---


def create_file(path: Path) -> File:
    """Create a file for writing, in the format that supports single-writer multiple-reader access."""
    return h5py.File(path, "w", libver="latest")


def open_file(path: Path) -> File:
    """Open a file for reading, also while another process is writing it."""
    return h5py.File(path, "r", libver="latest", swmr=True)


def start_single_writer_mode(file: File) -> None:
    """Switch to single-writer multiple-reader mode: readers may open the file, and nothing new may be created."""
    file.swmr_mode = True


type ColumnKind = Literal["float", "int", "str"]
type IntArray = np.ndarray[tuple[int], np.dtype[np.int64]]
type Column = FloatArray | IntArray | list[str]
"""What a column reads back as: floats and arrays as float arrays, ints as an int array, strings as a list."""


def create_column(group: Group, name: str, kind: ColumnKind, width: int | None = None, length: int = 64) -> None:
    """Create an empty column: a dataset unlimited along the rows, chunked, of scalars or of arrays of `width`.

    Strings are stored as fixed-length bytes of at most `length` characters: HDF5's variable-length strings cannot
    be read by another process while the file is being written.
    """
    dtype: object = {"float": np.float64, "int": np.int64, "str": f"S{length}"}[kind]
    shape: tuple[int, ...] = () if width is None else (width,)
    chunk = 256 if width is None else max(1, 65536 // (8 * width))  # about 2 KB of scalars or 512 KB of arrays
    group.create_dataset(  # type: ignore[reportUnknownMemberType]
        name, shape=(0, *shape), maxshape=(None, *shape), dtype=dtype, chunks=(chunk, *shape)
    )


def append_column(group: Group, name: str, values: list[Any]) -> None:
    """Append rows to a column and flush them, so that a reader can see them; strings are encoded."""
    dataset = cast(h5py.Dataset, group[name])
    if cast(str, cast(Any, dataset).dtype.kind) == "S":
        values = [str(v).encode() for v in values]
    n = int(cast(Any, dataset).shape[0])
    dataset.resize(n + len(values), axis=0)  # type: ignore[reportUnknownMemberType]
    dataset[n : n + len(values)] = values
    dataset.flush()


def read_column(group: Group, name: str) -> Column:
    """Read a whole column; what has been flushed."""
    dataset = cast(h5py.Dataset, group[name])
    values = cast(Any, dataset[()])
    kind = cast(str, cast(Any, dataset).dtype.kind)
    if kind == "S":
        return [v.decode() for v in cast(list[bytes], list(values))]
    if kind == "i":
        return np.asarray(values, dtype=np.int64).reshape(-1)
    return np.asarray(values, dtype=np.float64)


def column_names(group: Group) -> list[str]:
    """The names of a table's columns."""
    return [str(cast(object, name)) for name in cast(Any, group)]
