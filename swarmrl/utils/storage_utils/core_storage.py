"""Core abstractions for HDF5 trajectory storage."""

import pathlib
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import h5py
import numpy as np


class HDF5TrajectoryStorage(ABC):
    """
    Abstract base class for HDF5-based trajectory storage, handling
    - file initialization
    - dataset creation,
    - batch writing with proper indexing.
    """

    def __init__(
        self,
        out_folder: str = "./data",
        filename: str = "trajectory.hdf5",
        allow_existing_file: bool = False,
        write_chunk_size: int = 1,
    ):
        if write_chunk_size < 1:
            raise ValueError("write_chunk_size must be at least 1")

        self.out_folder = pathlib.Path(out_folder)
        self.h5_filename = self.out_folder / filename
        self.allow_existing_file = allow_existing_file
        self.write_chunk_size = write_chunk_size

        # Internal state
        self._is_initialized = False
        self._write_idx = 0
        self._data_holder = None
        self._h5_group_tag = None
        self._dataset_keys = []

    @abstractmethod
    def _get_dataset_specs(self, data_sample: Any) -> Dict[str, Dict[str, Any]]:
        """Return dataset specifications inferred from one sample."""
        pass

    @abstractmethod
    def _extract_sample(self, data_sample: Any) -> Dict[str, Any]:
        """Return one sample mapped by HDF5 dataset name."""
        pass

    def _initialize_data_holder(self) -> Dict[str, List]:
        """Create an empty in-memory holder for pending writes."""
        return {key: list() for key in self._dataset_keys}

    def _accumulate_sample(self, sample: Dict[str, Any]) -> None:
        if self._data_holder is None:
            self._dataset_keys = list(sample.keys())
            self._data_holder = self._initialize_data_holder()

        for key in self._dataset_keys:
            self._data_holder[key].append(sample[key])

    def _init_h5_output(self, data_sample: Any) -> None:
        if not self.allow_existing_file and self.h5_filename.exists():
            raise FileExistsError(
                f"Refusing to write to existing file: {self.h5_filename}. "
                "Set allow_existing_file=True if reusing it is intentional."
            )

        dataset_specs = self._get_dataset_specs(data_sample)
        self._dataset_keys = list(dataset_specs.keys())
        self.out_folder.mkdir(parents=True, exist_ok=True)
        self._data_holder = self._initialize_data_holder()

        with h5py.File(self.h5_filename.as_posix(), "a", libver="latest") as h5_outfile:
            group = h5_outfile.require_group(self._h5_group_tag)
            dataset_kwargs = dict(compression="gzip")

            for name, spec in dataset_specs.items():
                if name in group:
                    continue

                initial_shape = (0, *spec["shape"][1:])
                group.create_dataset(
                    name,
                    shape=initial_shape,
                    maxshape=spec["maxshape"],
                    dtype=spec["dtype"],
                    **dataset_kwargs,
                )

        self._is_initialized = True
        self._write_idx = 0

    def _write_to_h5(self) -> None:
        if not self._data_holder or all(
            len(v) == 0 for v in self._data_holder.values()
        ):
            return

        with h5py.File(self.h5_filename.as_posix(), "a", libver="latest") as h5_outfile:
            group = h5_outfile[self._h5_group_tag]

            for key in self._data_holder.keys():
                dataset = group[key]
                values = np.stack(self._data_holder[key], axis=0)
                n_new = values.shape[0]

                dataset.resize(self._write_idx + n_new, axis=0)
                dataset[self._write_idx : self._write_idx + n_new] = values

            self._write_idx += n_new

        self._data_holder = self._initialize_data_holder()

    def write(self, data: Any) -> None:
        if not self._is_initialized:
            self._init_h5_output(data)

        sample = self._extract_sample(data)
        self._accumulate_sample(sample)

        first_key = self._dataset_keys[0] if self._dataset_keys else None
        if (
            first_key is not None
            and len(self._data_holder[first_key]) >= self.write_chunk_size
        ):
            self._write_to_h5()

    def finalize(self) -> None:
        """Write any buffered samples to HDF5."""
        self._write_to_h5()

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    def write_accumulated_batch(self, accumulated_data: Dict[str, List]) -> None:
        if not accumulated_data or all(len(v) == 0 for v in accumulated_data.values()):
            return

        with h5py.File(self.h5_filename.as_posix(), "a", libver="latest") as h5_outfile:
            group = h5_outfile[self._h5_group_tag]

            n_new = None
            for key in accumulated_data.keys():
                dataset = group[key]
                values = np.stack(accumulated_data[key], axis=0)
                n_new = values.shape[0]

                dataset.resize(self._write_idx + n_new, axis=0)
                dataset[self._write_idx : self._write_idx + n_new] = values

            if n_new is not None:
                self._write_idx += n_new
