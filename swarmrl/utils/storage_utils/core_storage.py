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
    ):
        self.out_folder = pathlib.Path(out_folder)
        self.h5_filename = self.out_folder / filename

        # Internal state
        self._is_initialized = False
        self._write_idx = 0
        self._data_holder = None
        self._h5_group_tag = None

    @abstractmethod
    def _get_dataset_specs(self, data_sample: Any) -> Dict[str, Dict[str, Any]]:
        """Return dataset specifications inferred from one sample."""
        pass

    @abstractmethod
    def _initialize_data_holder(self) -> Dict[str, List]:
        """Create an empty in-memory holder for pending writes."""
        pass

    @abstractmethod
    def _accumulate_data(self, data: Any) -> None:
        """Accumulate one sample into the data holder."""
        pass

    def _init_h5_output(self, data_sample: Any) -> None:
        self.out_folder.mkdir(parents=True, exist_ok=True)
        self._data_holder = self._initialize_data_holder()

        dataset_specs = self._get_dataset_specs(data_sample)

        with h5py.File(self.h5_filename.as_posix(), "a", libver="latest") as h5_outfile:
            group = h5_outfile.require_group(self._h5_group_tag)
            dataset_kwargs = dict(compression="gzip")

            for name, spec in dataset_specs.items():
                group.require_dataset(
                    name,
                    shape=spec["shape"],
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

        self._accumulate_data(data)
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
