'''
Authors : Pablo Prietz, Kerstin Pieper
'''

import dataclasses
import enum
import logging
import pathlib
import typing as T
import itertools

import pandas as pd

from processing.shared.xdf_convert import STREAM_TYPES, FILE_SUFFIXES, OutputFormat

logger = logging.getLogger(__name__)




@dataclasses.dataclass
class Recording:
    directory: pathlib.Path
    '''Recording
    Collection of functions to handle recorded .xdf files. 
    Extracts data out of lsl streams and save it to parquet/csv.
    Combines raw physio data with marker data.
    '''

    @classmethod
    def find_by_pattern(
        cls, pattern: str, root: T.Optional[T.Union[pathlib.Path, str]] = None
    ) -> T.Iterable["Recording"]:

        if root is None:
            root = pathlib.Path().resolve()
        elif not isinstance(root, pathlib.Path):
            root = pathlib.Path(root).resolve()
            if not root.exists():
                raise FileNotFoundError(root)

        logger.debug(f"Root: {root}")
        logger.debug(f"Pattern: {pattern}")
        for match in root.rglob(pattern):
            if match.is_file():
                yield cls.from_incl_file(match)
            elif match.is_dir():
                yield cls(match)
            else:
                raise RuntimeError(f"Unknown path type: {match}")

    @classmethod
    def from_incl_file(cls, file: pathlib.Path) -> "Recording":
        assert file.is_file()
        return cls(file.resolve().parent)

    def marker_path(self, ext: OutputFormat = OutputFormat.PARQUET) -> pathlib.Path:
        pat = f"*{FILE_SUFFIXES[STREAM_TYPES.marker]}{ext.value}"
        return next(self.directory.glob(pat), None)

    def marker_path_missing(self, *args, **kwargs) -> pathlib.Path:
        """Path to marker file that includes missing markers"""
        return self._marker_path_fixed("_missing", *args, **kwargs)

    def marker_path_invalid(self, *args, **kwargs) -> pathlib.Path:
        """Path to marker file that includes invalid markers"""
        return self._marker_path_fixed("_invalid", *args, **kwargs)

    def _marker_path_fixed(self, suffix, *args, **kwargs):
        marker_path = self.marker_path(*args, **kwargs)
        if not marker_path:
            return
        stem_fixed = marker_path.stem + suffix
        suffix = marker_path.suffix
        path_fixed = marker_path.with_name(stem_fixed)
        path_fixed = path_fixed.with_suffix(suffix)
        return path_fixed

    def eye_tracking_path(
        self, ext: OutputFormat = OutputFormat.PARQUET
    ) -> pathlib.Path:
        pat = f"*{FILE_SUFFIXES[STREAM_TYPES.eye_tracking]}{ext.value}"
        return next(self.directory.glob(pat), None)

    def ecg_path(self, ext: OutputFormat = OutputFormat.PARQUET) -> pathlib.Path:
        pat = f"*{FILE_SUFFIXES[STREAM_TYPES.brainvision_eda]}{ext.value}"
        return next(self.directory.glob(pat), None)

    def eeg_path(self, ext: OutputFormat = OutputFormat.PARQUET) -> pathlib.Path:
        pat = f"*{FILE_SUFFIXES[STREAM_TYPES.g_tec]}{ext.value}"
        return next(self.directory.glob(pat), None)

    @staticmethod
    def read_csv(path: pathlib.Path) -> pd.DataFrame:
        return pd.read_csv(path, index_col="time_stamps")

    @staticmethod
    def read_parquet(path: pathlib.Path, *args, **kwargs) -> pd.DataFrame:
        df = pd.read_parquet(path, *args, **kwargs)
        if df.index.name != "time_stamps":
            df.set_index("time_stamps", inplace=True)
        return df

    def read_markers(self, include_fixes=True):
        try:
            df = self._read_markers(OutputFormat.PARQUET, include_fixes)
        except FileNotFoundError:
            df = self._read_markers(OutputFormat.CSV, include_fixes)
        column_mapping = {"0": "id", "1": "label"}
        marker_id_labels = df[["0", "1"]]
        marker_id_labels = marker_id_labels.rename(columns=column_mapping)
        marker_id_labels.id = marker_id_labels.id.astype(int)
        return marker_id_labels

    def _read_markers(self, ext: OutputFormat, include_fixes):
        read_method = {
            OutputFormat.PARQUET: self.read_parquet,
            OutputFormat.CSV: self.read_csv,
        }[ext]
        df = read_method(self.marker_path(ext))
        invalid_marker_path = self.marker_path_invalid(ext)
        if include_fixes and invalid_marker_path.exists():
            invalid_markers = read_method(invalid_marker_path)
            num_inv_markers = invalid_markers.shape[0]
            try:
                print(f"{__name__}! Dropping {num_inv_markers} invalid marker(s).")
                df = df.drop(index=invalid_markers.index)
            except KeyError as err:
                err_msg = (
                    "Invalid markers could not be found in recorded markers! ",
                    "Make sure to save the *exact* timestamp of the invalid marker.",
                )
                raise KeyError(err_msg) from err

        missing_marker_path = self.marker_path_missing(ext)
        if include_fixes and missing_marker_path.exists():
            missing_markers = read_method(missing_marker_path)
            num_mis_markers = missing_markers.shape[0]
            print(f"{__name__}! Inserting {num_mis_markers} missing marker(s).")
            df = pd.concat([df, missing_markers]).sort_index()
        return df

    @property
    def vp_code(self):
        return self.directory.name




