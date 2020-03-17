"""
Author: Pablo Prietz
Usage: python xd_convert.py [OPTIONS] [FILENAMES]...
"""
import enum
import pathlib
import typing as T

import click
import pyxdf
import pandas as pd

from shared.markers_v2 import MarkersV2


class STREAM_TYPES:
    eye_tracking = "pupil_capture"
    marker = "psychopy_marker"
    brainvision_eda = "BrainVision RDA"
    g_tec = "g.USBamp"


FILE_SUFFIXES = {
    STREAM_TYPES.eye_tracking: "_eye_tracking",
    STREAM_TYPES.marker: "_marker",
    STREAM_TYPES.brainvision_eda: "_brainvision",
    STREAM_TYPES.g_tec: "_gtec"
}


class OutputFormat(enum.Enum):
    CSV = ".csv"
    PARQUET = ".parquet"


@click.command()
@click.option("--parquet", "format_", default=True, flag_value=OutputFormat.PARQUET)
@click.option("--csv", "format_", flag_value=OutputFormat.CSV)
@click.argument("filenames", nargs=-1, type=click.Path(exists=True))
def xdf_convert(format_, filenames):
    """Extracts streams from given XDF files and saves them to a defined output format

    filenames: List of XDF file paths

    Output: Each stream will be stored as an individual file next to their corresponding XDF file.
    """
    start_times = {}
    filenames = sorted(pathlib.Path(fn).resolve() for fn in filenames)
    for path in filenames:
        print(f"Loading {path}")

        # Assumes that the parent folder's name is the subject
        # Eg. path: .../ARB42/sub-ARB42_ses-S001_task-T1_run-001_eeg.xdf
        # -> subject_id: ARB42
        subject_id = path.parent.name

        streams_to_convert = [
            STREAM_TYPES.eye_tracking,
            STREAM_TYPES.marker,
            STREAM_TYPES.brainvision_eda,
            STREAM_TYPES.g_tec,
        ]
        streams = xdf_load_streams_by_name(path, streams_to_convert)

        # In case of split recordings, we have multiple xdf files in the same directory,
        # i.e. in path.parent. In these cases, we need a common start time for all xdf
        # files in this folder.
        # Check, if we calculated a start time for path.parent before. If not, extract
        # from loaded data frames, see stream_data() for details.
        start_time = start_times.get(path.parent)
        previous_split_found = start_time is not None
        streams_by_name, start_time = stream_data(streams, start_time)
        # Save calculated start_time in case other splits follow.
        start_times[path.parent] = start_time
        for stream_type, df in streams_by_name.items():
            export_path = path.with_name(subject_id + FILE_SUFFIXES[stream_type])
            export_path = export_path.with_suffix(format_.value)
            if format_ is OutputFormat.CSV:
                if previous_split_found and export_path.exists():
                    # In this case append to existing csv file
                    print(f"Appending to {export_path}")
                    df.to_csv(
                        export_path, index_label="time_stamps", mode="a", header=False
                    )
                else:
                    print(f"Exporting to {export_path}")
                    df.to_csv(export_path, index_label="time_stamps")
            elif format_ is OutputFormat.PARQUET:
                df.index.rename("time_stamps", inplace=True)
                if previous_split_found and export_path.exists():
                    print(f"Appending to {export_path}")
                    prev_df = pd.read_parquet(export_path)
                    combined = pd.concat([prev_df, df], axis=0)
                    combined.to_parquet(export_path, index=True)
                else:
                    print(f"Exporting to {export_path}")
                    df.to_parquet(export_path, index=True)
            else:
                raise ValueError(f"Don't know how to handle format: {format_}")


def xdf_load_streams_by_name(path, names=None):
    if names is not None:
        names = [{"name": name} for name in names]
    streams = pyxdf.load_xdf(path, select_streams=names)[0]
    return streams


def stream_data(streams, start_time=None):
    streams_by_name = (dataframe_from_stream(stream) for stream in streams)
    streams_by_name = dict(streams_by_name)
    if start_time is None:
        start_time = extract_start_time(streams_by_name[STREAM_TYPES.marker])
    normalize_index(streams_by_name.values(), start_time)
    return streams_by_name, start_time


def dataframe_from_stream(
    stream, header_replacement=None
) -> T.Tuple[str, pd.DataFrame]:
    name = stream["info"]["name"][0]
    data = stream["time_series"]

    try:
        channels = stream["info"]["desc"][0]["channels"][0]["channel"]
        headers = [chan["label"][0] for chan in channels]
    except TypeError:
        headers = None

    index = stream["time_stamps"]
    df = pd.DataFrame(data, index=index, columns=headers)
    if header_replacement is not None:
        df.rename(columns=header_replacement, inplace=True)
    if None in df.columns:
        print(f"Dropping unlabeled column(s) in stream {name}.")
        df.drop(columns=[None], inplace=True)
    headers_as_str = {col: str(col) for col in df.columns if not isinstance(col, str)}
    if headers_as_str:
        df.rename(columns=headers_as_str, inplace=True)

    return name, df


def extract_start_time(marker_df):
    start_marker_mask = marker_df["0"] == str(MarkersV2.start.value)
    start_ts = marker_df[start_marker_mask]["0"].index[0]
    return start_ts


def normalize_index(dfs, start_time):
    print(f"Using {start_time} as start time:")
    for df in dfs:
        df.index -= start_time


if __name__ == "__main__":
    xdf_convert()
