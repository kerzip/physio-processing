'''
Author: Pablo Prietz
'''

import itertools
import logging
import typing as T
import pandas as pd

from processing.shared.markers_example import Periods
from processing.shared.markers_example import Markers

logger = logging.getLogger(__name__)


def select_from_data(
    data: pd.DataFrame, markers: pd.DataFrame, period: Periods
) -> T.Iterable[T.Tuple[pd.DataFrame, pd.DataFrame]]:
    """Yields subsections of data that correspond to period

    Input:
        data: Any extracted xdf stream
        markers: Extracted marker stream
        period: Period definition
    """

    intervals = intervals_from_period(markers, period)

    for start_ts, stop_ts in intervals:
        data_slice = data.loc[start_ts:stop_ts]
        marker_slice = markers.loc[start_ts:stop_ts]
        yield data_slice, marker_slice


def select_data_around(
    data: pd.DataFrame,
    markers: pd.DataFrame,
    event: Markers,
    before_s: float = 0.2,
    after_s: float = 0.5,
    relative_time: bool = False,
) -> T.Iterable[T.Tuple[pd.DataFrame, pd.DataFrame]]:
    markers_of_interest = _marker_entries_event(markers, event)
    timestamps = markers_of_interest.index.values
    start_ts_all = timestamps - before_s
    stop_ts_all = timestamps + after_s
    intervals = zip(start_ts_all, timestamps, stop_ts_all)
    for start_ts, event_ts, stop_ts in intervals:
        data_slice = data.loc[start_ts:stop_ts]
        if relative_time:
            data_slice = data_slice.set_index(data_slice.index - event_ts)
        yield data_slice


def intervals_from_period(
    markers: pd.DataFrame, period: Periods
) -> T.Iterator[T.Tuple[float, float]]:
    for start_mid, end_mid in period.value:
        markers_of_interest = _marker_entries_for_mids(markers, start_mid, end_mid)
        timestamps = markers_of_interest.index.values
        if timestamps.size % 2 != 0:
            timestamps = timestamps[:-1]
        timestamp_pairs = timestamps.reshape(-1, 2)
        yield from timestamp_pairs


def _marker_entries_event(markers, marker_id):
    if marker_id in Markers:
        marker_id = marker_id.value
    mask = markers.id == marker_id
    markers_of_interest = markers.loc[mask]
    return markers_of_interest


def _marker_entries_for_mids(markers, start_mid, end_mid):
    if start_mid in Markers:
        start_mid = start_mid.value
    if end_mid in Markers:
        end_mid = end_mid.value
    start_mask = markers.id == start_mid
    end_mask = markers.id == end_mid
    start_end_mask = start_mask | end_mask

    markers_of_interest = markers.loc[start_end_mask]

    # Drop consecutive duplicates:
    # https://stackoverflow.com/questions/19463985/pandas-drop-consecutive-duplicates
    shifted_MOI = markers_of_interest.id.shift()
    not_conscutively_duplicated = shifted_MOI != markers_of_interest.id
    markers_of_interest = markers_of_interest.loc[not_conscutively_duplicated]

    return markers_of_interest


def groupby_period(
    data: pd.DataFrame,
    markers: pd.DataFrame,
    period: Periods,
    return_markers: bool = False,
) -> pd.DataFrame:
    idc = combined_interval_index_from_period(data, markers, period)
    data_cut = pd.cut(data.index, idc)
    data_groups = data.groupby(data_cut)
    if return_markers:
        marker_cut = pd.cut(markers.index, idc)
        marker_groups = markers.groupby(marker_cut)
        return data_groups, marker_groups
    return data_groups


def combined_interval_index_from_period(
    data: pd.DataFrame, markers: pd.DataFrame, period: Periods
) -> pd.IntervalIndex:
    intervals = interval_indices_from_period(markers, period)

    # Initial interval might be empty
    combined_interval = next(intervals)
    for interval in intervals:
        # Only consider non-empty intervals
        if len(interval):
            if not len(combined_interval):
                # Replace initial interval if it was empty
                combined_interval = interval
            else:
                # Else append to initial interval
                combined_interval = combined_interval.append(interval)
    return combined_interval


def interval_indices_from_period(
    markers: pd.DataFrame, period: Periods
) -> T.Iterator[pd.IntervalIndex]:
    for start_mid, end_mid in period.value:
        markers_of_interest = _marker_entries_for_mids(markers, start_mid, end_mid)
        timestamps = markers_of_interest.index.values
        if timestamps.size % 2 != 0:
            timestamps = timestamps[:-1]
        timestamp_pairs = timestamps.reshape(-1, 2)
        interval_idc = pd.IntervalIndex.from_arrays(*timestamp_pairs.T, closed="left")
        yield interval_idc


def split(period_data, *, num_periods):
    start, end = period_data.index[[0, -1]]
    intervals = pd.interval_range(start, end, num_periods)
    cuts = pd.cut(period_data.index, intervals)
    groups = period_data.groupby(cuts)
    return groups


def extract_periods(block, num_task_subblocks=6):
    '''
    Divide data streams into periods which are defined by Periods.
    The task block is splitted into smaller subblocks to compare values over time.

    :param block:
    :param num_task_subblocks:
    :return:
    '''
    data, markers = block
    bline_h = next(select_data(data, markers, Periods.baseline_h))
    bline_l = next(select_data(data, markers, Periods.baseline_l))

    task, _ = next(select_data(data, markers, Periods.task))
    subblocks = split(task, num_periods=num_task_subblocks)
    subblocks = map(lambda groupby: groupby[1], subblocks)

    baselines = (bline_h[0], bline_l[0])
    all_periods = itertools.chain(baselines, subblocks)

    period_names = [ "baseline_high", "baseline_low"]
    period_names += [f"task_subblock_{idx}" for idx in range(num_task_subblocks)]
    period_concat = pd.concat(
        all_periods, keys=period_names, names=["period", data.index.name]
    )
    return period_concat


def yield_periods(block, num_task_subblocks=6):
    data, markers = block

    task_data, task_markers = next(select_data(data, markers, Periods.task))
    subblocks = split(task_data, num_periods=num_task_subblocks)
    subblocks = list(map(lambda groupby: groupby[1], subblocks))
    submarkers = split(task_markers, num_periods=num_task_subblocks)
    submarkers = list(map(lambda groupby: groupby[1], submarkers))

    period_names = [f"task_subblock_{idx}" for idx in range(num_task_subblocks)]
    yield from zip(period_names, subblocks, submarkers)
