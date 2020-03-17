'''
Authors: Kerstin Pieper, Pablo Prietz

Basic steps to extract parameters from ECG raw data

'''
import itertools
import pathlib

import click
import neurokit as nk
import pandas as pd

from processing.shared.markers_example import Periods
from processing.shared.select_data import select_data
from processing.shared.recording import Recording


def process_ecg(data):
    """Process_ecg uses neurokit to extract features from raw data.

    :param
    data: pandas DataFrame
        Recorded raw data is saved as DataFrame before
    :return:
    result_series: pandas Series
        Includes extracted features (HR, sdNN, RMSSD)
    """
    print("\tStarting ecg processing...")
    result = nk.ecg_process(
        data,
        sampling_rate=1000,     # must fit to current recording
    )

    result["df"].set_index(data.index, inplace=True)
    hrv_results = result["ECG"]["HRV"]
    sdnn = hrv_results["sdNN"]
    rMSSD = hrv_results["RMSSD"]
    hr_mean = result["df"].Heart_Rate
    print("\tFinished ecg processing.")

    result_series = pd.Series(
        [hr_mean, sdnn, rMSSD],
        index=["hr_mean", "sdnn", "rMSSD"],
    )
    return result_series


def split(period_data, *, num_periods):
    start, end = period_data.index[[0, -1]]
    intervals = pd.interval_range(start, end, num_periods)
    cuts = pd.cut(period_data.index, intervals)
    groups = period_data.groupby(cuts)
    return groups


def extract_periods(block, num_task_subblocks=6):
    data, markers = block
    basline_h = next(select_data(data, markers, Periods.baseline_h))
    basline_l = next(select_data(data, markers, Periods.baseline_l))

    task, _ = next(select_data(data, markers, Periods.task))
    subblocks = split(task, num_periods=num_task_subblocks)
    subblocks = map(lambda groupby: groupby[1], subblocks)

    baselines = (basline_h[0], basline_l[0])
    all_periods = itertools.chain(baselines, subblocks)

    period_names = ["baseline_h", "baseline_l"]
    period_names += [f"task_subblock_{idx}" for idx in range(num_task_subblocks)]
    period_concat = pd.concat(
        all_periods, keys=period_names, names=["period", data.index.name]
    )
    return period_concat



@click.command()
@click.argument("folders", nargs=-1, type=click.Path(exists=True))
def main(folders):
    print(folders)
    """Processes and extracts statistics from EDA data

    folders: List of folders containing processed eda csv files

    Output: Stattistics saved to folder/extracted_csv/eda_<vp_code>.csv
    """
    start_times = {}
    folders = sorted(pathlib.Path(fn).resolve() for fn in folders)
    for path in folders:
        R = Recording(path)
        print(f"Loading {path}")
        conditions = R.condition_order()
        print(f"\tFound vp_code `{R.vp_code}`:")
        print(f"\t\tBlock 0: {conditions.block0}")
        print(f"\t\tBlock 1: {conditions.block1}")
        print(f"\t\tBlock 2: {conditions.block2}")
        data = R.read_csv(R.ecg_path)
        markers = R.read_markers()
        print("\tData loaded. Starting processing...")

        blocks = select_data(data.ECG, markers, Periods.block)
        results = []
        for cond, block in zip(conditions, blocks):
            print(f"\tCalculating statistics for {cond.value}...")
            periods = extract_periods(block)
            stats = periods.groupby(level=0).apply(process_ecg)
            results.append(stats)
            print("\tFinished statistics.")

        print("\tCombining results...")
        condition_labels = [cond.value for cond in conditions]
        results = pd.concat(results, keys=condition_labels, names=["block"], sort=True)
        final = pd.concat([results], keys=[R.vp_code], names=["vp_code"])
        final_frame = final.to_frame()
        final_frame.reset_index(inplace=True)

        target_dir = R.directory / "extracted_csv"
        target_dir.mkdir(exist_ok=True)
        extracted_csv_name = f"ecg_{R.vp_code}.csv"
        extracted_csv_path = target_dir / extracted_csv_name
        print(f"\tWriting results to {extracted_csv_path}")
        final_frame.to_csv(extracted_csv_path, index=False)
        print("\tDone!")


if __name__ == "__main__":
       main()