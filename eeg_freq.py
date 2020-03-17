'''
Authors: Kerstin Pieper, Pablo Prietz
'''
import functools
import itertools
import pathlib
import click
import mne
import numpy as np
import pandas as pd

from eeg2mne import eeg2mne
from shared.markers_example import Periods
from shared.select_data import select_data
from shared.recording import Recording


def filter_raw(raw):
    '''
    Bandpass filter
    '''
    fmin, fmax = 1, 48  # to adjust

    raw_fir_filtered = raw.filter(
        fmin,
        fmax,
        l_trans_bandwidth="auto",
        h_trans_bandwidth="auto",
        filter_length="auto",
        method="fir",
        fir_window="hamming",
        fir_design="firwin",
        phase="zero",
    )

    return raw_fir_filtered


def check_for_bads(data_raw, b_ch):
    if not b_ch:
        print('no bad channels')
        return False
    else:
        data_raw.info['bads'].extend(b_ch)
        print(f"\tFound bad channels `{data_raw.info['bads']}`:")
        return True


def mod_chan_list(data_raw, vp_code):
    bad = data_raw.info['bads']
    channels = data_raw.info['ch_names']
    for x in bad:
        try:
            channels.remove(x)
        except ValueError:
            pass
    return channels


def do_welch(raw):
    psds_welch, freqs = mne.time_frequency.psd_welch(
        raw,
        fmin=1,
        fmax=48,
        average=None,
        reject_by_annotation=True,  # exclude bad channels
    )
    return psds_welch, freqs


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


def calc_eeg_stats_per_band(psds_welch, channels, subblock_idc):
    Freq_bands = {
        "Delta": [0, 4],
        "Theta": [4, 8],
        "Alpha": [8, 12],
        "Beta": [12, 30],
        "Gamma": [30, 45],
    }

    selection = subblock_idc.values.reshape(-1)

    eeg_stats = []
    for fmin, fmax in Freq_bands.values():
        psds_band = psds_welch[:, fmin:fmax, selection].mean(axis=-1).mean(axis=-1)
        eeg_stats.append(psds_band)
    columns = pd.Index(channels, name="Channels")
    index = pd.Index(Freq_bands.keys(), name="Freq Bands")

    aggregated = pd.DataFrame(eeg_stats, index=index, columns=columns)
    return aggregated


@click.command()
@click.argument("folders", nargs=-1, type=click.Path(exists=True))
def main(folders):
    print(folders)
    """Processes and extracts statistics from EEG data

    folders: List of folders containing processed eeg parquet files

    Output: Statistics saved to folder/extracted_csv/eeg_<vp_code>.csv
    """
    start_times = {}
    folders = sorted(pathlib.Path(fn).resolve() for fn in folders)
    for path in folders:
        R = Recording(path)
        print(f"Loading {path}")
        try:
            conditions = R.condition_order()
        except KeyError:
            print(f"\tUnknown vp_code {R.vp_code}. Aborting.")
            continue
        print(f"\tFound vp_code `{R.vp_code}`:")
        print(f"\t\tBlock 0: {conditions.block0}")
        print(f"\t\tBlock 1: {conditions.block1}")
        print(f"\t\tBlock 2: {conditions.block2}")
        data = R.read_parquet(R.eeg_path())
        data_raw = eeg2mne(data)
        markers = R.read_markers()
        print("\tData loaded. Starting processing...")

        print("\tFilter frequencies below 1Hz and above 48 Hz.")
        data_filtered = filter_raw(data_raw)

        print("\tCheck for bad channels")
        b_ch = R.get_bads()
        if check_for_bads(data_raw, b_ch) == True:
            psds_welch, freqs = do_welch(data_filtered)
            channels = mod_chan_list(data_raw, R.vp_code)
            print("\tCalc Welch without bads.")

        else:
            print("\tCalc Welch.")
            psds_welch, freqs = do_welch(data_filtered)
            channels = data_raw.info['ch_names']

        n_seg = psds_welch.shape[-1]
        eeg_ts_first = data.index[0]
        welch_idc = np.arange(n_seg)
        welch_ts = welch_idc + eeg_ts_first
        welch_idc_df = pd.DataFrame(welch_idc, index=welch_ts)
        blocks = select_data(welch_idc_df, markers, Periods.block)

        print("\tCalc Welch.")
        calc_eeg_stats_per_band_fixed = functools.partial(calc_eeg_stats_per_band, psds_welch, channels)

        results = []

        for cond, block in zip(conditions, blocks):
            print(f"\tCalculating statistics for {cond.value}...")
            periods = extract_periods(block)

            eeg_stats_per_band = periods.groupby(level=0).apply(calc_eeg_stats_per_band_fixed)
            channel_names = eeg_stats_per_band.columns
            eeg_stats_per_band_long = pd.concat(
                [eeg_stats_per_band[cn] for cn in channel_names],
                keys=channel_names,
            )
            eeg_stats_per_band_long.name = "Power"
            results.append(eeg_stats_per_band_long)

        print("\tFinished statistics.")

        print("\tCombining results...")
        condition_labels = [cond.value for cond in conditions]
        results = pd.concat(results, keys=condition_labels, names=["block"], sort=True)
        final = pd.concat([results], keys=[R.vp_code], names=["vp_code"])
        final_frame = final.to_frame()
        final_frame.reset_index(inplace=True)

        target_dir = R.directory / "extracted_csv"
        target_dir.mkdir(exist_ok=True)
        extracted_csv_name = f"eeg_freq_{R.vp_code}.csv"
        extracted_csv_path = target_dir / extracted_csv_name
        print(f"\tWriting results to {extracted_csv_path}")
        final_frame.to_csv(extracted_csv_path, index=False)
        print("\tDone!")


if __name__ == "__main__":
    main()


