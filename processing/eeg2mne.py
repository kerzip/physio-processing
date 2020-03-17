'''
Author: Pablo Prietz
'''
import pandas as pd
import mne



def eeg2mne(gtec_dataframe):
    electrode_sites = ['F3','Fz','F4','T3','C3','Cz','C4','T4','P3','Pz','P4','O1','Oz','O2']

    # Select columns with channels of interest
    eeg = gtec_dataframe[electrode_sites]
    # Convert data to array format
    eeg_array = eeg.to_numpy().transpose()
    # Create list with channel names
    channel_names = electrode_sites
    # Create list with channel types
    channel_types = ['eeg'] * len(electrode_sites)
    # Create sampling frequency
    sfreq = 256
    # Create MNE info file
    eeg_info = mne.create_info(channel_names, sfreq, channel_types)
    # Create raw file in MNE format
    eeg_mne_format = mne.io.RawArray(eeg_array, eeg_info)
    return eeg_mne_format