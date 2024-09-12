import os
import mne
import datetime

def get_filenames():
    data_path = "G:/laydat_to_raw/raw_fif/"
    filenames = ["Patient_1_split-01_raw.fif", "Patient_1_split-02_raw.fif", "Patient_1_split-03_raw.fif", "Patient_1_split-04_raw.fif", "Patient_1_split-05_raw.fif", "Patient_1_split-06_raw.fif", "Patient_1_split-07_raw.fif", "Patient_1_split-08_raw.fif", "Patient_1_split-09_raw.fif", "Patient_1_split-10_raw.fif", "Patient_1_split-11_raw.fif", "Patient_1_split-12_raw.fif", "Patient_1_split-13_raw.fif", "Patient_1_split-14_raw.fif", "Patient_1_split-15_raw.fif", "Patient_1_split-16_raw.fif", "Patient_1_split-17_raw.fif", "Patient_1_split-18_raw.fif", "Patient_1_split-19_raw.fif", "Patient_1_split-20_raw.fif", "Patient_1_split-21_raw.fif", "Patient_1_split-22_raw.fif", "Patient_1_split-23_raw.fif", "Patient_1_split-24_raw.fif", "Patient_1_split-25_raw.fif", "Patient_1_split-26_raw.fif", "Patient_1_split-27_raw.fif", "Patient_1_split-28_raw.fif", "Patient_1_split-29_raw.fif", "Patient_1_split-30_raw.fif", "Patient_1_split-31_raw.fif", "Patient_1_split-32_raw.fif", "Patient_1_split-33_raw.fif", "Patient_1_split-34_raw.fif", "Patient_1_split-35_raw.fif", "Patient_1_split-36_raw.fif", "Patient_1_split-37_raw.fif", "Patient_1_split-38_raw.fif", "Patient_1_split-39_raw.fif", "Patient_1_split-40_raw.fif", "Patient_1_split-41_raw.fif", "Patient_1_split-42_raw.fif", "Patient_1_split-43_raw.fif", "Patient_1_split-44_raw.fif"]
    return data_path, filenames

ieeg_data_dir, ieeg_fns = get_filenames()

# for idx, ieeg_fn in enumerate(ieeg_fns):
#     ieeg_filepath = ieeg_data_dir+ieeg_fn
#     new_fn = ieeg_data_dir + str(idx) + "_" + ieeg_fn
#     os.rename(ieeg_filepath, new_fn)
#     pass 

data_out_path = "G:/laydat_to_raw/edf_b/"
os.makedirs(data_out_path, exist_ok=True)
time_change = datetime.timedelta(seconds=0)

for idx, ieeg_fn in enumerate(ieeg_fns):
    ieeg_filepath = ieeg_data_dir + str(idx) + "_" + ieeg_fn
    assert os.path.isfile(ieeg_filepath), f"File {ieeg_filepath} does not exist"

    ieeg_data = mne.io.read_raw_fif(ieeg_filepath, on_split_missing='ignore',verbose=False)
    study_start_date = ieeg_data.info['meas_date']
    eeg_dur_s = (ieeg_data.n_times/ieeg_data.info['sfreq'])
    
    ieeg_data.set_meas_date(study_start_date+time_change)
    print(ieeg_data.info['meas_date'])
    time_change = time_change+datetime.timedelta(seconds=eeg_dur_s)

    time = ieeg_data.times
    nr_samples = ieeg_data.n_times
    fs = ieeg_data.info["sfreq"]

    print(ieeg_fn)
    print(f"Sampling Rate: {fs}")
    print(f"Duration min: {time[-1]/60}")

    out_ieeg = data_out_path+ieeg_fn.replace('_raw.fif', '.edf')
    mne.export.export_raw(fname=out_ieeg, raw=ieeg_data, fmt='edf')
    print("EDF export Done")

pass