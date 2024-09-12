import os
import mne
import socket

import pandas as pd 

def get_filenames():
    if socket.gethostname() == "LAPTOP-TFQFNF6U":
        data_path = "G:/ACH_Files/"
        filenames = ["Patient_1.lay"]  
    elif socket.gethostname() == "DLP":
        data_path = "G:/ACH_Files/"
        filenames = ["Patient_1.lay"]  
    
    return data_path, filenames

ieeg_data_dir, ieeg_fns = get_filenames()

for ieeg_fn in ieeg_fns:
    ieeg_filepath = ieeg_data_dir+ieeg_fn
    ieeg_data = mne.io.read_raw_persyst(ieeg_filepath, verbose=False)
    study_start_date = ieeg_data.info['meas_date']

    # ts = pd.Timestamp(ieeg_data.info['subject_info']['birthday'])
    # ieeg_data.info['subject_info']['birthday'] = ts.to_julian_date() 

    birthday = ieeg_data.info['subject_info']['birthday']
    ieeg_data.info['subject_info']['birthday'] = [birthday.year, birthday.month, birthday.day]

    time = ieeg_data.times
    nr_samples = ieeg_data.n_times
    fs = ieeg_data.info["sfreq"]

    print(ieeg_fn)
    print(f"Sampling Rate: {fs}")
    print(f"Duration min: {time[-1]/60}")

    out_dir= "G:/laydat_to_raw/"

    start_time = 0
    end_time = time[-1]
    out_raw_ieeg = out_dir+ieeg_fn.replace('.lay', '_raw.fif')
    
    try:
        ieeg_data.save(fname=out_raw_ieeg, picks=None, tmin=start_time, tmax=end_time, fmt='single', overwrite=False, split_size='2GB', split_naming='bids', verbose=None)
    except:
        print("Saving Raw file failed")
        os.remove(out_raw_ieeg)
    


    print(f"Raw export Done")

pass