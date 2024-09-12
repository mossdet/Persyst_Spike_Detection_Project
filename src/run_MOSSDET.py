import os
import subprocess
import numpy as np
import pandas as pd
from scipy.io import savemat

def run_mossdet(path_to_mossdet: str = None, fs:int=0, eeg_signal:np.ndarray=None, sig_name:str='SigA') -> None:

    eegSignal = np.transpose(eeg_signal)
    
    saved_signal_path = path_to_mossdet + sig_name + '.mat'
    eeg_signal_dict = {'GlobalDouble': eegSignal}
    savemat(saved_signal_path, eeg_signal_dict)

    # with open(savedSignalPath, "w") as sig_file_fid:
    #     eegSignal.tofile(sig_file_fid, sep='')
                    
    outputFolder = path_to_mossdet + sig_name + '/'

    params = {}
    params['mossdet_exe_path'] = path_to_mossdet +  'Test_ConsoleApp.exe'#"MOSSDET_c.exe", Test_ConsoleApp.exe
    params['saved_signal_path'] = saved_signal_path
    params['dec_functions_path'] = path_to_mossdet
    params['output_path'] = path_to_mossdet + sig_name + os.path.sep
    params['start_time'] = 0
    params['end_time'] = 60*60*243*65
    params['sampling_rate'] = fs
    params['eoi_type'] = 'HFO+IES'  # Options are 'HFO+IES' or 'SleepSpindles'
    params['mossdet_verbose'] = 1
    params['save_detections'] = 1

    os.makedirs(params['output_path'], exist_ok=True)

    command = [str(val) for key, val in params.items()]
    #command = ' '.join(command)
    #print(command)

    #result = subprocess.run(command, capture_output=True, text=True, shell=False)

    result = subprocess.run(["F:\Postdoc_Calgary\Research\Persyst_Project\Persyst_Spike_Detection_Project\src\MOSSDET\Test_ConsoleApp.exe"], capture_output=True, text=True, shell=False)

    result = subprocess.run(command, capture_output=True, text=True, shell=False)
    output = result.stdout

    result = subprocess.run(command, capture_output=True, text=True, check=True, shell=True)

    print(result.stdout)


    # if os.path.exists(saved_signal_path):
    #     os.remove(saved_signal_path)
    # else:
    #     print("The signal file does not exist") 


    pass

if __name__ == '__main__':

    this_module_path = os.path.dirname(os.path.abspath(__file__)) + os.path.sep
    path_to_mossdet = this_module_path+"MOSSDET" + os.path.sep
    fs = 2000
    eeg_signal = np.random.rand(1,fs*60*2)
    sig_name = 'mattest'
    run_mossdet(path_to_mossdet, fs, eeg_signal, sig_name)