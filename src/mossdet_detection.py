import os
import mne

from persyst_spikes_evaluator import PersystSpikesEvaluator
from montage_creator import MontageCreator

mossdet_path = "MOSSDET/MOSSDET_c.exe/"

# define path to EEG files
this_file_path = os.path.dirname(os.path.abspath(__file__))
workspacePath = this_file_path[: this_file_path.rfind(os.path.sep)] + os.path.sep
ieeg_data_dir = "F:/Postdoc_Calgary/Research/Persyst_Project/EEG_Clips/"
ieeg_fn = "Patient_1_anonym-clip.lay"
ieeg_filepath = ieeg_data_dir + ieeg_fn


ieeg_data = mne.io.read_raw_persyst(fname=ieeg_filepath, verbose=False)

evaluator = PersystSpikesEvaluator(ieeg_filepath=ieeg_filepath, images_path=None)
manual_marks = evaluator.parse_manual_eoi(manual_eoi_key="@Spike")
#auto_marks = evaluator.parse_auto_eoi(auto_eoi_key="XLSpike")

start_sample = int(0)
end_sample = int(evaluator.fs * 60 * 10)
bip_ieeg = evaluator.read_seeg_bip_data(start=start_sample, stop=end_sample, plot_ok=False)
for mtg_idx, mtg_name in enumerate(bip_ieeg['mtg_labels']):
    print(mtg_name)

    mtg_data = bip_ieeg['data'][mtg_idx,:]


    subprocess.run([mossdet_path, "my-python-script.py"])

    pass


pass