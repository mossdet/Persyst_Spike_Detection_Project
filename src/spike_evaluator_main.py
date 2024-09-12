import os
import mne
from persyst_spikes_evaluator import PersystSpikesEvaluator

this_file_path = os.path.dirname(os.path.abspath(__file__))
workspacePath = this_file_path[: this_file_path.rfind(os.path.sep)] + os.path.sep

ieeg_data_dir = "F:/Postdoc_Calgary/Research/Persyst_Project/EEG_Clips/"
ieeg_fn = "Patient_1_anonym-clip.lay"

ieeg_filepath = ieeg_data_dir + ieeg_fn

images_path = workspacePath + os.path.sep + "Images" + os.path.sep
os.makedirs(images_path, exist_ok=True)

evaluator = PersystSpikesEvaluator(ieeg_filepath=ieeg_filepath, images_path=images_path)
#mtg_ieeg = evaluator.read_seeg_bip_data()
# mtg_scalp_eeg = evaluator.read_scalp_eeg_data()
manual_marks = evaluator.parse_manual_eoi(manual_eoi_key="@Spike")
auto_marks = evaluator.parse_auto_eoi(auto_eoi_key="NotAvailable")

manual_marks = evaluator.correct_eoi_center_point(mtg_ieeg, manual_marks)
auto_marks = evaluator.correct_eoi_center_point(mtg_ieeg, auto_marks)


# Define which iEEG time-range to use for the calculation of the agreement between detectors. Events outside of this range will be ignored.
manual_eoi_min_time = 10
manual_eoi_max_time = evaluator.get_manual_eoi_max_time(manual_marks)
manual_eoi_max_time = 600

agreement_params = {
    "manual_eoi": auto_marks.center,
    "auto_eoi": auto_marks.center,
    "eoi_duration": 0.4,
    "min_time": manual_eoi_min_time,
    "max_time": manual_eoi_max_time,
    "bin_duration": 0.2,
}
perf_metrics = evaluator.measure_eoi_types_agreement(**agreement_params)

print("/nPerformance metrics:")
for k, v in perf_metrics.items():
    print(k, v)

pass
