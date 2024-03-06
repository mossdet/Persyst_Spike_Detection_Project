import os
from persyst_spikes_evaluator import PersystSpikesEvaluator

this_file_path = os.path.dirname(os.path.abspath(__file__))
workspacePath = this_file_path[:this_file_path.rfind(os.path.sep)]+os.path.sep

ieeg_data_dir = "C:/Users/HFO/Documents/Persyst_Project/iEEG_Archives/"
ieeg_fn = "Test.lay"
ieeg_filepath = ieeg_data_dir + ieeg_fn

images_path = workspacePath+os.path.sep+"Images"+os.path.sep
os.makedirs(images_path, exist_ok=True)

evaluator = PersystSpikesEvaluator(ieeg_filepath=ieeg_filepath)
evaluator.read_ieeg_data()
evaluator.parse_eoi(eoi_key='Spike', visual_eoi_key='elpi')


# Define which iEEG time-range to use for the calculation of the agreement between detectors. Events outside of this range will be ignored.
manual_eoi_min_time = 10
manual_eoi_max_time = evaluator.get_manual_eoi_max_time()
manual_eoi_max_time = 600

agreement_params = {'eoi_duration': 0.4,
                    'min_time': manual_eoi_min_time,
                    'max_time': manual_eoi_max_time,
                    'bin_duration': 0.1,
                    'min_bin_overlap': 0.5}
evaluator.measure_eoi_types_agreement(**agreement_params)
perf_metrics = evaluator.get_manual_vs_auto_performance()
evaluator.plot_manual_and_auto_masks(images_path=images_path)

print("\nPerformance metrics:")
for k, v in perf_metrics.items():
    print(k, v)

pass
