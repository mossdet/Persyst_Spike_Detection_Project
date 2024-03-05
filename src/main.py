from persyst_spikes_evaluator import PersystSpikesEvaluator

ieeg_data_dir = "C:/Users/HFO/Documents/Persyst_Project/iEEG_Archives/"
ieeg_fn = "Test_File.lay"
ieeg_filepath = ieeg_data_dir + ieeg_fn

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
                    'bin_duration': 0.001,
                    'min_bin_overlap': 0.5}
evaluator.measure_eoi_types_agreement(**agreement_params)
perf_metrics = evaluator.get_manual_vs_auto_performance()
evaluator.plot_manual_and_auto_masks()

print("Accuracy", perf_metrics[0])
print("F1-Score", perf_metrics[1])
print("Precision", perf_metrics[2])
print("Recall", perf_metrics[3])

pass
