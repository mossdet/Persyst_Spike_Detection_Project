import os
import mne
from persyst_spikes_evaluator import PersystSpikesEvaluator
from montage_creator import MontageCreator

this_file_path = os.path.dirname(os.path.abspath(__file__))
workspacePath = this_file_path[: this_file_path.rfind(os.path.sep)] + os.path.sep

ieeg_data_dir = "F:/Postdoc_Calgary/Research/Persyst_Project/EEG_Clips/"
ieeg_fn = "Patient_1_anonym-clip.lay"

ieeg_filepath = ieeg_data_dir + ieeg_fn

ieeg_data = mne.io.read_raw_persyst(fname=ieeg_filepath, verbose=False)
mntg_creator = MontageCreator(ieeg_data)
ref_mtg_labels = mntg_creator.get_intracranial_ref_mtg_labels()

evaluator = PersystSpikesEvaluator(ieeg_filepath=ieeg_filepath, images_path=None)
manual_marks = evaluator.parse_manual_eoi(manual_eoi_key="@Spike")

elpi_annots_fn = "Patient_1_anonym-clip_JJ_Spikes.mat"
elpi_annots_fn = ieeg_data_dir + elpi_annots_fn

evaluator.generate_elpi_annotations(annotations=manual_marks, elpi_file_destination=elpi_annots_fn)

pass