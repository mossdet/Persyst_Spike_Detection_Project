import mne
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy.typing as npt
import datetime

from scipy.io import savemat
from collections import namedtuple, defaultdict
from montage_creator import MontageCreator
from typing import List, Dict, Union, NamedTuple

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    cohen_kappa_score,
)


class PersystSpikesEvaluator:
    """
    This class provides methods to evaluate Spike annotations generated with Persyst.
    """

    def __init__(self, ieeg_filepath: str = None, images_path: str = None):
        """
        Initialize the PersystSpikesEvaluator class.

        Parameters
        ----------
        ieeg_filepath : str
            Path to the Persyst IEEG file.
        """
        self.ieeg_filepath = ieeg_filepath
        self.images_path = images_path
        self.ieeg_data = None
        self.fs = None
        self.time = None
        self.nr_samples = None
        self.time_bin_array = None

        """
        Read the header data from the Persyst EEG file.
        """
        self.ieeg_data = mne.io.read_raw_persyst(self.ieeg_filepath, verbose=False)
        self.time = self.ieeg_data.times
        self.nr_samples = self.ieeg_data.n_times
        self.fs = self.ieeg_data.info["sfreq"]

    def read_scalp_eeg_data(self):
        """
        Read the actual EEG from the file and generate the scalp montages.
        """
        mntg_creator = MontageCreator(self.ieeg_data)
        eeg_mtg_data = mntg_creator.scalp_longitudinal_bipolar()

        return eeg_mtg_data

    def read_intracranial_referential_montages(self):
        """
        Read the actual EEG from the file and generate the intracranial montages.
        """
        mntg_creator = MontageCreator(self.ieeg_data)
        bip_mtg_labels = mntg_creator.get_intracranial_referential_montage_labels()

        return bip_mtg_labels[['montageName', 'montageChNr']]
    
    def read_intracranial_bipolar_montages(self):
        """
        Read the actual EEG from the file and generate the intracranial montages.
        """
        mntg_creator = MontageCreator(self.ieeg_data)
        bip_mtg_labels = mntg_creator.get_intracranial_bipolar_montage_labels()

        return bip_mtg_labels[['montageName', 'montageChNr']]
    def read_seeg_bip_data(self, start=0, stop=None, plot_ok=False):
        """
        Read the actual EEG from the file and generate the intracranial montages.
        """
        ieeg_mtg_data = MontageCreator(self.ieeg_data).intracranial_bipolar(start=start, stop=stop, plot_ok=False)

        return ieeg_mtg_data

    def parse_manual_eoi(self, manual_eoi_key: str = "elpi"):
        """
        Parse the visually marked Spikes in the EEG data  and check if their channel label is correct.

        Parameters
        ----------
        manual_eoi_key : str, optional
            Keywords used to identify the manual EOIs, by default 'elpi'
        """

        ref_ch_labels = [ch.lower() for ch in self.ieeg_data.ch_names]
        annotations = pd.DataFrame(self.ieeg_data.annotations)

        manual_eoi = namedtuple("EOI", ["center", "channel", "type"])

        manual_eoi.center = []
        manual_eoi.channel = []
        manual_eoi.type = []
        incorrect_ch_labels = []

        for index, row in annotations.iterrows():
            eoi_onset = row["onset"]
            description = row["description"]
            if manual_eoi_key in description:

                eoi_type = description.split()[0]
                if manual_eoi_key == eoi_type: 

                    eoi_channel = description[description.find("c=")+2:]
                    chann_group = re.match("[a-zA-Z]+", eoi_channel)[0]

                    if eoi_channel.lower() in ref_ch_labels:
                        manual_eoi.center.append(eoi_onset)
                        manual_eoi.channel.append(eoi_channel)
                        manual_eoi.type.append(eoi_type)
                    else:
                        incorrect_ch_labels.append(eoi_channel)

        manual_eoi.center = np.array(manual_eoi.center, dtype=float)
        print(f"Number of manually detected EOI: {len(manual_eoi.center)}")
        print(f"Number of incorrect channel labels: {len(incorrect_ch_labels)}")
        print(np.unique(incorrect_ch_labels))

        return manual_eoi

    def parse_auto_eoi( self, auto_eoi_key: str = "Spike"):
        """
        Parse the automatically detected Spikes in the EEG data.

        Parameters
        ----------
        auto_eoi_key : str, optional
            Keywords used to identify the EOIs, by default 'Spike'
        """
        annotations = pd.DataFrame(self.ieeg_data.annotations)

        auto_eoi = namedtuple("EOI", ["center", "channel", "type"])

        auto_eoi.center = []
        auto_eoi.channel = []
        auto_eoi.type = []

        for index, row in annotations.iterrows():
            eoi_onset = row["onset"]
            description = row["description"]
            if auto_eoi_key in description:
                eoi_type = description.split()[0]

                eoi_channel = description.split()[1]
                channel_1 = eoi_channel.split("-")[0]
                channel_2 = eoi_channel.split("-")[1]
                chann_group = re.match("[a-zA-Z]+", channel_1)[0]
                channel_2 = chann_group + channel_2
                eoi_channel = channel_1 + "-" + channel_2

                if auto_eoi_key in eoi_type:
                    auto_eoi.center.append(eoi_onset)
                    auto_eoi.channel.append(eoi_channel)
                    auto_eoi.type.append(eoi_type)
                pass

        auto_eoi.center = np.array(auto_eoi.center, dtype=float)

        print(f"Number of automatically detected EOI: {len(auto_eoi.center)}")

        return auto_eoi

    def correct_eoi_center_point(
        self,
        eeg_data: Dict = None,
        eoi_struct: NamedTuple = None,
    ):
        fs = eeg_data["fs"]
        eeg_signals = eeg_data["data"]

        uncorrected_onsets = eoi_struct.center.copy()

        for i in range(eoi_struct.center.shape[0]):

            # Get EOI channel index
            eoi_channel = eoi_struct.channel[i]

            eoi_ch_idx = np.where(eeg_data["mtg_labels"] == eoi_channel)[0]
            assert (
                len(eoi_ch_idx) == 1
            ), "iEEG contains repeated channels in montage channel list"
            eoi_ch_idx = eoi_ch_idx[0]

            # Correct EOI center point
            center_point = eoi_struct.center[i]
            onset_sample = int(np.round((center_point - 0.1) * fs))
            offset_sample = int(np.round((center_point + 0.1) * fs))

            if onset_sample < 0:
                onset_sample = 0
            if offset_sample > eeg_data["n_samples"]:
                offset_sample = eeg_data["n_samples"]

            eoi_signal = eeg_signals[eoi_ch_idx, onset_sample:offset_sample]
            eoi_signal = eoi_signal - np.mean(eoi_signal)

            max_peak_idx = np.argmax(eoi_signal)
            min_peak_idx = np.argmin(eoi_signal)
            max_peak_prominence = np.abs(eoi_signal[max_peak_idx] - np.mean(eoi_signal))
            min_peak_prominence = np.abs(eoi_signal[min_peak_idx] - np.mean(eoi_signal))
            if max_peak_prominence > min_peak_prominence:
                eoi_struct.center[i] = (onset_sample + max_peak_idx) / fs
            else:
                eoi_struct.center[i] = (onset_sample + min_peak_idx) / fs

        corrected_onsets = eoi_struct.center

        # plt.plot(uncorrected_onsets, label="Uncorrected Spike Centers")
        # plt.plot(corrected_onsets, label="Corrected Spike Centers")
        # plt.xlabel("EOI #")
        # plt.ylabel("Center (s)")
        # plt.legend()
        # plt.title("Manual EOI")
        # plt.show()
        # plt.close()

        return eoi_struct

    def get_manual_eoi_max_time(
        self,
        manual_eoi: NamedTuple = None,
    ):
        return np.max(manual_eoi.center) + 1 if len(manual_eoi.center) > 0 else 0

    def measure_eoi_types_agreement(
        self,
        manual_eoi: np.ndarray,
        auto_eoi: np.ndarray,
        eoi_duration: float = 0.400,
        min_time: float = 0,
        max_time: float = float("inf"),
        bin_duration: float = 0.1,
    ):
        """
        Calculate the agreement between the manually and automatically detected EOI types.

        Parameters
        ----------
        eoi_duration : float, optional
            Duration of the events, by default 400 ms
        min_time : float, optional
            First time point to consider events for the agreement calculation, by default 0 s
        max_time : float, optional
            Last time point to consider events for the agreement calculation, by default float('inf'), i.e. full length of iEEG
        bin_duration : float, optional
            Width of the time bins to use for the agreement calculation, by default 100 ms
        """

        if max_time == float("inf"):
            max_time = self.time[-1]

        # Select events within min_time and max_time
        eval_manual_eoi = manual_eoi[
            (manual_eoi >= min_time) & (manual_eoi <= max_time)
        ]
        eval_auto_eoi = auto_eoi[(auto_eoi >= min_time) & (auto_eoi <= max_time)]

        self.time_bin_array, manual_eoi_mask = self.__fill_events_mask(
            eval_manual_eoi,
            eoi_duration,
            min_time,
            max_time,
            bin_duration,
        )

        self.time_bin_array, auto_eoi_mask = self.__fill_events_mask(
            eval_auto_eoi,
            eoi_duration,
            min_time,
            max_time,
            bin_duration,
        )

        perf_metrics = self.get_manual_vs_auto_performance(
            manual_eoi_mask, auto_eoi_mask
        )
        self.plot_manual_and_auto_masks(manual_eoi_mask, auto_eoi_mask)

        return perf_metrics

    def __fill_events_mask(
        self,
        eoi_array: np.ndarray = None,
        eoi_duration: float = 0.400,
        min_time: float = 0,
        max_time: float = float("inf"),
        bin_duration: float = 0.1,
    ):
        """
        Fills the events mask.

        Parameters
        ----------
        eoi_array : np.ndarray, optional
            Array of event onsets, by default None
        eoi_duration : float, optional
            Duration of the events, by default 0.400
        min_time : float, optional
            Minimum time to consider events, by default 0
        max_time : float, optional
            Maximum time to consider events, by default float("inf")
        bin_duration : float, optional
            Duration of each bin, by default 0.1

        Returns
        -------
        np.ndarray, np.ndarray
            Time bin array and event mask
        """

        eoi_half_dur = eoi_duration / 2

        time_bin_array = np.arange(min_time, max_time, bin_duration)
        nr_bins = len(time_bin_array)
        eoi_mask = np.zeros(nr_bins, dtype=bool)

        eoi_start = eoi_array - eoi_half_dur
        eoi_end = eoi_array + eoi_half_dur

        for start_t, end_t in zip(eoi_start, eoi_end):
            # Adjust for cases where start or end are outside the min_time and max_time range
            eoi_start_adj = np.maximum(start_t, min_time) - min_time
            eoi_end_adj = np.minimum(end_t, max_time) - min_time

            # The rounding guarantees that a bin must overlap at least 50% with EOI to be marked as True
            overlap_start = np.round((eoi_start_adj) / bin_duration).astype(int)
            overlap_end = np.round((eoi_end_adj) / bin_duration).astype(int) - 1

            eoi_mask[overlap_start : overlap_end + 1] = True

        return time_bin_array, eoi_mask

    def get_manual_vs_auto_performance(
        self,
        manual_eoi_mask: np.ndarray,
        auto_eoi_mask: np.ndarray,
    ):
        """
        Calculates the performance metrics between manually and automatically detected EOIs.

        Returns
        -------
        dict
            A dictionary containing the accuracy, F1-score, precision, recall, MCC, and specificity values between manually and automatically detected EOIs.
        """

        nr_eeg_mins = self.time[-1] / 60

        (tn, fp, fn, tp) = confusion_matrix(manual_eoi_mask, auto_eoi_mask).ravel()

        accuracy_val = float(tp + tn) / float(tn + fp + fn + tp)
        f1_val = (2 * tp) / (2 * tp + fp + fn)
        mcc_val = float((tp * tn) - (fp * fn)) / np.sqrt(
            (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        )
        specificity_val = float(tn) / float(tn + fp)
        precision_val = float(tp) / float(tp + fp)
        recall_val = float(tp) / float(tp + fn)

        accuracy_val = accuracy_score(manual_eoi_mask, auto_eoi_mask)
        f1_val = f1_score(manual_eoi_mask, auto_eoi_mask)
        precision_val = precision_score(manual_eoi_mask, auto_eoi_mask)
        recall_val = recall_score(manual_eoi_mask, auto_eoi_mask)
        kappa_val = cohen_kappa_score(manual_eoi_mask, auto_eoi_mask)
        fp_per_m = fp / nr_eeg_mins

        assert accuracy_val == accuracy_score(
            manual_eoi_mask, auto_eoi_mask
        ), "Wrong accuracy score"
        assert f1_val == f1_score(manual_eoi_mask, auto_eoi_mask), "Wrong f1 score"
        assert precision_val == precision_score(
            manual_eoi_mask, auto_eoi_mask
        ), "Wrong precision score"
        assert recall_val == recall_score(
            manual_eoi_mask, auto_eoi_mask
        ), "Wrong recall score"

        performance_metrics = {
            "Accuracy": accuracy_val,
            "F1_Score": f1_val,
            "MathewsCorrCoeff": mcc_val,
            "Specificity": specificity_val,
            "Precision": precision_val,
            "Recall": recall_val,
            "Kappa": kappa_val,
            "FalsePositivesPerMin": fp_per_m,
        }

        return performance_metrics

    def plot_manual_and_auto_masks(
        self,
        manual_eoi_mask: np.ndarray,
        auto_eoi_mask: np.ndarray,
    ):
        """
        Plots the manual and automatically detected event masks.

        Parameters
        ----------
        """
        perf_metrics = self.get_manual_vs_auto_performance(
            manual_eoi_mask, auto_eoi_mask
        )

        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(20, 10))
        ax[0].plot(self.time_bin_array, manual_eoi_mask)
        ax[0].set_xlabel("Time (s)")
        ax[0].set_title("Manual EOI Mask")

        ax[1].plot(self.time_bin_array, auto_eoi_mask)
        ax[1].set_xlabel("Time (s)")
        ax[1].set_title("Automatically Detected EOI Mask")

        fig.suptitle(
            f"Agreement between manually and automatically detected EOI\n \
            Accuracy: {perf_metrics['Accuracy']:.2f}, \
            Precision: {perf_metrics['Precision']:.2f}, \
            Recall: {perf_metrics['Recall']:.2f}, \
            Kappa: {perf_metrics['Kappa']:.2f}, \
            FalsePositives/m: {perf_metrics['FalsePositivesPerMin']:.2f}"
        )

        image_name = self.images_path + "Manual_vs_Auto_Detections_Mask_Compare.jpeg"
        plt.savefig(image_name)
        plt.close()

    def generate_elpi_annotations(self, annotations: NamedTuple = None, elpi_file_destination: str = None):

        # Create a dictionary with the EOI information so that it can be read in Elpi
        creation_time = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        pass

        elpi_eoi_dict = defaultdict(list)

        nr_eoi = len(annotations.type)
        elpi_eoi_dict["Channel"] = annotations.channel
        elpi_eoi_dict["Type"] = annotations.type
        elpi_eoi_dict["StartSec"] = annotations.center-0.1
        elpi_eoi_dict["EndSec"] = annotations.center+0.1
        elpi_eoi_dict["StartSample"] = [int(val*self.fs) for val in elpi_eoi_dict["StartSec"]]
        elpi_eoi_dict["EndSample"] = [int(val*self.fs) for val in elpi_eoi_dict["EndSec"]]
        elpi_eoi_dict["Comments"] = [self.ieeg_filepath] * nr_eoi
        elpi_eoi_dict["ChSpec"] = np.ones(nr_eoi, dtype=bool)
        elpi_eoi_dict["CreationTime"] = [creation_time] * nr_eoi
        elpi_eoi_dict["User"] = ["JJ"] * nr_eoi

        print([len(value) for key, value in elpi_eoi_dict.items()])

        savemat(elpi_file_destination, elpi_eoi_dict)

        pass


        # merged_eoi_dict = {
        #     "Channel": [elpi_ch_name] * nr_hfo,
        #     "Type": ["Pruned_HFO"] * nr_hfo,
        #     "StartSec": start_samples.astype(np.float64) / fs,
        #     "EndSec": end_samples.astype(np.float64) / fs,
        #     "StartSample": start_samples.astype(np.int64),
        #     "EndSample": end_samples.astype(np.int64),
        #     "Comments": [eoi_file] * nr_hfo,
        #     "ChSpec": np.ones(nr_hfo, dtype=bool),
        #     "CreationTime": [creation_time] * nr_hfo,
        #     "User": ["DLP_Prune_HFO"] * nr_hfo,
        # }

