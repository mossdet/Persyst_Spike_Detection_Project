import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

    def __init__(self, ieeg_filepath: str):
        """
        Initialize the PersystSpikesEvaluator class.

        Parameters
        ----------
        ieeg_filepath : str
            Path to the Persyst IEEG file.
        """
        self.ieeg_filepath = ieeg_filepath
        self.ieeg_data = None
        self.fs = None
        self.time = None
        self.nr_samples = None
        self.manual_eoi = []
        self.auto_eoi = []
        self.time_bin_array = None
        self.manual_eoi_mask = None
        self.auto_eoi_mask = None

    def read_ieeg_data(self):
        """
        Read the Persyst IEEG data from the file.
        """
        self.ieeg_data = mne.io.read_raw_persyst(self.ieeg_filepath, verbose=False)
        self.time = self.ieeg_data.times
        self.nr_samples = self.ieeg_data.n_times
        self.fs = self.ieeg_data.info["sfreq"]

    def parse_eoi(self, eoi_key: str = "Spike", visual_eoi_key: str = "elpi"):
        """
        Parse the events in the IEEG data to separate the manually and automatically detected EOIs.

        Parameters
        ----------
        eoi_key : str, optional
            Keywords used to identify the EOIs, by default 'Spike'
        visual_eoi_key : str, optional
            Keywords used to identify the visual EOIs, by default 'elpi'
        """
        annotations = pd.DataFrame(self.ieeg_data.annotations)
        eoi_sel_maks = annotations["description"].str.contains(eoi_key, case=False)
        manual_sel_mask = eoi_sel_maks & annotations["description"].str.contains(
            visual_eoi_key, case=False
        )
        auto_sel_mask = eoi_sel_maks & ~annotations["description"].str.contains(
            visual_eoi_key, case=False
        )
        self.manual_eoi = annotations.loc[manual_sel_mask, "onset"].to_numpy()
        self.auto_eoi = annotations.loc[auto_sel_mask, "onset"].to_numpy()
        print(f"Number of manually detected EOI: {len(self.manual_eoi)}")
        print(f"Number of automatically detected EOI: {len(self.auto_eoi)}")

    def get_manual_eoi_max_time(self):
        return np.max(self.manual_eoi) + 1 if len(self.manual_eoi) > 0 else 0

    def measure_eoi_types_agreement(
        self,
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
        eval_manual_eoi = self.manual_eoi[
            (self.manual_eoi >= min_time) & (self.manual_eoi <= max_time)
        ]
        eval_auto_eoi = self.auto_eoi[
            (self.auto_eoi >= min_time) & (self.auto_eoi <= max_time)
        ]

        self.time_bin_array, self.manual_eoi_mask = self.__fill_events_mask(
            eval_manual_eoi,
            eoi_duration,
            min_time,
            max_time,
            bin_duration,
        )

        self.time_bin_array, self.auto_eoi_mask = self.__fill_events_mask(
            eval_auto_eoi,
            eoi_duration,
            min_time,
            max_time,
            bin_duration,
        )

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

    def get_manual_vs_auto_performance(self):
        """
        Calculates the performance metrics between manually and automatically detected EOIs.

        Returns
        -------
        dict
            A dictionary containing the accuracy, F1-score, precision, recall, MCC, and specificity values between manually and automatically detected EOIs.
        """

        nr_eeg_mins = self.time[-1] / 60

        (tn, fp, fn, tp) = confusion_matrix(
            self.manual_eoi_mask, self.auto_eoi_mask
        ).ravel()

        accuracy_val = float(tp + tn) / float(tn + fp + fn + tp)
        f1_val = (2 * tp) / (2 * tp + fp + fn)
        mcc_val = float((tp * tn) - (fp * fn)) / np.sqrt(
            (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        )
        specificity_val = float(tn) / float(tn + fp)
        precision_val = float(tp) / float(tp + fp)
        recall_val = float(tp) / float(tp + fn)

        accuracy_val = accuracy_score(self.manual_eoi_mask, self.auto_eoi_mask)
        f1_val = f1_score(self.manual_eoi_mask, self.auto_eoi_mask)
        precision_val = precision_score(self.manual_eoi_mask, self.auto_eoi_mask)
        recall_val = recall_score(self.manual_eoi_mask, self.auto_eoi_mask)
        kappa_val = cohen_kappa_score(self.manual_eoi_mask, self.auto_eoi_mask)
        fp_per_m = fp / nr_eeg_mins

        assert accuracy_val == accuracy_score(
            self.manual_eoi_mask, self.auto_eoi_mask
        ), "Wrong accuracy score"
        assert f1_val == f1_score(
            self.manual_eoi_mask, self.auto_eoi_mask
        ), "Wrong f1 score"
        assert precision_val == precision_score(
            self.manual_eoi_mask, self.auto_eoi_mask
        ), "Wrong precision score"
        assert recall_val == recall_score(
            self.manual_eoi_mask, self.auto_eoi_mask
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

    def plot_manual_and_auto_masks(self, images_path):
        """
        Plots the manual and automatically detected event masks.

        Parameters
        ----------
        images_path : str
            Path to save the plotted images.
        """
        perf_metrics = self.get_manual_vs_auto_performance()

        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(20, 10))
        ax[0].plot(self.time_bin_array, self.manual_eoi_mask)
        ax[0].set_xlabel("Time (s)")
        ax[0].set_title("Manual EOI Mask")

        ax[1].plot(self.time_bin_array, self.auto_eoi_mask)
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

        image_name = images_path + "Manual_vs_Auto_Detections_Mask_Compare.jpeg"
        plt.savefig(image_name)
        plt.close()
