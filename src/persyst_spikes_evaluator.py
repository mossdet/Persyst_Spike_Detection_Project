import os
import re
import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score


class PersystSpikesEvaluator():
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
        self.ieeg_data = mne.io.read_raw_persyst(
            self.ieeg_filepath, verbose=False)
        self.time = self.ieeg_data.times
        self.nr_samples = self.ieeg_data.n_times
        self.fs = self.ieeg_data.info["sfreq"]

    def parse_eoi(self, eoi_key: str = 'Spike', visual_eoi_key: str = 'elpi'):
        """
        Parse the events in the IEEG data to separate the manually and automatically detected EOIs.

        Parameters
        ----------
        eoi_key : str, optional
            Keywords used to identify the EOIs, by default 'Spike'
        visual_eoi_key : str, optional
            Keywords used to identify the visual EOIs, by default 'elpi'
        """
        self.manual_eoi = []
        self.auto_eoi = []
        for annotation in self.ieeg_data.annotations:
            annot_label = annotation['description']
            annot_start_time = annotation['onset']
            print(annot_label)
            print(annot_start_time)

            if eoi_key.lower() in annot_label.lower():
                if visual_eoi_key in annot_label:
                    self.manual_eoi.append(annot_start_time)
                else:
                    self.auto_eoi.append(annot_start_time)

        self.manual_eoi = np.array(self.manual_eoi)
        self.auto_eoi = np.array(self.auto_eoi)
        print(f"Number of manually detected EOI: {len(self.manual_eoi)}")
        print(f"Number of automatically detected EOI: {len(self.auto_eoi)}")

    def get_manual_eoi_max_time(self):
        return np.max(self.manual_eoi)+1

    def measure_eoi_types_agreement(self, eoi_duration: float = 0.400, min_time: float = 0, max_time: float = float('inf'), bin_duration: float = 0.1, min_bin_overlap: float = 0.5):
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
        min_bin_overlap : float, optional
            Minimum overlap between adjacent time bins to consider them valid, by default 0.5%
        """

        if max_time == float('inf'):
            max_time = self.time[-1]

        # Select events within min_time and max_time
        eval_manual_eoi = self.manual_eoi[np.logical_and(
            self.manual_eoi >= min_time, self.manual_eoi <= max_time)]
        eval_auto_eoi = self.auto_eoi[np.logical_and(
            self.auto_eoi >= min_time, self.auto_eoi <= max_time)]

        self.time_bin_array, self.manual_eoi_mask = self.__fill_events_mask(eval_manual_eoi, eoi_duration,
                                                                            min_time, max_time, bin_duration, min_bin_overlap)

        self.time_bin_array, self.auto_eoi_mask = self.__fill_events_mask(eval_auto_eoi, eoi_duration,
                                                                          min_time, max_time, bin_duration, min_bin_overlap)

    def __fill_events_mask(self, eoi_array: np.ndarray = None, eoi_duration: float = 0.400, min_time: float = 0, max_time: float = float('inf'), bin_duration: float = 0.1, min_bin_overlap: float = 0.5):
        """
        This function fills a boolean mask indicating the presence of EOIs in a time series.

        Parameters
        ----------
        eoi_array : np.ndarray, optional
            Array of EOI times, by default None
        eoi_duration : float, optional
            Duration of the events, by default 400 ms
        min_time : float, optional
            First time point to consider events for the agreement calculation, by default 0 s
        max_time : float, optional
            Last time point to consider events for the agreement calculation, by default float('inf'), i.e. full length of iEEG
        bin_duration : float, optional
            Width of the time bins to use for the agreement calculation, by default 100 ms
        min_bin_overlap : float, optional
            Minimum overlap between adjacent time bins to consider them valid, by default 0.5%

        Returns
        -------
        eoi_mask : np.ndarray
            Boolean mask indicating the presence of EOIs in the time series
        """
        eoi_half_dur = eoi_duration/2
        min_overlap_dur = bin_duration*min_bin_overlap

        time_bin_array = np.arange(min_time, max_time, bin_duration)
        nr_bins = len(time_bin_array)
        eoi_mask = np.full(nr_bins, False, dtype=bool)

        # Fill the manual events mask
        for tbidx, tbin in enumerate(time_bin_array):
            bin_start = tbin
            bin_end = bin_start + bin_duration

            for eoi_idx, eoi in enumerate(eoi_array):
                eoi_start = eoi - eoi_half_dur
                eoi_end = eoi + eoi_half_dur

                # EOI extends beyond bin limits
                overlap_a = (eoi_start <= bin_start) and (eoi_end >= bin_end)

                # EOI start is within the bin and overlaps with the bin by at least min_bin_overlap%
                overlap_b = (eoi_start >= bin_start) and (eoi_start <= bin_end) and (
                    (bin_end-eoi_start) >= min_overlap_dur)

                # EOI end is within the bin and overlaps with the bin by at least min_bin_overlap%
                overlap_c = (eoi_end >= bin_start) and (eoi_end <= bin_end) and (
                    (eoi_end-bin_start) >= min_overlap_dur)

                if overlap_a or overlap_b or overlap_c:
                    eoi_mask[tbidx] = True
                    break

        return time_bin_array, eoi_mask

    def get_manual_vs_auto_performance(self):
        """
        Calculates the performance metrics between manually and automatically detected EOIs.

        Returns:
            list: A list containing the accuracy, F1-score, precision, and recall values between manually and automatically detected EOIs.
        """
        eval_acc_val = accuracy_score(self.manual_eoi_mask, self.auto_eoi_mask)
        eval_f1_val = f1_score(self.manual_eoi_mask, self.auto_eoi_mask)
        eval_prec_val = precision_score(
            self.manual_eoi_mask, self.auto_eoi_mask)
        eval_recall_val = recall_score(
            self.manual_eoi_mask, self.auto_eoi_mask)

        (tn, fp, fn, tp) = confusion_matrix(
            self.manual_eoi_mask, self.auto_eoi_mask).ravel()
        precision_val = float(tp)/float(tp+fp)
        recall_val = float(tp)/float(tp+fn)
        specificity_val = float(tn)/float(tn+fp)
        accuracy_val = float(tp+tn)/float(tn+fp+fn+tp)
        f1_val = (2*tp)/(2*tp+fp+fn)
        mcc_val = float((tp*tn)-(fp*fn)) / \
            np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))

        return [eval_acc_val, eval_f1_val, eval_prec_val, eval_recall_val]

    def plot_manual_and_auto_masks(self, images_path):

        perf_metrics = self.get_manual_vs_auto_performance()
        eval_acc_val = perf_metrics[0]
        eval_f1_val = perf_metrics[1]
        eval_prec_val = perf_metrics[2]
        eval_recall_val = perf_metrics[3]

        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(20, 10))
        ax[0].plot(self.time_bin_array, self.manual_eoi_mask)
        ax[0].set_xlabel("Time (s)")
        ax[0].set_title("Manual EOI Mask")

        ax[1].plot(self.time_bin_array, self.auto_eoi_mask)
        ax[1].set_xlabel("Time (s)")
        ax[1].set_title("Automatically Detected EOI Mask")

        fig.suptitle(
            f"Agreement between manually and automatically detected EOI\n \
            Accuracy:{eval_acc_val:.2f}, \
            F1-Score:{eval_f1_val:.2f}, \
            Precision:{eval_prec_val:.2f}, \
            Recall:{eval_recall_val:.2f},")

        image_name = images_path+"Manual_vs_Auto_Detections_Mask_Compare.jpeg"
        plt.savefig(image_name)
        # plt.show(block=True)
        # plt.close()
