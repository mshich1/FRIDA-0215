"""
See Jupyter notebooks for example usage
"""
import numpy as np
import pandas as pd
import itertools
import math
import sys

from collections import Counter
from tqdm import tqdm
from utils import convert_dataframe

from scipy.stats import pearsonr, kendalltau, spearmanr


DATAFRAME_ERROR = "Data input must be a pandas DataFrame"
ANNOTATORS_ERROR = "Invalid choice of annotators.\n Possible options: "
KRIPP_DATA_TYPE_ERROR = """Invalid 'data_type' input.\n Possible options are
(nominal, ordinal, interval, ratio)"""


def main_input_checks(df, labels):
    if not isinstance(df, pd.DataFrame):
        raise TypeError(DATAFRAME_ERROR)


class Metrics():
    def __init__(self, df):
        converted_data = convert_dataframe(df)
        self.df = converted_data[0]
        self.labels = converted_data[1]
        self.data_dict = converted_data[2]
        main_input_checks(self.df, self.labels)

    def joint_probability(self, ann1, ann2):
        """
        The joint probability of agreement between two annotators.
        The most basic (and least useful) statistic to measure pairwise
        annotator agreement for non-continuous labelling.

        Parameters
        ----------
        ann1: string
            Name of one of the annotators
        ann2: string
            Name of another annotator

        Returns
        -------
        Probability of the two annotators agreeing across all instances
        """
        all_anns = self.df.columns
        if (ann1 not in all_anns or ann2 not in all_anns):
            raise ValueError(ANNOTATORS_ERROR + str(list(all_anns)))

        df = self.df.dropna(subset=[ann1, ann2])
        ann1_labels = df[ann1].values.tolist()
        ann2_labels = df[ann2].values.tolist()
        zipped = zip(ann1_labels, ann2_labels)
        agree = [1 if label[0] == label[1] else 0 for label in zipped]

        return sum(agree) / len(agree)

    def cohens_kappa(self, ann1, ann2):
        """
        A statistic to measure pairwise annotator agreement for non-continuous
        labelling.

        Parameters
        ----------
        ann1: string
            Name of one of the annotators
        ann2: string
            Name of another annotator

        Returns
        -------
        Cohen's kappa statistic between the two annotators
        """
        all_anns = self.df.columns
        if (ann1 not in all_anns or ann2 not in all_anns):
            raise ValueError(ANNOTATORS_ERROR + str(list(all_anns)))

        df = self.df.dropna(subset=[ann1, ann2])
        ann1_labels = df[ann1].values.tolist()
        ann2_labels = df[ann2].values.tolist()
        num_instances = df.shape[0]
        num_categories = len(self.labels)

        ann1_num, ann2_num = [], []
        for label in self.labels:
            ann1_counter = Counter(ann1_labels)
            ann2_counter = Counter(ann2_labels)
            ann1_num.append(ann1_counter[label])
            ann2_num.append(ann2_counter[label])

        assert len(ann1_num) == len(self.labels)
        assert len(ann1_num) == len(self.labels)

        summation = 0
        for i in range(len(ann1_num)):
            summation += (ann1_num[i] * ann2_num[i])

        chance_agreement_prob = (1 / num_instances ** 2) * summation
        observed_agreement_prob = self.joint_probability(ann1, ann2)

        if chance_agreement_prob == 1:
            return 1.

        numerator = observed_agreement_prob - chance_agreement_prob
        denominator = 1. - chance_agreement_prob

        return numerator / denominator

    def df2table(self, df):
        # fleiss_kappa() helper function
        # Convert df(rows=instances, cols=annotators)
        # to df(rows=instances, cols=labels)
        n = len(self.labels)

        df_rows = []
        for idx, row in df.iterrows():
            labels = [0] * n
            for label in row:
                if not math.isnan(label):
                    labels[int(label)] += 1
            df_rows.append(labels)

        return pd.DataFrame(df_rows, columns=self.labels)

    def proportion_label_per_category(self, df):
        # fleiss_kappa() helper function
        # Formula for calculating the proportion of all annotator
        # labels to the j-th category (list of all j)
        num_assignments = list(df.sum(axis=0))
        normaliser = 1. / sum(num_assignments)

        return [normaliser * i for i in num_assignments]

    def rater_agreement_extent(self, df):
        # fleiss_kappa() helper function
        # Formula for calculating the extent to which annotators
        # agree on instance j (list of all j)
        # Returns 1 for full agreement
        total_labels = list(df.sum(axis=1))
        df2 = df ** 2
        total_labels_squared = list(df2.sum(axis=1))
        v1 = np.array(total_labels_squared)
        v2 = np.array(total_labels)
        summations = list(v1 - v2)

        final = []
        for i in range(len(total_labels)):
            try:
                normalise = 1. / (total_labels[i] * (total_labels[i] - 1.))
            except ZeroDivisionError:
                normalise = 0
            final.append(normalise * summations[i])

        return final

    def fleiss_kappa(self):
        """
        A statistic to measure agreement between any number of annotators
        for non-continuous labelling.

        Parameters
        ----------
        None

        Returns
        -------
        Fleiss' kappa statistic for all the annotators
        """
        labels_per_instance = []
        for i, row in self.df.iterrows():
            num_nans = sum(math.isnan(k) for k in row)
            labels_per_instance.append(len(row) - num_nans)

        num_instances = self.df.shape[0]
        fleiss_df = self.df2table(self.df)
        prop_labels_per_cat = self.proportion_label_per_category(fleiss_df)
        rater_agreement_extent = self.rater_agreement_extent(fleiss_df)

        mean_P = (1 / num_instances) * sum(rater_agreement_extent)
        mean_p = sum([i ** 2 for i in prop_labels_per_cat])

        if mean_p == 1:
            return 1.

        return (mean_P - mean_p) / (1 - mean_p)

    def correlation(self, ann1, ann2, measure="pearson"):
        """
        Computes the correlation coefficient as a statistic for
        the agreement between two annotators. This
        method uses the scipy.stats module.

        Only appropriate for datasets larger than 500 or so (see scipy
        documentation).

        Parameters
        ----------
        ann1: string
            Name of one of the annotators
        ann2: string
            Name of another annotator
        measure: string, ("kendall", "pearson", "spearman")
            Pearson r, or Kendall tau, or Spearman rho statistics
            Pearson: assumes continuously labelled data
            Kendall/Spearman: assumes ordinal data

        Returns
        -------
        Tuple, (correlation, p-value)
        """
        P = "pearson"
        S = "spearman"
        K = "kendall"
        if not (measure == P or measure == S or measure == K):
            raise ValueError("Input measure '" + str(measure) + "' is invalid.\n Possible options: (pearson, kendall, spearman)")

        all_anns = self.df.columns
        if (ann1 not in all_anns or ann2 not in all_anns):
            raise ValueError(ANNOTATORS_ERROR + str(list(all_anns)))

        ann1_labels = self.df[ann1].values.tolist()
        ann2_labels = self.df[ann2].values.tolist()

        ann1_, ann2_ = [], []
        for i, label in enumerate(ann1_labels):
            ann2_label = ann2_labels[i]
            if (not math.isnan(label) and not math.isnan(ann2_label)):
                ann1_.append(label)
                ann2_.append(ann2_label)

        if (len(ann1_) == 0 and len(ann2_) == 0):
            raise ValueError("Annotators " + str(ann1) + " and " + str(ann2) + " have not labelled any of the same instances.")

        if measure == "pearson":
            result = pearsonr(ann1_, ann2_)
            return (abs(result[0]), result[1])
        elif measure == "kendall":
            result = kendalltau(ann1_, ann2_)
            return (abs(result[0]), result[1])
        elif measure == "spearman":
            result = spearmanr(ann1_, ann2_)
            return (abs(result[0]), result[1])

def remove_nans(l):
    return [int(i) for i in l if not math.isnan(i)]
def coincidence_mat(df, labels):
    coincidence_matrix = np.zeros((len(labels), len(labels)))
    for row_num, labels in df.iterrows():
        labels = remove_nans(list(labels))
        num_annotations = len(labels)
        perms = itertools.permutations(labels, 2)
        for perm in perms:
            i, j = perm[0], perm[1]
            coincidence_matrix[i][j] += 1 / (num_annotations - 1)

    return coincidence_matrix


class Krippendorff():
    """
    Class for computing Krippendorff's alpha statistic between annotations
    agreements.

    Parameters
    ----------
    df: pandas DataFrame
        rows are data instances, columns are annotator labels

    Initialised
    -----------
    num_anns: float
        number of annotators in the data
    num_instances: float
        number of instances of labelled data
    A: numpy array
        matrix version of the dataframe transposed
    labels_per_instance: list
        list of len(num_instances)
        Each element is the number of times that instance was labelled
    coincidence_matrix: numpy array
        matrix computed in coincidence_mat()
    coincidence_matrix_sum: 1D numpy array
        sum of rows/columns in coincidence_matrix
    """
    def __init__(self, df, use_tqdm=False):
        df_original = df
        self.df, self.labels, self.data_dict = convert_dataframe(df)
        self.num_instances, self.num_anns = self.df.shape
        self.A = self.df.values.transpose()
        self.use_tqdm = use_tqdm
        self.labels_per_instance = []
        for i, row in self.df.iterrows():
            self.labels_per_instance.append(len(row) - sum(math.isnan(k) for k in row))

        self.coincidence_matrix = coincidence_mat(self.df, self.labels)
        self.coincidence_matrix_sum = np.sum(self.coincidence_matrix, axis=0)

    def delta_nominal(self, v1, v2):
        if v1 == v2:
            return 0
        else:
            return 1

    def delta_ordinal(self, v1, v2):
        v1, v2 = int(v1), int(v2)

        val = 0
        for g in range(v1, v2 + 1):
            element1 = self.coincidence_matrix_sum[g]
            val += element1

        element2 = (self.coincidence_matrix_sum[v1] + self.coincidence_matrix_sum[v2]) / 2.
        val = val - element2

        return val ** 2

    def delta_interval(self, v1, v2):
        v1, v2 = float(v1), float(v2)
        return (v1 - v2) ** 2

    def delta_ratio(self, v1, v2):
        v1, v2 = float(v1), float(v2)
        return ((v1 - v2) / (v1 + v2)) ** 2

    def disagreement(self, obs_or_exp, data_type):
        n = self.coincidence_matrix_sum

        result = 0
        for v1 in range(1, len(self.labels)):
            for v2 in range(v1):
                if data_type == "nominal":
                    delta = self.delta_nominal(str(v1), str(v2))
                elif data_type == "ordinal":
                    delta = self.delta_ordinal(str(v1), str(v2))
                elif data_type == "interval":
                    delta = self.delta_interval(str(v1), str(v2))
                elif data_type == "ratio":
                    delta = self.delta_ratio(str(v1), str(v2))

                if obs_or_exp == "observed":
                    result += (self.coincidence_matrix[v1][v2] * delta)
                else:
                    result += (n[v1] * n[v2] * delta)
        return result

    def alpha(self, data_type="nominal"):
        """
        Attribute used to produce Krippendorff's alpha

        Parameters
        ----------
        data_type: str, ("nominal", "ordinal", "interval", "ratio")

        Returns
        -------
        Krippendorff's alpha: float
        """
        if not (data_type == "nominal" or data_type == "ordinal" or data_type == "interval" or data_type == "ratio"):
            raise ValueError(KRIPP_DATA_TYPE_ERROR)

        observed_disagreement = self.disagreement(obs_or_exp="observed",
                                                  data_type=data_type)
        expected_disagreement = self.disagreement(obs_or_exp="expected",
                                                  data_type=data_type)

        if expected_disagreement == 0:
            return 1.

        n_total = sum(self.coincidence_matrix_sum)

        return 1. - (n_total - 1.) * (observed_disagreement / expected_disagreement)
