import networkx as nx
import pandas as pd
import numpy as np
import math
from sklearn.impute import KNNImputer

def stratified_disparity(valid_nodes, acc_list, group_labels, G, num_bin, bins_function, diff_function, log=True, data_log=True):
    '''
    Compute Stratified Disparity (SD) by partitioning nodes into N structural bins
    and measuring performance disparity within each bin.

    Parameters
    ----------
    valid_nodes : list[int]
        Indices of nodes included in the evaluation.

    acc_list : list[float]
        Prediction performance (e.g., accuracy, precision, recall, AUC score)
        associated with each valid node.

    group_labels : dictionary
        Protected attribute associated with each node.

    G : networkx.Graph
        Input graph.

    num_bin : int
        Number of bins used for stratification.

    bins_function : callable
        Function used to compute the structural attribute for stratification
        (e.g., degree, clustering coefficient, PageRank).

    diff_function : callable
        Function used to compute the performance disparity
        (e.g., standard deviation).

    log : bool, optional (default=True)
        Whether to partition nodes using logarithmic bin boundaries.
        If False, linear binning is used.

    data_log : bool, optional (default=True)
        Whether to print intermediate statistics and debugging information.

    Returns
    -------
    sd : float
        Overall Stratified Disparity score of StratifiedDisparity(N).
    '''
    
    # compute bin feature list for each valid node: feature_list = [ , , ...]
    feature_list = bins_function(G, valid_nodes)
    
    # sort acc and bin_feature into bins
    his, his_result, labels = bin_group_performance(valid_nodes, acc_list, feature_list, group_labels, num_bin, log, data_log)
    
    # compute prob
    prob = compute_bins_probability(his, num_bin, data_log)

    # compute disparity
    esti_Disparity = compute_estimate_disparity_general(his_result, num_bin, prob, diff_function, data_log)

    return esti_Disparity


class StratifiedDisparity:
    """
    Object-oriented interface for repeated Stratified Disparity evaluation.

    This class caches structural attributes for fixed graph/valid_nodes,
    so repeated evaluations do not recompute the stratification variable.
    """

    def __init__(
        self,
        G,
        valid_nodes,
        bins_function,
        diff_function,
        log=True,
        data_log=False,
    ):
        self.G = G
        self.valid_nodes = list(valid_nodes)
        self.bins_function = bins_function
        self.diff_function = diff_function
        self.log = log
        self.data_log = data_log

        self.feature_list = None
        self._is_fitted = False
        self.stabilized_point = None

    def find_stabilized_point(self):
        """
        Compute the stabilized point k*.
    
        k* is the maximum meaningful number of bins before
        additional partitions no longer increase structural resolution.
    
        Returns
        -------
        int
            Stabilized point k*.
        """
        
        values = np.asarray(self.feature_list, dtype=float)
    
        values = np.unique(np.sort(values))
    
        if len(values) <= 1:
            self.stabilized_point = 1
            return 0
    
        diffs = np.diff(values)
    
        nonzero_diffs = diffs[diffs > 0]
    
        if len(nonzero_diffs) == 0:
            self.stabilized_point = 1
            return 0
    
        delta = np.min(nonzero_diffs)
    
        k_star = int(
            np.ceil(
                (values.max() - values.min()) / delta
            )
        )
        
        self.stabilized_point = max(1, k_star)
        return 0

    def fit(self):
        """
        Precompute structural attributes for valid nodes.
        """
        self.feature_list = self.bins_function(self.G, self.valid_nodes)
        self._is_fitted = True
        self.find_stabilized_point()
        return self

    def score(self, acc_list, group_labels, num_bin):
        """
        Compute SD for one given number of bins.
        """
        if not self._is_fitted:
            self.fit()

        if len(acc_list) != len(self.valid_nodes):
            raise ValueError(
                "acc_list must have the same length as valid_nodes."
            )

        return stratified_disparity(
            valid_nodes=self.valid_nodes,
            acc_list=acc_list,
            group_labels=group_labels,
            G=self.G,
            num_bin=num_bin,
            bins_function=lambda G, valid_nodes: self.feature_list, # directly fetch from calculated result
            diff_function=self.diff_function,
            log=self.log,
            data_log=self.data_log,
        )

    def compute_curve(self, acc_list, group_labels, bin_list=None, max_bins=None):
        """
        Compute SD curve over multiple bin numbers.
        """
        if bin_list is None:
            if not self._is_fitted:
                self.fit()
            if max_bins is None:
                max_bins = self.stabilized_point

            upper = min(max_bins, self.stabilized_point)
            bin_list = (
                list(range(1, min(50, upper + 1)))
                + list(range(50, min(200, upper + 1), 5))
                + list(range(200, upper + 1, 20))
            )
            if upper not in bin_list:
                bin_list.append(upper)

        group_labels = self._resolve_group_labels(group_labels)

        records = []

        for num_bin in bin_list:
            sd = self.score(acc_list, group_labels, num_bin)
            records.append({
                "num_bins": num_bin,
                "stratified_disparity": sd,
            })

        return pd.DataFrame(records)

    def find_elbow(self, curve):
        """
        Find elbow point using maximum distance to the line connecting
        the first and last points.
        """
        x = curve["num_bins"].to_numpy(dtype=float)
        y = curve["stratified_disparity"].to_numpy(dtype=float)

        if len(x) < 3:
            raise ValueError("At least three points are required to find an elbow.")

        p1 = np.array([x[0], y[0]])
        p2 = np.array([x[-1], y[-1]])

        line = p2 - p1
        line_norm = np.linalg.norm(line)

        if line_norm == 0:
            elbow_idx = 0
        else:
            distances = []
            for xi, yi in zip(x, y):
                p = np.array([xi, yi])
                dist = np.abs(np.cross(line, p1 - p)) / line_norm
                distances.append(dist)

            elbow_idx = int(np.argmax(distances))

        return curve.iloc[elbow_idx].to_dict()

    def plot_curve(self, curve, elbow=None, ax=None):
        """
        Plot SD curve.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(
            curve["num_bins"],
            curve["stratified_disparity"],
            marker="o",
        )

        ax.set_xlabel("Number of bins")
        ax.set_ylabel("Stratified Disparity")
        ax.set_title("Stratified Disparity Curve")

        if elbow is not None:
            ax.scatter(
                elbow["num_bins"],
                elbow["stratified_disparity"],
                s=80,
                zorder=3,
            )
            ax.annotate(
                f"elbow={int(elbow['num_bins'])}",
                xy=(elbow["num_bins"], elbow["stratified_disparity"]),
                xytext=(5, 5),
                textcoords="offset points",
            )

        return ax
        
    def _resolve_group_labels(self, group_labels):
        if isinstance(group_labels, str):
            group_labels = nx.get_node_attributes(self.G, group_labels)
    
        if group_labels is None:
            raise ValueError("group_labels must be provided.")
    
        missing = [v for v in self.valid_nodes if v not in group_labels]
        if missing:
            raise ValueError(f"Missing group labels for {len(missing)} valid nodes.")
    
        return group_labels
        
    def permutation_baseline(self, acc_list, group_labels, n_perm=20, bin_list=None, max_bins=None, random_state=None, return_permutations=False):
        rng = np.random.default_rng(random_state)
        group_labels = self._resolve_group_labels(group_labels)

        if bin_list is None:
            if not self._is_fitted:
                self.fit()
            if max_bins is None:
                max_bins = self.stabilized_point

            upper = min(max_bins, self.stabilized_point)
            bin_list = (
                list(range(1, min(50, upper + 1)))
                + list(range(50, min(200, upper + 1), 5))
                + list(range(200, upper + 1, 20))
            )
            if upper not in bin_list:
                bin_list.append(upper)

        perm_curves = []

        labels = [group_labels[v] for v in self.G]

        for _ in range(n_perm):
            shuffled_values = labels.copy()
            rng.shuffle(shuffled_values)

            shuffled_labels = dict(zip(list(self.G), shuffled_values))

            curve = self.compute_curve(
                acc_list=acc_list,
                group_labels=shuffled_labels,
                bin_list=bin_list,
            )

            perm_curves.append(curve)

        baseline = average_curves(perm_curves)

        if return_permutations:
            return baseline, perm_curves
    
        return baseline
        
    def plot_bin_group_performance(self, acc_list, group_labels, num_bin, ax=None):
        import matplotlib.pyplot as plt

        his, his_result, labels = self.bin_group_performance(acc_list, group_labels, num_bin)

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))

        num_community = len(his_result.keys())
        # set bar width
        bar_width = 1/(num_community+1)
          
        # set x-asix position
        r = [[] for i in range(num_community)]
        r[0] = np.arange(len(his_result[0]))
        
        for i in range(1,num_community):
            r[i] = [x + i*bar_width for x in r[0]]
        # create bar
        for i, _label in enumerate(his_result.keys()):
            ax.bar(r[i], his_result[i], width=bar_width, edgecolor='grey', label='Group '+str(_label))
        # add plot information
        ax.set_ylim((0, 1))
        ax.set_xticks(r[int(num_community/2)])
        ax.set_xticklabels(labels, rotation=45) # put label in middle
        ax.set_ylabel("Average performance", fontsize=12, fontweight='bold')
        ax.set_xlabel("Structure bin", fontsize=12, fontweight='bold')
        ax.legend()
        ax.figure.tight_layout()
    
        return ax
        
    def bin_group_performance(self, acc_list, group_labels, num_bin):
        if not self._is_fitted:
            self.fit()

        if len(acc_list) != len(self.valid_nodes):
            raise ValueError(
                "acc_list must have the same length as valid_nodes."
            )

        group_labels = self._resolve_group_labels(group_labels)
    
        his, his_result, labels = bin_group_performance(self.valid_nodes, acc_list, self.feature_list, group_labels, num_bin, self.log, self.data_log)
    
        return his, his_result, labels

def compute_step_log_general(feature_dic, num_bin, log=True):
    feature_list = []
    for att in feature_dic.keys():
        feature_list += feature_dic[att]
    max_feature = max(feature_list)
    min_feature = min(feature_list)

    max_log = math.log(max_feature,10)
    min_log = math.log(min_feature,10)

    plot_step = (max_log - min_log+0.00001)/num_bin
    if log: print(plot_step)
    return plot_step, max_log, min_log

def compute_step_general(feature_dic, num_bin, log=True):
    feature_list = []
    for att in feature_dic.keys():
        feature_list += feature_dic[att]
    max_feature = max(feature_list)
    min_feature = min(feature_list)

    plot_step = (max_feature - min_feature+0.00001)/num_bin
    if log: print(plot_step)
    return plot_step, max_feature, min_feature

def compute_avg_acc_bins_log(acc_dic, num_bin, plot_step, degree_dic, min_log, log=True):
    his = {}
    his_result = {}
    for k in list(acc_dic.keys()): # attribute 0,1,2
        his[k] = [[] for i in range(num_bin)]
        his_result[k] = []
        # 按照社区内前百分之10， 前百分之20划分
        #max_degree = max(degree_dic[k])
        #min_degree = min(degree_dic[k])
        for i in range(len(degree_dic[k])):
            # 按照社区内前百分之10， 前百分之20划分
            #his[k][int(degree_dic[k][i]/math.ceil((max_degree-min_degree)/num_bin))].append(acc_dic[k][i])
            # 按照真实degree 划分
            his[k][int((math.log(degree_dic[k][i],10)-min_log)/plot_step)].append(acc_dic[k][i])
        for l in his[k]: 
            if len(l)!= 0:
                his_result[k].append(sum(l)/len(l))
            else:
                his_result[k].append(0)
    return his, his_result

def compute_avg_acc_bins(acc_dic, num_bin, plot_step, degree_dic, min_log, log=True):
    his = {}
    his_result = {}
    for k in list(acc_dic.keys()): # attribute 0,1,2
        his[k] = [[] for i in range(num_bin)]
        his_result[k] = []
        # 按照社区内前百分之10， 前百分之20划分
        #max_degree = max(degree_dic[k])
        #min_degree = min(degree_dic[k])
        for i in range(len(degree_dic[k])):
            # 按照社区内前百分之10， 前百分之20划分
            #his[k][int(degree_dic[k][i]/math.ceil((max_degree-min_degree)/num_bin))].append(acc_dic[k][i])
            # 按照真实degree 划分
            his[k][int((degree_dic[k][i]-min_log)/plot_step)].append(acc_dic[k][i])
        for l in his[k]: 
            if len(l)!= 0:
                his_result[k].append(sum(l)/len(l))
            else:
                his_result[k].append(0)
    return his, his_result

def compute_bins_probability(his, num_bin, log=True):
    prob = [0 for i in range(num_bin)]
    for i in range(num_bin):
        for k in list(his.keys()):
            assert num_bin == len(his[k])
            prob[i] += len(his[k][i])
    prob = [i/sum(prob) for i in prob]
    if log: print(prob)
    return prob

def compute_disparity_general(his_result, num_bin, prob, diff_function, log=True):
    # compute new disparity
    Disparity = 0
    for i in range(num_bin):
        # For multi-communities use Variance
        _acc = [his_result[k][i] for k in range(len(his_result.keys()))]
        Disparity += prob[i] * diff_function(_acc)
    if log: print("Disparity:", Disparity)
    return Disparity

def compute_estimate_disparity_general(his_result, num_bin, prob, diff_function, log=True):
    # use KNN Imputer to estimate, it should KNN by each group, not each bin, it makes more sense
    df = pd.DataFrame([[ his_result[k][i] for k in range(len(his_result.keys()))] for i in range(num_bin)])
    df = df.replace(0,None)
    imputer = KNNImputer(n_neighbors=2)
    his_result_re = imputer.fit_transform(df)
    
    esti_Disparity = 0
    for i in range(num_bin):
        _acc = his_result_re[i]
        esti_Disparity += prob[i] * diff_function(_acc)
    if log: print("estimate Disparity:", esti_Disparity)
    return esti_Disparity
    
def average_curves(curves):
    if len(curves) == 0:
        raise ValueError("curves must contain at least one DataFrame.")

    base_bins = curves[0]["num_bins"].to_list()

    for curve in curves[1:]:
        if curve["num_bins"].to_list() != base_bins:
            raise ValueError("All curves must have the same num_bins values.")

    values = np.vstack([
        curve["stratified_disparity"].to_numpy()
        for curve in curves
    ])

    return pd.DataFrame({
        "num_bins": base_bins,
        "stratified_disparity": values.mean(axis=0),
    })
    
def bin_group_performance(valid_nodes, acc_list, feature_list, group_labels, num_bin, log, data_log):
    # sort acc and bin_feature into dictionary
    acc_dic = {}
    feature_dic = {}
    for i, n in enumerate(valid_nodes):
        if group_labels[n] not in acc_dic.keys():
            acc_dic[group_labels[n]] = [acc_list[i]]
            feature_dic[group_labels[n]] = [feature_list[i]]
        else:
            acc_dic[group_labels[n]].append(acc_list[i])
            feature_dic[group_labels[n]].append(feature_list[i])
        
    # compute plot step
    if log:
        plot_step, max_feature, min_feature = compute_step_log_general(feature_dic, num_bin, data_log)
        his, his_result = compute_avg_acc_bins_log(acc_dic, num_bin, plot_step, feature_dic, min_feature, data_log)
        label_list_cell = [math.ceil(10**( i * plot_step + min_feature)) for i in range(num_bin+1)]
    else:
        plot_step, max_feature, min_feature = compute_step_general(feature_dic, num_bin, data_log)
        his, his_result = compute_avg_acc_bins(acc_dic, num_bin, plot_step, feature_dic, min_feature, data_log)
        label_list_cell = [math.ceil(i * plot_step + min_feature) for i in range(num_bin+1)]
    
    labels = [f"{label_list_cell[i]}-{label_list_cell[i+1]-1}" for i in range(len(label_list_cell)-1)]
    return his, his_result, labels