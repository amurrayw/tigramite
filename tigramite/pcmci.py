"""Tigramite causal discovery for time series."""

# Author: Jakob Runge <jakob@jakob-runge.com>
#
# License: GNU General Public License v3.0

from __future__ import print_function
import itertools
from collections import defaultdict
from copy import deepcopy
from scipy import stats
from tigramite.independence_tests import GPDC, CMIknn
import tigramite.data_processing as pp
from tigramite.independence_tests import ParCorr
import numpy as np
import networkx as nx
import pandas as pd

def _create_nested_dictionary(depth=0, lowest_type=dict):
    """Create a series of nested dictionaries to a maximum depth.  The first
    depth - 1 nested dictionaries are defaultdicts, the last is a normal
    dictionary.

    Parameters
    ----------
    depth : int
        Maximum depth argument.
    lowest_type: callable (optional)
        Type contained in leaves of tree.  Ex: list, dict, tuple, int, float ...
    """
    new_depth = depth - 1
    if new_depth <= 0:
        return defaultdict(lowest_type)
    return defaultdict(lambda: _create_nested_dictionary(new_depth))


def _nested_to_normal(nested_dict):
    """Transforms the nested default dictionary into a standard dictionaries

    Parameters
    ----------
    nested_dict : default dictionary of default dictionaries of ... etc.
    """
    if isinstance(nested_dict, defaultdict):
        nested_dict = {k: _nested_to_normal(v) for k, v in nested_dict.items()}
    return nested_dict


class PCMCI():
    r"""PCMCI causal discovery for time series datasets.

    PCMCI is a 2-step causal discovery method for large-scale time series
    datasets. The first step is a condition-selection followed by the MCI
    conditional independence test. The implementation is based on Algorithms 1
    and 2 in [1]_.

    PCMCI allows:

       * different conditional independence test statistics adapted to
         continuously-valued or discrete data, and different assumptions about
         linear or nonlinear dependencies
       * hyperparameter optimization
       * easy parallelization
       * handling of masked time series data
       * false discovery control and confidence interval estimation


    Notes
    -----

    .. image:: mci_schematic.*
       :width: 200pt

    The PCMCI causal discovery method is comprehensively described in [1]_,
    where also analytical and numerical results are presented. Here we briefly
    summarize the method.

    In the PCMCI framework, the dependency structure of a set of
    time series variables is represented in a *time series graph* as shown in
    the Figure. The nodes of a time series graph are defined as the variables at
    different times and a link exists if two lagged variables are *not*
    conditionally independent given the past of the whole process. Assuming
    stationarity, the links are repeated in time. The parents
    :math:`\mathcal{P}` of a variable are defined as the set of all nodes with a
    link towards it (blue and red boxes in Figure). Estimating these parents
    directly by testing for conditional independence on the whole past is
    problematic due to high-dimensionality and because conditioning on
    irrelevant variables leads to biases [1]_.

    PCMCI estimates causal links by a two-step procedure:

    1.  Condition-selection: For each variable :math:`j`, estimate a
        *superset*  of parents :math:`\tilde{\mathcal{P}}(X^j_t)` with the
        iterative PC1 algorithm , implemented as ``run_pc_stable``.

    2.  *Momentary conditional independence* (MCI)

        .. math:: X^i_{t-\tau} ~\perp~ X^j_{t} ~|~ \tilde{\mathcal{P}}(X^j_t),
                                        \tilde{\mathcal{P}}(X^i_{t-{\tau}})

    here implemented as ``run_mci``. The condition-selection step reduces the
    dimensionality and avoids conditioning on irrelevant variables.

    PCMCI can be flexibly combined with any kind of conditional independence
    test statistic adapted to the kind of data (continuous or discrete) and its
    assumed dependency structure. Currently, implemented in Tigramite are
    ParCorr as a linear test, GPACE allowing nonlinear additive dependencies,
    and CMI with different estimators making no assumptions about the
    dependencies. The classes in ``tigramite.independence_tests`` also handle
    masked data.

    The main free parameters of PCMCI (in addition to free parameters of the
    conditional independence test statistic) are the maximum time delay
    :math:`\tau_{\max}` (``tau_max``) and the significance threshold in the
    condition- selection step :math:`\alpha` (``pc_alpha``). The maximum time
    delay depends on the application and should be chosen according to the
    maximum causal time lag expected in the complex system. We recommend a
    rather large choice that includes peaks in the lagged cross-correlation
    function (or a more general measure). :math:`\alpha` should not be seen as a
    significance test level in the condition-selection step since the iterative
    hypothesis tests do not allow for a precise confidence level. :math:`\alpha`
    rather takes the role of a regularization parameter in model-selection
    techniques. The conditioning sets :math:`\tilde{\mathcal{P}}`  should
    include the true parents and at the same time be small in size to reduce the
    estimation dimension of the MCI test and improve its power. But including
    the true  parents is typically more important. If a list of values is given
    or ``pc_alpha=None``, :math:`\alpha` is optimized using model selection
    criteria.

    Further optional parameters are discussed in [1]_.

    References
    ----------
    .. [1] J. Runge et al. (2018): Detecting Causal Associations in Large 
           Nonlinear Time Series Datasets.
           https://arxiv.org/abs/1702.07007v2

    Examples
    --------
    >>> import numpy
    >>> from tigramite.pcmci import PCMCI
    >>> from tigramite.independence_tests import ParCorr
    >>> import tigramite.data_processing as pp
    >>> numpy.random.seed(42)
    >>> # Example process to play around with
    >>> # Each key refers to a variable and the incoming links are supplied as a
    >>> # list of format [((driver, lag), coeff), ...]
    >>> links_coeffs = {0: [((0, -1), 0.8)],
                        1: [((1, -1), 0.8), ((0, -1), 0.5)],
                        2: [((2, -1), 0.8), ((1, -2), -0.6)]}
    >>> data, _ = pp.var_process(links_coeffs, T=1000)
    >>> # Data must be array of shape (time, variables)
    >>> print data.shape
    (1000, 3)
    >>> dataframe = pp.DataFrame(data)
    >>> cond_ind_test = ParCorr()
    >>> pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test)
    >>> results = pcmci.run_pcmci(tau_max=2, pc_alpha=None)
    >>> pcmci._print_significant_links(p_matrix=results['p_matrix'],
                                         val_matrix=results['val_matrix'],
                                         alpha_level=0.05)
    ## Significant parents at alpha = 0.05:
        Variable 0 has 1 parent(s):
            (0 -1): pval = 0.00000 | val = 0.623
        Variable 1 has 2 parent(s):
            (1 -1): pval = 0.00000 | val = 0.601
            (0 -1): pval = 0.00000 | val = 0.487
        Variable 2 has 2 parent(s):
            (2 -1): pval = 0.00000 | val = 0.597
            (1 -2): pval = 0.00000 | val = -0.511

    Parameters
    ----------
    dataframe : data object
        This is the Tigramite dataframe object. It has the attributes
        dataframe.values yielding a numpy array of shape (observations T,
        variables N) and optionally a mask of  the same shape.

    cond_ind_test : conditional independence test object
        This can be ParCorr or other classes from the tigramite package or an
        external test passed as a callable. This test can be based on the class
        tigramite.independence_tests.CondIndTest. If a callable is passed, it
        must have the signature::

            class CondIndTest():
                # with attributes
                # * measure : str
                #   name of the test
                # * use_mask : bool
                #   whether the mask should be used

                # and functions
                # * run_test(X, Y, Z, tau_max) : where X,Y,Z are of the form
                #   X = [(var, -tau)]  for non-negative integers var and tau
                #   specifying the variable and time lag
                #   return (test statistic value, p-value)
                # * set_dataframe(dataframe) : set dataframe object

                # optionally also

                # * get_model_selection_criterion(j, parents) : required if
                #   pc_alpha parameter is to be optimized. Here j is the
                #   variable index and parents a list [(var, -tau), ...]
                #   return score for model selection
                # * get_confidence(X, Y, Z, tau_max) : required for
                #   return_confidence=True
                #   estimate confidence interval after run_test was called
                #   return (lower bound, upper bound)

    selected_variables : list of integers, optional (default: range(N))
        Specify to estimate parents only for selected variables. If None is
        passed, parents are estimated for all variables.

    verbosity : int, optional (default: 0)
        Verbose levels 0, 1, ...

    Attributes
    ----------
    all_parents : dictionary
        Dictionary of form {0:[(0, -1), (3, -2), ...], 1:[], ...} containing the
        conditioning-parents estimated with PC algorithm.

    val_min : dictionary
        Dictionary of form val_min[j][(i, -tau)] = float
        containing the minimum test statistic value for each link estimated in
        the PC algorithm.

    p_max : dictionary
        Dictionary of form p_max[j][(i, -tau)] = float containing the maximum
        p-value for each link estimated in the PC algorithm.

    iterations : dictionary
        Dictionary containing further information on algorithm steps.

    N : int
        Number of variables.

    T : int
        Time series sample length.

    """

    def __init__(self, dataframe,
                 cond_ind_test,
                 selected_variables=None,
                 verbosity=0):
        # Set the data for this iteration of the algorithm
        self.dataframe = dataframe
        # Set the conditional independence test to be used
        self.cond_ind_test = cond_ind_test
        self.cond_ind_test.set_dataframe(self.dataframe)
        # Set the verbosity for debugging/logging messages
        self.verbosity = verbosity
        # Set the variable names 
        self.var_names = self.dataframe.var_names

        # Store the shape of the data in the T and N variables
        self.T, self.N = self.dataframe.values.shape
        # Set the selected variables
        self.selected_variables = \
            self._set_selected_variables(selected_variables)

    def _set_selected_variables(self, selected_variables):
        """Helper function to set and check the selected variables argument

        Parameters
        ----------
        selected_variables : list or None
            List of variable ID's from the input data set

        Returns
        -------
        selected_variables : list
            Defaults to a list of all given variable IDs [0..N-1]
        """
        # Set the default selected variables if none are set
        _int_selected_variables = deepcopy(selected_variables)
        if _int_selected_variables is None:
            _int_selected_variables = range(self.N)
        # Some checks
        if _int_selected_variables is not None and \
                (np.any(np.array(_int_selected_variables) < 0) or
                 np.any(np.array(_int_selected_variables) >= self.N)):
            raise ValueError("selected_variables must be within 0..N-1")
        # Ensure there are only unique values
        _int_selected_variables = sorted(list(set(_int_selected_variables)))
        # Return the selected variables
        return _int_selected_variables

    def _set_sel_links(self, selected_links, tau_min, tau_max):
        """Helper function to set and check the selected links argument

        Parameters
        ----------
        selected_links : dict or None
            Dictionary of form {0:[(0, -1), (3, -2), ...], 1:[], ...}
            specifying whether only selected links should be tested. If None is
            passed, all links are returned
        tau_mix : int
            Minimum time delay to test
        tau_max : int
            Maximum time delay to test

        Returns
        -------
        selected_variables : list
            Defaults to a list of all given variable IDs [0..N-1]
        """
        # Copy and pass into the function
        _int_sel_links = deepcopy(selected_links)
        # Set the default selected links if none are set
        _vars = list(range(self.N))
        if _int_sel_links is None:
            _int_sel_links = {}
            # Set the default as all combinations of the selected variables
            for j in _vars:
                # If it is in selected variables, select all possible links
                if j in self.selected_variables:
                    _int_sel_links[j] = [(var, -lag) for var in _vars
                                         for lag in range(tau_min, tau_max + 1)]
                # If it is not, make it an empty list
                else:
                    _int_sel_links[j] = []
        # Otherwise, check that our selection is sane
        # Check that the selected links refer to links that are inside the
        # data range
        _key_set = set(_int_sel_links.keys())
        valid_entries = _key_set.issubset(_vars)
        valid_entries = valid_entries and \
                        set(var for parents in _int_sel_links.values()
                            for var, _ in parents).issubset(_vars)
        if not valid_entries:
            raise ValueError("Out of range variable defined in \n",
                             _int_sel_links,
                             "\nMust be in range [0, ", self.N - 1, "]")
        # Warning if keys in selected links not in selected_variables
        if self.verbosity > 0 and \
                not _key_set.issubset(set(self.selected_variables)):
            print("Warning: Link specified in selected links that is outside " + \
                  "the scope of the selected variables")
        ## Note: variables are scoped by selected_variables first, and then
        ## by selected links.  Add to docstring?
        # Return the selected links
        return _int_sel_links

    def _iter_conditions(self, parent, conds_dim, all_parents):
        """Yield next condition.

        Yields next condition from lexicographically ordered conditions.

        Parameters
        ----------
        parent : tuple
            Tuple of form (i, -tau).
        conds_dim : int
            Cardinality in current step.
        all_parents : list
            List of form [(0, -1), (3, -2), ...]

        Yields
        -------
        cond :  list
            List of form [(0, -1), (3, -2), ...] for the next condition.
        """
        all_parents_excl_current = [p for p in all_parents if p != parent]
        for cond in itertools.combinations(all_parents_excl_current, conds_dim):
            yield list(cond)

    def _sort_parents(self, parents_vals):
        """Sort current parents according to test statistic values.

        Sorting is from strongest to weakest absolute values.

        Parameters
        ---------
        parents_vals : dict
            Dictionary of form {(0, -1):float, ...} containing the minimum test
            statistic value of a link

        Returns
        -------
        parents : list
            List of form [(0, -1), (3, -2), ...] containing sorted parents.
        """
        if self.verbosity > 1:
            print("\n    Sorting parents in decreasing order with "
                  "\n    weight(i-tau->j) = min_{iterations} |I_{ij}(tau)| ")
        # Get the absoute value for all the test statistics
        abs_values = {k: np.abs(parents_vals[k]) for k in list(parents_vals)}
        return sorted(abs_values, key=abs_values.get, reverse=True)

    def _dict_to_matrix(self, val_dict, tau_max, n_vars):
        """Helper function to convert dictionary to matrix formart.

        Parameters
        ---------
        val_dict : dict
            Dictionary of form {0:{(0, -1):float, ...}, 1:{...}, ...}
        tau_max : int
            Maximum lag.

        Returns
        -------
        matrix : array of shape (N, N, tau_max+1)
            Matrix format of p-values and test statistic values.
        """
        matrix = np.ones((n_vars, n_vars, tau_max + 1))
        for j in val_dict.keys():
            for link in val_dict[j].keys():
                k, tau = link
                matrix[k, j, abs(tau)] = val_dict[j][link]
        return matrix

    def _print_link_info(self, j, index_parent, parent, num_parents):
        """Print info about the current link being tested

        Parameters
        ----------
        j : int
            Index of current node being tested
        index_parent : int
            Index of the current parent
        parent : tuple
            Standard (i, tau) tuple of parent node id and time delay
        num_parents : int
            Total number of parents
        """
        print("\n    Link (%s %d) --> %s (%d/%d):" % (
            self.var_names[parent[0]], parent[1], self.var_names[j],
            index_parent + 1, num_parents))

    def _print_cond_info(self, Z, comb_index, pval, val):
        """Print info about the condition

        Parameters
        ----------
        Z : list
            The current condition being tested
        comb_index : int
            Index of the combination yielding this condition
        pval : float
            p-value from this condition
        val : float
            value from this condition
        """
        var_name_z = ""
        for i, tau in Z:
            var_name_z += "(%s %d) " % (self.var_names[i], tau)
        print("    Combination %d: %s --> pval = %.5f / val = %.3f" %
              (comb_index, var_name_z, pval, val))

    def _print_a_pc_result(self, pval, pc_alpha, conds_dim, max_combinations):
        """
        Print the results from the current iteration of conditions.

        Parameters
        ----------
        pval : float
            pval to check signficance
        pc_alpha : float
            lower bound on what is considered significant
        conds_dim : int
            Cardinality of the current step
        max_combinations : int
            Maximum number of combinations of conditions of current cardinality
            to test.
        """
        # Start with an indent
        print_str = "    "
        # Determine the body of the text
        if pval > pc_alpha:
            print_str += "Non-significance detected."
        elif conds_dim > max_combinations:
            print_str += "Still conditions of dimension" + \
                         " %d left," % (conds_dim) + \
                         " but q_max = %d reached." % (max_combinations)
        else:
            print_str += "No conditions of dimension %d left." % (conds_dim)
        # Print the message
        print(print_str)

    def _print_converged_pc_single(self, converged, j, max_conds_dim):
        """
        Print statement about the convergence of the pc_stable_single algorithm.

        Parameters
        ----------
        convergence : bool
            true if convergence was reacjed
        j : int
            Variable index.
        max_conds_dim : int
            Maximum number of conditions to test
        """
        if converged:
            print("\nAlgorithm converged for variable %s" %
                  self.var_names[j])
        else:
            print(
                "\nAlgorithm not yet converged, but max_conds_dim = %d"
                " reached." % max_conds_dim)

    def _run_pc_stable_single(self, j,
                              selected_links=None,
                              tau_min=1,
                              tau_max=1,
                              save_iterations=False,
                              pc_alpha=0.2,
                              max_conds_dim=None,
                              max_combinations=1):
        """PC algorithm for estimating parents of single variable.

        Parameters
        ----------
        j : int
            Variable index.

        selected_links : list, optional (default: None)
            List of form [(0, -1), (3, -2), ...]
            specifying whether only selected links should be tested. If None is
            passed, all links are tested

        tau_min : int, optional (default: 1)
            Minimum time lag to test. Useful for variable selection in
            multi-step ahead predictions. Must be greater zero.

        tau_max : int, optional (default: 1)
            Maximum time lag. Must be larger or equal to tau_min.

        save_iterations : bool, optional (default: False)
            Whether to save iteration step results such as conditions used.

        pc_alpha : float or None, optional (default: 0.2)
            Significance level in algorithm. If a list is given, pc_alpha is
            optimized using model selection criteria provided in the
            cond_ind_test class as get_model_selection_criterion(). If None,
            a default list of values is used.

        max_conds_dim : int, optional (default: None)
            Maximum number of conditions to test. If None is passed, this number
            is unrestricted.

        max_combinations : int, optional (default: 1)
            Maximum number of combinations of conditions of current cardinality
            to test. Defaults to 1 for PC_1 algorithm. For original PC algorithm
            a larger number, such as 10, can be used.

        Returns
        -------
        parents : list
            List of estimated parents.

        val_min : dict
            Dictionary of form {(0, -1):float, ...} containing the minimum test
            statistic value of a link.

        p_max : dict
            Dictionary of form {(0, -1):float, ...} containing the maximum
            p-value of a link across different conditions.

        iterations : dict
            Dictionary containing further information on algorithm steps.
        """
        # Initialize the dictionaries for the p_max, val_min parents_values
        # results
        p_max = dict()
        val_min = dict()
        parents_values = dict()
        # Initialize the parents values from the selected links, copying to
        # ensure this initial argument is unchagned.
        parents = deepcopy(selected_links)
        # Define a nested defaultdict of depth 4 to save all information about
        # iterations
        iterations = _create_nested_dictionary(4)
        # Ensure tau_min is atleast 1
        tau_min = max(1, tau_min)

        # Loop over all possible condition dimentions
        max_conds_dim = self._set_max_condition_dim(max_conds_dim,
                                                    tau_min, tau_max)
        # Iteration through increasing number of conditions, i.e. from 
        # [0,max_conds_dim] inclusive
        converged = False
        for conds_dim in range(max_conds_dim + 1):
            # (Re)initialize the list of non-significant links
            nonsig_parents = list()
            # Check if the algorithm has converged
            if len(parents) - 1 < conds_dim:
                converged = True
                break
            # Print information about
            if self.verbosity > 1:
                print("\nTesting condition sets of dimension %d:" % conds_dim)

            # Iterate through all possible pairs (that have not converged yet)
            for index_parent, parent in enumerate(parents):
                # Print info about this link
                if self.verbosity > 1:
                    self._print_link_info(j, index_parent, parent, len(parents))
                # Iterate through all possible combinations
                for comb_index, Z in \
                        enumerate(self._iter_conditions(parent, conds_dim,
                                                        parents)):
                    # Break if we try too many combinations
                    if comb_index >= max_combinations:
                        break
                    # Perform independence test
                    val, pval = self.cond_ind_test.run_test(X=[parent],
                                                            Y=[(j, 0)],
                                                            Z=Z,
                                                            tau_max=tau_max)
                    # Print some information if needed
                    if self.verbosity > 1:
                        self._print_cond_info(Z, comb_index, pval, val)
                    # Keep track of maximum p-value and minimum estimated value
                    # for each pair (across any condition)
                    parents_values[parent] = \
                        min(np.abs(val), parents_values.get(parent,
                                                            float("inf")))
                    p_max[parent] = \
                        max(np.abs(pval), p_max.get(parent, -float("inf")))
                    val_min[parent] = \
                        min(np.abs(val), val_min.get(parent, float("inf")))
                    # Save the iteration if we need to
                    if save_iterations:
                        a_iter = iterations['iterations'][conds_dim][parent]
                        a_iter[comb_index]['conds'] = deepcopy(Z)
                        a_iter[comb_index]['val'] = val
                        a_iter[comb_index]['pval'] = pval
                    # Delete link later and break while-loop if non-significant
                    if pval > pc_alpha:
                        nonsig_parents.append((j, parent))
                        break

                # Print the results if needed
                if self.verbosity > 1:
                    self._print_a_pc_result(pval, pc_alpha,
                                            conds_dim, max_combinations)

            # Remove non-significant links
            for _, parent in nonsig_parents:
                del parents_values[parent]
            # Return the parents list sorted by the test metric so that the
            # updated parents list is given to the next cond_dim loop
            parents = self._sort_parents(parents_values)
            # Print information about the change in possible parents
            if self.verbosity > 1:
                print("\nUpdating parents:")
                self._print_parents_single(j, parents, parents_values, p_max)

        # Print information about if convergence was reached
        if self.verbosity > 1:
            self._print_converged_pc_single(converged, j, max_conds_dim)
        # Return the results
        return {'parents': parents,
                'val_min': val_min,
                'p_max': p_max,
                'iterations': _nested_to_normal(iterations)}

    def _print_pc_params(self, selected_links, tau_min, tau_max, pc_alpha,
                         max_conds_dim, max_combinations):
        """
        Print the setup of the current pc_stable run

        Parameters
        ----------
        selected_links : dict or None
            Dictionary of form specifying which links should be tested.
        tau_min : int, default: 1
            Minimum time lag to test.
        tau_max : int, default: 1
            Maximum time lag to test
        pc_alpha : float or list of floats
            Significance level in algorithm.
        max_conds_dim : int
            Maximum number of conditions to test.
        max_combinations : int
            Maximum number of combinations of conditions to test.
        """
        print("\n##\n## Running Tigramite PC algorithm\n##"
              "\n\nParameters:")
        if len(self.selected_variables) < self.N:
            print("selected_variables = %s" % self.selected_variables)
        if selected_links is not None:
            print("selected_links = %s" % selected_links)
        print("independence test = %s" % self.cond_ind_test.measure
              + "\ntau_min = %d" % tau_min
              + "\ntau_max = %d" % tau_max
              + "\npc_alpha = %s" % pc_alpha
              + "\nmax_conds_dim = %s" % max_conds_dim
              + "\nmax_combinations = %d" % max_combinations)
        print("\n")

    def _print_pc_sel_results(self, pc_alpha, results, j, score, optimal_alpha):
        """
        Print the results from the pc_alpha selection

        Parameters
        ----------
        pc_alpha : list
            Tested significance levels in algorithm.
        results : dict
            Results from the tested pc_alphas
        score : array of floats
            scores from each pc_alpha
        j : int
            Index of current variable.
        optimal_alpha : float
            Optimal value of pc_alpha
        """
        print("\n# Condition selection results:")
        for iscore, pc_alpha_here in enumerate(pc_alpha):
            names_parents = "[ "
            for pari in results[pc_alpha_here]['parents']:
                names_parents += "(%s %d) " % (
                    self.var_names[pari[0]], pari[1])
            names_parents += "]"
            print("    pc_alpha=%s got score %.4f with parents %s" %
                  (pc_alpha_here, score[iscore], names_parents))
        print("\n--> optimal pc_alpha for variable %s is %s" %
              (self.var_names[j], optimal_alpha))

    def _check_tau_limits(self, tau_min, tau_max):
        """
        Check the tau limits adhere to 0 <= tau_min <= tau_max

        Parameters
        ----------
        tau_min : float
            Minimum tau value.
        tau_max : float
            Maximum tau value.
        """
        if not 0 <= tau_min <= tau_max:
            raise ValueError("tau_max = %d, " % (tau_max) + \
                             "tau_min = %d, " % (tau_min) + \
                             "but 0 <= tau_min <= tau_max")

    def _set_max_condition_dim(self, max_conds_dim, tau_min, tau_max):
        """
        Set the maximum dimension of the conditions. Defaults to self.N*tau_max

        Parameters
        ----------
        max_conds_dim : int
            Input maximum condition dimension
        tau_max : int
            Maximum tau.

        Returns
        -------
        max_conds_dim : int
            Input maximum condition dimension or default
        """
        # Check if an input was given
        if max_conds_dim is None:
            max_conds_dim = self.N * (tau_max - tau_min + 1)
        # Check this is a valid
        if max_conds_dim < 0:
            raise ValueError("maximum condition dimension must be >= 0")
        return max_conds_dim

    def run_pc_stable(self,
                      selected_links=None,
                      tau_min=1,
                      tau_max=1,
                      save_iterations=False,
                      pc_alpha=0.2,
                      max_conds_dim=None,
                      max_combinations=1):
        """PC algorithm for estimating parents of all variables.

        Parents are made available as self.all_parents

        Parameters
        ----------
        selected_links : dict or None
            Dictionary of form {0:[(0, -1), (3, -2), ...], 1:[], ...}
            specifying whether only selected links should be tested. If None is
            passed, all links are tested

        tau_min : int, default: 1
            Minimum time lag to test. Useful for multi-step ahead predictions.
            Must be greater zero.

        tau_max : int, default: 1
            Maximum time lag. Must be larger or equal to tau_min.

        save_iterations : bool, default: False
            Whether to save iteration step results such as conditions used.

        pc_alpha : float or list of floats, default: [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
            Significance level in algorithm. If a list or None is passed, the
            pc_alpha level is optimized for every variable across the given
            pc_alpha values using the score computed in
            cond_ind_test.get_model_selection_criterion()

        max_conds_dim : int or None
            Maximum number of conditions to test. If None is passed, this number
            is unrestricted.

        max_combinations : int, default: 1
            Maximum number of combinations of conditions of current cardinality
            to test. Defaults to 1 for PC_1 algorithm. For original PC algorithm
            a larger number, such as 10, can be used.

        Returns
        -------
        all_parents : dict
            Dictionary of form {0:[(0, -1), (3, -2), ...], 1:[], ...}
            containing estimated parents.
        """
        # Create an internal copy of pc_alpha
        _int_pc_alpha = deepcopy(pc_alpha)
        # Check if we are selecting an optimal alpha value
        select_optimal_alpha = True
        # Set the default values for pc_alpha
        if _int_pc_alpha is None:
            _int_pc_alpha = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        elif not isinstance(_int_pc_alpha, (list, tuple, np.ndarray)):
            _int_pc_alpha = [_int_pc_alpha]
            select_optimal_alpha = False
        # Check the limits on tau_min
        self._check_tau_limits(tau_min, tau_max)
        tau_min = max(1, tau_min)
        # Check that the maximum combinatiosn variable is correct
        if max_combinations <= 0:
            raise ValueError("max_combinations must be > 0")
        # Implement defaultdict for all p_max, val_max, and iterations
        p_max = defaultdict(dict)
        val_min = defaultdict(dict)
        iterations = defaultdict(dict)
        # Print information about the selected parameters
        if self.verbosity > 0:
            print("\n##\n## Running Tigramite PC algorithm\n##"
                  "\n\nParameters:")
            if len(self.selected_variables) < self.N:
                print("selected_variables = %s" % self.selected_variables)
            if selected_links is not None:
                print("selected_links = %s" % selected_links)
            print("independence test = %s" % self.cond_ind_test.measure
                  + "\ntau_min = %d" % tau_min
                  + "\ntau_max = %d" % tau_max
                  + "\npc_alpha = %s" % pc_alpha
                  + "\nmax_conds_dim = %s" % max_conds_dim
                  + "\nmax_combinations = %d" % max_combinations)
            print("\n")

        if selected_links is None:
            selected_links = {}
            for j in range(self.N):
                if j in self.selected_variables:
                    selected_links[j] = [(var, -lag)
                                         for var in range(self.N)
                                         for lag in range(tau_min, tau_max + 1)
                                         ]
                else:
                    selected_links[j] = []

        all_parents = selected_links

        if max_conds_dim is None:
            max_conds_dim = self.N * tau_max

        if max_conds_dim < 0:
            raise ValueError("max_conds_dim must be >= 0")

            self._print_pc_params(selected_links, tau_min, tau_max,
                                  _int_pc_alpha, max_conds_dim,
                                  max_combinations)
        # Set the selected links
        _int_sel_links = self._set_sel_links(selected_links, tau_min, tau_max)
        # Initialize all parents
        all_parents = dict()
        # Set the maximum condition dimension
        max_conds_dim = self._set_max_condition_dim(max_conds_dim,
                                                    tau_min, tau_max)

        # Loop through the selected variables
        for j in self.selected_variables:
            # Print the status of this variable
            if self.verbosity > 0:
                print("\n## Variable %s" % self.var_names[j])
                if self.verbosity > 1:
                    print("\nIterating through pc_alpha = %s:" % _int_pc_alpha)
            # Initialize the scores for selecting the optimal alpha
            score = np.zeros_like(_int_pc_alpha)
            # Initialize the result
            results = {}
            for iscore, pc_alpha_here in enumerate(_int_pc_alpha):
                # Print statement about the pc_alpha being tested
                if self.verbosity > 1:
                    print("\n# pc_alpha = %s (%d/%d):" % (pc_alpha_here,
                                                          iscore + 1,
                                                          score.shape[0]))
                # Get the results for this alpha value
                results[pc_alpha_here] = \
                    self._run_pc_stable_single(j,
                                               selected_links=_int_sel_links[j],
                                               tau_min=tau_min,
                                               tau_max=tau_max,
                                               save_iterations=save_iterations,
                                               pc_alpha=pc_alpha_here,
                                               max_conds_dim=max_conds_dim,
                                               max_combinations=max_combinations)
                # Figure out the best score if there is more than one pc_alpha
                # value
                if select_optimal_alpha:
                    score[iscore] = \
                        self.cond_ind_test.get_model_selection_criterion(
                            j, results[pc_alpha_here]['parents'], tau_max)
            # Record the optimal alpha value
            optimal_alpha = _int_pc_alpha[score.argmin()]
            # Only print the selection results if there is more than one
            # pc_alpha
            if self.verbosity > 1 and select_optimal_alpha:
                self._print_pc_sel_results(_int_pc_alpha, results, j,
                                           score, optimal_alpha)
            # Record the results for this variable
            all_parents[j] = results[optimal_alpha]['parents']
            val_min[j] = results[optimal_alpha]['val_min']
            p_max[j] = results[optimal_alpha]['p_max']
            iterations[j] = results[optimal_alpha]['iterations']
            # Only save the optimal alpha if there is more than one pc_alpha
            if select_optimal_alpha:
                iterations[j]['optimal_pc_alpha'] = optimal_alpha
        # Save the results in the current status of the algorithm
        self.all_parents = all_parents
        self.val_matrix = self._dict_to_matrix(val_min, tau_max, self.N)
        self.p_matrix = self._dict_to_matrix(p_max, tau_max, self.N)
        self.iterations = iterations
        self.val_min = val_min
        self.p_max = p_max
        # Print the results
        if self.verbosity > 0:
            print("\n## Resulting condition sets:")
            self._print_parents(all_parents, val_min, p_max)
        # Return the parents
        return all_parents

    def _print_parents_single(self, j, parents, val_min, p_max):
        """Print current parents for variable j.

        Parameters
        ----------
        j : int
            Index of current variable.
        parents : list
            List of form [(0, -1), (3, -2), ...]
        val_min : dict
            Dictionary of form {(0, -1):float, ...} containing the minimum test
            statistic value of a link
        p_max : dict
            Dictionary of form {(0, -1):float, ...} containing the maximum
            p-value of a link across different conditions.
        """
        if len(parents) < 20 or hasattr(self, 'iterations'):
            print("\n    Variable %s has %d parent(s):" % (
                self.var_names[j], len(parents)))
            if (hasattr(self, 'iterations')
                    and 'optimal_pc_alpha' in list(self.iterations[j])):
                print("    [pc_alpha = %s]" % (
                    self.iterations[j]['optimal_pc_alpha']))
            for p in parents:
                print("        (%s %d): max_pval = %.5f, min_val = %.3f" % (
                    self.var_names[p[0]], p[1], p_max[p],
                    val_min[p]))
        else:
            print("\n    Variable %s has %d parent(s):" % (
                self.var_names[j], len(parents)))

    def _print_parents(self, all_parents, val_min, p_max):
        """Print current parents.

        Parameters
        ----------
        all_parents : dictionary
            Dictionary of form {0:[(0, -1), (3, -2), ...], 1:[], ...} containing
            the conditioning-parents estimated with PC algorithm.
        val_min : dict
            Dictionary of form {0:{(0, -1):float, ...}} containing the minimum
            test statistic value of a link
        p_max : dict
            Dictionary of form {0:{(0, -1):float, ...}} containing the maximum
            p-value of a link across different conditions.
        """
        for j in [var for var in list(all_parents)]:
            self._print_parents_single(j, all_parents[j],
                                       val_min[j], p_max[j])

    def _mci_condition_to_string(self, conds):
        """Convert the list of conditions into a string

        Parameters
        ----------
        conds : list
            List of conditions
        """
        cond_string = "[ "
        for k, tau_k in conds:
            cond_string += "(%s %d) " % (self.var_names[k], tau_k)
        cond_string += "]"
        return cond_string

    def _print_mci_conditions(self, conds_y, conds_x_lagged,
                              j, i, tau, count, n_parents):
        """Print information about the conditions for the MCI algorithm

        Parameters
        ----------
        conds_y : list
            Conditions on node
        conds_x_lagged : list
            Conditions on parent
        j : int
            Current node
        i : int
            Parent node
        tau : int
            Parent time delay
        count : int
            Index of current parent
        n_parents : int
            Total number of parents
        """
        # Remove the current parent from the conditions
        conds_y_no_i = [node for node in conds_y if node != (i, tau)]
        # Get the condition string for parent
        condy_str = self._mci_condition_to_string(conds_y_no_i)
        # Get the condition string for node
        condx_str = self._mci_condition_to_string(conds_x_lagged)
        # Formate and print the information
        indent = "\n        "
        print_str = indent + "link (%s %d) " % (self.var_names[i], tau)
        print_str += "--> %s (%d/%d):" % (self.var_names[j], count + 1, n_parents)
        print_str += indent + "with conds_y = %s" % (condy_str)
        print_str += indent + "with conds_x = %s" % (condx_str)
        print(print_str)

    def _get_int_parents(self, parents):
        """Get the input parents dictionary

        Parameters
        ----------
        parents : dict or None
            Dictionary of form {0:[(0, -1), (3, -2), ...], 1:[], ...}
            specifying the conditions for each variable. If None is
            passed, no conditions are used.

        Returns
        -------
        int_parents : defaultdict of lists
            Internal copy of parents, respecting default options
        """
        int_parents = deepcopy(parents)
        if int_parents is None:
            int_parents = defaultdict(list)
        else:
            int_parents = defaultdict(list, int_parents)
        return int_parents

    def _iter_indep_conds(self,
                          parents,
                          selected_variables,
                          selected_links,
                          max_conds_py,
                          max_conds_px):
        """Iterate through the conditions dictated by the arguments, yielding
        the needed arguments for conditional independence functions.

        Parameters
        ----------
        parents : dict
            Dictionary of form {0:[(0, -1), (3, -2), ...], 1:[], ...}
            specifying the conditions for each variable.
        selected_variables : list of integers, optional (default: range(N))
            Specify to estimate parents only for selected variables.
        selected_links : dict
            Dictionary of form {0:[(0, -1), (3, -2), ...], 1:[], ...}
            specifying whether only selected links should be tested.
        max_conds_py : int
            Maximum number of conditions of Y to use.
        max_conds_px : int
            Maximum number of conditions of Z to use.

        Yields
        ------
        i, j, tau, Z : list of tuples
            (i, tau) is the parent node, (j, 0) is the current node, and Z is of
            the form [(var, tau + tau')] and specifies the condition to test
        """
        # Loop over the selected variables
        for j in selected_variables:
            # Get the conditions for node j
            conds_y = parents[j][:max_conds_py]
            # Create a parent list from links seperated in time and by node
            parent_list = [(i, tau) for i, tau in selected_links[j]
                           if (i, tau) != (j, 0)]
            # Iterate through parents (except those in conditions)
            for cnt, (i, tau) in enumerate(parent_list):
                # Get the conditions for node i
                conds_x = parents[i][:max_conds_px]
                # Shift the conditions for X by tau
                conds_x_lagged = [(k, tau + k_tau) for k, k_tau in conds_x]
                # Print information about the mci conditions if requested
                if self.verbosity > 1:
                    self._print_mci_conditions(conds_y, conds_x_lagged, j, i,
                                               tau, cnt, len(parent_list))
                # Construct lists of tuples for estimating
                # I(X_t-tau; Y_t | Z^Y_t, Z^X_t-tau)
                # with conditions for X shifted by tau
                Z = [node for node in conds_y if node != (i, tau)]
                # Remove overlapped nodes between conds_x_lagged and conds_y
                Z += [node for node in conds_x_lagged if node not in Z]
                # Yield these list
                yield j, i, tau, Z

    def get_lagged_dependencies(self,
                                selected_links=None,
                                tau_min=0,
                                tau_max=1,
                                parents=None,
                                max_conds_py=None,
                                max_conds_px=None):
        """Returns matrix of lagged dependence measure values.

        Parameters
        ----------
        selected_links : dict or None
            Dictionary of form {0:[(0, -1), (3, -2), ...], 1:[], ...}
            specifying whether only selected links should be tested. If None is
            passed, all links are tested
        tau_min : int, default: 0
            Minimum time lag.
        tau_max : int, default: 1
            Maximum time lag. Must be larger or equal to tau_min.
        parents : dict or None
            Dictionary of form {0:[(0, -1), (3, -2), ...], 1:[], ...}
            specifying the conditions for each variable. If None is
            passed, no conditions are used.
        max_conds_py : int or None
            Maximum number of conditions from parents of Y to use. If None is
            passed, this number is unrestricted.
        max_conds_px : int or None
            Maximum number of conditions from parents of X to use. If None is
            passed, this number is unrestricted.

        Returns
        -------
        val_matrix : array
            The matrix of shape (N, N, tau_max+1) containing the lagged
            dependencies.
        """
        # Check the limits on tau
        self._check_tau_limits(tau_min, tau_max)
        # Set the selected links
        _int_sel_links = self._set_sel_links(selected_links, tau_min, tau_max)
        # Print status message
        if self.verbosity > 0:
            print("\n## Estimating lagged dependencies")
        # Set the maximum condition dimension for Y and X
        max_conds_py = self._set_max_condition_dim(max_conds_py,
                                                   tau_min, tau_max)
        max_conds_px = self._set_max_condition_dim(max_conds_px,
                                                   tau_min, tau_max)
        # Get the parents that will be checked
        _int_parents = self._get_int_parents(parents)
        # Initialize the returned val_matrix
        val_matrix = np.zeros((self.N, self.N, tau_max + 1))
        # Get the conditions as implied by the input arguments
        for j, i, tau, Z in self._iter_indep_conds(_int_parents,
                                                   self.selected_variables,
                                                   _int_sel_links,
                                                   max_conds_py,
                                                   max_conds_px):
            # Set X and Y (for clarity of code)
            X = [(i, tau)]
            Y = [(j, 0)]
            # Run the independence test
            val = self.cond_ind_test.get_measure(X, Y, Z=Z, tau_max=tau_max)
            # Record the value
            val_matrix[i, j, abs(tau)] = val
            # Print the results
            if self.verbosity > 1:
                self.cond_ind_test._print_cond_ind_results(val=val)
        # Return the value matrix
        return val_matrix

    def _print_mci_parameters(self, tau_min, tau_max,
                              max_conds_py, max_conds_px):
        """Print the parameters for this MCI algorithm

        Parameters
        ----------
        tau_min : int
            Minimum time delay
        tau_max : int
            Maximum time delay
        max_conds_py : int
            Maximum number of conditions of Y to use.
        max_conds_px : int
            Maximum number of conditions of Z to use.
        """
        print("\n##\n## Running Tigramite MCI algorithm\n##"
              "\n\nParameters:")
        print("\nindependence test = %s" % self.cond_ind_test.measure
              + "\ntau_min = %d" % tau_min
              + "\ntau_max = %d" % tau_max
              + "\nmax_conds_py = %s" % max_conds_py
              + "\nmax_conds_px = %s" % max_conds_px)

    def run_mci(self,
                selected_links=None,
                tau_min=0,
                tau_max=1,
                parents=None,
                max_conds_py=None,
                max_conds_px=None):
        """MCI conditional independence tests.

        Implements the MCI test (Algorithm 2 in [1]_). Returns the matrices of
        test statistic values,  p-values, and confidence intervals.

        Parameters
        ----------
        selected_links : dict or None
            Dictionary of form {0:all_parents (3, -2), ...], 1:[], ...}
            specifying whether only selected links should be tested. If None is
            passed, all links are tested
        tau_min : int, default: 0
            Minimum time lag to test. Note that zero-lags are undirected.
        tau_max : int, default: 1
            Maximum time lag. Must be larger or equal to tau_min.
        parents : dict or None
            Dictionary of form {0:[(0, -1), (3, -2), ...], 1:[], ...}
            specifying the conditions for each variable. If None is
            passed, no conditions are used.
        max_conds_py : int or None
            Maximum number of conditions of Y to use. If None is passed, this
            number is unrestricted.
        max_conds_px : int or None
            Maximum number of conditions of Z to use. If None is passed, this
            number is unrestricted.

        Returns
        -------
        results : dictionary of arrays of shape [N, N, tau_max+1]
            {'val_matrix':val_matrix, 'p_matrix':p_matrix} are always returned
            and optionally conf_matrix which is of shape [N, N, tau_max+1,2]
        """
        # Check the limits on tau
        self._check_tau_limits(tau_min, tau_max)
        # Set the selected links
        _int_sel_links = self._set_sel_links(selected_links, tau_min, tau_max)
        # Print information about the input parameters
        if self.verbosity > 0:
            self._print_mci_parameters(tau_min, tau_max,
                                       max_conds_py, max_conds_px)

        # Set the maximum condition dimension for Y and X
        max_conds_py = self._set_max_condition_dim(max_conds_py,
                                                   tau_min, tau_max)
        max_conds_px = self._set_max_condition_dim(max_conds_px,
                                                   tau_min, tau_max)
        # Get the parents that will be checked
        _int_parents = self._get_int_parents(parents)
        # Initialize the return values
        val_matrix = np.zeros((self.N, self.N, tau_max + 1))
        p_matrix = np.ones((self.N, self.N, tau_max + 1))
        # Initialize the optional return of the confidance matrix
        conf_matrix = None
        if self.cond_ind_test.confidence is not False:
            conf_matrix = np.zeros((self.N, self.N, tau_max + 1, 2))

        # Get the conditions as implied by the input arguments
        for j, i, tau, Z in self._iter_indep_conds(_int_parents,
                                                   self.selected_variables,
                                                   _int_sel_links,
                                                   max_conds_py,
                                                   max_conds_px):
            # Set X and Y (for clarity of code)
            X = [(i, tau)]
            Y = [(j, 0)]
            # Run the independence tests and record the results
            val, pval = self.cond_ind_test.run_test(X, Y, Z=Z, tau_max=tau_max)
            val_matrix[i, j, abs(tau)] = val
            p_matrix[i, j, abs(tau)] = pval
            # Get the confidance value, returns None if cond_ind_test.confidance
            # is False
            conf = self.cond_ind_test.get_confidence(X, Y, Z=Z, tau_max=tau_max)
            # Record the value if the conditional independence requires it
            if self.cond_ind_test.confidence:
                conf_matrix[i, j, abs(tau)] = conf
            # Print the results if needed
            if self.verbosity > 1:
                self.cond_ind_test._print_cond_ind_results(val,
                                                           pval=pval,
                                                           conf=conf)
        # Return the values as a dictionary
        return {'val_matrix': val_matrix,
                'p_matrix': p_matrix,
                'conf_matrix': conf_matrix}

    def get_corrected_pvalues(self, p_matrix,
                              fdr_method='fdr_bh',
                              exclude_contemporaneous=True):
        """Returns p-values corrected for multiple testing.

        Currently implemented is Benjamini-Hochberg False Discovery Rate
        method. Correction is performed either among all links if
        exclude_contemporaneous==False, or only among lagged links.

        Parameters
        ----------
        p_matrix : array-like
            Matrix of p-values. Must be of shape (N, N, tau_max + 1).
        fdr_method : str, optional (default: 'fdr_bh')
            Correction method, currently implemented is Benjamini-Hochberg
            False Discovery Rate method.     
        exclude_contemporaneous : bool, optional (default: True)
            Whether to include contemporaneous links in correction.

        Returns
        -------
        q_matrix : array-like
            Matrix of shape (N, N, tau_max + 1) containing corrected p-values.
        """

        def _ecdf(x):
            '''no frills empirical cdf used in fdrcorrection
            '''
            nobs = len(x)
            return np.arange(1, nobs + 1) / float(nobs)

        # Get the shape paramters from the p_matrix
        _, N, tau_max_plusone = p_matrix.shape
        # Create a mask for these values
        mask = np.ones((N, N, tau_max_plusone), dtype='bool')
        # Ignore values from autocorrelation indices
        mask[range(N), range(N), 0] = False
        # Exclude all contemporaneous values if requested
        if exclude_contemporaneous:
            mask[:, :, 0] = False
        # Create the return value
        q_matrix = np.array(p_matrix)
        # Use the multiple tests function
        if fdr_method is None or fdr_method == 'none':
            pass
        elif fdr_method == 'fdr_bh':
            pvs = p_matrix[mask]
            # q_matrix[mask] = multicomp.multipletests(pvs, method=fdr_method)[1]

            pvals_sortind = np.argsort(pvs)
            pvals_sorted = np.take(pvs, pvals_sortind)

            ecdffactor = _ecdf(pvals_sorted)
            # reject = pvals_sorted <= ecdffactor*alpha

            pvals_corrected_raw = pvals_sorted / ecdffactor
            pvals_corrected = np.minimum.accumulate(pvals_corrected_raw[::-1])[::-1]
            del pvals_corrected_raw

            pvals_corrected[pvals_corrected > 1] = 1
            pvals_corrected_ = np.empty_like(pvals_corrected)
            pvals_corrected_[pvals_sortind] = pvals_corrected
            del pvals_corrected

            q_matrix[mask] = pvals_corrected_

        else:
            raise ValueError('Only FDR method fdr_bh implemented')

        # Return the new matrix
        return q_matrix

    def return_significant_parents(self,
                                   pq_matrix,
                                   val_matrix,
                                   alpha_level=0.05):
        """Returns list of significant parents as well as a boolean matrix.

        Significance based on p-matrix, or q-value matrix with corrected
        p-values.

        Parameters
        ----------
        pq_matrix : array-like
            p-matrix, or q-value matrix with corrected p-values. Must be of
            shape (N, N, tau_max + 1).
        val_matrix : array-like
            Matrix of test statistic values. Must be of shape (N, N, tau_max +
            1).
        alpha_level : float, optional (default: 0.05)
            Significance level.

        Returns
        -------
        all_parents : dict
            Dictionary of form {0:[(0, -1), (3, -2), ...], 1:[], ...}
            containing estimated parents.

        link_matrix : array, shape [N, N, tau_max+1]
            Boolean array with True entries for significant links at alpha_level
        """
        # Initialize the return value
        all_parents = dict()
        for j in self.selected_variables:
            # Get the good links
            good_links = np.argwhere(pq_matrix[:, j, 1:] <= alpha_level)
            # Build a dictionary from these links to their values
            links = {(i, -tau - 1): np.abs(val_matrix[i, j, abs(tau) + 1])
                     for i, tau in good_links}
            # Sort by value
            all_parents[j] = sorted(links, key=links.get, reverse=True)
        # Return the significant parents
        return {'parents': all_parents,
                'link_matrix': pq_matrix <= alpha_level}

    def print_significant_links(self,
                                p_matrix,
                                val_matrix,
                                conf_matrix=None,
                                q_matrix=None,
                                alpha_level=0.05):
        """Prints significant parents.

        Parameters
        ----------
        alpha_level : float, optional (default: 0.05)
            Significance level.

        p_matrix : array-like
            Must be of shape (N, N, tau_max + 1).

        val_matrix : array-like
            Must be of shape (N, N, tau_max + 1).

        q_matrix : array-like, optional (default: None)
            Adjusted p-values. Must be of shape (N, N, tau_max + 1).

        conf_matrix : array-like, optional (default: None)
            Matrix of confidence intervals of shape (N, N, tau_max+1, 2)
        """
        if q_matrix is not None:
            sig_links = (q_matrix <= alpha_level)
        else:
            sig_links = (p_matrix <= alpha_level)
        print("\n## Significant links at alpha = %s:" % alpha_level)
        for j in self.selected_variables:
            links = {(p[0], -p[1]): np.abs(val_matrix[p[0], j, abs(p[1])])
                     for p in zip(*np.where(sig_links[:, j, :]))}
            # Sort by value
            sorted_links = sorted(links, key=links.get, reverse=True)
            n_links = len(links)
            string = ("\n    Variable %s has %d "
                      "link(s):" % (self.var_names[j], n_links))
            for p in sorted_links:
                string += ("\n        (%s %d): pval = %.5f" %
                           (self.var_names[p[0]], p[1],
                            p_matrix[p[0], j, abs(p[1])]))
                if q_matrix is not None:
                    string += " | qval = %.5f" % (
                        q_matrix[p[0], j, abs(p[1])])
                string += " | val = %.3f" % (
                    val_matrix[p[0], j, abs(p[1])])
                if conf_matrix is not None:
                    string += " | conf = (%.3f, %.3f)" % (
                        conf_matrix[p[0], j, abs(p[1])][0],
                        conf_matrix[p[0], j, abs(p[1])][1])
            print(string)

    ## Takes the time series rep. and converts to a conventional graph.

    #
    #
    def pcalg_skeleton(self, X, knowledge, sig_level=0.1, pmax=100, qmax=100,
                       ci_test='par_corr',
                       verbosity=0):
        N = X.shape[1]

        # Form complete graph
        #    graph = np.ones((N, N), dtype='int')
        #   graph[range(N),range(N)] = 0

        ## Use prior knowledge as the starting graph.
        graph = build_adj_mat(time_graph=knowledge, add_contemp=True, plot=False)
        graph = nx.adjacency_matrix(graph, nodelist=sorted(graph.nodes())).todense()
        # Define adj
        adj = self.get_adj(graph)

        print(adj)

        print("graph")
        print(graph)
        print("graph.shape")
        print(graph.shape)
        print("zip(*np.where(graph))")
        print(zip(*np.where(graph)))

        # Define sepset
        sepset = dict([((i, j), []) for i in range(N) for j in range(N)])

        # TODO: Fix bug where:
        #  File "<input>", line 1533, in pcalg_skeleton
        #  File "<input>", line 2014, in CI
        #  IndexError: index 3 is out of bounds for axis 1 with size 3

        p = 0
        # print [len(adj[j]) for j in range(N)]
        if verbosity > 0:
            print(graph)
        print("HERE")
        while (np.any([len(adj[j]) - 1 >= p for j in range(N)]) and p <= pmax):
            if verbosity > 0:
                print("\n:::::::::::::::::::::::: p = %d" % p)

            # Store already computed CI test results
            ci_results = {}

            for (i, j) in zip(*np.where(graph)):
                # for (i, j) in itertools.combinations(range(N), 2):
                # for j in range(N):
                if graph[i, j] != 0 and len(adj[j]) - 1 >= p:
                    if verbosity > 0:
                        print("\n\t%d -- %d" % (i, j))

                    conditions = list(itertools.combinations([l for l in adj[j] if l != i], p))
                    if verbosity > 0:
                        print("\tIterate through conditions %s" % conditions)
                    for q, S in enumerate(conditions):
                        if q > qmax:
                            break

                        check = self.is_in_set(i, j, S, ci_results)
                        if type(check) != bool:
                            pval = check
                        else:
                            pval = self.CI(X, i, j, S, ci_test)[0]

                        if verbosity > 0:
                            print("\t\t%d _|_ %d | %s : pval=%.4f %s  %s" % (
                                i, j, S, pval,
                                {0: '<= %s: dep' % sig_level, 1: '> %s: indep' % sig_level}[pval > sig_level],
                                {False: '[recycled]', True: ''}[type(check) == bool]))
                        if pval > sig_level:
                            graph[i, j] = graph[j, i] = 0
                            sepset[(i, j)] = sepset[(j, i)] = list(S)
                            break
                        else:
                            ci_results[((i, j), S)] = pval

            if verbosity > 0:
                print("\nUpdated graph")
                print(graph)
            # print sepset

            # Increase condition cardinality
            p += 1

            # Re-compute adj
            adj = self.get_adj(graph)
            # dict([(j, list(np.where(graph[:,j] != 0)[0])) for j in range(N)])
            # print "updated ", adj
        return {'graph': graph,
                'sepset': sepset}

        #     def pcalg_skeleton_timeseries(self, X, knowledge, sig_level=0.1,
        #                                   tau_min=0, tau_max=1,
        #                                   pmax=100, qmax=100,
        #                                   ci_test='par_corr',
        #                                   verbosity=0):
        #         N = X.shape[1]
        #
        #         # Form complete graph
        #         #graph =
        #         #graph[range(N), range(N), 0] = 0
        #
        #         graph = knowledge
        #
        # # TODO: Need to add effects for contemporary variables, but unclear how he stores this in his representation...
        # # TODO: Ask Jakob for some test cases -- unclear how to tell if results are correct at this stage (since things run).
        #         # Make complete graph for contemp. variables, then zero main diagonal.
        #         # for z in range(0, graph.shape[2]):
        #         #     graph[:,:,z][np.where(graph[:,:,z]==0)]=-1 ## Failed attempt1.
        #
        #         print(graph.shape)
        #         print(np.ones((N, N, tau_max + 1), dtype='int').shape)
        #         # Define adj
        #         adjt = self.get_adj_time_series(graph)
        #         # print adjt
        #         # Define sepset
        #         sepset = dict([(((i, -tau), j), []) for tau in range(tau_max + 1) for i in range(N) for j in range(N)])
        #
        #         if verbosity > 0:
        #             print("\n--------------------------")
        #             print("Skeleton discovery phase")
        #             print("--------------------------")
        #
        #         p = 0
        #         # print [len(adj[j]) for j in range(N)]
        #         # if verbosity > 0:
        #         #     print (graph)
        #
        #         while (np.any([len(adjt[j]) - 1 >= p for j in range(N)]) and p <= pmax):
        #             if verbosity > 0:
        #                 print("\n:::::::::::::::::::::::: p = %d" % p)
        #
        #             # Store already computed CI test results
        #             ci_results = {}
        #             for (i, j, abstau) in zip(*np.where(graph)):
        #                 if graph[i, j, abstau] != 0 and len(adjt[j]) - 1 >= p:
        #                     if verbosity > 0:
        #                         print("\n\t(%d, %d) o--o %d" % (i, -abstau, j))
        #
        #                     conditions = list(itertools.combinations([(k, tauk) for (k, tauk) in adjt[j]
        #                                                               if not (k == i and tauk == -abstau)], p))
        #                     if verbosity > 0:
        #                         print("\tIterate through conditions %s" % conditions)
        #
        #                     for q, S in enumerate(conditions):
        #                         if q > qmax:
        #                             break
        #
        #                         check = self.is_in_set_timeseries(i, abstau, j, S, ci_results)
        #                         if type(check) != bool:
        #                             pval = check
        #                         else:
        #                             pval = self.CI_timeseries(X, i, abstau, j, S, ci_test)[0]
        #
        #                         if verbosity > 0:
        #                             print("\t\t(%d, %d) _|_ %d | %s : pval=%.4f %s  %s" % (i, -abstau, j, S, pval,
        #                                                                                    {0: '<= %s: dep' % sig_level,
        #                                                                                     1: '> %s: indep' % sig_level}[
        #                                                                                        pval > sig_level],
        #                                                                                    {False: '[recycled]', True: ''}[
        #                                                                                        type(check) == bool]))
        #                         if pval > sig_level:
        #                             if abstau == 0:
        #                                 graph[i, j, 0] = graph[j, i, 0] = 0
        #                                 sepset[((i, 0), j)] = sepset[((j, 0), i)] = list(S)
        #                             else:
        #                                 graph[i, j, abstau] = 0
        #                                 sepset[((i, -abstau), j)] = list(S)
        #                             break
        #                         else:
        #                             ci_results[(((i, -abstau), j), S)] = pval
        #
        #             # Increase condition cardinality
        #             p += 1
        #
        #             # Re-compute adj
        #             adjt = self.get_adj_time_series(graph)
        #             # dict([(j, list(np.where(graph[:,j] != 0)[0])) for j in range(N)])
        #             # print "updated ", adjt
        #             if verbosity > 0:
        #                 print("\nUpdated adjacencies")
        #                 print(adjt)
        #
        #         return {'graph': graph,
        #                 'sepset': sepset}

    def get_adj(self, graph):
        N = len(graph)
        return dict([(j, list(np.where(graph[:, j] != 0)[0])) for j in range(N)])

    def get_adj_time_series(self, graph):
        N, N, tau_max_plusone = graph.shape
        adjt = {}
        for j in range(N):
            where = np.where(graph[:, j, :] != 0)
            adjt[j] = list(zip(*(where[0], -where[1])))

        return adjt
    def build_adj_mat(time_graph, add_contemp=True, plot=False):
        print(time_graph)
        print(type(time_graph))
        dimensions = np.empty(0, dtype=int)
        # print("HERE")
        # print(dimensions.shape)
        # Getting n.dimensions in time graph.
        for i in time_graph.shape:
            dimensions = np.append(dimensions, i)

        G = nx.Graph()  # creating empty graph.

        # creating empty matrix
        # adj_mat=np.zeros((np.sum(dimensions[0]), np.sum(dimensions)))
        # adding time series edges.
        for z in range(0, dimensions[1]):
            for i in range(0, dimensions[0]):
                # print("X*_0")
                G.add_node("X" + str(i) + "_" + str(0))
                for j in range(0, dimensions[dimensions.size - 1]):
                    G.add_node("X" + str(i) + "_" + str(j))
                    # print("X*_*")
                    # print("X"+str(i) + "_" +str(j))
                    if time_graph[i, z, j] == 1:
                        G.add_edge(("X" + str(i) + "_" + str(0)), ("X" + str(i) + "_" + str(j)))
        # Now Need to add in simultanious edges ("1" for all vars on same time step...)
        if add_contemp:
            time_lag = dimensions[dimensions.size - 1]
            n_vars = dimensions[0]
            for lag_degree in range(0, time_lag):
                for node1 in G.nodes:
                    for node2 in G.nodes:
                        if ((str("_") + str(lag_degree)) in node1 and (str("_") + str(lag_degree)) in node2):
                            G.add_edge(node1, node2)
        if plot:
            nx.draw_networkx(G)  # plotting for testing purposes.
        return G

    # zxcv=build_adj_mat(results_3['graph'], add_contemp=False)
    # print(nx.adjacency_matrix(zxcv, nodelist=sorted(zxcv.nodes())).todense()) # prints out adj.mat

    def get_adj_time_series_contemp(self, graph):
        N, N, tau_max_plusone = graph.shape
        adjt = {}
        for j in range(N):
            adjt[j] = list(np.where(graph[:, j, 0] != 0)[0])

        return adjt

    def pcalg_colliders(self, graph, sepset, verbosity=0):
        N = graph.shape[0]

        # Check symmetry
        if not np.all(graph == graph.transpose()):
            raise ValueError("Graph not symmetric")

        adj = self.get_adj(graph)

        # Find unshielded triples
        triples = []
        for i in range(N):
            for k in adj[i]:
                for j in adj[k]:
                    if j > i and i != k and k != j:
                        if graph[i, j] == 0 and graph[j, i] == 0:
                            triples.append((i, k, j))

        for ikj in triples:
            i, k, j = ikj
            # print "triples ", ikj, sepset[(i,j)]
            if k not in sepset[(i, j)]:
                # print 'not in sep'
                graph[k, i] = 0
                graph[k, j] = 0

        if verbosity > 0:
            print("Updated graph")
            print(graph)

        return {'graph': graph,
                'sepset': sepset}

    def pcalg_colliders_timeseries(self, graph, sepset, verbosity=0):
        N = graph.shape[0]

        if verbosity > 0:
            print("\n----------------------------")
            print("Collider orientation phase")
            print("----------------------------")

        # Check symmetry
        if not np.all(graph[:, :, 0] == graph[:, :, 0].transpose()):
            raise ValueError("Graph not symmetric")

        adjt = self.get_adj_time_series(graph)

        # Find unshielded triples
        triples = []
        for j in range(N):
            for (k, tauk) in adjt[j]:
                if tauk == 0:
                    for (i, taui) in adjt[k]:
                        if not (k == j or (taui == 0 and (i == k or i == j))):
                            if ((taui == 0 and graph[i, j, 0] == 0 and graph[j, i, 0] == 0)
                                    or taui < 0 and graph[i, j, abs(taui)] == 0):
                                triples.append(((i, taui), k, j))

        for itaukj in triples:
            print(itaukj)
            (i, tau), k, j = itaukj
            if (k, 0) not in sepset[((i, tau), j)]:
                print("(%d, 0) not in sepset, Remove %d --> %d " % (k, k, j))
                graph[k, j, 0] = 0
                if tau == 0:
                    print("tau=0, Remove %d --> %d " % (k, i))
                    graph[k, i, 0] = 0

        if verbosity > 0:
            adjt = self.get_adj_time_series(graph)
            print("\nUpdated adjacencies")
            print(adjt)

        return {'graph': graph,
                'sepset': sepset}

    def pcalg_rules(self, graph, sepset, verbosity=0):
        N = graph.shape[0]

        # adj = get_adj(graph)

        def rule1(self, graph):
            # Find triples i --> k -- j with i -/- j
            adj = self.get_adj(graph)
            triples = []
            for k in range(N):
                for i in adj[k]:
                    if k not in adj[i]:
                        for j in adj[k]:
                            if k in adj[j]:
                                if graph[i, j] == 0 and graph[j, i] == 0:
                                    triples.append((i, k, j))

            # orient as i --> k --> j
            for ikj in triples:
                i, k, j = ikj
                graph[j, k] = 0

            return len(triples) > 0, graph

        def rule2(self, graph):
            # Find triples i --> k --> j with i -- j
            adj = self.get_adj(graph)
            triples = []
            for j in range(N):
                for k in adj[j]:
                    if j not in adj[k]:
                        for i in adj[k]:
                            if k not in adj[i]:
                                if graph[i, j] == 1 and graph[j, i] == 1:
                                    triples.append((i, k, j))

            # orient as i --> j
            for ikj in triples:
                i, k, j = ikj
                graph[j, i] = 0

            return len(triples) > 0, graph

        def rule3(self, graph):
            # Find chains i -- k --> j and  i -- l --> j with i -- j
            # and k -/- l
            adj = self.get_adj(graph)
            pairs = []
            for j in range(N):
                for i in adj[j]:
                    if graph[j, i] == 1:
                        for k in adj[j]:
                            for l in adj[j]:
                                if (k != l and k != i and l != i and j not in adj[k] and j not in adj[l]
                                        and graph[k, l] == 0 and graph[l, k] == 0):
                                    if i in adj[l] and i in adj[k]:
                                        if graph[k, i] == 1 and graph[l, i] == 1:
                                            pairs.append((i, j))

            # orient as i --> j
            for ij in pairs:
                # print ij
                i, j = ij
                graph[j, i] = 0

            return len(pairs) > 0, graph

        graph_new = np.copy(graph)
        any1 = any2 = any3 = True
        while (any1 or any2 or any3):
            any1, graph_new = rule1(graph_new)
            any2, graph_new = rule2(graph_new)
            any3, graph_new = rule3(graph_new)

        if verbosity > 0:
            print("Updated graph")
            print(graph_new)

        return {'graph': graph_new,
                'sepset': sepset}

    def pcalg_rules_timeseries(self, graph, sepset, verbosity=0):
        N = graph.shape[0]

        def rule1(graph):

            adjt = self.get_adj_time_series(graph)

            # Find triples i_tau --> k_t -- j_t with i_tau -/- j_t
            # Orient as i_tau --> k_t --> j_t
            triples = []
            for j in range(N):
                for (k, tauk) in adjt[j]:
                    if tauk == 0 and graph[j, k, 0]:
                        for (i, taui) in adjt[k]:
                            if not (k == j or (taui == 0 and (i == k or i == j))):
                                if ((taui == 0 and graph[i, j, 0] == 0
                                     and graph[j, i, 0] == 0
                                     and graph[k, i, 0] == 0)
                                        or taui < 0 and graph[i, j, abs(taui)] == 0):
                                    triples.append(((i, taui), k, j))

            for itaukj in triples:
                (i, tau), k, j = itaukj
                print("Found triple %s, Removing (%d, 0) --> %d" % (itaukj, j, k))
                graph[j, k, 0] = 0

            return len(triples) > 0, graph

        def rule2(self, graph):

            # Find triples i_t --> k_t --> j_t with i_t -- j_t
            # Orient as i_t --> j_t
            adjtcont = self.get_adj_time_series_contemp(graph)

            triples = []
            for j in range(N):
                for k in adjtcont[j]:
                    if j not in adjtcont[k]:
                        for i in adjtcont[k]:
                            if k not in adjtcont[i]:
                                if graph[i, j, 0] == 1 and graph[j, i, 0] == 1:
                                    triples.append((i, k, j))

            # orient as i --> j
            for ikj in triples:
                i, k, j = ikj
                graph[j, i, 0] = 0

            return len(triples) > 0, graph

        def rule3(self, graph):
            # Find chains i_t -- k_tauk --> j_t and  i_t -- l_taul --> j_t with i_t -- j_t
            # and k_tauk -/- l_taul
            # Orient as i_t --> j_t
            adjt = self.get_adj_time_series(graph)

            pairs = []
            for j in range(N):
                for (i, taui) in adjt[j]:
                    if taui == 0 and graph[j, i, 0] == 1:
                        for (k, tauk) in adjt[j]:
                            for (l, taul) in adjt[j]:
                                # Nodes should not be identical
                                if not ((k == l and tauk == taul)
                                        or (k == i and tauk == taui)
                                        or (l == i and taul == taui)):
                                    # There should be an arrowhead from k and l to j
                                    if (j, 0) not in adjt[k] and (j, 0) not in adjt[l]:
                                        # Check that i is adjacent to k and l
                                        if (k, tauk) in adjt[i] and (l, taul) in adjt[i]:
                                            # Check that not both have arrow towards i
                                            if (i, 0) in adjt[k] or (i, 0) in adjt[l]:
                                                # k and l should not be adjacent
                                                if ((taul < tauk and graph[l, k, abs(taul) - abs(tauk)] == 0)
                                                        or (taul > tauk and graph[k, l, abs(tauk) - abs(taul)] == 0)
                                                        or (taul == tauk and graph[k, l, 0] == 0 and graph[
                                                            l, k, 0] == 0)
                                                ):
                                                    pairs.append((i, j))

                                                    # orient as i --> j
            for ij in pairs:
                # print ij
                i, j = ij
                graph[j, i, 0] = 0

            return len(pairs) > 0, graph

        if verbosity > 0:
            print("\n----------------------------")
            print("Rule orientation phase")
            print("----------------------------")

        graph_new = np.copy(graph)
        any1 = any2 = any3 = True
        while (any1 or any2 or any3):
            if verbosity > 0:
                print("Apply rule(s) %s" % (np.where(np.array([any1, any2, any3]))))
            any1, graph_new = rule1(graph_new)
            any2, graph_new = rule2(graph_new)
            any3, graph_new = rule3(graph_new)

        if verbosity > 0:
            adjt = self.get_adj_time_series(graph_new)
            print("\nUpdated adjacencies")
            print(adjt)

        return {'graph': graph_new,
                'sepset': sepset}

    def skeleton(self, graph):
        skele = np.copy(graph + graph.transpose())
        return (skele > 0).astype('bool').astype('int')

    def is_in_set(self, i, j, S, pval_list):
        for ijS in pval_list.keys():
            if set([i, j]) == set(ijS[0]) and set(S) == set(ijS[1]):
                return pval_list[ijS]

        return False

    def is_in_set_timeseries(self, i, abstau, j, S, pval_list):
        if abstau == 0:
            for itaujS in pval_list.keys():
                (((i_stored, tau_stored), j_stored), S_stored) = itaujS
                if tau_stored == 0:
                    # print i, j, S, ijS, set([i, j]) == set(ijS[0]), set(S) == set(ijS[1])
                    if set([i, j]) == set([i_stored, j_stored]) and set(S) == set(S_stored):
                        return pval_list[itaujS]

        return False

    def CI(self, data, i, j, S, test='par_corr'):
        if len(S) > 0:
            z = data[:, S]
        else:
            z = None

        x = data[:, i]
        y = data[:, j]

        if test == 'par_corr':
            df = len(x) - 2
            # print z.shape, len(z)
            if z is not None and z.shape[1] > 0:
                assert np.ndim(z) > 1
                df -= z.shape[1]
                beta_hat_x = np.linalg.lstsq(z, x, rcond=None)[0]
                xresid = x - np.dot(z, beta_hat_x)
                beta_hat_y = np.linalg.lstsq(z, y, rcond=None)[0]
                yresid = y - np.dot(z, beta_hat_y)
            else:
                xresid = x
                yresid = y

            value = np.corrcoef(xresid, yresid)[0, 1]

            # Two-sided significance level
            trafo_val = value * np.sqrt(df / (1. - value ** 2))
            pval = stats.t.sf(np.abs(trafo_val), df) * 2

        elif test == 'gpdc':
            cond_ind_test = GPDC()
            n = len(x)
            if z is not None and z.shape[1] > 0:
                array = np.hstack((x.reshape(n, 1), y.reshape(n, 1),
                                   z)).T
                xyz = np.array([0, 1, ] + [2 for i in range(z.shape[1])])
            else:
                array = np.vstack((x, y))
                xyz = np.array([0, 1])

            dim, n = array.shape
            value = cond_ind_test.get_dependence_measure(array, xyz)
            pval = cond_ind_test.get_analytic_significance(value, n, dim)

        elif test == 'cmi_knn':
            cond_ind_test = CMIknn()
            n = len(x)
            if z is not None and z.shape[1] > 0:
                array = np.hstack((x.reshape(n, 1), y.reshape(n, 1),
                                   z)).T
                xyz = np.array([0, 1, ] + [2 for i in range(z.shape[1])])
            else:
                array = np.vstack((x, y))
                xyz = np.array([0, 1])

            dim, n = array.shape

            value = cond_ind_test.get_dependence_measure(array, xyz)
            pval = cond_ind_test.get_shuffle_significance(array, xyz, value)

        return pval, value

    def CI_timeseries(self, data, i, abstau, j, S, test='par_corr'):

        X = [(i, -abstau)]
        Y = [(j, 0)]
        Z = list(S)

        maxlag = abstau
        for cond in S:
            # print cond
            maxlag = max(maxlag, abs(cond[1]))

        dataframe = pp.DataFrame(data)

        # Here tau_max = maxlag, which is not good since then
        # the samplesize is different for different conditioning
        # sets
        array, xyz = dataframe.construct_array(X, Y, Z, maxlag,
                                               mask=None,
                                               mask_type=None,
                                               return_cleaned_xyz=False,
                                               do_checks=True,
                                               cut_off='2xtau_max',
                                               verbosity=0)

        i_here = 0
        j_here = 1
        S_here = list(np.where(xyz == 2)[0])

        return self.CI(array.T, i_here, j_here, S_here, test=test)

    def run_pcmci(self,
                  selected_links=None,
                  tau_min=0,
                  tau_max=1,
                  save_iterations=False,
                  pc_alpha=0.05,
                  pc_alpha_contemp=.05,
                  max_conds_dim=None,
                  max_combinations=1,
                  max_conds_py=None,
                  max_conds_px=None,
                  test_contemp=False,
                  fdr_method='none'):
        """Run full PCMCI causal discovery for time series datasets.

                Wrapper around PC-algorithm function and MCI function.

                Parameters
                ----------
                selected_links : dict or None
                    Dictionary of form {0:all_parents (3, -2), ...], 1:[], ...}
                    specifying whether only selected links should be tested. If None is
                    passed, all links are tested

                tau_min : int, optional (default: 0)
                  Minimum time lag to test. Note that zero-lags are undirected.

                tau_max : int, optional (default: 1)
                  Maximum time lag. Must be larger or equal to tau_min.

                save_iterations : bool, optional (default: False)
                  Whether to save iteration step results such as conditions used.

                pc_alpha : float, optional (default: 0.05)
                  Significance level in algorithm.

                pc_alpha_contemp : float, optional (default: 0.05)
                  Significance level in contemp version of algorithm.

                max_conds_dim : int, optional (default: None)
                  Maximum number of conditions to test. If None is passed, this number
                  is unrestricted.

                max_combinations : int, optional (default: 1)
                  Maximum number of combinations of conditions of current cardinality
                  to test. Defaults to 1 for PC_1 algorithm. For original PC algorithm
                  a larger number, such as 10, can be used.

                max_conds_py : int, optional (default: None)
                    Maximum number of conditions of Y to use. If None is passed, this
                    number is unrestricted.

                max_conds_px : int, optional (default: None)
                    Maximum number of conditions of Z to use. If None is passed, this
                    number is unrestricted.

                fdr_method : str, optional (default: 'none')
                    Correction method, default is Benjamini-Hochberg False Discovery
                    Rate method.

                Returns
                -------
                results : dictionary of arrays of shape [N, N, tau_max+1]
                    {'val_matrix':val_matrix, 'p_matrix':p_matrix} are always returned
                    and optionally q_matrix and conf_matrix which is of shape
                    [N, N, tau_max+1,2]
                """
        # Get the parents from run_pc_stable
        all_parents = self.run_pc_stable(selected_links=selected_links,
                                         tau_min=tau_min,
                                         tau_max=tau_max,
                                         save_iterations=save_iterations,
                                         pc_alpha=pc_alpha,
                                         max_conds_dim=max_conds_dim,
                                         max_combinations=max_combinations)
        if test_contemp:  # Runs PC on mci outputed graph, then returns graph and sepsets.
            results = self.run_contemp_mci(selected_links=selected_links,
                                           tau_min=tau_min,
                                           tau_max=tau_max,
                                           parents=all_parents,
                                           max_conds_py=max_conds_py,
                                           max_conds_px=max_conds_px,
                                           pc_alpha=pc_alpha,
                                           pc_alpha_contemp=pc_alpha_contemp)

            return_dict = {'graph': results['graph'],
                           'sepset': results['sepset']}

        else:
            # Get the results from run_mci, using the parents as the input
            results = self.run_mci(selected_links=selected_links,
                                   tau_min=tau_min,
                                   tau_max=tau_max,
                                   parents=all_parents,
                                   max_conds_py=max_conds_py,
                                   max_conds_px=max_conds_px)
            # Get the values and p-values
            val_matrix = results['val_matrix']
            p_matrix = results['p_matrix']
            # Initialize and fill the the confidance matrix if the confidance test
            # says it should be returned

            conf_matrix = None
            if self.cond_ind_test.confidence is not False:
                conf_matrix = results['conf_matrix']
            # Initialize and fill the q_matrix if there is a fdr_method
            q_matrix = None
            if fdr_method != 'none':
                q_matrix = self.get_corrected_pvalues(p_matrix,
                                                      fdr_method=fdr_method)
            # Store the parents in the pcmci member
            self.all_parents = all_parents
            # Cache the resulting values in the return dictionary
            return_dict = {'val_matrix': val_matrix,
                           'p_matrix': p_matrix,
                           'q_matrix': q_matrix,
                           'conf_matrix': conf_matrix}
        # Print the information
        if self.verbosity > 0:
            self.print_results(return_dict)
        # Return the dictionary
        return return_dict

    def run_contemp_mci(self,
                        selected_links=None,
                        tau_min=0,
                        tau_max=1,
                        parents=None,
                        max_conds_py=None,
                        max_conds_px=None,
                        pc_alpha_contemp=.05,
                        pc_alpha=.05):
        """MCI conditional independence tests.

        Implements the MCI test (Algorithm 2 in [1]_). Returns the matrices of
        test statistic values,  p-values, and confidence intervals.

        Parameters
        ----------
        selected_links : dict or None
            Dictionary of form {0:all_parents (3, -2), ...], 1:[], ...}
            specifying whether only selected links should be tested. If None is
            passed, all links are tested
        tau_min : int, default: 0
            Minimum time lag to test. Note that zero-lags are undirected.
        tau_max : int, default: 1
            Maximum time lag. Must be larger or equal to tau_min.
        parents : dict or None
            Dictionary of form {0:[(0, -1), (3, -2), ...], 1:[], ...}
            specifying the conditions for each variable. If None is
            passed, no conditions are used.
        max_conds_py : int or None
            Maximum number of conditions of Y to use. If None is passed, this
            number is unrestricted.
        max_conds_px : int or None
            Maximum number of conditions of Z to use. If None is passed, this
            number is unrestricted.

        Returns
        -------
        results : dictionary of arrays of shape [N, N, tau_max+1]
            {'val_matrix':val_matrix, 'p_matrix':p_matrix} are always returned
            and optionally conf_matrix which is of shape [N, N, tau_max+1,2]
        """
        # Check the limits on tau
        self._check_tau_limits(tau_min, tau_max)
        # Set the selected links
        _int_sel_links = self._set_sel_links(selected_links, tau_min, tau_max)
        # Print information about the input parameters
        if self.verbosity > 0:
            self._print_mci_parameters(tau_min, tau_max,
                                       max_conds_py, max_conds_px)

        # Set the maximum condition dimension for Y and X
        max_conds_py = self._set_max_condition_dim(max_conds_py,
                                                   tau_min, tau_max)
        max_conds_px = self._set_max_condition_dim(max_conds_px,
                                                   tau_min, tau_max)
        # Get the parents that will be checked
        _int_parents = self._get_int_parents(parents)
        # Initialize the return values
        val_matrix = np.zeros((self.N, self.N, tau_max + 1))
        p_matrix = np.ones((self.N, self.N, tau_max + 1))
        # Initialize the optional return of the confidance matrix
        conf_matrix = None
        if self.cond_ind_test.confidence is not False:
            conf_matrix = np.zeros((self.N, self.N, tau_max + 1, 2))

        # Get the conditions as implied by the input arguments
        for j, i, tau, Z in self._iter_indep_conds(_int_parents,
                                                   self.selected_variables,
                                                   _int_sel_links,
                                                   max_conds_py,
                                                   max_conds_px):
            # Set X and Y (for clarity of code)
            X = [(i, tau)]
            Y = [(j, 0)]
            # Run the independence tests and record the results
            val, pval = self.cond_ind_test.run_test(X, Y, Z=Z, tau_max=tau_max)
            val_matrix[i, j, abs(tau)] = val
            p_matrix[i, j, abs(tau)] = pval
            # Get the confidance value, returns None if cond_ind_test.confidance
            # is False
            conf = self.cond_ind_test.get_confidence(X, Y, Z=Z, tau_max=tau_max)
            # Record the value if the conditional independence requires it
            if self.cond_ind_test.confidence:
                conf_matrix[i, j, abs(tau)] = conf
            # Print the results if needed
            if self.verbosity > 1:
                self.cond_ind_test._print_cond_ind_results(val,
                                                           pval=pval,
                                                           conf=conf)

        pc_results = self.pcalg_skeleton(X=dataframe.values, knowledge=((p_matrix < pc_alpha) + 0),
                                         sig_level=pc_alpha_contemp,
                                         pmax=100,
                                         qmax=100,
                                         ci_test='par_corr',
                                         verbosity=0)

        # Note: bad practice to add an extra fn for mci. Only a few lines change vs regular mci. Should
        # look into better design in future.

        return {'graph': pc_results['graph'],
                'sepset': pc_results['sepset']}
        # Return the values as a dictionary
        # return {'val_matrix': val_matrix,
        #         'p_matrix': p_matrix,
        #         'conf_matrix': conf_matrix}
        return

        def print_results(self,
                          return_dict,
                          alpha_level=0.05):
            """Prints significant parents from output of MCI or PCMCI algorithms.

            Parameters
            ----------
            return_dict : dict
                Dictionary of return values, containing keys
                    * 'p_matrix'
                    * 'val_matrix'
                    * 'conf_matrix'
                'q_matrix' can also be included in keys, but is not necessary.

            alpha_level : float, optional (default: 0.05)
                Significance level.
            """
            # Check if q_matrix is defined.  It is returned for PCMCI but not for
            # MCI
            q_matrix = None
            q_key = 'q_matrix'
            if q_key in return_dict:
                q_matrix = return_dict[q_key]
            # Check if conf_matrix is defined
            conf_matrix = None
            conf_key = 'conf_matrix'
            if conf_key in return_dict:
                conf_matrix = return_dict[conf_key]
            # Wrap the already defined function
            self.print_significant_links(return_dict['p_matrix'],
                                         return_dict['val_matrix'],
                                         conf_matrix=conf_matrix,
                                         q_matrix=q_matrix,
                                         alpha_level=alpha_level)

    if __name__ == '__main__':
        from tigramite.independence_tests import ParCorr
        import tigramite.data_processing as pp

    # In this case the vars should all be indep. of one another.
    #     dataframe = pp.DataFrame(np.random.randn(100, 3), )
    #     pcmci = PCMCI(dataframe, ParCorr())
    #
    #     test_result = pcmci.run_pcmci(test_contemp=True, pc_alpha_contemp=.05, pc_alpha=.05)
    #
    #     test_result

    # Test case based on online example in documentation. (only checking if runs without error).
np.random.seed(42)
links_coeffs = {0: [((0, -1), 0.8)], 1: [((1, -1), 0.8), ((0, -1), 0.5)], 2: [((2, -1), 0.8), ((1, -2), -0.6)]}
data, _ = pp.var_process(links_coeffs, T=1000)
dataframe = pp.DataFrame(data)
cond_ind_test = ParCorr()
pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test)

# results_0 = pcmci.run_pcmci(tau_max=0, pc_alpha=.05, test_contemp=True)
# results_1 = pcmci.run_pcmci(tau_max=1, pc_alpha=.05, test_contemp=True)
# results_2 = pcmci.run_pcmci(tau_max=2, pc_alpha=.05, test_contemp=True)
results_3 = pcmci.run_pcmci(tau_max=3, pc_alpha=.05, test_contemp=True)

    # results_0['graph']
    #
    # results_1['graph']
    #
    # results_2['graph']
    #
    # results_3['graph']
    # (pcmci.run_pcmci(tau_max=3, pc_alpha=.05, test_contemp=False)['p_matrix']<=.05)+0

    # pcmci.get_corrected_pvalues(np.random.rand(2, 2, 2))


