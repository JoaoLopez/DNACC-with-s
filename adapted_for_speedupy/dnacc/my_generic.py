"""
A generic DNA-coated colloid (DNACC) system.

At an abstract level, a DNACC system consists of a set of N tethers that can
bind pairwise with binding free energy :math:`\\Delta G_{ij}`.  This binding
energy may be infinite if i and j physically cannot bind.  Otherwise, it
consists of the solution hybridisation free energy of the sticky end
sequences of i and j along with the configuration free energy cost of
binding.
"""
__all__ = ['Generic']
import numpy as np
import scipy.integrate
from math import log, exp
from .utils import csr_matrix_items, SyntheticList, default_zero_dict, is_csr_matrix_symmetric
import sys

class Generic(object):
    """A generic DNA-coated colloid (DNACC) system.

    Everywhere, N is the number of tethers in the DNACC system.  This object
    is immutable, i.e., once the parameters are specified at creation, they
    cannot be changed later.

    Parameters
      an NxN symmetric matrix, with boltz_bind[i,j] = :math:`K_{ij} =
      \\exp(-\\beta \\Delta G_{ij})`, i.e. the ratio of partition functions
      of the tethers i and j when bound vs. unbound.  The easiest way to
      create this matrix is by first building a dictionary of entries, then
      passing it to :func:`.csr_matrix_from_dict`.

    weights : None, or list of N floats
      If not None, a weight to assign to each tether's probability of being
      free (see Notes for details).  This is used by
      :class:`.PlatesMeanField` to take into account the grafting densities
      of each type of tether.  The calculated :attr:`p_free` and
      :attr:`p_bound` are also weighted.

    Other Parameters
    ----------------
    self_consistent_max_delta : float
      Threshold change in any :math:`p_i` to terminate fixed-point
      iteration.
    self_consistent_max_steps : integer
      Maximum number of fixed-point iterations.

    Attributes
    ----------
    avg_num_bonds
      average number of bonds formed
    p_free
      p_free[i] = probability that tether i is unbound.
    p_bound
      p_bound[i,j] = probability that tethers i and j are bound.
    binding_free_energy
      free energy of binding

    Notes
    -----
    If weights are not specified, solve the following self-consistent
    equations for the probability of tether i being unbound, :math:`p_i`:

    .. math:: p_i = 1 / (1 + \\sum_j K_{ij} p_j),

    whereby the probability of i and j being bound is:

    .. math:: p_{ij} = p_i K_{ij} p_j.

    If weights are specified, then solve the following self-consistent
    equations for the weighted probability of tether i being unbound,
    :math:`s_i = w_i p_i`:

    .. math:: s_i = w_i / (1 + \\sum_j K_{ij} s_j)

    whereby the weighted probability of i and j being bound is:

    .. math:: s_{ij} = s_i K_{ij} s_j.

    The accuracy of the fixed point iteration can be controlled using the
    attributes ``self_consistent_max_delta`` and
    ``self_consistent_max_steps``.

    The binding energy is computed using the following thermodynamic
    integration:

    .. math::
       F_{att} = - \\int_0^{10\\,k_B T - \\min(\\Delta G_{ij})}
                           d\\lambda\\, \\sum_{i < j} p_{ij}(\\lambda)

    where :math:`p_{ij}(\\lambda)` is the probability that tethers i and j
    are bound after shifting all binding energies :math:`\\Delta G_{ij}` up
    by an amount :math:`\\lambda`.

    Previously, the thermodynamic integral was evaluated explicitly.  Now,
    we have an exact expression for the result of this integral, namely

    .. math::
       F_{att} = \\sum_{i < j} p_{ij} + \\sum_i \\ln p_i

    or, for weighted tethers,

    .. math::
       f_{att} = \\sum_{i < j} s_{ij} + \\sum_i w_i \\ln p_i
    """

    def README(self):
        """Read source for docstring (Cython doesn't export docstrings well)"""
        pass

    def __init__(self, boltz_bind, weights=None, self_consistent_max_delta=1e-07, self_consistent_max_steps=10001):
        (M, N) = boltz_bind.shape
        assert M == N
        self._boltz_bind = boltz_bind
        self._weights = None if weights is None else np.array(weights)
        self._self_consistent_max_delta = self_consistent_max_delta
        self._self_consistent_max_steps = self_consistent_max_steps
        self.avg_num_bonds = -1
        self.p_free = np.zeros(N)
        self.p_bound = SyntheticList(fgetitem=lambda pair: self.p_free[pair[0]] * self._boltz_bind[pair] * self.p_free[pair[1]])
        self._calculate()

    def _calculate(self):
        """Recalculate self-consistent values of p_free and avg_num_bonds.
        """
        W = self._boltz_bind
        indices = W.indices
        indptr = W.indptr
        data = W.data
        max_delta = self._self_consistent_max_delta
        max_steps = self._self_consistent_max_steps
        p_free = self.p_free
        weights = self._weights
        N = p_free.shape[0]
        p_free[:] = 1.0
        delta = 1.0
        step = 0
        while delta > max_delta and step < max_steps:
            delta = 0.0
            for i in range(N):
                oldP = p_free[i]
                x = 0.0
                for jj in range(indptr[i], indptr[i + 1]):
                    j = indices[jj]
                    Kij = data[jj]
                    x += Kij * p_free[j]
                w = weights[i] if weights is not None else 1.0
                p_free[i] = w / (1.0 + x)
                delta = max(delta, abs(oldP - p_free[i]) / (w or 1.0))
            step += 1
        if self._weights is None:
            self.avg_num_bonds = 0.5 * (N - np.sum(self.p_free))
        else:
            self.avg_num_bonds = 0.5 * np.sum(self._weights - self.p_free)
        if self._weights is None:
            threshold_p_i = exp(log(100 * sys.float_info.min) / N)
            sum_ln_p_i = log(np.prod(self.p_free[self.p_free > threshold_p_i])) + np.sum(np.log(self.p_free[self.p_free <= threshold_p_i]))
            self.binding_free_energy = self.avg_num_bonds + sum_ln_p_i
        else:
            sum_weighted_ln_p_i = np.sum(self._weights * np.log(self.p_free / self._weights))
            self.binding_free_energy = self.avg_num_bonds + sum_weighted_ln_p_i

    def count_bonds(self, i_set, j_set):
        """Counts bonds between tethers in i_set and j_set.

        Parameters
        ----------
        i_set : set of int
            One set of tether ids
        j_set : set of int
            Another set of tether ids

        Notes
        -----
        For performance, it's *essential* that i_set and j_set be sets and
        not lists.
        """
        mtx = self._boltz_bind
        p = self.p_free
        count = 0.0
        for i in i_set:
            if p[i] != 0.0:
                partial_count = 0.0
                for ((gobble, j), Kij) in csr_matrix_items(mtx, row=i):
                    if j in j_set and (j not in i_set or j > i):
                        partial_count += Kij * p[j]
                count += p[i] * partial_count
        return count