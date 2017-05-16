import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator

from pymbar import BAR as BAR_


class BAR(BaseEstimator):
    """Bennett acceptance ratio (BAR).

    Parameters
    ----------

    Attributes
    ----------

    delta_f_ : DataFrame
        The estimated dimensionless free energy difference between each state.

    d_delta_f_ : DataFrame
        The estimated statistical uncertainty (one standard deviation) in
        dimensionless free energy differences.

    theta_ : DataFrame
        The theta matrix.

    states_ : list
        Lambda states for which free energy differences were obtained.

    """

    def __init__(self, maximum_iterations=500, relative_tolerance=1.0e-11,
                 initial_f_k=None, verbose=False):

        self.maximum_iterations = maximum_iterations
        self.relative_tolerance = relative_tolerance
        self.initial_f_k = initial_f_k
        self.method = (dict(method=method), )
        self.verbose = verbose

        # handle for pymbar.MBAR object
        self._bar = None

    def fit(self, u_nk):
        """
        Compute overlap matrix of reduced potentials using multi-state
        Bennett acceptance ratio.

        Parameters
        ----------
        u_nk : DataFrame 
            u_nk[n,k] is the reduced potential energy of uncorrelated
            configuration n evaluated at state k.

        """
        # sort by state so that rows from same state are in contiguous blocks
        u_nk = u_nk.sort_index(level=u_nk.index.names[1:])


        # for each lambda window sampled, get difference with neighbor
        
        ## USE THIS BELOW TO DO IT
        groups = u_nk.groupby(level=u_nk.index.names[1:])
        N_k = [(len(groups.get_group(i)) if i in groups.groups else 0) for i in u_nk.columns]

        def W(k, m, groups):
            """Get work values from state k evaluated at state m
                
            Parameters
            ----------
            k : float
                State k -- the state for which the trajectory was generated
            m : float
                State m -- the state for which the hamiltonian was evaluated
            """
            forward_state = groups.get_group(k)[m].values
            current_state = groups.get_group(k)[k].values
            return forward_state - current_state

        # evaluate all of the forward work values
        wFs = [W(i, i+1, groups) for i, val in enumerate(u_nk.columns)]
        # evaluate all of the reverse work values
        wRs = [W(i+1, i, groups) for i, val in enumerate(u_nk.columns)]

        self.states_ = u_nk.columns.values.tolist()

        deltas = np.zeros((len(u_nk.columns),len(u_nk.columns)))
        d_deltas = np.zeros_like(deltas)

        for j in range(len(deltas.shape[0])):
            out = []
            dout = []
            for i in range(j, len(deltas.shape[0])):
                if i == 0 and j == 0:
                    out.append(0)
                    continue
                wF = wFs[j:i+1]
                wR = wRs[j:i+1]
                delta_cell = []
                d_delta_cell = []
                for k in range(len(wF)):
                    delta_f, d_delta_f = BAR_(wF[k], wR[k], 
                               maximum_iterations=self.maximum_iterations,
                               relative_tolerance=self.relative_tolerance,
                               DeltaF=self.initial_f_k, verbose=self.verbose,
                               method=self.method)
                    delta_cell.append(delta_f)
                    d_delta_cell.append(d_delta_f)
                out.append(np.sum(delta_cell))
                dout.append(np.sum(d_delta_cell))
            deltas[j,:] = np.array(out)
            d_deltas[j,:] = np.array(dout)
                

        deltas = deltas - deltas.T
        d_deltas = d_deltas + d_deltas.T

        self.delta_f_ = pd.Dataframe(deltas, columns=u_nk.columns, 
                               index=u_nk.columns, dtype=float)

        self.d_delta_f_= pd.DataFrame(d_deltas, columns=u_nk.columns,
                               index=u_nk.columns, dtype=float)
        
        return self

    def predict(self, u_ln):
        pass
