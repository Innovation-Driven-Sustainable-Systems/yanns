# imports
import numpy as np
from typing import Union
from scipy.linalg import block_diag

# generate equivalent mp_MPC matrices from linear MPC
def make_LQR_MPC_QP(
        stat_mat: np.ndarray, ctrl_mat: np.ndarray, ricc_wgt: np.ndarray,
        stat_wgt: np.ndarray, ctrl_wgt: np.ndarray, num_step: int,
        all_t_input_cmat: np.ndarray, all_t_input_cvec: np.ndarray,
        initl_state_cmat: np.ndarray, initl_state_cvec: np.ndarray,
        all_t_state_cmat: Union[np.ndarray, None]=None,
        all_t_state_cvec: Union[np.ndarray, None]=None,
        final_state_cmat: Union[np.ndarray, None]=None,
        final_state_cvec: Union[np.ndarray, None]=None) \
    -> np.ndarray:

    if all_t_state_cmat is None or all_t_state_cvec is None:
        all_t_state_cmat = np.empty((0, stat_mat.shape[0]), float)
        all_t_state_cvec = np.empty((0,), float)
    if final_state_cmat is None or final_state_cvec is None:
        final_state_cmat = np.empty((0, stat_mat.shape[0]), float)
        final_state_cvec = np.empty((0,), float)

    Bk = np.zeros((ctrl_mat.shape[0], ctrl_mat.shape[1]*num_step), float)

    num_inpt = all_t_input_cvec.size

    alts_off = 0
    inpt_off = alts_off + all_t_state_cvec.size*num_step
    cnst_off = inpt_off + all_t_input_cvec.size*num_step
    G = np.zeros((cnst_off + final_state_cvec.size, ctrl_mat.shape[1]*num_step), float)
    S = np.zeros((cnst_off + final_state_cvec.size, stat_mat.shape[1]), float)

    Ak = stat_mat
    cycslice = np.r_[-ctrl_mat.shape[1]:(ctrl_mat.shape[1]*num_step + -ctrl_mat.shape[1])]
    Bk[:,cycslice[:ctrl_mat.shape[1]]] = ctrl_mat

    H = Bk[:,cycslice].T@stat_wgt@Bk[:,cycslice]
    Z = Bk[:,cycslice].T@(stat_wgt + stat_wgt.T)@Ak
    M = Ak.T@stat_wgt@Ak

    G[:all_t_state_cvec.size,:] = all_t_state_cmat@Bk[:,cycslice]

    S[:all_t_state_cvec.size,:] = -all_t_state_cmat@Ak

    for indx in range(1, num_step - 1, 1):

        Ak = stat_mat@Ak
        indy = -(indx + 1)
        cycslice = np.r_[indy*ctrl_mat.shape[1]:(indy*ctrl_mat.shape[1] + ctrl_mat.shape[1]*num_step)]
        Bk[:,cycslice[:ctrl_mat.shape[1]]] = stat_mat@Bk[:,cycslice[ctrl_mat.shape[1]:2*ctrl_mat.shape[1]]]

        H += Bk[:,cycslice].T@stat_wgt@Bk[:,cycslice]
        Z += Bk[:,cycslice].T@(stat_wgt + stat_wgt.T)@Ak
        M += Ak.T@stat_wgt@Ak

        G[indx*all_t_state_cvec.size:(indx + 1)*all_t_state_cvec.size,:] \
            = all_t_state_cmat@Bk[:,cycslice]

        S[indx*all_t_state_cvec.size:(indx + 1)*all_t_state_cvec.size,:] \
            = -all_t_state_cmat@Ak

    Ak = stat_mat@Ak
    Bk[:,:ctrl_mat.shape[1]] = stat_mat@Bk[:,ctrl_mat.shape[1]:2*ctrl_mat.shape[1]]

    H += Bk.T@ricc_wgt@Bk
    H += block_diag(*((ctrl_wgt,)*num_step))
    Z += Bk.T@(ricc_wgt + ricc_wgt.T)@Ak
    M += Ak.T@ricc_wgt@Ak

    G[(inpt_off - all_t_state_cvec.size):inpt_off,:] = all_t_state_cmat@Bk
    G[inpt_off:cnst_off,:] = block_diag(*((all_t_input_cmat,)*num_step))
    G[cnst_off:] = final_state_cmat@Bk

    S[(inpt_off - all_t_state_cvec.size):inpt_off,:] = -all_t_state_cmat@Ak
    S[inpt_off:cnst_off,:] = np.zeros((all_t_input_cvec.size*num_step, stat_mat.shape[1]), float)
    S[cnst_off:] = -final_state_cmat@Ak

    W = np.block([np.tile(all_t_state_cvec, (num_step,)),
        np.tile(all_t_input_cvec, (num_step,)), final_state_cvec])[:,np.newaxis]

    return H, Z, M, G, W, S, initl_state_cmat, initl_state_cvec[:,np.newaxis]