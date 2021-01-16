import numpy as np
import cv2
from ssim3d_utils import ssim3d, msssim2_1d

from scipy.io import loadmat, savemat
from scipy.stats import spearmanr, pearsonr
from scipy.optimize import curve_fit

import os
import argparse

import progressbar

parser = argparse.ArgumentParser(description="Code to generate SSIM data for Netflix Public Database")
parser.add_argument("--path", help="Path to database", required=True)
args = parser.parse_args()

path = args.path
f = loadmat('data/nflx_repo_scores.mat')
scores = f['scores'].squeeze()
scores = (scores - np.min(scores))/(np.max(scores) - np.min(scores))

ref_file_list = os.listdir(os.path.join(path, 'ref', 'rgb'))
ref_file_list = sorted([v for v in ref_file_list if v[-3:] == 'mp4'], key=lambda v: v.lower())
n_ref_files = len(ref_file_list)

dist_file_list = os.listdir(os.path.join(path, 'dis', 'rgb'))
dist_file_list = sorted([v for v in dist_file_list if v[-3:] == 'mp4'], key=lambda v: v.lower())
n_dist_files = len(dist_file_list)

widgets = [
            progressbar.ETA(),
            progressbar.Bar(),
            ' ', progressbar.DynamicMessage('file'),
            '/', progressbar.DynamicMessage('total')
            ]

ks = [1, 3, 5, 7, 10, 15, 20]
n_ks = len(ks)

multiscale_mssim = np.zeros((n_dist_files, n_ks))
mssim = np.zeros((n_dist_files, n_ks))

ssim_pccs = np.zeros((n_ks,))
ssim_sroccs = np.zeros((n_ks,))
ssim_rmses = np.zeros((n_ks,))

msssim_pccs = np.zeros((n_ks,))
msssim_sroccs = np.zeros((n_ks,))
msssim_rmses = np.zeros((n_ks,))

i_dist = 0
with progressbar.ProgressBar(max_value=n_dist_files, widgets=widgets) as bar:
    for i_ref in range(n_ref_files):
        ref_filename = ref_file_list[i_ref][:-4].split('_')[0]
        v_ref = cv2.VideoCapture(os.path.join(path, 'ref', 'rgb', ref_file_list[i_ref]))
        while(i_dist < n_dist_files and ref_filename in dist_file_list[i_dist]):
            v_dist = cv2.VideoCapture(os.path.join(path, 'dis', 'rgb', dist_file_list[i_dist]))

            for i_k, kt in enumerate(ks):
                ksize = np.array([11, 11, kt])
                mssim[i_dist, i_k] = np.mean(ssim3d(v_ref, v_dist, ksize, 0.01, 0.03))

                v_ref.set(cv2.CAP_PROP_POS_FRAMES, 0)
                v_dist.set(cv2.CAP_PROP_POS_FRAMES, 0)

                multiscale_mssim[i_dist, i_k] = np.mean(msssim2_1d(v_ref, v_dist, ksize, 5, 0.01, 0.03))

                v_ref.set(cv2.CAP_PROP_POS_FRAMES, 0)
                v_dist.set(cv2.CAP_PROP_POS_FRAMES, 0)

            i_dist += 1
            v_ref.set(cv2.CAP_PROP_POS_FRAMES, 0)
            bar.update(i_dist-1, file=i_dist, total=n_dist_files)

savemat(os.path.join('data', 'nflx_repo_ssim3d_ssims.mat'),
        {'mssim': mssim, 'multiscale_mssim': multiscale_mssim})

for i_k in range(n_ks):
    # Fitting logistic function to SSIM
    [[b0, b1, b2, b3, b4], _] = curve_fit(lambda t, b0, b1, b2, b3, b4: b0 * (0.5 - 1.0/(1 + np.exp(b1*(t - b2))) + b3 * t + b4),
                                          1 - mssim[:, i_k], scores, p0=0.5*np.ones((5,)), maxfev=200000)

    scores_pred = b0 * (0.5 - 1.0/(1 + np.exp(b1*((1 - mssim[:, i_k]) - b2))) + b3 * (1 - mssim[:, i_k]) + b4)

    ssim_pccs[i_k] = pearsonr(scores_pred, scores)[0]
    ssim_sroccs[i_k] = spearmanr(scores_pred, scores)[0]
    ssim_rmses[i_k] = np.sqrt(np.mean((scores_pred - scores)**2))

    [[b0, b1, b2, b3, b4], _] = curve_fit(lambda t, b0, b1, b2, b3, b4: b0 * (0.5 - 1.0/(1 + np.exp(b1*(t - b2))) + b3 * t + b4),
                                          1 - multiscale_mssim[:, i_k], scores, p0=0.5*np.ones((5,)), maxfev=20000)

    scores_pred = b0 * (0.5 - 1.0/(1 + np.exp(b1*((1 - multiscale_mssim[:, i_k]) - b2))) + b3 * (1 - multiscale_mssim[:, i_k]) + b4)

    msssim_pccs[i_k] = pearsonr(scores_pred, scores)[0]
    msssim_sroccs[i_k] = spearmanr(scores_pred, scores)[0]
    msssim_rmses[i_k] = np.sqrt(np.mean((scores_pred - scores)**2))

savemat(os.path.join('data', 'nflx_repo_ssim3d_ssims.mat'),
        {'mssim': mssim, 'multiscale_mssim': multiscale_mssim,
         'ssim_pccs': ssim_pccs, 'ssim_sroccs': ssim_sroccs, 'ssim_rmses': ssim_rmses,
         'msssim_pccs': msssim_pccs, 'msssim_sroccs': msssim_sroccs, 'msssim_rmses': msssim_rmses})
