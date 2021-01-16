import numpy as np
import cv2
from ssim3d_utils import ssim3d, msssim2_1d

from scipy.io import savemat
from scipy.stats import spearmanr, pearsonr
from scipy.optimize import curve_fit
import pandas as pd

import os
import argparse

import progressbar

parser = argparse.ArgumentParser(description="Code to generate SSIM data for LIVE VQA Database")
parser.add_argument("--path", help="Path to database", required=True)
args = parser.parse_args()

df = pd.read_csv(os.path.join(args.path, 'live_video_quality_data.txt'), delimiter='\t', header=None, engine='python')
scores = df.values[:, 0]

scores = (scores - np.min(scores))/(np.max(scores) - np.min(scores))

df = pd.read_csv(os.path.join(args.path, 'live_video_quality_seqs.txt'), header=None, engine='python')
file_list = df.values[:, 0]

refs = ["pa", "rb", "rh", "tr", "st", "sf", "bs", "sh", "mc", "pr"]
fps = [25, 25, 25, 25, 25, 25, 25, 50, 50, 50]
fps = [str(f) + 'fps' for f in fps]

n_refs = len(refs)

widgets = [
            progressbar.ETA(),
            progressbar.Bar(),
            ' ', progressbar.DynamicMessage('file'),
            '/', progressbar.DynamicMessage('total')
            ]

ks = [1, 3, 5, 7, 10, 15, 20]
n_ks = len(ks)

multiscale_mssim = np.zeros((n_refs*15, n_ks))
mssim = np.zeros((n_refs*15, n_ks))

ssim_pccs = np.zeros((n_ks,))
ssim_sroccs = np.zeros((n_ks,))
ssim_rmses = np.zeros((n_ks,))

msssim_pccs = np.zeros((n_ks,))
msssim_sroccs = np.zeros((n_ks,))
msssim_rmses = np.zeros((n_ks,))

k = 0
with progressbar.ProgressBar(max_value=n_refs*15, widgets=widgets) as bar:
    for i_ref, ref in enumerate(refs):
        v_ref = cv2.VideoCapture(os.path.join(args.path, 'videos', ref + '_Folder', 'rgb', ref + '1' + '_' + fps[i_ref] + '.mp4'))
        for i_dist in range(2, 17):
            v_dist = cv2.VideoCapture(os.path.join(args.path, 'videos', ref + '_Folder', 'rgb', ref + str(i_dist) + '_' + fps[i_ref] + '.mp4'))

            for i_k, kt in enumerate(ks):
                mssim[k, i_k] = np.mean(ssim3d(v_ref, v_dist, [11, 11, kt], 0.01, 0.03))

                v_ref.set(cv2.CAP_PROP_POS_FRAMES, 0)
                v_dist.set(cv2.CAP_PROP_POS_FRAMES, 0)

                multiscale_mssim[k, i_k] = np.mean(msssim2_1d(v_ref, v_dist, [11, 11, kt], 5, 0.01, 0.03))

                v_ref.set(cv2.CAP_PROP_POS_FRAMES, 0)
                v_dist.set(cv2.CAP_PROP_POS_FRAMES, 0)

            k += 1
            bar.update(i_ref*15 + i_dist - 2, file=i_ref*15+i_dist-1, total=n_refs*15)

for i_k in range(n_ks):
    # Fitting logistic function to SSIM
    [[b0, b1, b2, b3, b4], _] = curve_fit(lambda t, b0, b1, b2, b3, b4: b0 * (0.5 - 1.0/(1 + np.exp(b1*(t - b2))) + b3 * t + b4),
                                          1 - mssim[:, i_k], scores, p0=0.5*np.ones((5,)), maxfev=20000)

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

savemat(os.path.join('data', 'live_vqa_ssim3d_ssims.mat'),
        {'mssim': mssim, 'multiscale_mssim': multiscale_mssim,
         'ssim_pccs': ssim_pccs, 'ssim_sroccs': ssim_sroccs, 'ssim_rmses': ssim_rmses,
         'msssim_pccs': msssim_pccs, 'msssim_sroccs': msssim_sroccs, 'msssim_rmses': msssim_rmses})
