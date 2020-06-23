import numpy as np
import cv2
# from skimage.transform.integral import integral_image
from scipy.signal import convolve2d


def integral_image(x):
    M, N = x.shape
    int_x = np.zeros((M+1, N+1))
    int_x = np.cumsum(np.cumsum(x, 0), 1)
    return int_x


def ssim_buff(buff_ref_sum_1, buff_ref_sum_2, buff_dist_sum_1, buff_dist_sum_2, buff_cross_sum, k_size, K1, K2, mode='partial'):
    C1 = (K1*255)**2
    C2 = (K2*255)**2

    kh = k_size[0]
    kw = k_size[1]

    k_norm = np.prod(k_size)

    temp_sum_1_ref = buff_ref_sum_1.copy()
    temp_sum_1_dist = buff_dist_sum_1.copy()

    int_1_ref = integral_image(temp_sum_1_ref)
    int_1_dist = integral_image(temp_sum_1_dist)

    # int_1_ref = np.cumsum(np.cumsum(temp_sum_1_ref, axis=0), axis=1)
    # int_1_dist = np.cumsum(np.cumsum(temp_sum_1_dist, axis=0), axis=1)

    temp_sum_2_ref = buff_ref_sum_2.copy()
    temp_sum_2_dist = buff_dist_sum_2.copy()

    int_2_ref = integral_image(temp_sum_2_ref)
    int_2_dist = integral_image(temp_sum_2_dist)

    # int_2_ref =	np.cumsum(np.cumsum(temp_sum_2_ref, axis=0), axis=1)
    # int_2_dist = np.cumsum(np.cumsum(temp_sum_2_dist, axis=0), axis=1)

    temp_sum_cross = buff_cross_sum.copy()
    int_cross = integral_image(temp_sum_cross)
    # int_cross = np.cumsum(np.cumsum(temp_sum_cross, axis=0), axis=1)

    mu_ref_local = (int_1_ref[:-kh, :-kw] - int_1_ref[:-kh, kw:] - int_1_ref[kh:, :-kw] + int_1_ref[kh:, kw:]) / k_norm
    mu_dist_local = (int_1_dist[:-kh, :-kw] - int_1_dist[:-kh, kw:] - int_1_dist[kh:, :-kw] + int_1_dist[kh:, kw:]) / k_norm

    mu_sq_ref_local = mu_ref_local**2
    mu_sq_dist_local = mu_dist_local**2

    var_ref_local = (int_2_ref[:-kh, :-kw] - int_2_ref[:-kh, kw:] - int_2_ref[kh:, :-kw] + int_2_ref[kh:, kw:]) / k_norm - mu_sq_ref_local
    var_dist_local = (int_2_dist[:-kh, :-kw] - int_2_dist[:-kh, kw:] - int_2_dist[kh:, :-kw] + int_2_dist[kh:, kw:]) / k_norm - mu_sq_dist_local

    cov_local = (int_cross[:-kh, :-kw] - int_cross[:-kh, kw:] - int_cross[kh:, :-kw] + int_cross[kh:, kw:]) / k_norm - mu_ref_local*mu_dist_local

    ssim_buff_map = (2*cov_local + C2) / (var_ref_local + var_dist_local + C2)
    if mode == "full":
        ssim_buff_map = ssim_buff_map * (2*mu_ref_local*mu_dist_local + C1) / (mu_sq_ref_local + mu_sq_dist_local + C1)

    mssim_buff = np.mean(ssim_buff_map)
    return mssim_buff


def ssim3d(v_ref, v_dist, k_size, K1, K2):

    if v_ref.get(cv2.CAP_PROP_FRAME_COUNT) != v_dist.get(cv2.CAP_PROP_FRAME_COUNT) or \
       v_ref.get(cv2.CAP_PROP_FRAME_HEIGHT) != v_dist.get(cv2.CAP_PROP_FRAME_HEIGHT) or \
       v_ref.get(cv2.CAP_PROP_FRAME_WIDTH) != v_dist.get(cv2.CAP_PROP_FRAME_WIDTH):
        print("Videos should have the same dimensions")
        mssim = -np.Inf
    else:
        N = int(v_ref.get(cv2.CAP_PROP_FRAME_COUNT))
        H = int(v_ref.get(cv2.CAP_PROP_FRAME_HEIGHT))
        W = int(v_ref.get(cv2.CAP_PROP_FRAME_WIDTH))

        kt = k_size[2]

        mssim = np.zeros((N-kt+1,))

        buff_ref = np.zeros((H, W, kt))
        buff_dist = np.zeros((H, W, kt))

        for i in range(kt-1):
            buff_ref[:, :, i+1] = cv2.cvtColor(v_ref.read()[1], cv2.COLOR_BGR2YUV)[:, :, 0].astype('float32')
            buff_dist[:, :, i+1] = cv2.cvtColor(v_dist.read()[1], cv2.COLOR_BGR2YUV)[:, :, 0].astype('float32')

        i = kt - 1

        buff_ref_sum_1 = np.sum(buff_ref, -1)
        buff_ref_sum_2 = np.sum(buff_ref**2, -1)
        buff_dist_sum_1 = np.sum(buff_dist, -1)
        buff_dist_sum_2 = np.sum(buff_dist**2, -1)

        buff_cross_sum = np.sum(buff_ref*buff_dist, -1)

        while i < N:
            i += 1
            temp_ref = cv2.cvtColor(v_ref.read()[1], cv2.COLOR_BGR2YUV)[:, :, 0].astype('float32')
            temp_dist = cv2.cvtColor(v_dist.read()[1], cv2.COLOR_BGR2YUV)[:, :, 0].astype('float32')

            buff_ref_sum_1 = buff_ref_sum_1 - buff_ref[:, :, i % kt] + temp_ref
            buff_ref_sum_2 = buff_ref_sum_2 - buff_ref[:, :, i % kt]**2 + temp_ref**2

            buff_dist_sum_1 = buff_dist_sum_1 - buff_dist[:, :, i % kt] + temp_dist
            buff_dist_sum_2 = buff_dist_sum_2 - buff_dist[:, :, i % kt]**2 + temp_dist**2

            buff_cross_sum = buff_cross_sum - buff_ref[:, :, i % kt]*buff_dist[:, :, i % kt] + temp_ref*temp_dist

            mssim[i-kt] = ssim_buff(buff_ref_sum_1, buff_ref_sum_2, buff_dist_sum_1, buff_dist_sum_2, buff_cross_sum, k_size, K1, K2, 'full')

            buff_ref[:, :, i % kt] = temp_ref
            buff_dist[:, :, i % kt] = temp_dist

    return mssim


def msssim2_1d(v_ref, v_dist, k_size, levels, K1, K2):

    if v_ref.get(cv2.CAP_PROP_FRAME_COUNT) != v_dist.get(cv2.CAP_PROP_FRAME_COUNT) or \
       v_ref.get(cv2.CAP_PROP_FRAME_HEIGHT) != v_dist.get(cv2.CAP_PROP_FRAME_HEIGHT) or \
       v_ref.get(cv2.CAP_PROP_FRAME_WIDTH) != v_dist.get(cv2.CAP_PROP_FRAME_WIDTH):
        print("Videos should have the same dimensions")
        mssim = -np.Inf
    else:
        N = int(v_ref.get(cv2.CAP_PROP_FRAME_COUNT))
        H = int(v_ref.get(cv2.CAP_PROP_FRAME_HEIGHT))
        W = int(v_ref.get(cv2.CAP_PROP_FRAME_WIDTH))

        sizes = np.zeros((levels, 2), dtype='uint32')
        sizes[0, 0], sizes[0, 1] = [H, W]
        for i in range(levels - 1):
            sizes[i+1, 0], sizes[i+1, 1] = np.ceil((sizes[i, 0] - 1)/2), np.ceil((sizes[i, 1] - 1)/2)

        kt = k_size[2]

        exponents = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])

        downsample_window = np.ones((2, 2)) / 4

        mssim = np.zeros((N-kt+1,))
        ssim_temp = np.zeros((levels,))

        buff_ref = np.empty((levels,), dtype=object)
        buff_dist = np.empty((levels,), dtype=object)

        buff_ref_sum_1 = np.empty((levels,), dtype=object)
        buff_ref_sum_2 = np.empty((levels,), dtype=object)
        buff_dist_sum_1 = np.empty((levels,), dtype=object)
        buff_dist_sum_2 = np.empty((levels,), dtype=object)
        buff_cross_sum = np.empty((levels,), dtype=object)

        for i in range(levels):
            buff_ref[i] = np.zeros(tuple(sizes[i, :]) + (kt,))
            buff_dist[i] = np.zeros(tuple(sizes[i, :]) + (kt,))

            buff_ref_sum_1[i] = np.zeros(tuple(sizes[i, :]))
            buff_ref_sum_2[i] = np.zeros(tuple(sizes[i, :]))
            buff_dist_sum_1[i] = np.zeros(tuple(sizes[i, :]))
            buff_dist_sum_2[i] = np.zeros(tuple(sizes[i, :]))
            buff_cross_sum[i] = np.zeros(tuple(sizes[i, :]))

        for i in range(kt-1):
            buff_ref[0][:, :, i+1] = cv2.cvtColor(v_ref.read()[1], cv2.COLOR_BGR2YUV)[:, :, 0].astype('float32')
            buff_dist[0][:, :, i+1] = cv2.cvtColor(v_dist.read()[1], cv2.COLOR_BGR2YUV)[:, :, 0].astype('float32')

            for level in range(1, levels):
                temp_ref_frame = convolve2d(buff_ref[level-1][:, :, i+1], downsample_window, 'valid')
                buff_ref[level][:, :, i+1] = temp_ref_frame[::2, ::2]

                temp_dist_frame = convolve2d(buff_dist[level-1][:, :, i+1], downsample_window, 'valid')
                buff_dist[level][:, :, i+1] = temp_dist_frame[::2, ::2]

        i = kt - 1

        for level in range(levels):
            buff_ref_sum_1[level] = np.sum(buff_ref[level], -1)
            buff_dist_sum_1[level] = np.sum(buff_dist[level], -1)

            buff_ref_sum_2[level] = np.sum(buff_ref[level]**2, -1)
            buff_dist_sum_2[level] = np.sum(buff_dist[level]**2, -1)

            buff_cross_sum[level] = np.sum(buff_ref[level]*buff_dist[level], -1)

        while i < N:
            i += 1
            temp_ref = cv2.cvtColor(v_ref.read()[1], cv2.COLOR_BGR2YUV)[:, :, 0].astype('float32')
            temp_dist = cv2.cvtColor(v_dist.read()[1], cv2.COLOR_BGR2YUV)[:, :, 0].astype('float32')

            for level in range(levels-1):
                buff_ref_sum_1[level] = buff_ref_sum_1[level] - buff_ref[level][:, :, i % kt] + temp_ref
                buff_ref_sum_2[level] = buff_ref_sum_2[level] - buff_ref[level][:, :, i % kt]**2 + temp_ref**2

                buff_dist_sum_1[level] = buff_dist_sum_1[level] - buff_dist[level][:, :, i % kt] + temp_dist
                buff_dist_sum_2[level] = buff_dist_sum_2[level] - buff_dist[level][:, :, i % kt]**2 + temp_dist**2

                buff_cross_sum[level] = buff_cross_sum[level] - buff_ref[level][:, :, i % kt]*buff_dist[level][:, :, i % kt] + temp_ref*temp_dist

                ssim_temp[level] = ssim_buff(buff_ref_sum_1[level], buff_ref_sum_2[level], buff_dist_sum_1[level], buff_dist_sum_2[level], buff_cross_sum[level], k_size, K1, K2)

                buff_ref[level][:, :, i % kt] = temp_ref
                buff_dist[level][:, :, i % kt] = temp_dist

                temp_ref_temp = convolve2d(temp_ref, downsample_window, 'valid')
                temp_ref = temp_ref_temp[::2, ::2]

                temp_dist_temp = convolve2d(temp_dist, downsample_window, 'valid')
                temp_dist = temp_dist_temp[::2, ::2]

            buff_ref_sum_1[levels-1] = buff_ref_sum_1[levels-1] - buff_ref[levels-1][:, :, i % kt] + temp_ref
            buff_ref_sum_2[levels-1] = buff_ref_sum_2[levels-1] - buff_ref[levels-1][:, :, i % kt]**2 + temp_ref**2

            buff_dist_sum_1[levels-1] = buff_dist_sum_1[levels-1] - buff_dist[levels-1][:, :, i % kt] + temp_dist
            buff_dist_sum_2[levels-1] = buff_dist_sum_2[levels-1] - buff_dist[levels-1][:, :, i % kt]**2 + temp_dist**2

            buff_cross_sum[levels-1] = buff_cross_sum[levels-1] - buff_ref[levels-1][:, :, i % kt]*buff_dist[levels-1][:, :, i % kt] + temp_ref*temp_dist

            ssim_temp[levels-1] = ssim_buff(buff_ref_sum_1[levels-1], buff_ref_sum_2[levels-1], buff_dist_sum_1[levels-1], buff_dist_sum_2[levels-1], buff_cross_sum[levels-1], k_size, K1, K2, 'full')

            buff_ref[levels-1][:, :, i % kt] = temp_ref
            buff_dist[levels-1][:, :, i % kt] = temp_dist

            mssim[i-kt] = np.prod(ssim_temp ** exponents[:levels])

    return mssim
