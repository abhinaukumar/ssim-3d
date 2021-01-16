# SSIM 3D
An efficient extension of SSIM and Multiscale SSIM to 3D volumes, with applications in Video Quality Assessment (VQA). Local statistics (means, variances, and covariances) are calculated in local 3D neighbourhoods. These statistics are then used in SSIM and MS-SSIM expressions to obtain quality estimates of spatio-temporal regions of the video.

The extension of Multiscale SSIM is termed MS-SSIM 2+1D, since it is only multiscale in space, not time.

## Python implementation
`ssim3d_utils.py` contains all the functions needed to implement 3D. The function signatures are

```
scores = ssim3d(v_ref, v_dist, k_size, K1, K2)
```

```
scores = msssim2_1d(v_ref, v_dist, k_size, levels, K1, K2)
```

```
scores = ssim_frame(frame_ref, frame_dist, k, K1, K2)
```

where

* `v_ref` and `v_dist` are `cv2.VideoCapture` objects corresponding to the reference and distorted videos.
* `k_size` is a three-tuple `(kh, kw, kt)` corresponding to the height, width, and length of a 3D cuboidal averaging window used to define the spatio-temporal neighbourhoods.
* `K1` and `K2` are the regularization constants of SSIM.
* `levels` is the number of levels of the multiscale implementation - must be <= 5.

Both `ssim3d` and `msssim2_1d` return frame-wise quality scores.

`ssim_frame` is an implementation of the standard SSIM model for Image Quality Assessment (IQA), where `frame_ref` and `frame_dist` are the reference and distorted frames respectively.

## MATLAB implementation 
The function signatures are

```
scores = ssim3d(v_ref, v_dist, k_size, K1, K2)
```

```
scores = msssim2_1d(v_ref, v_dist, k_size, levels, K1, K2)
```

The parameters and returned values are almost identical to those of the Python implementation, with the only change being that `v_ref` and `v_dist` are `VideoReader` objects.

While SSIM 3D and MS-SSIM 2+1D operate on spatio-temporal neighbourhoods, the computational complexity of the algorithm does not grow with the temporal size of the filter.
In fact, the complexity does not grow with the spatial dimensions either, owing to the use of integral images and circular buffers to implement cuboidal averaging windows.
