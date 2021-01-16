function mssim_buff = ssim_buff(buff_ref_sum_1, buff_ref_sum_2, buff_dist_sum_1, buff_dist_sum_2, buff_cross_sum, k_size, K1, K2, mode)

    if nargin < 8
        disp("Too few arguments");
        mssim_buff = -Inf;
        return;
    elseif nargin == 8
        mode = "partial";
    end

    C1 = (K1*255)^2;
    C2 = (K2*255)^2;
    
    kh = k_size(1);
    kw = k_size(2);

    k_norm = prod(k_size);
    
    temp_sum_1_ref = buff_ref_sum_1;
    temp_sum_1_dist = buff_dist_sum_1;

    int_1_ref = integralImage(temp_sum_1_ref);
    int_1_dist = integralImage(temp_sum_1_dist);

    temp_sum_2_ref = buff_ref_sum_2;
    temp_sum_2_dist = buff_dist_sum_2;

    int_2_ref = integralImage(temp_sum_2_ref);
    int_2_dist = integralImage(temp_sum_2_dist);

    temp_sum_cross = buff_cross_sum;
    int_cross = integralImage(temp_sum_cross);

    mu_ref_local = (int_1_ref(1:end-kh, 1:end-kw) - int_1_ref(1:end-kh, kw+1:end) - int_1_ref(kh+1:end, 1:end-kw) + int_1_ref(kh+1:end, kw+1:end)) ./ k_norm;
    mu_dist_local = (int_1_dist(1:end-kh, 1:end-kw) - int_1_dist(1:end-kh, kw+1:end) - int_1_dist(kh+1:end, 1:end-kw) + int_1_dist(kh+1:end, kw+1:end)) ./ k_norm;

    mu_sq_ref_local = mu_ref_local.^2;
    mu_sq_dist_local = mu_dist_local.^2;

    var_ref_local = (int_2_ref(1:end-kh, 1:end-kw) - int_2_ref(1:end-kh, kw+1:end) - int_2_ref(kh+1:end, 1:end-kw) + int_2_ref(kh+1:end, kw+1:end)) ./ k_norm - mu_sq_ref_local;
    var_dist_local = (int_2_dist(1:end-kh, 1:end-kw) - int_2_dist(1:end-kh, kw+1:end) - int_2_dist(kh+1:end, 1:end-kw) + int_2_dist(kh+1:end, kw+1:end)) ./ k_norm - mu_sq_dist_local;
    cov_local = (int_cross(1:end-kh, 1:end-kw) - int_cross(1:end-kh, kw+1:end) - int_cross(kh+1:end, 1:end-kw) + int_cross(kh+1:end, kw+1:end)) ./ k_norm - mu_ref_local .* mu_dist_local;

    mask_ref = (var_ref_local <= 0);
    mask_dist = (var_dist_local <= 0);
    
    var_ref_local(mask_ref) = 0;
    var_dist_local(mask_dist) = 0;
    cov_local(mask_ref | mask_dist) = 0;

    map = (2 .* cov_local + C2) ./ (var_ref_local + var_dist_local + C2);
    if mode == "full"
        map = map .* (2 .* mu_ref_local .* mu_dist_local + C1) ./ (mu_sq_ref_local + mu_sq_dist_local + C1);
    end
    
    mssim_buff = mean(map(:));