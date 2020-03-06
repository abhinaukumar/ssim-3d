function mssim = ssim3d(v_ref, v_dist, k_size, K1, K2)

    if v_ref.NumFrames ~= v_dist.NumFrames || v_ref.Height ~= v_dist.Height || v_ref.Width ~= v_dist.Width
        disp("Videos should have the same dimensions");
        mssim = -Inf;
    else
        N = v_ref.NumFrames;
        H = v_ref.Height;
        W = v_ref.Width;

%         kh = k_size(1);
%         kw = k_size(2);
        kt = k_size(3);
        
%         k_norm = prod(k_size);
%         avg_window = ones(k_size) ./ prod(k_size);
        mssim = zeros([N - kt + 1,1]);
        
%         mu_ref_local = zeros([H - kh + 1, W - kw + 1]);
%         mu_dist_local = zeros([H - kh + 1, W - kw + 1]);
%         
%         var_ref_local = zeros([H - kh + 1, W - kw + 1]);
%         var_dist_local = zeros([H - kh + 1, W - kw + 1]);
%         
%         cov_local = zeros([H - kh + 1, W - kw + 1]);

        C1 = (K1*255)^2;
        C2 = (K2*255)^2;

        buff_ref = zeros([H, W, kt]);
        buff_dist = zeros([H, W, kt]);

        for i = 1:kt-1
            buff_ref(:,:,i+1) = rgb2gray(readFrame(v_ref));
            buff_dist(:,:,i+1) = rgb2gray(readFrame(v_dist)); 
        end
        if kt == 1
            i = 0;
        end
        
        buff_ref_sum_1 = sum(buff_ref, 3);
        buff_ref_sum_2 = sum(buff_ref.^2, 3);
        buff_dist_sum_1 = sum(buff_dist, 3);
        buff_dist_sum_2 = sum(buff_dist.^2, 3);
        
        buff_cross_sum = sum(buff_ref .* buff_dist, 3);
        
        while hasFrame(v_ref) && hasFrame(v_dist)
            i = i + 1;
            temp_ref = double(rgb2gray(readFrame(v_ref)));
            temp_dist = double(rgb2gray(readFrame(v_dist)));
            
            buff_ref_sum_1 = buff_ref_sum_1 - buff_ref(:,:,mod(i,kt) + 1) +  temp_ref;
            buff_ref_sum_2 = buff_ref_sum_2 - buff_ref(:,:,mod(i,kt) + 1).^2 +  temp_ref.^2;

            buff_dist_sum_1 = buff_dist_sum_1 - buff_dist(:,:,mod(i,kt) + 1) +  temp_dist;
            buff_dist_sum_2 = buff_dist_sum_2 - buff_dist(:,:,mod(i,kt) + 1).^2 +  temp_dist.^2;

            buff_cross_sum = buff_cross_sum - buff_ref(:,:,mod(i,kt) + 1).* buff_dist(:,:,mod(i,kt) + 1) + temp_ref.*temp_dist;
            
            mssim(i - kt + 1) = ssim_buff(buff_ref_sum_1, buff_ref_sum_2, buff_dist_sum_1, buff_dist_sum_2, buff_cross_sum, k_size, K1, K2, 'full');
            
            buff_ref(:,:,mod(i,kt) + 1) = temp_ref;
            buff_dist(:,:,mod(i,kt) + 1) = temp_dist;
%             mu_ref_local = convn(buff_ref, avg_window, 'valid');
%             mu_dist_local = convn(buff_dist, avg_window, 'valid');
%             
%             mu_sq_ref_local = mu_ref_local.^2;
%             mu_sq_dist_local = mu_dist_local.^2;
            
%             var_ref_local = convn(buff_ref.^2, avg_window, 'valid') - mu_sq_ref_local;
%             var_dist_local = convn(buff_dist.^2, avg_window, 'valid') - mu_sq_dist_local;
            
%             cov_local = convn(buff_ref .* buff_dist, avg_window, 'valid') - mu_ref_local.* mu_dist_local;
            
%             mssim(i - kt + 1) = mean(mean(((2 .* mu_ref_local .* mu_dist_local + C1) .* (2 .* cov_local + C2)) ./ ((mu_sq_ref_local + mu_sq_dist_local + C1) .* (var_ref_local + var_dist_local + C2))));
        end
        
%         mssim = mssim / (N - kt + 1);
    end
end