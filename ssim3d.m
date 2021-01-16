function mssim = ssim3d(v_ref, v_dist, k_size, K1, K2)

    if v_ref.NumFrames ~= v_dist.NumFrames || v_ref.Height ~= v_dist.Height || v_ref.Width ~= v_dist.Width
        disp("Videos should have the same dimensions");
        mssim = -Inf;
    else
        N = v_ref.NumFrames;
        H = v_ref.Height;
        W = v_ref.Width;

        kt = k_size(3);

        mssim = zeros([N - kt + 1,1]);

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
        end
    end
end