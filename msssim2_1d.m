function mssim = msssim2_1d(v_ref, v_dist, k_size, levels, K1, K2)

    if v_ref.NumFrames ~= v_dist.NumFrames || v_ref.Height ~= v_dist.Height || v_ref.Width ~= v_dist.Width
        disp("Videos should have the same dimensions");
        mssim = -Inf;
    else
        N = v_ref.NumFrames;
        H = v_ref.Height;
        W = v_ref.Width;

        sizes = zeros([levels, 2]);
        sizes(1,:) = [H, W];
        for i = 1:levels - 1
            sizes(i+1, :) = ceil((sizes(i,:) - 1)./2);
        end
        
        kt = k_size(end);
        
        exponents = [0.0448; 0.2856; 0.3001; 0.2363; 0.1333];
        
        avg_window = ones(k_size) ./ prod(k_size);
        downsample_window = ones(2) ./ 4;
        
        mssim = zeros([N - kt + 1,1]);
        ssim_temp = zeros([levels, 1]);
        
        C1 = (K1*255)^2;
        C2 = (K2*255)^2;

%         buff_ref = zeros([H, W, kt]);
%         buff_dist = zeros([H, W, kt]);
        
        buff_ref = cell([levels, 1]);
        buff_dist = cell([levels, 1]);
        
        buff_ref_sum_1 = cell([levels,1]);
        buff_ref_sum_2 = cell([levels,1]);
        buff_dist_sum_1 = cell([levels,1]);
        buff_dist_sum_2 = cell([levels,1]);
        buff_cross_sum = cell([levels,1]);
        
        for i = 1:levels
            buff_ref{i} = zeros([sizes(i, :) kt]);
            buff_dist{i} = zeros([sizes(i, :) kt]);
            
            buff_ref_sum_1{i} = zeros(sizes(i, :));
            buff_ref_sum_2{i} = zeros(sizes(i, :));
            buff_dist_sum_1{i} = zeros(sizes(i, :));
            buff_dist_sum_2{i} = zeros(sizes(i, :));
            buff_cross_sum{i} = zeros(sizes(i, :));
            
        end
        
%         disp(size(buff_ref{2}))
        for i = 1:kt-1
            buff_ref{1}(:,:,i+1) = double(rgb2gray(readFrame(v_ref)));
            buff_dist{1}(:,:,i+1) = double(rgb2gray(readFrame(v_dist)));

            for level = 2:levels
                temp_ref_frame = convn(buff_ref{level-1}(:,:,i+1), downsample_window, 'valid');
                buff_ref{level}(:,:,i+1) = temp_ref_frame(1:2:end, 1:2:end);
                
                temp_dist_frame = convn(buff_dist{level-1}(:,:,i+1), downsample_window, 'valid');
                buff_dist{level}(:,:,i+1) = temp_dist_frame(1:2:end, 1:2:end);
            end

            
%             for level = 1:levels
%                 buff_ref_sum_1{level} = buff_ref_sum_1{level} +  buff_ref{level}(:,:,i+1);
%                 buff_dist_sum_1{level} = buff_dist_sum_1{level} +  temp_dist_frame;
%                 
%                 buff_ref_sum_2{level} = buff_ref_sum_2{level} +  temp_ref_frame.^2;
%                 buff_dist_sum_2{level} = buff_dist_sum_2{level} +  temp_dist_frame.^2;
%                 
%                 buff_cross_sum{level} = buff_cross_sum{level} + temp_ref_frame .* temp_dist_frame;
%                 
%                 temp_ref_frame = convn(temp_ref_frame, downsample_window, 'valid');
%                 temp_dist_frame = convn(temp_dist_frame, downsample_window, 'valid');
%             end
        end
        if kt == 1
            i = 0;
        else
            for level = 1:levels
                
%                 disp(size(buff_ref{level}))
                buff_ref_sum_1{level} = sum(buff_ref{level}, 3);
                buff_dist_sum_1{level} = sum(buff_dist{level}, 3);

                buff_ref_sum_2{level} = sum(buff_ref{level}.^2, 3);
                buff_dist_sum_2{level} = sum(buff_dist{level}.^2, 3);

                buff_cross_sum{level} = sum(buff_ref{level}.*buff_dist{level}, 3);
            end
        end
            
%         disp(size(buff_ref_sum_1{1}))
        while hasFrame(v_ref) && hasFrame(v_dist)
            i = i + 1;
            temp_ref = double(rgb2gray(readFrame(v_ref)));
            temp_dist = double(rgb2gray(readFrame(v_dist)));
            
%             temp_ref_frame = buff_ref(:,:,mod(i,kt) + 1);
%             temp_dist_frame = buff_dist(:,:,mod(i,kt) + 1);
            
%             buff_ref_sum_1 = buff_ref_sum_1 -  buff_ref(:,:,mod(i+1,kt) + 1) +  buff_ref(:,:,mod(i,kt) + 1);
%             buff_ref_sum_2 = buff_ref_sum_2 -  buff_ref(:,:,mod(i+1,kt) + 1).^2 +  buff_ref(:,:,mod(i,kt) + 1).^2;
%             
%             buff_dist_sum_1 = buff_dist_sum_1 -  buff_dist(:,:,mod(i+1,kt) + 1) +  buff_dist(:,:,mod(i,kt) + 1);
%             buff_dist_sum_2 = buff_dist_sum_2 -  buff_dist(:,:,mod(i+1,kt) + 1).^2 +  buff_dist(:,:,mod(i,kt) + 1).^2;
%             
%             buff_cross_sum = buff_cross_sum - buff_ref(:,:,mod(i+1,kt) + 1).* buff_dist(:,:,mod(i+1,kt) + 1) + buff_ref(:,:,mod(i,kt) + 1) .* buff_dist(:,:,mod(i,kt) + 1);
            
            for level = 1:levels-1

%                 disp(size(buff_ref{level}))
                buff_ref_sum_1{level} = buff_ref_sum_1{level} -  buff_ref{level}(:,:,mod(i,kt) + 1) +  temp_ref;
                buff_ref_sum_2{level} = buff_ref_sum_2{level} -  buff_ref{level}(:,:,mod(i,kt) + 1).^2 +  temp_ref.^2;

                buff_dist_sum_1{level} = buff_dist_sum_1{level} -  buff_dist{level}(:,:,mod(i,kt) + 1) +  temp_dist;
                buff_dist_sum_2{level} = buff_dist_sum_2{level} -  buff_dist{level}(:,:,mod(i,kt) + 1).^2 +  temp_dist.^2;

                buff_cross_sum{level} = buff_cross_sum{level} - buff_ref{level}(:,:,mod(i,kt) + 1).* buff_dist{level}(:,:,mod(i,kt) + 1) + temp_ref.*temp_dist;

%                 mu_ref_local = convn(temp_buff_ref, avg_window, 'valid');
%                 mu_dist_local = convn(temp_buff_dist, avg_window, 'valid');
% 
%                 mu_sq_ref_local = mu_ref_local.^2;
%                 mu_sq_dist_local = mu_dist_local.^2;
% 
%                 var_ref_local = convn(temp_buff_ref.^2, avg_window, 'valid') - mu_sq_ref_local;
%                 var_dist_local = convn(temp_buff_dist.^2, avg_window, 'valid') - mu_sq_dist_local;
% 
%                 cov_local = convn(temp_buff_ref .* temp_buff_dist, avg_window, 'valid') - mu_ref_local.* mu_dist_local;
%                 
%                 ssim_temp(level) = mean(mean((2 .* cov_local + C2) ./ (var_ref_local + var_dist_local + C2)));
                
                ssim_temp(level) = ssim_buff(buff_ref_sum_1{level}, buff_ref_sum_2{level}, buff_dist_sum_1{level}, buff_dist_sum_2{level}, buff_cross_sum{level}, k_size, K1, K2);
                
                buff_ref{level}(:,:,mod(i,kt) + 1) = temp_ref;
                buff_dist{level}(:,:,mod(i,kt) + 1) = temp_dist;
%                 temp_buff_ref = convn(temp_buff_ref, downsample_window, 'valid');
%                 temp_buff_dist = convn(temp_buff_dist, downsample_window, 'valid');
%                 
%                 temp_buff_ref = temp_buff_ref(1:2:end, 1:2:end, :);
%                 temp_buff_dist = temp_buff_dist(1:2:end, 1:2:end, :);

                temp_ref_temp = convn(temp_ref, downsample_window, 'valid');
                temp_ref = temp_ref_temp(1:2:end, 1:2:end);
                
                temp_dist_temp = convn(temp_dist, downsample_window, 'valid');
                temp_dist = temp_dist_temp(1:2:end, 1:2:end);
            end
            
            buff_ref_sum_1{levels} = buff_ref_sum_1{levels} -  buff_ref{levels}(:,:,mod(i,kt) + 1) +  temp_ref;
            buff_ref_sum_2{levels} = buff_ref_sum_2{levels} -  buff_ref{levels}(:,:,mod(i,kt) + 1).^2 +  temp_ref.^2;

            buff_dist_sum_1{levels} = buff_dist_sum_1{levels} -  buff_dist{levels}(:,:,mod(i,kt) + 1) +  temp_dist;
            buff_dist_sum_2{levels} = buff_dist_sum_2{levels} -  buff_dist{levels}(:,:,mod(i,kt) + 1).^2 +  temp_dist.^2;

            buff_cross_sum{levels} = buff_cross_sum{levels} - buff_ref{levels}(:,:,mod(i,kt) + 1).* buff_dist{levels}(:,:,mod(i,kt) + 1) + temp_ref.*temp_dist;

%                 mu_ref_local = convn(temp_buff_ref, avg_window, 'valid');
%                 mu_dist_local = convn(temp_buff_dist, avg_window, 'valid');
% 
%                 mu_sq_ref_local = mu_ref_local.^2;
%                 mu_sq_dist_local = mu_dist_local.^2;
% 
%                 var_ref_local = convn(temp_buff_ref.^2, avg_window, 'valid') - mu_sq_ref_local;
%                 var_dist_local = convn(temp_buff_dist.^2, avg_window, 'valid') - mu_sq_dist_local;
% 
%                 cov_local = convn(temp_buff_ref .* temp_buff_dist, avg_window, 'valid') - mu_ref_local.* mu_dist_local;
%                 
%                 ssim_temp(level) = mean(mean((2 .* cov_local + C2) ./ (var_ref_local + var_dist_local + C2)));

            ssim_temp(levels) = ssim_buff(buff_ref_sum_1{levels}, buff_ref_sum_2{levels}, buff_dist_sum_1{levels}, buff_dist_sum_2{levels}, buff_cross_sum{levels}, k_size, K1, K2, 'full');

            buff_ref{levels}(:,:,mod(i,kt) + 1) = temp_ref;
            buff_dist{levels}(:,:,mod(i,kt) + 1) = temp_dist;

%             ssim_temp(levels) = mean(mean(((2 .* mu_ref_local .* mu_dist_local + C1) .* (2 .* cov_local + C2)) ./ ((mu_sq_ref_local + mu_sq_dist_local + C1) .* (var_ref_local + var_dist_local + C2))));
            mssim(i - kt + 1) = prod(ssim_temp .^ exponents(1:levels));
        end
        
%         mssim = mssim / (N - kt + 1);
    end
end