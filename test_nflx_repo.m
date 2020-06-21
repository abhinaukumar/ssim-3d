clc;
clearvars;

db_dir = "/media/abhinau/ext_hard_drive/databases/NFLX/NFLX-Repo/";
ref_file_list = dir(db_dir + "ref/" + "rgb/");
dist_file_list = dir(db_dir + "dis/" + "rgb/");

n_ref_files = length(ref_file_list);
n_dist_files = length(dist_file_list);

ks = [1, 3, 5, 7, 10, 15, 20];
n_ks = length(ks);

% ssims = cell([n_folders*15,1]);
multiscale_mssim = zeros([n_dist_files,n_ks]);
mssim = zeros([n_dist_files,n_ks]);

ssim_pccs = zeros([n_ks, 1]);
ssim_sroccs = zeros([n_ks, 1]);
ssim_rmses = zeros([n_ks, 1]);

msssim_pccs = zeros([n_ks, 1]);
msssim_sroccs = zeros([n_ks, 1]);
msssim_rmses = zeros([n_ks, 1]);

load('/home/abhinau/projects/fb_live/ssim-comparison/data/nflx_repo_scores.mat');
scores = scores.';
scores = (scores - min(scores)) ./ (max(scores) - min(scores));

i_dist = 3;

for i_ref = 3:n_ref_files
    ref_filename = split(ref_file_list(i_ref).name, '_');
    ref_filename = ref_filename{1};
    
    v_ref = VideoReader(db_dir + "ref/" + "rgb/" + ref_file_list(i_ref).name);
    
    while(contains(dist_file_list(i_dist).name, ref_filename))
        v_dist = VideoReader(db_dir + "dis/" + "rgb/" + dist_file_list(i_dist).name);
        
        for i_k = 1:n_ks
            disp([i_ref-2, i_dist-2, i_k])
            tic;

            mssim(i_dist-2, i_k) = mean(ssim3d(v_ref, v_dist, [11, 11, ks(i_k)], 0.01, 0.03));

            v_ref.CurrentTime = 0;
            v_dist.CurrentTime = 0;

            multiscale_mssim(i_dist-2, i_k) = mean(msssim2_1d(v_ref, v_dist, [11, 11, ks(i_k)], 5 , 0.01, 0.03));

            v_ref.CurrentTime = 0;
            v_dist.CurrentTime = 0;

            toc;
        end
        i_dist = i_dist + 1;

        v_ref.CurrentTime = 0;
    end
end

mssim = real(mssim);
multiscale_mssim = real(multiscale_mssim);

for i_k = 1:n_ks
    modelfun = @(b,x)(b(1) .* (0.5 - 1./(1 + exp(b(2)*(x - b(3))))) + b(4) .* x + b(5));
    b_fit = nlinfit(mssim(:,i_k), scores,modelfun,[1,1,1,1,1]);
    quality = modelfun(b_fit,mssim(:,i_k));
    [ssim_sroccs(i_k), ~] = corr(quality,scores,'Type','Spearman');
    [ssim_pccs(i_k), ~] = corr(quality,scores,'Type','Pearson');
    ssim_rmses(i_k) = sqrt(mean((quality - scores).^2));
    
    modelfun = @(b,x)(b(1) .* (0.5 - 1./(1 + exp(b(2)*(x - b(3))))) + b(4) .* x + b(5));
    b_fit = nlinfit(multiscale_mssim(:,i_k), scores,modelfun,[1,1,1,1,1]);
    quality = modelfun(b_fit,multiscale_mssim(:,i_k));
    [msssim_sroccs(i_k), ~] = corr(quality,scores,'Type','Spearman');
    [msssim_pccs(i_k), ~] = corr(quality,scores,'Type','Pearson');
    msssim_rmses(i_k) = sqrt(mean((quality - scores).^2));
end

figure;
plot(ks, ssim_pccs, 'b-o')
hold on
plot(ks, msssim_pccs, 'g-o')
plot(ks, ssim_sroccs, 'b-x')
hold on
plot(ks, msssim_sroccs, 'g-x')
legend(["SSIM - PCC", "MS-SSIM - PCC", "SSIM - SROCC", "MS-SSIM - SROCC"])
xlabel("K_t")
ylabel("Correlation")