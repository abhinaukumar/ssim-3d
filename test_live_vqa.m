clc;
clearvars;

refs = ["pa", "rb", "rh", "tr", "st", "sf", "bs", "sh", "mc", "pr"];
fps = ["25fps", "25fps", "25fps", "25fps", "25fps", "25fps", "25fps", "50fps", "50fps", "50fps"];

n_folders = length(refs);
db_dir = "/media/abhinau/ext_hard_drive/databases/LIVE_VQA/";

ks = [1, 3, 5, 7, 10, 15, 20];
n_ks = length(ks);

% ssims = cell([n_folders*15,1]);
multiscale_mssim = zeros([n_folders*15,n_ks]);
mssim = zeros([n_folders*15,n_ks]);

ssim_pccs = zeros([n_ks, 1]);
ssim_sroccs = zeros([n_ks, 1]);

msssim_pccs = zeros([n_ks, 1]);
msssim_sroccs = zeros([n_ks, 1]);

load('/home/abhinau/projects/fb_live/ssim-comparison/data/live_vqa_scores.mat');
scores = scores.';
    

k = 1;
for i_ref = 1:n_folders
    ref = refs(i_ref);
    v_ref = VideoReader(db_dir + "videos/" + ref + "_Folder/" + "rgb/" + ref + "1" + "_" + fps(i_ref) + ".mp4");

    for i_dist = 2:16

        v_dist = VideoReader(db_dir + "videos/" + ref + "_Folder/" + "rgb/" + ref + int2str(i_dist) + "_" + fps(i_ref) + ".mp4");

        for i_k = 1:n_ks
            disp([i_ref, i_dist, i_k])
            tic;

            mssim(k, i_k) = mean(ssim3d(v_ref, v_dist, [11, 11, ks(i_k)], 0.01, 0.03));

            v_ref.CurrentTime = 0;
            v_dist.CurrentTime = 0;

            multiscale_mssim(k, i_k) = mean(msssim2_1d(v_ref, v_dist, [11, 11, ks(i_k)], 5 , 0.01, 0.03));

            v_ref.CurrentTime = 0;
            v_dist.CurrentTime = 0;

            toc;
        end
        k = k + 1;

        v_ref.CurrentTime = 0;
    end
end

%     for i = 1:n_folders*15
%         mssim(i) = mean(ssims{i});
%     end
mssim = real(mssim);
multiscale_mssim = real(multiscale_mssim);

for i_k = 1:n_ks
    modelfun = @(b,x)(b(1) .* (0.5 - 1./(1 + exp(b(2)*(x - b(3))))) + b(4) .* x + b(5));
    b_fit = nlinfit(mssim(:,i_k), scores,modelfun,[1,1,1,1,1]);
    quality = modelfun(b_fit,mssim(:,i_k));
    [ssim_sroccs(i_k), ~] = corr(quality,scores,'Type','Spearman');
    [ssim_pccs(i_k), ~] = corr(quality,scores,'Type','Pearson');
    
    modelfun = @(b,x)(b(1) .* (0.5 - 1./(1 + exp(b(2)*(x - b(3))))) + b(4) .* x + b(5));
    b_fit = nlinfit(multiscale_mssim(:,i_k), scores,modelfun,[1,1,1,1,1]);
    quality = modelfun(b_fit,multiscale_mssim(:,i_k));
    [msssim_sroccs(i_k), ~] = corr(quality,scores,'Type','Spearman');
    [msssim_pccs(i_k), ~] = corr(quality,scores,'Type','Pearson');
end