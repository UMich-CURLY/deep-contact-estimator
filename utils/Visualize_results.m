clear;
load('/media/justin/CURLY_Justin/DockerFolder/code/deep-contact-estimator/inference_results/0316_ws150_lr1e-4_2block_drop_out_best_val_acc.mat')

% plot(1:size(contacts,1),contacts(:,1),'*');
% hold on
% plot(1:size(ground_truth,1),ground_truth(:,1),'*');
% 
% legend('prediction','groundtruth')

contacts = logical(contacts);
ground_truth = logical(ground_truth);

for i = 1:4
    figure(i)
    
    plot(1:size(p,1),p(:,3*i));
    hold on
    plot(find(contacts(:,i)),p(contacts(:,i),3*i), "g*");
    hold on
    plot(find(ground_truth(:,i)),p(ground_truth(:,i),3*i), "r*");
    
    legend("foot_pos","contacts","gt");
    title("foot\_position"+i)
end