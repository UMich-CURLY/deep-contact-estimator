% clear;
% load('/media/justin/CURLY_Justin/DockerFolder/code/deep-contact-estimator/inference_results/0316_ws150_lr1e-4_2block_drop_out_best_val_acc.mat')

% plot(1:size(contacts,1),contacts(:,1),'*');
% hold on
% plot(1:size(ground_truth,1),ground_truth(:,1),'*');
% 
% legend('prediction','groundtruth')

contacts_est = logical(contacts_est);
contacts_gt = logical(contacts_gt);
diff = contacts_est ~= contacts_gt;

for i = 1:4
    figure(i)
    
    plot(imu_time,p(:,3*i));
    hold on
    plot(imu_time(contacts_est(:,i)),p(contacts_est(:,i),3*i), "g*");
    hold on
    plot(imu_time(contacts_gt(:,i)),p(contacts_gt(:,i),3*i), "r*");
    
%     legend("foot_pos","contacts");
    legend("foot_pos","contacts","gt");
    title("foot\_position"+i)
end

% for i = 1:4
%     figure(i+4)
%     
%     plot(1:size(p,1),p(:,3*i));
%     hold on
%     plot(find(diff(:,i)),p(diff(:,i),3*i), "b*");
%     
% %     legend("foot_pos","contacts");
%     legend("foot_pos","diff");
%     title("foot\_position"+i)
% end