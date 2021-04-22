% clear;
% load('/media/justin/CURLY_Justin/DockerFolder/code/deep-contact-estimator/inference_results/0316_ws150_lr1e-4_2block_drop_out_best_val_acc.mat')

% plot(1:size(contacts,1),contacts(:,1),'*');
% hold on
% plot(1:size(ground_truth,1),ground_truth(:,1),'*');
% 
% legend('prediction','groundtruth')
%%
contacts_est = logical(contacts_est);
contacts_gt = logical(contacts_gt);
diff = contacts_est ~= contacts_gt;

%%
for i = 1:4
    figure(i)
    
    plot(1:size(p),p(:,3*i));
    hold on
    plot(find(contacts_est(:,i)),p(contacts_est(:,i),3*i), "g*");
    hold on
    plot(find(contacts_gt(:,i)),p(contacts_gt(:,i),3*i), "r*");
    
%     legend("foot_pos","contacts");
    legend("foot_pos","contacts","gt");
    title("foot\_position"+i)
end

%%
for i = 1:4
    figure(i)
    
    plot(1:size(F),F(:,3*i),'Color',[0.4940 0.1840 0.5560]);
    hold on
    plot(find(contacts_est(:,i)),F(contacts_est(:,i),3*i),'o','MarkerEdgeColor',[0.9290 0.6940 0.1250]);
    hold on
    plot(find(contacts_gt(:,i)),F(contacts_gt(:,i),3*i),'.','MarkerEdgeColor',[0 0.4470 0.7410]);
    
%     legend("foot_pos","contacts");
    legend("GRF","contacts","gt");
    title("foot\_position"+i)
end


%%
for i = 1:4
    figure(i+4)
    
    plot(1:size(contacts_est(:,i)),contacts_est(:,i));
    hold on
    plot(1:size(contacts_gt(:,i)),contacts_gt(:,i));
    
%     legend("foot_pos","contacts");
    ylim([-0.5 1.5])
    legend("estimated contacts","gt contacts");
    title("foot\_position"+i)
end

%%
for i = 1:4
    figure(i+8)
    
    plot(1:size(p,1),p(:,3*i));
    hold on
    plot(find(diff(:,i)),p(diff(:,i),3*i), "b*");
    
%     legend("foot_pos","contacts");
    legend("foot_pos","diff");
    title("foot\_position"+i)
end