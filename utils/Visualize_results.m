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
for i=1:4
    v_mag(:,i) = sqrt(sum(v(:,3*(i-1)+1:3*i).^2,2));
end

%%
plot(imu_time(1,:),p(:,1));
hold on
plot(imu_time(1,:),p(:,2));
hold on
plot(imu_time(1,:),p(:,3));

%%
plot(imu_time(1,:),v(:,1));
hold on
plot(imu_time(1,:),v(:,2));
hold on
plot(imu_time(1,:),v(:,3));

%%
plot(imu_time(1,:),imu_acc(:,1));
hold on
plot(imu_time(1,:),imu_acc(:,2));
hold on
plot(imu_time(1,:),imu_acc(:,3));

%% plot with imu
for i = 1:3
    figure(i)
    
    plot(1:size(p),imu_acc(:,i));
    hold on
    plot(find(contacts_est(:,i)),imu_acc(contacts_est(:,i),i), "g*");
    hold on
    plot(find(contacts_gt(:,i)),imu_acc(contacts_gt(:,i),i), "r*");
    
%     legend("foot_pos","contacts");
    legend("acceleration","contacts","gt");
    title("acceleration"+i)
end

%% plot with foot position
for i = 1:4
    figure(i)
    
    plot(1:size(p),p(:,3*i));
    hold on
    plot(find(contacts_est(:,i)),p(contacts_est(:,i),3*i), "g*");
    hold on
%     plot(find(contacts_gt(:,i)),p(contacts_gt(:,i),3*i), "r*");
    
%     legend("foot_pos","contacts");
    legend("foot_pos","contacts","gt");
    title("foot\_position"+i)
end

%% plot with foot velocity
for i = 1:4
    figure(i)
    
    plot(1:size(v),v(:,3*i));
    hold on
    plot(find(contacts_est(:,i)),v(contacts_est(:,i),3*i), "g*");
    hold on
    plot(find(contacts_gt(:,i)),v(contacts_gt(:,i),3*i), "r*");
    
%     legend("foot_pos","contacts");
    legend("foot_pos","contacts","gt");
    title("foot\_position"+i)
end

%% plot with foot velocity
for i = 1:4
    figure(i)
    
    plot(imu_time(1,:),v_mag(:,i));
    hold on
    plot(imu_time(1,contacts_est(:,i)),v_mag(contacts_est(:,i),i), "g*");
    hold on
    plot(imu_time(1,contacts_gt(:,i)),v_mag(contacts_gt(:,i),i), "r*");
    
%     legend("foot_pos","contacts");
    legend("foot_velocity","contacts","gt");
    title("foot\_velocity"+i)
end

%% plot with GRF
for i = 1:4
    figure(i)
    
    plot(1:size(F),F(:,3*i),'Color',[0.4940 0.1840 0.5560]);
    hold on
    plot(find(contacts_est(:,i)),F(contacts_est(:,i),3*i),'o','MarkerEdgeColor',[0.9290 0.6940 0.1250]);
    hold on
    plot(find(contacts_gt(:,i)),F(contacts_gt(:,i),3*i),'.','MarkerEdgeColor',[0 0.4470 0.7410]);
    
%     legend("foot_pos","contacts");
    legend("GRF","contacts","gt");
    title("GRF"+i)
end


%% plot only contacts
for i = 1:4
    figure(i+4)
    
    plot(1:size(contacts_est(:,i)),contacts_est(:,i));
    hold on
    plot(1:size(contacts_gt(:,i)),contacts_gt(:,i));
    
%     legend("foot_pos","contacts");
    ylim([-0.5 1.5])
    legend("estimated contacts","gt contacts");
    title("Contacts"+i)
end

%% plot diff
for i = 1:4
    figure(i+8)
    
    plot(1:size(p,1),p(:,3*i));
    hold on
    plot(find(diff(:,i)),p(diff(:,i),3*i), "b*");
    
%     legend("foot_pos","contacts");
    legend("foot_pos","diff");
    title("foot\_position"+i)
end


%% 
start_idx = 3000;
end_idx = 4000;
range = start_idx:end_idx;
l = ['RF','LF','RH','LH'];

for i = 4
    figure(i+12);
    
    
    gt_idx = find(contacts_gt(range,i));
    est_idx = find(contacts_est(range,i));
    
    subplot(4,1,1);
%     plot(imu_time(range),v(range,3*i-2),'linewidth',2,'Color',[0 0.4470 0.7410]);
%     hold on
%     plot(imu_time(range),v(range,3*i-1),'linewidth',2,'Color',[0.4660 0.6740 0.1880]);
%     hold on
    plot(imu_time(range),p(range,3*i),'linewidth',2,'Color',[0.4660 0.6740 0.1880]);
%     hold on
%     plot(imu_time(range),v(range,3*i),'linewidth',2,'Color',[0.4940 0.1840 0.5560]);
    for j = 1:size(gt_idx)
        idx = gt_idx(j)+start_idx;
        if idx >= start_idx && idx<=end_idx
            patch('XData',imu_time([idx,idx+1,idx+1,idx]), 'YData',[max(ylim) max(ylim) min(ylim) min(ylim)],'FaceColor',[0.9290 0.6940 0.1250],'FaceAlpha',0.3, 'EdgeColor', 'none')
        end
    end
    title("Foot Velocity with Ground Truth Contact",'FontSize',14)
    legend({"v_{z}",'Ground Truth Contacts'},'FontSize',14);
%     legend({'v_{z}','Ground Truth Contacts'},'FontSize',14)
    ylabel('Velocity (m/s)');
    
    subplot(4,1,2);
%     plot(imu_time(range),v(range,3*i-2),'linewidth',2,'Color',[0 0.4470 0.7410]);
%     hold on
%     plot(imu_time(range),v(range,3*i-1),'linewidth',2,'Color',[0.4660 0.6740 0.1880]);
%     hold on
    plot(imu_time(range),p(range,3*i),'linewidth',2,'Color',[0.4660 0.6740 0.1880]);
%     hold on
%     plot(imu_time(range),v(range,3*i),'linewidth',2,'Color',[0.4940 0.1840 0.5560]);
    for j = 1:size(est_idx)
        idx = est_idx(j)+start_idx;
        if idx >= start_idx && idx<=end_idx
            patch('XData',imu_time([idx,idx+1,idx+1,idx]), 'YData',[max(ylim) max(ylim) min(ylim) min(ylim)],'FaceColor',[0 0.4470 0.7410],'FaceAlpha',0.3, 'EdgeColor', 'none')
        end
    end
    title("Foot Velocity with Estimated Contact",'FontSize',14)
    legend({"v_{z}",'Estimated Contacts'}, 'FontSize', 14);
%     legend({'v_{z}','Estimated Contacts'},'FontSize',14)
    ylabel('Velocity (m/s)');
    
    subplot(4,1,3);
    plot(imu_time(range),F(range,3*i-2),'linewidth',2,'Color',[0 0.4470 0.7410]);
    hold on
    plot(imu_time(range),F(range,3*i-1),'linewidth',2,'Color',[0.4660 0.6740 0.1880]);
    hold on
    plot(imu_time(range),F(range,3*i),'linewidth',2,'Color',[0.8500 0.3250 0.0980]);
    title("Ground Reaction Force with Ground Truth Contacts",'FontSize',14)
    for j = 1:size(gt_idx)
        idx = gt_idx(j)+start_idx;
        if idx >= start_idx && idx<=end_idx
            patch('XData',imu_time([idx,idx+1,idx+1,idx]), 'YData',[max(ylim) max(ylim) min(ylim) min(ylim)],'FaceColor',[0.9290 0.6940 0.1250],'FaceAlpha',0.3, 'EdgeColor', 'none')
        end
    end
    legend({'F_{x}','F_{y}','F_{z}','Ground Truth Contacts'},'FontSize',14);
    ylabel('Force (N)');
    
    subplot(4,1,4);
    plot(imu_time(range),F(range,3*i-2),'linewidth',2,'Color',[0 0.4470 0.7410]);
    hold on
    plot(imu_time(range),F(range,3*i-1),'linewidth',2,'Color',[0.4660 0.6740 0.1880]);
    hold on
    plot(imu_time(range),F(range,3*i),'linewidth',2,'Color',[0.8500 0.3250 0.0980]);
    title("Ground Reaction Force with Estimated Contacts",'FontSize',14)
    for j = 1:size(est_idx)
        idx = est_idx(j)+start_idx;
        if idx >= start_idx && idx<=end_idx
            patch('XData',imu_time([idx,idx+1,idx+1,idx]), 'YData',[max(ylim) max(ylim) min(ylim) min(ylim)],'FaceColor',[0 0.4470 0.7410],'FaceAlpha',0.3, 'EdgeColor', 'none')
        end
    end
    legend({'F_{x}','F_{y}','F_{z}','Estimated Contacts'},'FontSize',14);
    ylabel('Force (N)');
    xlabel('Time (s)');
%     subplot(3,1,3);
%     plot(imu_time(range),contacts_est(range,i));
%     hold on
%     plot(imu_time(range),contacts_gt(range,i));
%     title("Contacts")
%     ylim([-0.5 1.5])
%     legend("estimated contacts","gt contacts");
end

