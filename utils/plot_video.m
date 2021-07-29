
%%
fr = 200;
T_start = 32;
T_end = 34;
l=3;

data_name = 'forest';

p_yaxis_max = -0.22;
p_yaxis_min = -0.32;

v_yaxis_max = 2;
v_yaxis_min = -2;

f_yaxis_max = 100;
f_yaxis_min = -20;

%%
contacts_est = logical(contacts_est);
contacts_gt = logical(contacts_gt);
diff = contacts_est ~= contacts_gt;
est_idx = find(contacts_est(:,l));

%%
VideoName = strcat('contact_estimation_',data_name,'_position');
Video = VideoWriter(VideoName);          % Create the video
Video.FrameRate = fr;                  % Set frame rate

frame = figure(1);clf
set(frame,'Units','centimeters','position',[1 1 30 25]);
open(Video);


[start_idx_list, imu_t_start] = knnsearch(imu_time(1,:)',T_start);
start_idx = start_idx_list(1);
i = 1;
cur_idx = start_idx+(1000/200)*(i);
while imu_time(cur_idx) < T_end
    clf;
    
    range = start_idx:cur_idx;
    
    plot(imu_time(range),p(range,3*l),'linewidth',2,'Color',[0.4660 0.6740 0.1880]);
    
    axis([T_start-0.1 T_end+0.1 p_yaxis_min p_yaxis_max]);
    
    frame.Position = [1 1 30 25];
    hold on
    
%     plot(imu_time(find(contacts_est(range,l))+start_idx), p(contacts_est(range,l),3*l), "g*");
    for j = 1:size(est_idx)
        idx = est_idx(j);
        if idx >= start_idx+(1000/200) && idx<=cur_idx
            patch('XData',imu_time([idx,idx+1,idx+1,idx]), 'YData',[max(ylim) max(ylim) min(ylim) min(ylim)],'FaceColor',[0 0.4470 0.7410],'FaceAlpha',0.3, 'EdgeColor', 'none')
            frame.Position = [1 1 30 25];
        end
    end
    title("Foot Position with Estimated Contact",'FontSize',14)
    legend({"p_{z}",'Estimated Contacts'}, 'FontSize', 14);
    ylabel('Position (m)');
    xlabel('Time (sec)');
    

    frame.Position = [1 1 30 25];
    writeVideo(Video, getframe(frame));
    
    i = i + 1;
    cur_idx = start_idx+(1000/200)*(i);
end

close(Video);

%%
VideoName = strcat('contact_estimation_',data_name,'_velocity');
Video = VideoWriter(VideoName);          % Create the video
Video.FrameRate = fr;                  % Set frame rate

frame = figure(1);clf
set(frame,'Units','centimeters','position',[1 1 30 25]);
open(Video);


[start_idx_list, imu_t_start] = knnsearch(imu_time(1,:)',T_start);
start_idx = start_idx_list(1);
i = 1;
cur_idx = start_idx+(1000/200)*(i);
while imu_time(cur_idx) < T_end
    clf;
    
    range = start_idx:cur_idx;
    
    plot(imu_time(range),v(range,3*l),'linewidth',2,'Color',[0.4940 0.1840 0.5560]);
    
    axis([T_start-0.1 T_end+0.1 v_yaxis_min v_yaxis_max]);
    
    frame.Position = [1 1 30 25];
    hold on
    
%     plot(imu_time(find(contacts_est(range,l))+start_idx), p(contacts_est(range,l),3*l), "g*");
    for j = 1:size(est_idx)
        idx = est_idx(j);
        if idx >= start_idx+(1000/200) && idx<=cur_idx
            patch('XData',imu_time([idx,idx+1,idx+1,idx]), 'YData',[max(ylim) max(ylim) min(ylim) min(ylim)],'FaceColor',[0 0.4470 0.7410],'FaceAlpha',0.3, 'EdgeColor', 'none')
            frame.Position = [1 1 30 25];
        end
    end
    title("Foot Velocity with Ground Truth Contact",'FontSize',14)
    legend({"v_{z}",'Estimated Contacts'},'FontSize',14);
    ylabel('Velocity (m/s)');
    xlabel('Time (sec)');
    

    frame.Position = [1 1 30 25];
    writeVideo(Video, getframe(frame));
    
    i = i + 1;
    cur_idx = start_idx+(1000/200)*(i);
end

close(Video);

%% force
% VideoName = strcat('contact_estimation_',data_name,'_force');
% Video = VideoWriter(VideoName);          % Create the video
% Video.FrameRate = fr;                  % Set frame rate
% 
% frame = figure(1);clf
% set(frame,'Units','centimeters','position',[1 1 30 25]);
% open(Video);
% 
% 
% [start_idx_list, imu_t_start] = knnsearch(imu_time(1,:)',T_start);
% start_idx = start_idx_list(1);
% i = 1;
% cur_idx = start_idx+(1000/200)*(i);
% while imu_time(cur_idx) < T_end
%     clf;
%     
%     range = start_idx:cur_idx;
%     plot(imu_time(range),F(range,3*l-2),'linewidth',2,'Color',[0 0.4470 0.7410]);
%     hold on
%     plot(imu_time(range),F(range,3*l-1),'linewidth',2,'Color',[0.4660 0.6740 0.1880]);
%     hold on
%     plot(imu_time(range),F(range,3*l),'linewidth',2,'Color',[0.8500 0.3250 0.0980]);
%     axis([T_start-0.1 T_end+0.1 f_yaxis_min f_yaxis_max]);
%     
%     frame.Position = [1 1 30 25];
%     hold on
%     
%     
% %     plot(imu_time(find(contacts_est(range,l))+start_idx), p(contacts_est(range,l),3*l), "g*");
%     for j = 1:size(est_idx)
%         idx = est_idx(j);
%         if idx >= start_idx+(1000/200) && idx<=cur_idx
%             patch('XData',imu_time([idx,idx+1,idx+1,idx]), 'YData',[max(ylim) max(ylim) min(ylim) min(ylim)],'FaceColor',[0 0.4470 0.7410],'FaceAlpha',0.3, 'EdgeColor', 'none')
%             frame.Position = [1 1 30 25];
%         end
%     end
%     title("Ground Reaction Force with Estimated Contacts",'FontSize',14)
%     legend({'F_{x}','F_{y}','F_{z}','Estimated Contacts'},'FontSize',14);
%     ylabel('Force (N)');
%     xlabel('Time (s)');
%     
% 
%     frame.Position = [1 1 30 25];
%     writeVideo(Video, getframe(frame));
%     
%     i = i + 1;
%     cur_idx = start_idx+(1000/200)*(i);
% end
% 
% close(Video);