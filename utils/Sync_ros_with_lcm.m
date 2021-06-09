%%
% 5 short
lcm_start_time = 10.5162;
lcm_end_time = 57.1235;



%%

robot_time = robot_raw.joint_state.bag_time-robot_raw.joint_state.bag_time(1);
t_diff = robot_time(r_idx) - control_time(l_idx);

new_robot_time = robot_time - t_diff;

% [start_idx, end_idx, robot_t_cropped, robot_q2] = crop_data(robot_time,robot.joint_state.position2,)

r_start_time_ros = robot_raw.joint_state.bag_time(1) + t_diff + control_time(150)
r_end_time_ros = r_start_time_ros + l_end_time

%%
figure(2)
plot(new_robot_time,robot_raw.joint_state.position2);
hold on
plot(control_time,q(:,3));



%%

figure(3)
plot(robot.joint_state.bag_time-robot.joint_state.bag_time(1),robot.joint_state.position2);
hold on
plot(0.001*(1:size(q(:,3))),q(:,3));





function [start_idx, end_idx, t, x] = crop_data(t_init, x_init, start_t, end_t)
    start_idx = knnsearch(t_init', start_t);
    end_idx = knnsearch(t_init', end_t);
    x = x_init(start_idx:end_idx, :);
    t = t_init(start_idx:end_idx) - t_init(start_idx);
end