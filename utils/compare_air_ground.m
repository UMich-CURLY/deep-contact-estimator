%%
for i = 1:3
    figure(i)
    plot(imu_time(1,:),v(:,i));
    hold on 
    plot(imu_time_air(1,:),v_air(:,i));
    legend("v_{ground}","v_{air}");
    title("foot velocity comparison "+i);
end

%%

for i = 1:3
    figure(i+3)
    plot(imu_time(1,:),p(:,i));
    hold on 
    plot(imu_time_air(1,:),p_air(:,i));
    legend("p_{ground}","p_{air}");
    title("foot position comparison "+i);
end
%%
for i = 1:3
    figure(i+6)
    plot(imu_time(1,:),q(:,i));
    hold on 
    plot(imu_time_air(1,:),q_air(:,i));
    legend("q_{ground}","q_{air}");
    title("joint encoder comparison "+i);
end

%%
for i = 1:3
    figure(i+6)
    plot(imu_time(1,:),qd(:,i));
    hold on 
    plot(imu_time_air(1,:),qd_air(:,i));
    legend("qd_{ground}","qd_{air}");
    title("joint encoder velocity comparison "+i);
end

%%
for i = 1:3
    figure(i+9)
    plot(imu_time(1,:),imu_acc(:,i));
    hold on 
    plot(imu_time_air(1,:),imu_acc_air(:,i));
    legend("acc_{ground}","acc_{air}");
    title("acceleration comparison "+i);
end

%%
for i = 1:3
    figure(i+12)
    plot(imu_time(1,:),imu_omega(:,i));
    hold on 
    plot(imu_time_air(1,:),imu_omega_air(:,i));
    legend("omega_{ground}","omega_{air}");
    title("angular velocity comparison "+i);
end


%%
for i = 1:3
    figure(i+15)
    plot(imu_time(1,:),tau_est(:,i));
    hold on 
    plot(imu_time_air(1,:),tau_est_air(:,i));
    legend("tau_est_{ground}","tau_est_{air}");
    title("torque estimate comparison "+i);
end

%%
for i = 1:3
    figure(i+18)
    plot(imu_time(1,:),F(:,i));
    hold on 
    plot(imu_time_air(1,:),F_air(:,i));
    legend("F_{ground}","F_{air}");
    title("estimated GRF comparison "+i);
end

