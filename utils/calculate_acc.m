num_data = size(F_contacts,1);
start_idx = floor(0.15*num_data);
end_idx = floor(0.3*num_data);

%%
F_correct = (F_contacts==contacts);
F_correct = F_correct(start_idx:end_idx,:);
F_acc = sum(F_correct,1)/size(F_correct,1);

gait_correct = (gait_cycle_contacts==contacts);
gait_correct = gait_correct(start_idx:end_idx,:);
gait_acc = sum(gait_correct,1)/size(gait_correct,1);



%%
% contacts_est = logical(contacts_est);
% contacts_gt = logical(contacts_gt);
% contact_correct = (contacts_est==contacts_gt);
% contact_acc = sum(contact_correct,1)/size(contacts_est,1);