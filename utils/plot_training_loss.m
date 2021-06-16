figure(1)

plot(acc(:,2),acc(:,3),'Color',[0 0.4470 0.7410], 'linewidth',2);
title("Training Accuracy",'FontSize',14)
xlabel("Epoch",'FontSize',14)
ylabel("Accuracy (%)",'FontSize',14)

%%

figure(2)

plot(loss(:,2),loss(:,3),'Color',[0 0.4470 0.7410], 'linewidth',2);
title("Training Loss",'FontSize',20)
xlabel("Epoch",'FontSize',20)
ylabel("Loss",'FontSize',20)

%%
figure(3)
plot(loss(:,2),loss(:,3),'Color',[0 0.4470 0.7410], 'linewidth',2);
hold on
plot(acc(:,2),acc(:,3),'Color',[0.8500, 0.3250, 0.0980], 'linewidth',2);
title("Training Loss and Accuracy",'FontSize',20)
legend("Traning Loss","Training Accuracy")
xlabel("Epoch",'FontSize',20)
ylabel("Loss/Accuracy",'FontSize',20)
ylim([0 1.1])

%%

figure(4)

plot(loss(:,1),loss(:,2),'Color',[0 0.4470 0.7410], 'linewidth',2);
title("Validation Loss",'FontSize',20)
xlabel("Epoch",'FontSize',20)
ylabel("Loss",'FontSize',20)
