function [out] = main
% output: 
% out.Error: Error
% out.reError: relative Error
% out.fval: objective function value
% out.iter: iteration number;
% out.time: running time
% Reference:
% Semi-Supervised Discriminant Multi-Manifold Analysis for Action Recognition, TNNLS2019
 
%% IDT+FV
clc;clear;  
opts=load('../data/IDTFV.mat'); 
disp('========================IDT+FV+ALS========================');
out = ALS(opts)
i=1:out.iter;
set(gcf,'color','w');
plot(i,out.fval(i),'bx-','LineWidth',2);
hold on; 
disp('========================IDT+FV+SPG========================');
out = SPG(opts)
i=1:out.iter;
plot(i,out.fval(i),'mp-','LineWidth',2);
hold on;

%% TDD+FV
clear;
opts=load('../data/TDDFV.mat'); 
disp('========================TDD+FV+ALS========================');
out = ALS(opts)
i=1:out.iter;
plot(i,out.fval(i),'ro-','LineWidth',2);
hold on;
disp('========================TDD+FV+SPG========================');
out = SPG(opts)
i=1:out.iter;
plot(i,out.fval(i),'g^-','LineWidth',2);
legend('IDT+FV+ALS','IDT+FV+SPG','TDD+FV+ALS','TDD+FV+SPG');
set(gca, 'linewidth',2,'Fontsize',15);
xlabel('Iteration Number')
ylabel('Objective Function Value')
hold on;

%% SAVE
% print -f1 -r300 -djpeg myfigure % ok
saveas(gcf, 'plot.jpg');
end