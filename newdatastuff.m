%%
close all;
clear all;
data = open('180829_Lasso_Results.mat');
reaches = data.Reach_reg;
mkdir('DReachNew');
%%
figure; hold on;

for i = 1:length(reaches)
    r = reaches(i).rast_kin(300:700,:);
    %r = r(578:618,:);
    r(:,1) = r(:,1) - r(1,1);
    %plot(r(:,1),r(:,2))
    %plot(r(:,2))
    csvwrite('DReachNew/trajectory_' + string(i) + '.csv', r);
end


figure; hold on
plot(pos_x)