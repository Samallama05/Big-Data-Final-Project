% function [realdata] = SK_LASSORealData(Reach_reg,Bin10smooth)
clear all;
load 180829_Lasso_Results.mat
% lasso regression model
tic 

Reach_reg(1).rast_tot = [];
 %for i=1:length(Reach_reg)
 trigger = 1;
 for i=1:90
     if Reach_reg(i).exclude == 0
        switch trigger
            case 1
                Reach_reg(1).rast_tot = Reach_reg(i).rast_kin(300:700,:);
                %print = 1
                model_info = importdata('C:\University\Al Borno Lab\merged code\solutions\torque_solution' + string(i) + '.sto').data;
                model_info = model_info(:,[2,3,4,8,9,10]);
                Reach_reg(1).synth_tot = model_info;
                %size(model_info)
                trigger = 0;
            otherwise
                Reach_reg(1).rast_tot = vertcat(Reach_reg(1).rast_tot, Reach_reg(i).rast_kin(300:700,:));
                model_info = importdata('C:\University\Al Borno Lab\merged code\solutions\torque_solution' + string(i) + '.sto').data;
                model_info = model_info(:,[2,3,4,8,9,10]);
                Reach_reg(1).synth_tot = vertcat(Reach_reg(1).synth_tot, model_info);
                size(model_info);
        end
    end
 end
%%
% pull out predictors 
pos_x = Reach_reg(1).rast_tot(:,2);
pos_y = Reach_reg(1).rast_tot(:,3);
pos_z = Reach_reg(1).rast_tot(:,4);
vel = Reach_reg(1).rast_tot(:,5);

vel_x = Reach_reg(1).rast_tot(:,6);
vel_x_up_la = Reach_reg(1).rast_tot(:,6) >= 0;
vel_x_up = Reach_reg(1).rast_tot(:,6).*vel_x_up_la;
vel_x_down_la = Reach_reg(1).rast_tot(:,6) < 0;
vel_x_down = Reach_reg(1).rast_tot(:,6).*vel_x_down_la;

clear vel_x_up_la
clear vel_x_down_la

vel_y = Reach_reg(1).rast_tot(:,7);
vel_y_up_la = Reach_reg(1).rast_tot(:,7) >= 0;
vel_y_up = Reach_reg(1).rast_tot(:,7).*vel_y_up_la;
vel_y_down_la = Reach_reg(1).rast_tot(:,7) < 0;
vel_y_down = Reach_reg(1).rast_tot(:,7).*vel_y_down_la;

clear vel_y_up_la
clear vel_y_down_la

vel_z = Reach_reg(1).rast_tot(:,8);
vel_z_up_la = Reach_reg(1).rast_tot(:,8) >= 0;
vel_z_up = Reach_reg(1).rast_tot(:,8).*vel_z_up_la;
vel_z_down_la = Reach_reg(1).rast_tot(:,8) < 0;
vel_z_down = Reach_reg(1).rast_tot(:,8).*vel_z_down_la;

clear vel_z_up_la
clear vel_z_down_la

acc = Reach_reg(1).rast_tot(:,9);

acc_x = Reach_reg(1).rast_tot(:,10);
acc_x_up_la = Reach_reg(1).rast_tot(:,10) >= 0;
acc_x_up = Reach_reg(1).rast_tot(:,10).*acc_x_up_la;
acc_x_down_la = Reach_reg(1).rast_tot(:,10) < 0;
acc_x_down = Reach_reg(1).rast_tot(:,10).*acc_x_down_la;

clear acc_x_up_la
clear acc_x_down_la

acc_y = Reach_reg(1).rast_tot(:,11);
acc_y_up_la = Reach_reg(1).rast_tot(:,11) >= 0;
acc_y_up = Reach_reg(1).rast_tot(:,11).*acc_y_up_la;
acc_y_down_la = Reach_reg(1).rast_tot(:,11) < 0;
acc_y_down = Reach_reg(1).rast_tot(:,11).*acc_y_down_la;

clear acc_y_up_la
clear acc_y_down_la

acc_z = Reach_reg(1).rast_tot(:,12);
acc_z_up_la = Reach_reg(1).rast_tot(:,12) >= 0;
acc_z_up = Reach_reg(1).rast_tot(:,12).*acc_z_up_la;
acc_z_down_la = Reach_reg(1).rast_tot(:,12) < 0;
acc_z_down = Reach_reg(1).rast_tot(:,12).*acc_z_down_la;

clear acc_z_up_la
clear acc_z_down_la

%% Sama adds model information

% Obviously you need to process this
% model_info = xlsread('torque_solution.sto')
m_time = model_info(:,1);
m_shoulder_el = model_info(:,2);
m_shoulder_an = model_info(:,3);
m_elbow_an = model_info(:,4);
m_elbow_rot = model_info(:,5);
m_wrist_an = model_info(:,6);


Predictors = [pos_x pos_y pos_z vel vel_x vel_x_up vel_x_down...
    vel_y vel_y_up vel_y_down vel_z vel_z_up vel_z_down acc...
    acc_x acc_x_up acc_x_down acc_y acc_y_up acc_y_down acc_z acc_z_up acc_z_down];

clear pos_x
clear pos_y
clear pos_z
clear vel
clear vel_x
clear vel_x_up
clear vel_x_down
clear vel_y
clear vel_y_up
clear vel_y_down
clear vel_z
clear vel_z_up
clear vel_z_down
clear acc
clear acc_x
clear acc_x_up
clear acc_x_down
clear acc_y
clear acc_y_up
clear acc_y_down
clear acc_z
clear acc_z_up
clear acc_z_down

% regess for all reaches

lag_t = 60;
step_size = 1;

%now lag steps
lag = lag_t/step_size;

% find indices of beginning and end of each reach in firing rate data
% % % % % idx1 = zeros(numel(Reach(:).exclude == 0),1);

for i=1:length(Reach_reg)
    if Reach_reg(i).exclude == 0 
        idx1(i,1)=knnsearch(Bin10smooth(:,1),Reach_reg(i).rast_kin(300,1));
        idx2(i,1)=knnsearch(Bin10smooth(:,1),Reach_reg(i).rast_kin(700,1));
    end
end

%Regression

a=1;
for ii=-lag:0
tic
% concatenate FRs
Gaussian(1).cat_fire = [];
for i=1:90
    if Reach_reg(i).exclude == 0 
        switch i 
            case 1
%             Gaussian(1).cat_fire = Gaussian(1).s_sub(idx1(i,1)+(ii*step_size):idx2(i,1)+(ii*step_size),:);
            Gaussian(1).cat_fire = Bin10smooth(idx1(i,1)+(ii*step_size):idx2(i,1)+(ii*step_size),:);
            otherwise 
%             Gaussian(1).cat_fire = vertcat(Gaussian(1).cat_fire, Gaussian(1).s_sub(idx1(i,1)+(ii*step_size):idx2(i,1)+(ii*step_size),:));
            Gaussian(1).cat_fire = vertcat(Gaussian(1).cat_fire, Bin10smooth(idx1(i,1)+(ii*step_size):idx2(i,1)+(ii*step_size),:));
        end
    end
end

Responses = Gaussian(1).cat_fire(:,2); 

tic

% do regression
% ask for R2
%%
torque_names = ["m_shoulder_el", "m_shoulder_an", "m_elbow_an", ...
    "m_shoulder_el_speed", "m_shoulder_an_speed", "m_elbow_speed"]; %sama does this
Predictor_names = ["pos_x", "pos_y", "pos_z", "vel", "vel_x", "vel_x_up",  ...
    "vel_x_down", "vel_y", "vel_y_up", "vel_y_down", "vel_z", "vel_z_up", ...
    "vel_z_down", "acc", "acc_x", "acc_x_up", "acc_x_down", "acc_y", ...
    "acc_y_up", "acc_y_down", "acc_z", "acc_z_up", "acc_z_down"];
[B,FitInfo] = lasso(Predictors,Responses,'Standardize',true,'CV',10,'DFmax',23,'PredictorNames',Predictor_names);
%%

% put regression results into struct
Regression(a).B=B;
Regression(a).FitInfo=FitInfo;

a=a+1

toc
end    

toc

%% Get MSEs from regression with all variables and from regression with 3 variables

figure
hold on
for i = 1:length(Regression)
    % MSEs from all steps at minimum MSE number of variables then 3 variables 
    MSE(i,1) = min(Regression(i).FitInfo.MSE(1,:))

    % if want to plot variables
    scatter(i,min(Regression(i).FitInfo.MSE(1,:)))
end

% plot MSE and variables for 3 variables
[val,idx3] = min(MSE(1:end,1));
idx4 = Regression(idx3).FitInfo.IndexMinMSE;
lagmax = -(lag-idx3);

indexed_MSE = Regression(idx3).FitInfo.MSE;
STE = std(indexed_MSE)/sqrt(length(indexed_MSE));

figure
hold on
xlabel('log(Lambda)')
yyaxis left
ylabel('MSE')
plot(log(Regression(idx3).FitInfo.Lambda),Regression(idx3).FitInfo.MSE(1,:))
plot(log(Regression(idx3).FitInfo.Lambda),Regression(idx3).FitInfo.MSE(1,:)+STE)
plot([log(Regression(idx3).FitInfo.Lambda(1)) log(Regression(idx3).FitInfo.Lambda(end))]...
    ,[Regression(idx3).FitInfo.MSE(1,1)+STE Regression(idx3).FitInfo.MSE(1,1)+STE], 'r--')
Lambda_Stop = min(find(Regression(idx3).FitInfo.MSE(1,:) > Regression(idx3).FitInfo.MSE(1,1)+STE));

yyaxis right
ylabel('# variables included')
plot(log(Regression(idx3).FitInfo.Lambda),Regression(idx3).FitInfo.DF)
plot([log(Regression(idx3).FitInfo.Lambda(Lambda_Stop)) log(Regression(idx3).FitInfo.Lambda(Lambda_Stop))],...
    [min(Regression(idx3).FitInfo.DF) max(Regression(idx3).FitInfo.DF)], 'k--')

number_of_variables = Regression(idx3).FitInfo.DF(Lambda_Stop);

[B,FitInfo] = lasso(Predictors,Responses,'Standardize',true,'CV',10,'DFmax',number_of_variables,'PredictorNames',Predictor_names);

[~,minix] = min(FitInfo.MSE);
best_Beta = B(:,minix);

coef0 = FitInfo.Intercept(minix);
P= sum(Predictors'.*best_Beta) + coef0;
f = find(best_Beta ~= 0)
Predictor_names(f)

figure();
hold on;
plot(Responses,'k')
plot([1:length(P)]-lagmax,P,'r')

mdl = fitlm(P',Responses)
r2 = mdl.Rsquared.Adjusted
estr2 = corr(P',Responses)^2;

Reach_reg(i).start

%%

ii = 2;
endInd = [0];
Gaussian(1).cat_fire = [];
for i=1:90
    if Reach_reg(i).exclude == 0 
        switch ii 
            case 1
            Gaussian(1).cat_fire = Bin10smooth(idx1(i,1)+(ii*step_size):idx2(i,1)+(ii*step_size),:);
            endInd(ii) = length(Gaussian(1).cat_fire);
            ii = ii + 1;
            otherwise 
            Gaussian(1).cat_fire = vertcat(Gaussian(1).cat_fire, Bin10smooth(idx1(i,1)+(ii*step_size):idx2(i,1)+(ii*step_size),:));
            endInd(ii) = length(Gaussian(1).cat_fire);
            ii = ii + 1;
        end
    end
end

Ralt = [];
Palt = [];
R2results = [];
for i = 1:length(endInd) - 1
    s = endInd(i) + 1;
    e = endInd(i+1);
    Ralt(i,:) = Responses(s:e);
    Palt(i,:) = P(s:e);  
    R2results(i) = corr(Ralt(i,:)',Palt(i,:)')^2;
end

figure; hold on
set(gcf,'color','w') % This makes the figure white instead of grey.
plot(mean(Ralt), 'k');
plot(mean(Ralt) + std(Ralt)./sqrt(length(endInd)-1), 'Color', [0 0 0 0.2])
plot(mean(Ralt) - std(Ralt)./sqrt(length(endInd)-1), 'Color', [0 0 0 0.2])
plot(mean(Palt), 'r');
plot(mean(Palt) + std(Palt)./sqrt((length(endInd)-1)^2), 'Color', [1 0 0 0.2])
plot(mean(Palt) - std(Palt)./sqrt((length(endInd)-1)^2), 'Color', [1 0 0 0.2])
% Add a legend to the figure:
legend("Empirical FR","","","Predicted FR","","")
xlabel("Time (samples)")
ylabel("Firing Rate (FR)")
title("Mean FR alligned to reach in empirical data")

figure; hold on
boxplot(R2results);
% We need to show the mean too (Boxplots show medians and quartiles)
scatter(1,mean(R2results))
set(gcf,'color','w')
% This sets the 'ticks' on the x axis to a custom value:
xticks(1)
xticklabels(["R^2 of Data"])
% Give this a y axis label and title:
ylabel("R^2 of fit")
title("Fit to emperical data with kinematics")
% TODO: Show trial shuffled and time shuffled results


%%
figure(); hold on;
[sortedBeta,bestix] = sort(abs(best_Beta));
plot(sortedBeta(sortedBeta>0))
xticks(1:length(sortedBeta(sortedBeta>0)))
xticklabels(Predictor_names(bestix(sortedBeta>0)))

figure; hold on
plot(best_Beta)
xticks(1:length(best_Beta))
xticklabels(Predictor_names)
