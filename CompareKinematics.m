%% Set up the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 7);

% Specify range and delimiter
for i = 84:84
    opts.DataLines = [1, Inf];
    opts.Delimiter = ",";
    opts.VariableNames = ["VarName1", "VarName2", "VarName3", "VarName4", "VarName5", "VarName6", "VarName7"];
    opts.VariableTypes = ["double", "double", "double", "double", "double", "double", "double"];
    opts.ExtraColumnsRule = "ignore";
    opts.EmptyLineRule = "read";


    trajectory1 = readtable("DReachNew/trajectory_" + string(i) + ".csv", opts);
    Tab = table2array(trajectory1);
    
    clear opts
    
    opts = delimitedTextImportOptions("NumVariables", 7);
    
    % Specify range and delimiter
    opts.DataLines = [6, Inf];
    opts.Delimiter = ["\t", ","];
    
    % Specify column names and types
    opts.VariableNames = ["Time", "VarName2", "VarName3", "VarName4", "VarName5", "VarName6", "VarName7"];
    opts.VariableTypes = ["double", "double", "double", "double", "double", "double", "double"];
    
    % Specify file level properties
    opts.ExtraColumnsRule = "ignore";
    opts.EmptyLineRule = "read";
    torquetraj = readtable("solutions\torque_traj" + string(i) + ".sto", opts);
    
    figure; hold on;
    subplot(1,2,1); hold on
    plot(Tab(:,2),Tab(:,3))
    title("Actual kinematics")
    xlabel("x pos (pixels)")
    ylabel("y pos (pixels)")
    legend("Hand Position")
    
    subplot(1,2,2); hold on
    plot(torquetraj.VarName2,torquetraj.VarName3)
    plot(torquetraj.VarName5,torquetraj.VarName6)
    xlabel("x pos (m)")
    ylabel("y pos (m)")
    title("Opensim synthesized kinematics")
    legend(["Hand Position","Elbow Position"])
    set(gcf,'color','w')
    %%
    figure; hold on; 
    plot(torquetraj.Time,torquetraj.VarName2);
    comparison_x = Tab(:,2) - min(Tab(:,2));
    max_X = 0.027;
    max_Y = .002;
    max_Z = 0;
    start_X = 0.018965;
    start_Y = -0.002905;
    start_Z = -0.0064;
    comparison_x = comparison_x*(max_X - start_X)/max(comparison_x);
    comparison_x = comparison_x + min(torquetraj.VarName2(:));
    plot(Tab(:,1),comparison_x)
    legend(["synthesized","real"])
    
    interp_x = interp1(Tab(:,1),comparison_x,torquetraj.Time);
    error = mean(sqrt((torquetraj.VarName2 - interp_x).^2));
    disp("Trajectory " + string(i) + " : " + error);
end