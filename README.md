# Big-Data-Final-Project
Determine if mouse forelimb model provides extra data which can better explain Purkinje Cell firing rates


FILES:  
working_mouse_model_2023_2 -> this is the current working model which is used in reach optimization  
180829_Lasso_Results -> The data file containing the mouse kinematics and the Purkinje Cell firing rates which were used in the analysis  
helpermouse -> helper functions for optimization code  
trajectory_forelimb -> code containing the model initialization and cost function for optimization. Loops over several reaches and produces optimizations  
CompareKinematics -> compares the actual reach to the simulated reach and produces a side by side visual comparison of the paw movement and kinematics. Also outputs an error value for each reach   
newdatastuff -> iterates through the data file and extracts individual reaches for optimization  
SK_LASSOoriginal -> LASSO regression on Purkinje Cell firing rate and real kinematics only  
SK_LASSORealData -> LASSO regression on Purkinje Cell firing rates with real and synthesized kinematics  

Code Configuration and Execution:  
Download and install OpenSim on computer  
Place all the files in the same directory  
Run newdatastuff to extract reaches for optimization  
run trajectory_forelimb to optimize reaches and output torque files  
run CompareKinematics to get error values for reaches to ensure accuracy  
run SK_LASSOoriginal to get R^2 value of regression of Purkinje Cell firing rates with real kinematics  
run SK_LASSOoriginal to get R^2 value of regression of Purkinje Cell firing rates with synthesized kinematics
