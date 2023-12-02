# Import dependencies:
import matplotlib.pyplot as plt
import opensim as osim
import sys
import os
import numpy as np
import helpersmouse as helpers
import math

# Scaling:
max_X = 0.027
max_Y = .002
start_X = 0.018965
start_Y = -0.002905
elbow_max_X = 0.018322
elbow_max_Y = -0.000933
elbow_start_X = 0.010785
elbow_start_Y = -0.005334

# Load OpenSim model:
model_filename = 'working_model_2023_2.osim'

# Load the trajectory data as a timeseries:
for traj in range(1,91):
    trajectoryData = np.genfromtxt('DReachNew/trajectory_' + str(traj) + '.csv', delimiter=',')
    X = trajectoryData[:,1]

    Y = trajectoryData[:,2]
    Z = trajectoryData[:,3]*0 - 0.003685

    #plt.plot(X,Y)
    #plt.show()

    mX = max(X)
    mnX = min(X)
    scalar = (mX - mnX)/(max_X - start_X)
    print(scalar)
    X = X - mnX
    X = X / scalar
    X = X + start_X 

    mX = max(Y)
    mnX = min(Y)
    scalar = (mX - mnX)/(max_Y - start_Y)
    print(scalar)
    Y = Y - Y[0]
    Y = Y / scalar
    Y = Y  + start_Y 

    #plt.plot(X,Y)
    #plt.show()

    eX = trajectoryData[:,4]
    eY = trajectoryData[:,5]
    eZ = trajectoryData[:,6]*0 - 0.004079
    mX = max(eX)
    mnX = min(eX)
    scalar = (mX - mnX)/(elbow_max_X - elbow_start_X)
    print(scalar)
    eX = eX - mnX
    eX = eX / scalar
    eX = eX + elbow_start_X 

    mX = max(eY)
    mnX = min(eY)
    scalar = (mX - mnX)/(elbow_max_Y - elbow_start_Y)
    print(scalar)
    eY = eY - Y[0]
    eY = eY / scalar
    eY = eY  + elbow_start_Y 

    # First, optimize the torque bassed model:
    #model = helpers.getMuscleDrivenModel(model_filename)
    model = helpers.getTorqueDrivenModel(model_filename)

    # Get the references for the hand marker:
    marker = model.getMarkerSet().get("handm")

    # Setup the MoCo study:
    study = osim.MocoStudy()
    problem = study.updProblem()
    problem.setModel(model)

    finalTime = trajectoryData[-1,0]
    problem.setTimeBounds(0, finalTime)

    # Set the initial conditions and limits for the model:
    problem.setStateInfo('/jointset/shoulder/elv_angle/value', [],1.357936)
    problem.setStateInfo('/jointset/shoulder/extension_angle/value', [],0.265133)
    problem.setStateInfo('/jointset/humerus_ulna/elbow_flex/value', [],-1.144866)
    #problem.setStateInfoPattern('/jointset/.*/speed', [-3,3], 0, 0)

    # Read in the limb trajectory:
    markerTrajectories = osim.TimeSeriesTableVec3()
    markerTrajectories.setColumnLabels(["/markerset/handm","/markerset/elbow"])
    ixm = 0
    step = 1

    for ix in range(int(np.floor(len(trajectoryData[:,0])/step))):
    #for ix in range(len(trajectoryData[:,0])):
        X2 = X[ixm]
        Y2 = Y[ixm]
        Z2 = Z[ixm]
        T = trajectoryData[ixm,0]
        m0 = osim.Vec3(X2,Y2,Z2)
        eX2 = eX[ixm]
        eY2 = eY[ixm]
        eZ2 = eZ[ixm]
        m1 = osim.Vec3(eX2,eY2,eZ2)
        markerTrajectories.appendRow(T,
        osim.RowVectorVec3([m0, m1]))
        ixm = ixm + step

    # Assign a weight to each marker.
    markerWeights = osim.SetMarkerWeights()
    markerWeights.cloneAndAppend(osim.MarkerWeight("/markerset/handm", 200))
    markerWeights.cloneAndAppend(osim.MarkerWeight("/markerset/elbow", 200))

    ref = osim.MarkersReference(markerTrajectories, markerWeights)
    markerTracking = osim.MocoMarkerTrackingGoal()
    markerTracking.setMarkersReference(ref)

    problem.addGoal(markerTracking)      

    problem.addGoal(osim.MocoControlGoal('myeffort',2)) 

    solver = study.initCasADiSolver()
    solver.set_num_mesh_intervals(int((len(X)-1)/2))
    #solver.set_num_mesh_intervals(200)
    solver.set_optim_convergence_tolerance(1e-6)
    solver.set_optim_constraint_tolerance(1e-6)
    solver.set_optim_max_iterations(250)
    #solver.set_optim_max_iterations(2)

    predictSolution = study.solve()
    solutionUnsealed = predictSolution.unseal()
    filename = 'solutions/torque_solution' + str(traj) + '.sto'
    solutionUnsealed.write(filename)

    model.initSystem()
    states = predictSolution.exportToStatesTable()

    statesTraj = osim.StatesTrajectory.createFromStatesTable(model, states)

    markerTrajectories = osim.TimeSeriesTableVec3()
    markerTrajectories.setColumnLabels(["/markerset/handm","/markerset/elbow"])
    for state in statesTraj:
      model.realizePosition(state)
      m0 = model.getComponent("markerset/handm")
      m1 = model.getComponent("markerset/elbow")
      markerTrajectories.appendRow(state.getTime(),
      osim.RowVectorVec3([m0.getLocationInGround(state), m1.getLocationInGround(state)]))

    filename = 'solutions/torque_traj' + str(traj) + '.sto'
    osim.STOFileAdapterVec3.write(markerTrajectories,filename)

## Sama you can clip all of this off

#model = helpers.getMuscleDrivenModel(model_filename)

## Get the references for the hand marker:
#marker = model.getMarkerSet().get("handm")

## Setup the MoCo study:
#study = osim.MocoStudy()
#problem = study.updProblem()
#problem.setModel(model)

#finalTime = trajectoryData[-1,0]
#problem.setTimeBounds(0, finalTime)

## Set the initial conditions and limits for the model:
#problem.setStateInfo('/jointset/shoulder/elv_angle/value', [],1.357936)
#problem.setStateInfo('/jointset/shoulder/extension_angle/value', [],0.265133)
#problem.setStateInfo('/jointset/humerus_ulna/elbow_flex/value', [],-1.144866)
##problem.setStateInfoPattern('/jointset/.*/speed', [-3,3], 0, 0)

## Read in the limb trajectory:
#markerTrajectories = osim.TimeSeriesTableVec3()
#markerTrajectories.setColumnLabels(["/markerset/handm","/markerset/elbow"])
#ixm = 0
#step = 1
#for ix in range(int(np.floor(len(trajectoryData[:,0])/step))):
##for ix in range(len(trajectoryData[:,0])):
#    X2 = X[ixm]
#    Y2 = Y[ixm]
#    Z2 = Z[ixm]
#    T = trajectoryData[ixm,0]
#    m0 = osim.Vec3(X2,Y2,Z2)
#    eX2 = eX[ixm]
#    eY2 = eY[ixm]
#    eZ2 = eZ[ixm]
#    m1 = osim.Vec3(eX2,eY2,eZ2)
#    markerTrajectories.appendRow(T,
#    osim.RowVectorVec3([m0, m1]))
#    ixm = ixm + step

## Assign a weight to each marker.
#markerWeights = osim.SetMarkerWeights()
#markerWeights.cloneAndAppend(osim.MarkerWeight("/markerset/handm", 200))
#markerWeights.cloneAndAppend(osim.MarkerWeight("/markerset/elbow", 200))

#ref = osim.MarkersReference(markerTrajectories, markerWeights)
#markerTracking = osim.MocoMarkerTrackingGoal()
#markerTracking.setMarkersReference(ref)

#problem.addGoal(markerTracking)      

#problem.addGoal(osim.MocoControlGoal('myeffort',2)) 

#solver = study.initCasADiSolver()
#solver.set_num_mesh_intervals(int(len(X)*4))
#solver.set_optim_convergence_tolerance(1e-5)
#solver.set_optim_constraint_tolerance(1e-5)
#solver.set_optim_max_iterations(2500)

#predictSolution = study.solve()
#solutionUnsealed = predictSolution.unseal()
#filename = 'solutions/muscle_solution.sto'
#solutionUnsealed.write(filename)

##
#filename = 'solutions/torque_solution.sto'
#dirName = 'solutions/'
#tableProcessor = osim.TableProcessor(filename)
#print(tableProcessor)
#inverse = osim.MocoInverse()

#modelProcessor = osim.ModelProcessor(model_filename)
#modelProcessor.append(osim.ModOpIgnoreTendonCompliance())
#modelProcessor.append(osim.ModOpIgnorePassiveFiberForcesDGF())

#modelProcessor.append(osim.ModOpAddReserves(.000001))

#inverse.setModel(modelProcessor)
#inverse.setKinematics(tableProcessor)

#inverse.set_initial_time(0)
#inverse.set_mesh_interval(finalTime/120)
#inverse.set_max_iterations(1000)
#inverse.set_constraint_tolerance(.000001)
#inverse.set_minimize_sum_squared_activations(True)
#inverse.set_kinematics_allow_extra_columns(True)
#inverse.set_minimize_sum_squared_activations(True)

#inverseSolution = inverse.solve()
#solution = inverseSolution.getMocoSolution()
#inverseSolutionUnsealed = solution.unseal()
#inverse_filename = 'solutions/inverse_Solution.sto'
#inverseSolutionUnsealed.write(inverse_filename)