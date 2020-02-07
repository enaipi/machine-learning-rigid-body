# machine-learning-rigid-body
A simple collection of python(3) scripts that:
(i) simulate evolution of rigid body rotation
(ii) Learn the energy of the rigid body from the simulated points

Rigid body is simulated using the Poisson integrator developed in [1]. 

Using linear regression, the energy is then learned from the trajectory obtained in the simulation, more is explained in [2]. 
A simple usage is shown in learn_rigid_body.py script.
Learning using a distribution of initial conditions is done in script statistical_comparison.py, where also figures are plotted.


[1] Michal Pavelka, Václav Klika and Miroslav Grmela, Ehrenfest regularization of Hamiltonian systems, Physica D: Nonlinear phenomena, 399, 193-210, 2019
[2] Francisco Chinesta, Elias Cueto, Miroslav Grmela, Beatriz Moya, Michal Pavelka, Learning Physics from Data: a Thermodynamic Interpretation, arXiv:1909.01074, submitted to Physica D

