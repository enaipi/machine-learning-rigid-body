from LearnRB import LearnRBEnergy
import numpy as np

learner = LearnRBEnergy()

Ix = 10.0
Iy = 20.0
Iz = 40.0
learner.update_exact(Ix, Iy, Iz)

learner.load_trajectory_from_file("m.xyz", readevery = 10, stopat = 10*3)
learner.print_trajectory(stopat=10)
learner.fit()

print "\n----Comparing matrices: "
print "Original energy matrix: \n", np.array_str(np.array(learner.d2E_exact))
print "tr_exact = ", np.trace(learner.d2E_exact)
print "Spectrum exact: ", learner.spectrum_exact()
print "Moments of inertia exact: ", learner.moments_of_inertia_exact()

print "\n"
print "Learned energy matrix: \n", learner.print_d2E()
print "Spectrum: ", learner.spectrum()
learner.tr(verbose=True)
print "Moments of inertia: ", learner.moments_of_inertia()

print "\n"
learner.normalize()
print "Normalized energy matrix: \n", learner.print_d2E()
print "Spectrum: ", learner.spectrum()
learner.tr(verbose=True)
print "Moments of inertia: ", learner.moments_of_inertia()
print "Residual: ", learner.energy_residual()
print "Energy Score: ", learner.energy_score()

print "\n----Comparing trajectories: "

print learner.trajectory[0]
print learner.trajectory[1]
print "Using learned: "
learner.predict(learner.trajectory[1][0]-learner.trajectory[0][0], learner.trajectory[0][1], learner.trajectory[0][2],learner.trajectory[0][3])
print "Using exact: "
learner.predict(learner.trajectory[1][0]-learner.trajectory[0][0], learner.trajectory[0][1], learner.trajectory[0][2],learner.trajectory[0][3], using_exact = True)


