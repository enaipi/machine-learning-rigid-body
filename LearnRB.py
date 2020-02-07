import numpy as np
from sklearn.linear_model import LinearRegression
from sympy import LeviCivita

class LearnRBEnergy(object):
    def __init__(self):
        self.d2E = [[0,0,0],[0,0,0],[0,0,0]]
        self.d2E_exact = [[0,0,0],[0,0,0],[0,0,0]]
        self.trajectory = []

    def load_trajectory_from_file(self, file_name, readevery=1, stopat=0, verbose = False):
        #Assumed format: t, mx, my, mz \n 
        traj_file = open(file_name,"r")
        lines = traj_file.readlines()
        print("Length of data: ", len(lines))

        if stopat == 0: 
            stopat = len(lines) #read the whole file

        self.trajectory = [[float(j) for j in lines[0].split(" ")]] #contains entry at t=0
        for i in range(1,min(stopat,len(lines))):
            if (i+1) % readevery == 0:
                self.trajectory.append([float(j) for j in lines[i].split(" ")])

        if verbose:
            print("Trajectory shape: ", np.array(self.trajectory).shape)
            print("Number of points: ", len(self.trajectory))

    def load_trajectory(self, trajectory, readevery = 1, stopat = 0, verbose = False):
        #Assumed format: t, mx, my, mz \n 

        if stopat == 0: 
            stopat = len(trajectory) #read the whole trajectory

        self.trajectory = [trajectory[0]]
        for i in range(1,min(stopat,len(trajectory))):
            if (i+1) % readevery == 0:
                self.trajectory.append(trajectory[i])

        if verbose:
            print("Trajectory shape: ", np.array(self.trajectory).shape)
            print("Number of points: ", len(self.trajectory))

    def print_trajectory(self, stopat = 0):
        if stopat == 0: 
            stopat = len(self.trajectory)

        print("Prining the trajectory")
        for i in range(min(len(self.trajectory),stopat)):
            print(self.trajectory[i])

    def fit(self, verbose = False):
        x = []
        y = []
        for i in range(len(self.trajectory)-1):
            y.append(self.trajectory[i+1][1]-self.trajectory[i][1])
            y.append(self.trajectory[i+1][2]-self.trajectory[i][2])
            y.append(self.trajectory[i+1][3]-self.trajectory[i][3])

            dt = self.trajectory[i+1][0] - self.trajectory[i][0]

            m1 = self.trajectory[i][1]
            m2 = self.trajectory[i][2]
            m3 = self.trajectory[i][3]

            x_vector1 = dt*np.array([0, -m3*m1, m2*m1, -m3*m2, -m3*m3+m2*m2, m2*m3])
            x_vector2 = dt*np.array([m3*m1, m3*m2, m3*m3-m1*m1, 0, -m1*m2, -m1*m3])
            x_vector3 = dt*np.array([-m2*m1, -m2*m2+m1*m1, -m2*m3, m1*m2, m1*m3, 0])

            x.append(x_vector1)
            x.append(x_vector2)
            x.append(x_vector3)
     
        if verbose:
            print("x.shape: ", np.array(x).shape)
            print("y.shape: ", np.array(y).shape)

        model = LinearRegression(fit_intercept=False)
        #Fitting by linear regression...
        model.fit(x,y)
        [E11, E12, E13, E22, E23, E33] = model.coef_
        self.d2E = [[E11, E12, E13], [E12, E22, E23], [E13, E23, E33]]
       
        score = model.score(x,y)#1-(residual)/(total sum of squares). Close to 1.0 is good, negative bad.
        if verbose:
            print("Score: ", score) 
        if score < 0:
            print("Warning: linear regression did not coverge.")


    def tr(self, verbose = False):
        tr = np.trace(self.d2E)
        if verbose:
            print("tr d2E = ", tr)
        return tr 

    def det(self, verbose = False):
        det = np.linalg.det(self.d2E)
        if verbose:
            print("det d2E = ", det)
        return det 

 
    def print_d2E(self): 
        print("d2E = \n", np.array_str(np.array(self.d2E)))

    def spectrum(self, verbose = False):
        spectrum, eigvec = np.linalg.eig(self.d2E)
        if verbose: 
            print("Eigenvalues(d2E) = ", spectrum)
        return spectrum

    def spectrum_exact(self, verbose = False):
        spectrum, eigvec = np.linalg.eig(self.d2E_exact)
        if verbose: 
            print("Eigenvalues(d2E) = ", spectrum)
        return spectrum

    def predict(self, dt, m1, m2, m3, using_exact = False):
        m = [m1,m2,m3]
        m1_new = m1
        for j in range(1,4):
            for k in range(1,4):
                for l in range(1,4):
                    if not using_exact:
                        m1_new -= dt * m[k-1] * m[l-1] * LeviCivita(1,j,k) * self.d2E[j-1][l-1]
                    else:
                        m1_new -= dt * m[k-1] * m[l-1] * LeviCivita(1,j,k) * self.d2E_exact[j-1][l-1]
        
        m2_new = m2
        for j in range(1,4):
            for k in range(1,4):
                for l in range(1,4):
                    if not using_exact:
                        m2_new -= dt * m[k-1] * m[l-1] * LeviCivita(2,j,k) * self.d2E[j-1][l-1]
                    else:
                        m2_new -= dt * m[k-1] * m[l-1] * LeviCivita(2,j,k) * self.d2E_exact[j-1][l-1]

        m3_new = m3
        for j in range(1,4):
            for k in range(1,4):
                for l in range(1,4):
                    if not using_exact:
                        m3_new -= dt * m[k-1] * m[l-1] * LeviCivita(3,j,k) * self.d2E[j-1][l-1]
                    else:
                        m3_new -= dt * m[k-1] * m[l-1] * LeviCivita(3,j,k) * self.d2E_exact[j-1][l-1]
        result = [m1_new, m2_new, m3_new]

        print(result)
        return result

    def normalize(self): 
        #SO3 dynamics is not canonical and has Casimirs. Therefore, we can only learn energy up to the Casimirs.
        #Casimirs are multiples of |m|^2. 
        #This method normalized the learned d2E by adding a the unit matrix multiplied by a constant\
        #so that it matches the exact energy.
       
        tr = np.trace(self.d2E)
        tr_exact = np.trace(self.d2E_exact)

        self.d2E[0][0] += (tr_exact-tr)/3.0
        self.d2E[1][1] += (tr_exact-tr)/3.0
        self.d2E[2][2] += (tr_exact-tr)/3.0

    def energy_residual(self): #sum of squares of differences between entries of the exact and learned matrices
        result = 0
        for i in range(3):
            for j in range(3):
                result += (self.d2E[i][j]-self.d2E_exact[i][j])**2
        return result

    def energy_score(self): #1-u/v, where u is the energy residual and v is square of the exact matrix
        #Close to 1.0 is good, negative is bad
        square_exact = 0
        for i in range(3):
            for j in range(3):
                square_exact += (self.d2E_exact[i][j])**2
        return 1-self.energy_residual()/square_exact

    def moments_of_inertia(self):
        spectrum = self.spectrum()
        return [0.5/spectrum[0], 0.5/spectrum[1], 0.5/spectrum[2]]

    def moments_of_inertia_exact(self):
        spectrum = self.spectrum_exact()
        return [0.5/spectrum[0], 0.5/spectrum[1], 0.5/spectrum[2]]

    def update_exact(self, Ix, Iy, Iz):
        self.d2E_exact= [[1/Ix, 0, 0],[0, 1/Iy, 0], [0, 0, 1/Iz]]
