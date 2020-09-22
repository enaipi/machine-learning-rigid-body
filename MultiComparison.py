from RigidBody import *
from LearnRB import LearnRBEnergy
import numpy as np
import math
import matplotlib.pyplot as plt

class MultiComparison:
    def __init__(self, N, Ix, Iy, Iz, dt):
        self.N = N#number of steps
        print("N = ", N)

        #Moments of inertia
        self.Ix = Ix
        self.Iy = Iy
        self.Iz = Iz
        print("Ix = ", Ix, ", Iy = ", Iy, ", Iz = ", Iz)

        #time steps
        self.dt = dt
        self.tau = dt * 1.0
        print("dt = ", dt)
        print("tau = ", self.tau)

        #Preparing the Learner
        self.learner = LearnRBEnergy()
        self.learner.update_exact(Ix, Iy, Iz)

        self.store_file = None
        self.statistics = []
    
    def open_file(self, name = "statistics.xyz"):
        self.store_file = open(name,'w')

    def close_file(self):
        self.store_file.close()

    def store_to_file(self, values):
        result = str(values[0])
        for i in range(1, len(values)):
            result += " "+str(values[i])
        result += "\n"
        self.store_file.write(result)


    def compare(self, readevery = 1, stopat = 0, filename = "statistics.xyz"):
        #Preparing the sampling domain
        r = np.linspace(0,1, 50)

        def mz(r_, phi_, M = 1):
            return math.sqrt(M-r_**2)

        if self.store_file != None:
            self.open_file(filename)
        
        try:
            self.statistics = []
            if stopat == 0:
                stopat = self.N
            for r_ in r:
                print("r = ", r_)
                phi = np.linspace(0, 2*math.pi, round(10+r_*190))
                for phi_ in phi:
                    mx_ = r_*math.cos(phi_)
                    my_ = r_*math.sin(phi_)
                    mz_ = mz(r_,phi_)
    
                    trajectory = [[0.0, mx_, my_, mz_]]
                    RB = RBSeReFE(self.Ix, self.Iy, self.Iz, mx_, my_, mz_, self.dt, self.tau)
                    E_init  = RB.energy()
    
                    #Calculate evolution
                    for i in range(1,self.N-1):
                        RB.m_new()
                        t = self.dt * i
                        trajectory.append([t, RB.mx,RB.my,RB.mz])
                    E_final= RB.energy()
                    if abs(E_final-E_init)/E_init > 0.01:
                        print("Warning: For r=", r_ , ", phi=", phi_, " energy was shifted by more than 1 percent.")
                    #Evolution finished
             
                    #Learning
                    self.learner.load_trajectory(trajectory, readevery = readevery, stopat = stopat, verbose=False)
                    self.learner.fit()
                    self.learner.normalize()
                    score = self.learner.energy_score()
             
                    result = [mx_, my_, score]
                    self.statistics.append(result)
                    if self.store_file != None:
                        self.store_to_file(result)
        
            print("Statistics has ", len(self.statistics), " entries")
            if self.store_file != None:
                print("Written to file.")
        finally:
            if self.store_file != None:
                self.close_file()

    def find_max_score(self):
        max_score = self.statistics[0][2]
        max_i = 0
        for i in range(len(self.statistics)):
            if self.statistics[i][2] > max_score:
                max_score = self.statistics[i][2]
                max_i = i
        return max_i, max_score

    def find_min_score(self):
        min_score = self.statistics[0][2]
        min_i = 0
        for i in range(len(self.statistics)):
            if self.statistics[i][2] < min_score:
                min_score = self.statistics[i][2]
                min_i = i
        return min_i, min_score

    def filter_statistics(self, threshold, verbose = False):
        new_statistics = []
        for i in range(len(self.statistics)):
            for j in range(3):
                if self.statistics[i][2] < threshold:
                    new_statistics.append(self.statistics[i])
        if verbose:
            print("Length of statistics with threshold ", threshold," is: ", len(new_statistics))
        return new_statistics

    def plot(self, file_name="statistics.png", title = "Scores", verbose = False):
        plt.style.use("classic")

        s = np.array(self.filter_statistics(1.0))
        if len(s) == 0:
            print("No statistics. Not plotting.")
            return

        mx = s[:,0]
        my = s[:,1]
        scores = s[:,2]
        #importance = 100*(0.1*np.ones(len(scores))+ 10*np.heaviside(-s[:,2],np.zeros(len(scores))))

        fig = plt.figure()
        ax = plt.axes()
        #plt.scatter(mx, my, c = scores, s = importance, alpha=0.2, cmap='viridis')
        plt.scatter(mx, my, c = scores, alpha=0.1, edgecolors='none', cmap='viridis')
        plt.colorbar(); # show color scale

        s = np.array(self.filter_statistics(0, verbose=False))
        if len(s) == 0:
            print("No statistics. Not plotting.")
        else:
            mx = s[:,0]
            my = s[:,1]
            scores = s[:,2]
            plt.scatter(mx, my, c = scores, alpha=1.0, cmap='viridis')

        ax.set(title = title, xlabel = "$m_x$", ylabel = "$m_y$")

        if file_name != None:
            fig.savefig(file_name)

        if verbose:
            plt.show()
 


