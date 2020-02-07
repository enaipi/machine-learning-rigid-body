#This file contains the forward Euler solver for the self-regularized rigid body motion and the Crank-Nicolson solver for the energetic self-regularization of rigid body motion
#Author: Michal Pavelka; pavelka@karlin.mff.cuni.cz

from scipy.optimize import fsolve
from math import *

class RigidBody(object): #Parent Rigid body class
    def __init__(self, Ix, Iy, Iz, mx, my, mz, dt, tau, T=100, verbose = False):
        self.Ix = Ix
        self.Iy = Iy
        self.Iz = Iz
       
        self.Jx = 1/Iz - 1/Iy
        self.Jy = 1/Ix - 1/Iz
        self.Jz = 1/Iy - 1/Ix

        self.mx = mx
        self.my = my 
        self.mz = mz 

        self.mx0 = mx
        self.my0 = my 
        self.mz0 = mz 

        self.dt = dt
        self.tau = tau

        self.hbar = 1.0545718E-34 #reduced Planck constant [SI]
        self.rho = 8.92E+03 #for copper
        self.myhbar = self.hbar * self.rho #due to rescaled mass
        self.kB = 1.38064852E-23 #Boltzmann constant
        self.umean = 4600 #mean sound speed in the low temperature solid (Copper) [SI]
        self.Einconst = pi**2/10 * pow(15/(2* pi**2), 4.0/3) * self.hbar * self.umean * pow(self.kB, -4.0/3) #Internal energy prefactor, Characterisitic volume = 1
        if verbose: 
            print("Internal energy prefactor = ", self.Einconst)

        self.sin = self.ST(T) #internal entropy
        if verbose:
            print("Internal entropy set to Sin = ", self.sin, " at T=",T," K")
       
        self.Ein_init = 1
        self.Ein_init = self.Ein()
        self.sin_init = self.sin
        if verbose: 
            print("Initial total energy = ", self.Etot())

        if verbose: 
            print("RB set up.")

    def energy_x(self):
        return 0.5*self.mx*self.mx/self.Ix

    def energy_y(self):
        return 0.5*self.my*self.my/self.Iy

    def energy_z(self):
        return 0.5*self.mz*self.mz/self.Iz

    def energy(self):#returns kinetic energy
        return 0.5*(self.mx*self.mx/self.Ix+self.my*self.my/self.Iy+self.mz*self.mz/self.Iz)

    def omega_x(self):
        return self.mx/self.Ix

    def omega_y(self):
        return self.my/self.Iy

    def omega_z(self):
        return self.mz/self.Iz

    def m2(self):#returns m^2 
        return self.mx*self.mx+self.my*self.my+self.mz*self.mz

    def mx2(self):#returns mx^2 
        return self.mx*self.mx

    def my2(self):#returns my^2 
        return self.my*self.my

    def mz2(self):#returns mz^2 
        return self.mz*self.mz

    def m_magnitude(self):#returns |m|
        return sqrt(self.m2())

    def Ein(self):#returns normalized internal energy
        #return exp(2*(self.sin-1))/self.Iz
        return self.Einconst*pow(self.sin,4.0/3)/self.Ein_init

    def Ein_s(self): #returns normalized derivative of internal energy with respect to entropy (inverse temperature)
        #return 2*exp(2*(self.sin-1))/self.Iz
        return self.Einconst*4.0/3*pow(self.sin, 1.0/3) / self.Ein_init

    def ST(self, T): #returns entropy of a Copper body with characteristic volume equal to one (Debye), [T] = K 
        return 2 * pi**2/15 * self.kB * (self.kB/self.hbar *T/self.umean)**3

    def Etot(self):#returns normalized total energy
        return self.energy() + self.Ein()

    def Sin(self): #returns normalized internal entorpy
        return self.sin/self.sin_init

    def S_x(self):#kinetic entropy for rotation around x, beta = 1/4Iz
        m2 = self.m2()
        return -m2/self.Ix - 0.5*0.25/self.Iz*(m2-self.mx0*self.mx0)**2

    def S_z(self):#kinetic entropy for rotation around z
        m2 = self.m2()
        return -m2/self.Iz - 0.5*0.25/self.Iz*(m2-self.mz0*self.mz0)**2

    def Phi_x(self): #Returns the Phi potential for rotation around the x-axis
        return self.energy() + self.S_x()

    def Phi_z(self):
        return self.energy() + self.S_z()

class RBESeReCN(RigidBody):#E-SeRe with Crank Nicolson
    def __init__(self, Ix, Iy, Iz, mx, my, mz, dt, tau):
        super(RBESeReCN, self).__init__(Ix, Iy, Iz, mx, my, mz, dt, tau)

    def f(self, m_new):#defines the function f zero of which is sought
        mx = self.mx
        my = self.my
        mz = self.mz

        dt = self.dt
        tau = self.tau
       
        Ix = self.Ix
        Iy = self.Iy
        Iz = self.Iz

        Jx = self.Jx
        Jy = self.Jy
        Jz = self.Jz

        mx_new = m_new[0]
        my_new = m_new[1]
        mz_new = m_new[2]

        fx = mx_new - mx \
                - 0.5*dt*(my*mz*Jx + 0.5*dt*tau*mx*(my*my * Jz/Iz - mz*mz*Jy/Iy)) \
                - 0.5*dt*(my_new*mz_new*Jx + 0.5*dt*tau*mx_new*(my_new*my_new * Jz/Iz - mz_new*mz_new*Jy/Iy))
        fy = my_new - my \
                - 0.5*dt*(mz*mx*Jy + 0.5*dt*tau*my*(mz*mz * Jx/Ix - mx*mx*Jz/Iz)) \
                - 0.5*dt*(mz_new*mx_new*Jy + 0.5*dt*tau*my_new*(mz_new*mz_new * Jx/Ix - mx_new*mx_new*Jz/Iz))
        fz = mz_new - mz \
                - 0.5*dt*(mx*my*Jz + 0.5*dt*tau*mz*(mx*mx * Jy/Iy - my*my*Jx/Ix)) \
                - 0.5*dt*(mx_new*my_new*Jz + 0.5*dt*tau*mz_new*(mx_new*mx_new * Jy/Iy - my_new*my_new*Jx/Ix))

        return (fx, fy, fz)

    def m_new(self, with_entropy = False): #return new m and update RB
        #calculate
        m_new = fsolve(self.f, (self.mx, self.my, self.mz))

        #update
        self.mx = m_new[0]
        self.my = m_new[1]
        self.mz = m_new[2]

        return m_new

class RBSeReFE(RigidBody):#SeRe forward Euler
    def __init__(self, Ix, Iy, Iz, mx, my, mz, dt, tau):
        super(RBSeReFE, self).__init__(Ix, Iy, Iz, mx, my, mz, dt, tau)

    def m_new(self, with_entropy = False):
        mx = self.mx
        my = self.my
        mz = self.mz

        dt = self.dt
        tau = self.tau
       
        Ix = self.Ix
        Iy = self.Iy
        Iz = self.Iz

        Jx = self.Jx
        Jy = self.Jy
        Jz = self.Jz

        #calculate
        mx_new = mx + dt*my*mz*Jx + 0.5*dt*tau*mx*(my*my * Jx*Jz + mz*mz * Jx*Jy)
        my_new = my + dt*mz*mx*Jy + 0.5*dt*tau*my*(mz*mz * Jy*Jx + mx*mx * Jy*Jz)
        mz_new = mz + dt*mx*my*Jz + 0.5*dt*tau*mz*(mx*mx * Jz*Jy + my*my * Jz*Jx)

        #update
        self.mx = mx_new
        self.my = my_new
        self.mz = mz_new
    
        if with_entropy: #calculate new entropy using explicit forward Euler
            sin_new = self.sin+ 0.5*(tau-dt)*dt/self.Ein_s() * ((my*mz*Jx)**2/Ix + (mz*mx*Jy)**2/Iy + (mx*my*Jz)**2/Iz)
            self.sin = sin_new

        return (mx_new, my_new, mz_new)


