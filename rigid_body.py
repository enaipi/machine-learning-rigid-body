import numpy as np
import math
from RigidBody import RBESeReCN
#import random

N = 2*100000 #number of steps

#initial condition
mx = 1.0
my = 0.3
mz = 0.3
print "Initial |m| = ", math.sqrt(mx**2+my**2+mz**2)

si = 0.0
print "Initial internal entropy =", si
print "Using E = E_kin + Ein(si)."
def Ein(s):
    return s
    #return math.exp(s)
def Ein_s(s):
    return 1.0
    #return math.exp(s)
print "Initial internal energy = ", Ein(si)

#Inertia
I = 10
Ix = 1.0*I
Iy = 2.0*I
Iz = 4.0*I
print "Initial Omega_x = ", mx/Ix
print "Initial Omega_y = ", my/Iy
print "Initial Omega_z = ", mz/Iz

print "Initial E_kin_x = ", 0.5*mx**2/Ix
print "Initial E_kin_y = ", 0.5*my**2/Iy
print "Initial E_kin_z = ", 0.5*mz**2/Iz

print "Initial kinetic E = ", 0.5*(mx**2/Ix + my**2/Iy + mz**2/Iz)
Jx = 1/Iz - 1/Iy
Jy = 1/Ix - 1/Iz
Jz = 1/Iy - 1/Ix
#Jxy = 1/Ix-1/Iy #=-Jz
#Jyx = -Jxy
#Jyz = 1/Iy-1/Iz = #=-Jx
#Jzy = -Jyz
#Jzx = 1/Iz-1/Ix #=-Jy
#Jxz = -Jzx

def Phi(mx_, my_, mz_):
    m2x = mx_**2
    m2y = my_**2
    m2z = mz_**2
    return -(m2x + m2y + m2z)/(2*Ix) + m2x/(2*Ix)+ m2y/(2*Iy)+ m2z/(2*Iz)
print "Initial Phi = ", Phi(mx, my, mz)


#time steps
#dt= 0.01 * 1/max([mx/Ix,my/Iy,mz/Iz])
dt= 0.05
dtau = 0.1*dt
print "dt = ", dt
print "dtau = ", dtau

solver = RBESeReCN(Ix,Iy,Iz,mx,my,mz,dt,dtau)

#preparing files
mfile = open("m.xyz",'w')
def store_m(values):
    result = str(values[0])
    for i in range(1, len(values)):
        result += " "+str(values[i])
    result += "\n"
    mfile.write(result)

store_m([0, mx,my,mz])
store_each = 1

try:
    #calculate evolution
    for i in range(N-1):
        #SeRe:
        #mx_new = mx + dt*my*mz*Jx + 0.5*dt*dtau*mx*(my*my * Jx*Jz + mz*mz * Jx*Jy)
        #my_new = my + dt*mz*mx*Jy + 0.5*dt*dtau*my*(mz*mz * Jy*Jx + mx*mx * Jy*Jz)
        #mz_new = mz + dt*mx*my*Jz + 0.5*dt*dtau*mz*(mx*mx * Jz*Jy + my*my * Jz*Jx)

        #Energetic SeRe:
        mx_new = mx + dt*my*mz*Jx + 0.5*dt*dtau*mx*(my*my * Jz/Iz - mz*mz*Jy/Iy)
        my_new = my + dt*mz*mx*Jy + 0.5*dt*dtau*my*(mz*mz * Jx/Ix - mx*mx*Jz/Iz)
        mz_new = mz + dt*mx*my*Jz + 0.5*dt*dtau*mz*(mx*mx * Jy/Iy - my*my*Jx/Ix)
        #Energetic SeRe with CN
        #mx_new, my_new, mz_new = solver.m_new()

        #Entropic SeRe:
        #mx_new = mx + dt*my*mz*Jx + 0.5*dt*dtau*mx*(-my*my * Jz/Iy + mz*mz*Jy/Iz)
        #my_new = my + dt*mz*mx*Jy + 0.5*dt*dtau*my*(-mz*mz * Jx/Iz + mx*mx*Jz/Ix)
        #mz_new = mz + dt*mx*my*Jz + 0.5*dt*dtau*mz*(-mx*mx * Jy/Ix + my*my*Jx/Iy)

        si_new = si+ 0.5*(dtau-dt)*dt/Ein_s(si) * ((my*mz*Jx)**2/Ix + (mz*mx*Jy)**2/Iy + (mx*my*Jz)**2/Iz)

        mx = mx_new
        my = my_new
        mz = mz_new
        si = si_new

        if i % store_each == 0:
            t = dt * i
            mx2 = mx*mx
            my2 = my*my
            mz2 = mz*mz
            #store_m([t, mx,my,mz, math.sqrt(mx2+my2+mz2), 10*0.5*mx2/Ix+0.5*my2/Iy+0.5*mz2/Iz,si,Ein(si), Phi(mx, my, mz)])
            store_m([t, mx,my,mz])
        if i % (N/10) == 0:
            print (i+0.0)/N*100, "%"
            print "|m| = ", math.sqrt(mx**2+my**2+mz**2)
            print "E = ", 0.5*(mx**2/Ix + my**2/Iy + mz**2/Iz)

finally:
    mfile.close()

print "Final time = ", N*dt
print "Final |m| = ", math.sqrt(mx**2+my**2+mz**2)
print "Final kinetic E = ", 0.5*(mx**2/Ix + my**2/Iy + mz**2/Iz)
print "Final total energy = ", Ein(si) + 0.5*(mx**2/Ix + my**2/Iy + mz**2/Iz)
print "Final E_kin_x = ", 0.5*mx**2/Ix
print "Final E_kin_y = ", 0.5*my**2/Iy
print "Final E_kin_z = ", 0.5*mz**2/Iz
print "Final Phi = ", Phi(mx,my,mz)


#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

#def print_mm(filter_step = 1):
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection='3d')
#
#    ax.plot(mx, my, mz, label='m')
#    #ax.plot(filter(mx,filter_step), filter(my,filter_step), filter(mz,filter_step), label='m')
#    ax.set_xlabel("mx")
#    ax.set_ylabel("my")
#    ax.set_zlabel("mz")
#    ax.legend()
#
#    plt.plot(m)
#plt.show()

print "SO3 simulation finished"
