from scipy.optimize import fsolve

class Solver(object):
    def __init__(self, Ix, Iy, Iz, dt, tau):
        self.Ix = Ix
        self.Iy = Iy
        self.Iz = Iz
        
        self.Jx = 1/Iz - 1/Iy
        self.Jy = 1/Ix - 1/Iz
        self.Jz = 1/Iy - 1/Ix

        self.dt = dt
        self.tau = tau

        self.mx = 0
        self.my = 0
        self.mz = 0

class CrankNicolsonESeRe(Solver):
    def __init__(self, Ix, Iy, Iz, dt, tau):
        super(CrankNicolsonESeRe, self).__init__(Ix, Iy, Iz, dt, tau)

    def f(self, m_new):
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

    def m_new(self, m):
        self.mx = m[0]
        self.my = m[1]
        self.mz = m[2]

        return fsolve(self.f, m)

    


