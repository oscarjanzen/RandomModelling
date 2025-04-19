#1d_linear_with_common_loss.py
#oscarjanzen
#20250418

import numpy as np
from scipy.integrate import solve_ivp 
from scipy import interpolate
import matplotlib.pyplot as plt

#%% Define Material class and example materials
class Material:
    def __init__(self,k,c,rho,epsilon):
        #Constructor
        self.k = k              #Thermal Conductivity [W/m*K]
        self.c = c              #Specific Heat Capacity [J/kg*K]
        self.rho = rho          #Density [kg/m^3]
        self.epsilon = epsilon  #Emissivity []
    
    def get_k(self,T=273.15):
        #Return thermal conductivity [W/m*K] at temperature T [K]
        if isinstance(self.k, np.ndarray):
            f = interpolate.interp1d(self.k[:,0],self.k[:,1],fill_value="extrapolate")
            return f(T)
        else:
            return self.k
        
    def get_c(self):
        #Return specific heat capacity [J/kg*K]
        return self.c
    
    def get_rho(self):
        #Return density [kg/m^3]
        return self.rho
    
    def get_epsilon(self):
        #Return total hemispherical emissivity []
        return self.epsilon

#Define copper material properties
#Source: https://www.matweb.com/search/DataSheet.aspx?MatGUID=3c78d450e90f48c48d68e2d17a8e51f7
#Source: https://www.omega.co.uk/literature/transactions/volume1/emissivitya.html
k_c110 = np.array([[-268.8, 300],[700, 300],[300,370],[100,380],[20,388],[0,390],[-79,400],[-196,550],[-253,1300]])
k_c110[:,0] = k_c110[:,0] + 273.15
C110 = Material(k_c110,c=385,rho=8890,epsilon=0.22)

#Define tungsten material properties
#Source: https://www.matweb.com/search/datasheet_print.aspx?matguid=41e0851d2f3c417ba69ea0188fa570e3
#Source: https://pubs.aip.org/aip/jap/article-abstract/31/8/1382/163099/Spectral-Emissivity-Total-Emissivity-and-Thermal?redirectedFrom=fulltext
k_W = np.array([[200,146],[600,128],[1000,117]])
k_W[:,0] = k_W[:,0] + 273.15
W = Material(k_W,c=134,rho=19300,epsilon=0.3)

#Define alumina material properties
#Source: https://www.omega.co.uk/literature/transactions/volume1/emissivityb.html
#Source: https://www.matweb.com/search/datasheet.aspx?matguid=0654701067d147e88e8a38c646dda195
Al2O3 = Material(k=30,c=880,rho=3900,epsilon=0.7)

#Define inconel 
#Source: https://asm.matweb.com/search/specificmaterial.asp?bassnum=ninc34
#Source: https://www.sciencedirect.com/science/article/abs/pii/S0029549315001004
Inconel718 = Material(k=11.4,c=435,rho=8190,epsilon=0.2)

#%% Define Tube class for storing geometric information and returning characteristic values
# Tube class has ID, OD, Length
# Tube class constructor takes a Material object, ID, OD, Length

class Tube:
    def __init__(self,Material,ID,OD,L):
        self.Material = Material
        self.ID = ID 
        self.OD = OD 
        self.L = L 
    
    def get_Ao(self):
        #Return area of outer diameter [m^2]
        return np.pi * self.OD * self.L
    
    def get_Ai(self):
        #Return area of inner diameter [m^2]
        return np.pi * self.ID * self.L 
    
    def get_A_end(self):
        #Return cross sectional area of tube end [m^2]
        return np.pi * ( ((self.OD/2)**2) - ((self.ID/2)**2) )
    
    def calc_C(self):
        #Return heat capacity [J/K]
        V = self.get_A_end() * self.L
        m = self.Material.get_rho() * V 
        c = self.Material.get_c()
        return c*m
    
    def calc_R_axial(self):
        L = self.L
        k = self.Material.get_k()
        A = self.get_A_end()
        return L/(k*A)

#%% Define enclosed two surface radiation resistance function

def R_twoSurfRadiation(innerTube,outerTube,F_12):
    
    if (innerTube.OD > outerTube.ID):
        ValueError("innerTube OD must be smaller than outerTube ID.")
    
    R_1 = (1-innerTube.Material.get_epsilon())/(innerTube.Material.get_epsilon() * innerTube.get_Ao())
    R_12 = 1/(innerTube.get_Ai() * F_12)
    R_2 = (1-outerTube.Material.get_epsilon())/(outerTube.Material.get_epsilon() * innerTube.get_Ai())
    
    sigma = 5.670374419e-8 #[W m^-2 K^-4]
    R_tot = (R_1 + R_12 + R_2) / sigma
    
    return R_tot

#%% Tube definitions
tube0 = Tube(Al2O3,0.010,0.020,0.010)
tube1 = Tube(W,0.021,0.022,0.010)

tube1_base = tube1
tube1_base.L = 0.050

tube2 = Tube(C110,0.025,0.026,0.010)
tube3 = Tube(C110,0.027,0.028,0.010)
tube4 = Tube(C110,0.029,0.030,0.010)

tube5 = Tube(Inconel718,0.040,0.046,0.010)

tube5_base = tube5
tube5_base.L = 0.050

#%% Construct Matricies

C_vect = np.array([-tube1.calc_C(),
                   -tube2.calc_C(),
                   -tube3.calc_C(),
                   -tube4.calc_C(),
                   -tube5.calc_C()])

K_vect = np.array([0,   #K_0-1
                   0,   #K_1-2
                   0,   #K_2-3
                   0,   #K_3-4
                   0])  #K_4-5

Kf_vect = np.array([1/tube1_base.calc_R_axial(),    #Kf_1
                    0,                              #Kf_2
                    0,                              #Kf_3
                    0,                              #Kf_4
                    1/tube5_base.calc_R_axial()])   #Kf_5

H_vect = np.array([1/R_twoSurfRadiation(tube0,tube1,1),   #H_0-1
                   1/R_twoSurfRadiation(tube1,tube2,1),   #H_1-2
                   1/R_twoSurfRadiation(tube2,tube3,1),   #H_2-3
                   1/R_twoSurfRadiation(tube3,tube4,1),   #H_3-4
                   1/R_twoSurfRadiation(tube4,tube5,1)])  #H_4-5

q = np.array([0,0,0,0,0])

C = np.diag(C_vect)
K = np.diag(K_vect[1:],1) + np.diag(K_vect[1:],-1) - (np.diag(K_vect) + np.diag(Kf_vect) + np.diag(np.append(K_vect[1:],0)))
H = np.diag(H_vect[1:],1) + np.diag(H_vect[1:],-1) - (np.diag(H_vect) + np.diag(np.append(H_vect[1:],0)))

B = np.zeros((np.size(C_vect,0),3))
B[0,0] = K_vect[0]
B[0,1] = H_vect[0]
B[:,2] = Kf_vect

#%% Define derivative function

T0 = 1000 + 273.15
Tf = 100 + 273.15
sim_inputs = {'q': q,
              'C': C,
              'K': K,
              'H': H,
              'B': B,
              'T0': T0,
              'Tf': Tf}

init_conds = (273.15 + 100) * np.ones_like(q)
Tmax = 1000
reltol = 1e-8
solver = 'LSODA'
dt = 0.1

def fun(t,T,inputs):
    q,C,K,H,B,T0,Tf = inputs.values()

    T4 = T**4
    
    dTdt = np.linalg.inv(-C) @ (q + K@T + H@T4 + B@np.array([T0,T0**4,Tf]))
    return dTdt

#%% Solve 

sol = solve_ivp(fun, [0, Tmax], max_step=dt, args=(sim_inputs,), y0=init_conds, rtol=reltol, method=solver)

q_in = ((T0 - sol.y[0]) * K_vect[0]) +  ((T0**4 - sol.y[0]**4) * H_vect[0])
q_loss = (sol.y[0] - Tf) * Kf_vect[0]
q_next = ((sol.y[0] - sol.y[1]) * K_vect[1]) +  ((sol.y[0]**4 - sol.y[1]**4) * H_vect[1])
q_store = -C_vect[0] * np.gradient(sol.y[0])/np.gradient(sol.t)

#%% Visualization

fig, ax = plt.subplots(3, 1, sharex=True)
p = 0
ax[p].plot(sol.t, sol.y[0], label='T1')
ax[p].plot(sol.t, sol.y[1], label='T2')
ax[p].plot(sol.t, sol.y[2], label='T3')
ax[p].plot(sol.t, sol.y[3], label='T4')
ax[p].plot(sol.t, sol.y[4], label='T5')
ax[p].grid('enable')
ax[p].set_ylabel('Temperature [K]')
ax[p].legend()

p = 1
ax[p].plot(sol.t, q_in, label='q_in')
ax[p].plot(sol.t, q_loss, label='q_loss')
ax[p].plot(sol.t, q_next, label='q_next')
ax[p].plot(sol.t, q_store, label='q_store')
ax[p].grid('enable')
ax[p].set_ylabel('Heat Flow [W]')
ax[p].legend()

p = 2
ax[p].plot(sol.t, q_in - q_loss - q_next - q_store)
ax[p].grid('enable')
ax[p].set_ylabel('Heat Balance [W]')
ax[p].set_xlabel('Time [s]')

#%% Check heat flow balance
q0 = (sol.y[3] - sol.y[4]) * K_vect[4]          #Conduction in
q1 = (sol.y[3]**4 - sol.y[4]**4) * H_vect[4]    #Radiation in
q2 = (sol.y[4] - Tf) * Kf_vect[4]               #Conduction out
q3 = C_vect[4] * np.gradient(sol.y[4])          #Storage out

fig, ax = plt.subplots(2, 1, sharex=True)
p = 0
ax[p].plot(sol.t, q0, label='q0')
ax[p].plot(sol.t, q1, label='q1')
ax[p].plot(sol.t, q0 + q1, label='q_in')
ax[p].plot(sol.t, q2, label='q2')
ax[p].plot(sol.t, q3, label='q3')

ax[p].grid('enable')
ax[p].set_ylabel('Heat Flow [W]')
ax[p].legend()

p = 1
ax[p].plot(sol.t, q0 + q1 - q2 - q3, label='Error')
ax[p].grid('enable')
ax[p].set_ylabel('Erroneous Heat Flow [W]')
#ax[p].legend()