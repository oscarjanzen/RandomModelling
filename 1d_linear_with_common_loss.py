#1d_linear_with_common_loss.py
#oscarjanzen
#20250418

import numpy as np
#from scipy.integrate import 
from scipy import interpolate

class Material:
    def __init__(self,k,c,rho,epsilon):
        self.k = k              #Thermal Conductivity [W/m*K]
        self.c = c              #Specific Heat Capacity [J/kg*K]
        self.rho = rho          #Density [kg/m^3]
        self.epsilon = epsilon  #Emissivity []
    
    def get_k(self,T=273.15):
        if isinstance(self.k, np.ndarray):
            f = interpolate.interp1d(self.k[:,0],self.k[:,1],fill_value="extrapolate")
            return f(T)
        else:
            return self.k
        
    def get_c(self):
        return self.c
    
    def get_rho(self):
        return self.rho
    
    def get_epsilon(self):
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

