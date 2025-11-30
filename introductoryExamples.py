"""
Created on Sun November 30 2025

@author: Oscar Janzen
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

#Start with constituitive equation table for different domains
#Note that numerical modelling is built around solving systems of equations for both simple and complex systems
#You are setting up the equations as a representation of a conserved quantity moving through the system, such as energy or mass
#Give example of non-dynamic (is that the term?) equations and ones with derivatives
#You end up with a system of equations, a system of differential equations, or a combination called a system of Differential-Algebraic Equations (DAEs) or Partial Differential Algebraic Equations (PDAEs)
#Once you have your equations, you need to solve them to answer some kind of question. You may also need to solve for an initial condition. That won't be covered here. This will focus on situations where you know or assume an initial condition
#Give preferred SI units for all constants


#%% Example 0: Electrical Resistor Network
#Simple network, show how the system of equations is algebraic and can be solved by hand using row reduction, or numerically using linalg.solve()
#NOTE: It's best to set up your equations with variable coefficients, rather than with 
#Verify simple network equivalent resistance using manual resistor reduction
#Present a more complex system and imagine solving by hand; show the fast solution time with linalg.solve() and validate with SPICE.

#Simple Network parameters
R = np.array((1,1,1,1))
I_in = 1
V_2 = 0

#Simple Network, solved with np.linalg.fzero()
#Note that this requires a guess, which is annoying.
def resistorNetwork(x,*params):
    #note that the elements of x are:
        #x[0] = V_0
        #x[1] = V_1
        #x[2] = I_out
        
    #additional arguments need to be unpacked
    R,I_in,V_2 = params
    
    #construct equations
    out = np.zeros((3,))
    out[0] = I_in + ( (-1/R[0]) + (-1/R[1]) )*x[0] + ( (1/R[0]) + (1/R[1]) )*x[1]
    out[1] = ( (1/R[0]) + (1/R[1]) )*x[0] + ( (-1/R[0]) + (-1/R[1]) + (-1/R[2]) + (-1/R[3]) )*x[1]+ ( (1/R[2]) + (1/R[3]) )*V_2
    out[2] = ( (1/R[2]) + (1/R[3]) )*x[1] + ( (-1/R[2]) + (-1/R[3]) )*V_2 - x[2]
    
    return out

x0 = np.array((1,1,1))
x = fsolve(resistorNetwork,x0,args=(R,I_in,V_2))

print("-----Result using scipy.optimize.fsolve-----")
print("x="+str(x))
print("V_0   = "+str(x[0]))
print("V_1   = "+str(x[1]))
print("I_out = "+str(x[2]))

#Simple Network, solved with np.linalg.solve()
#Note that this did not require a guess. Aren't matrix methods nice?

#b = Ax
b = np.array((
                (-I_in),
              (((-1/R[2])+(-1/R[3]))*V_2),
              (((1/R[2])+(1/R[3]))*V_2)
             ))

A = np.array((
                ((-1/R[0])+(-1/R[1]),   ((1/R[0])+(1/R[1])),                        0),
                (((1/R[0])+(1/R[1])),   (-1/R[0])+(-1/R[1])+(-1/R[2])+(-1/R[3]),    0),
                (0,                     ((1/R[2])+(1/R[3])),                        -1)
            ))

x = np.linalg.solve(A, b)

print("-----Result using np.linalg.solve-----")
print("x="+str(x))
print("V_0   = "+str(x[0]))
print("V_1   = "+str(x[1]))
print("I_out = "+str(x[2]))

#Simple Network, solved with np.linalg.fzero()

#Complex Network


#%% Example 1: First-order Thermal RC System
#Introduce solving an equation with a time derivative and the types of inputs

#%% Example 2: Second-order Mass-Spring-Damper Mechanical Vibrations
#Model initial state and ring-down; check with closed form equations
#Model step input; check with closed form equations
#Create bode plot for system, do TDS to verify amplitude and phase, show stabilization time and when the transient response dies out (also use steady state equation)
#Verify natural frequency using closed form equations and bode plot

#%% Example 3: Cantilever Beam Vibration
#Use this example to illustrate matrix representation
#Introduce state-space as a framework in contrast to Mass Spring and Damping matricies
#Illustrate automated M, K, and C matrix construction
#Try to find closed form equation for cantilever beam vibration and validate sim
#Pretty plot of nodes during ring-down

#%% Example 4: Complex Thermal Network with Nonlinear Terms (Fan Convection and Radiation)
#Model as a 3x3 square grid, heaters at the edges, TEC in the middle, sensors on all 9 nodes. Convective cooling in plane of plate from one side
#Do a simple steady state transient, showing how the system behaves with a fixed input
#Illustrate the power of arbitrary inputs with pretty IO plots
#Introduce radiation to illustrate how state space does not permit non-linear terms; empahsize it works for LINEAR systems. Also convergence
#Introduce the idea of control of a system like this; if we can heat and cool, could we design a controller using this model?
#Introduce the idea of multi-physics simulations where the heater q input could come from a thermo-electric model of the heater with its own dynamics based on current and temperature

#%% Example 5: Fluid Flow with Nonlinear Terms and Complex Constituitive Equations
#Illustrate how coolprop can be used to model a two stage regulator with orifices
#Note the difficulty of convergence
#Note use of events

#%% Example 6: Second-order Mass-Spring-Damper with Position Control (SISO)
#Write out equations, create matrix model, for a mass on a cart with viscous damping between the cart and ground, and a spring between the prime mover and mass
#Show illustration of how to conceptually translate the cart to a lumped parameter illustration
#Introduce PI controller, show how to integrate into the TDS
#Write out coupled dynamics and controller
#Show bode plot and what it means
#Show pole-zero plot and what it means (compare TDS with pole zero)
#Show hand-tuning, and overlaid plots of step response with different K and I
#Show LQR and optimal response

#%% Example 7: Complex Thermal Network with Temperature Control (MIMO, Optimal Full State Feedback Control)
#Re-use convective plate model 
#Write out non-linear system of equations, do TDS with arbitrary inputs
#Linearize the system about a chosen operating point, design controller for this point
#Check controller performance when full non-linear system is modelled
#Check controller performance when full non-linear system is modelled, and there is a disturbance

#%% Example 8: Non-cartesian coordinate systems
# Introduce the value of alternate coordinate systems by asking how one might efficiently model a hot, multilayer sphere cooling due to cross-flow
# Imagine a sensitive device surrounded by insulation with an outer metal shell
# Get convective heat transfer equation for a sphere in cross flow
# Simulate transient response of the layers and point out that it would be more work to model as a cube surrounded by cubes; you end up with more nodes
# Emphasize that the lumped thermal resistance is DIFFERENT for a unit element of cubic and spherical shape; use diagram from Incropera
# Show how, assuming biot number looks ok, you can reduce it to 3 nodes
# You can answer the question of heating time

#%% Thoughts and extentions:
# Why use numerical modelling? I don't like math, so why don't I just build the system and tune by hand?
#   Perfectly reasonable approach, though depends on the situation
#   You can get away with a "buy it and try it" approach for cheap easy-to-build systems. Might even be faster if you are not confident with the modelling approace
#   You could basically empirically find the effects of K and I, as in example 6
#   What if you have expensive hardware and make the system unstable? How fast can you hit an e-stop before your hardware is irreparably damaged?
#   If you are faced with example 7, it can be nearly impossible to get the performance you want by hand tuning if the dynamics are closely coupled and/or your performance requirements are tight
#   Hand tuning can also be hard for systems that respond very slowly. Update example and show stabilization times of 1hr. Point out that you would consume an 8 hour day before you were 4 iterations into your tuning particularly if cooldown is slow
#   It is also worth remembering that everything is a model, and all models have errors and simplifications that will make the sim result different from reality. Controllers may still need to be fine-tuned by hand on real hardware, but you usually save a ton of time doing coarse tuning in sims. 
#   If you are able to do this kind of modelling, you are prepared to use more advanced techniques like system identification, MPC, and use real-world data to tune your model. Doing so can get you a very accurate and useful model. 
# Why am I writing my own code to do simulations when I could just use Ansys or a similar software package? Won't mathworks simscape do the same thing? Couldn't I use Simulink once I have my constituitive equations?
#   Apart from obvious things like license cost and license availablity in a large organization...
#   FEM packages are great and can be used in conjunction with lumped or reduced order models. You might need CFD to get Cv factors that could feed into a TDS sim. Or FEA to get an equivalent stiffness of a complex mechanical structure
#   You do not need to use one or the other, and the function of your own code and commercial code can be complementary.
#   However, these techniques are valuable for more advanced techniques like observability and controllability analysis, and the design of Kalman filters to simplify a sensor arrangement or reduce DAQ equiptment capabilty/cost. Ansys can't do this out of the box, but you might be able to export the matricies it generates or do equivalent processing using APDL.
#   You can use a reduced order model for control design (ex LQR) then export the actuator output from your sim and into Ansys, then check that the reduced order model agrees with the finely meshed model
#   Reduced order and lumped models will run much faster than FEM ones, and depending on the answer you are trying to get, several hours of solve time on an expensive computer could be traded for seconds on a modest computer to answer questions like stabilization time or to get approximate controller parameters.
#   Suggest the work of Steve Brunton on cross flow mode shapes and how much faster it is to simulate than full CFD
# Where can I find constituitive equations? What if I want to model a device with physics that don't neatly fall into the electrical, thermal, mechanical, or fluid buckets?
#   Suggest textbooks or wikipedia pages for thermal, fluids, electrical devices, etc.
#   For complex or niche devices, like piezoelectrics or TECs, 
#   Suggest google keywords that turn up handy results
#   Suggest mathworks pages on devices; they kindly give the equations at the core of their device models as well as literature references
#   Suggest energy balance as a framework for understanding devices
# Extentions:
#   All of the above is in the continuous time domain. For systems involving digital elements or digital control, you need to look into z-transforms and delay.
# Cool additional resources??
#   Steve Brunton's book and youtube
#   Brian Douglas
#   Modellica language
#   