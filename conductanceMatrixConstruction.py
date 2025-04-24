# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 11:51:03 2025

@author: Oscar Janzen
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

#Declare Minimal Conductance Matrix
A = np.array([[0,  1,  2,  3,  4],
              [0, 11, 12, 13, 14],
              [0,  0, 22, 23, 24],
              [0,  0,  0, 33, 34],
              [0,  0,  0,  0, 44]])

#Declare inputList, get stateList
nodeList = np.arange(0,np.size(A,axis=0),1)
inputList = np.array([1,3])
stateBool = np.ones_like(nodeList,dtype=bool)
stateBool[inputList] = False
stateList = nodeList[stateBool]
print(inputList)
print(stateList)

#Construct K_global, the global conductance matrix
diagK = np.sum(A,1)

K_global = np.triu(A,k=1) + np.transpose(np.triu(A,k=1)) + np.diag(diagK)

print(K_global)

#%% Constructing the Conductance Matrix
#Construct K (option 1)
K_rowsRemoved = np.delete(K_global,inputList,axis=0)
K_colsRemoved = np.delete(K_rowsRemoved,inputList,axis=1)
K = K_colsRemoved

print(K)

#Construct K (option 2)
K = K_global[np.ix_(stateList,stateList)]
print(K)

#%% Constructing the Input Matrix
#Construct B (option 1)
B = np.zeros((np.size(K,axis=0),np.size(inputList)))
for i in range(np.size(inputList)):
    keepIdx = np.ones(np.size(K_global,axis=0), dtype=bool)
    keepIdx[inputList] = False
    B[:,i] = K_global[inputList[i],keepIdx]

print(B)

#Construct B (option 2)
B_withDiagonals = K_global[:,inputList]
B = np.delete(B_withDiagonals,inputList,axis=0)
print(B)

#Construct B (option 3)
B = K_global[np.ix_(stateList,inputList)]
print(B)

#%% Function to Return K and B from A (Minimal Conductance Matrix) and inputList

def K_B_Constructor(A,inputList):
    #A must be square
    #Size of inputList must be between 0 and size(A)-1
    
    #Construct K_global, the global conductance matrix
    diagK = np.sum(A,1)
    K_global = np.triu(A,k=1) + np.transpose(np.triu(A,k=1)) - np.diag(diagK)
    
    #Create stateList and inputList
    nodeList = np.arange(0,np.size(K_global,axis=0),1)
    stateBool = np.ones_like(nodeList,dtype=bool)
    stateBool[inputList] = False
    stateList = nodeList[stateBool]
    
    #Construct K
    K = K_global[np.ix_(stateList,stateList)]
    
    #Construct B
    B = K_global[np.ix_(stateList,inputList)]
    
    return K,B

K,B = K_B_Constructor(A,inputList)
print(K)
print(B)

#%% Example use
# 5 nodes across, 2 high, lower row and top left node are inputs

A = np.array([[0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
              [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

inputList = np.array([0,5,6,7,8,9])

K,B = K_B_Constructor(A,inputList)
print(K)
print(B)

C = np.diag(np.ones(4))
q = np.zeros(4)

x0 = 0
T0 = 100
Tf = 0
sim_inputs = {'q': q,
              'C': C,
              'K': K,
              'B': B,
              'T0': T0,
              'Tf': Tf}

init_conds = Tf * np.ones_like(q)
Tmax = 1000
reltol = 1e-8
solver = 'LSODA'
dt = 0.1

def fun_transient(t,x,inputs):
    q,C,K,B,T0,Tf = inputs.values()
    
    dxdt = np.linalg.inv(C) @ (q + K@x + B@np.array([T0,Tf,Tf,Tf,Tf,Tf]))
    return dxdt

Tmax = 10
sol = solve_ivp(fun_transient, [0, Tmax], args=(sim_inputs,), y0=init_conds)

fig, ax = plt.subplots(1, 1, sharex=True)
ax.plot(sol.t, sol.y[0], label='T1')
ax.plot(sol.t, sol.y[1], label='T2')
ax.plot(sol.t, sol.y[2], label='T3')
ax.plot(sol.t, sol.y[3], label='T4')
plt.legend()
