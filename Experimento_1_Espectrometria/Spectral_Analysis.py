# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 21:56:02 2022

Experimento 1: Espectrometría

Objetivo: Determinar la constante de Rydberg con datos de las líneas espectrales de la serie 
de Balmer exportados por Astrosurf IRIS

Updated on Sat Feb 26 07:23:59 2022:
    Correciones:
        - Corregido reporte de barras de error
        - Corregido cálculo de incertidumbre de pendiente usando stats.linregress

@author: Lucas Nieto
"""

# %% Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Al abrir en un notebook de Jupyter, sacar la línea siguiente del comentario
#%matplotlib inline

# %% Plots pixel vs grayscale value para hidrógeno y mercurio

"""Hidrógeno"""
HFile = pd.read_csv("Corrected_Hydrogen.csv",sep = ";")

DataH = HFile.to_numpy().transpose().tolist()

x_valuesH = np.array(DataH[0])
y_valuesH = np.array(DataH[1])

plt.scatter(x_valuesH, y_valuesH, c="crimson", s = 10)
plt.title("Espectro experimental del hidrógeno")
plt.xlabel("Pixels")
plt.ylabel("Grayscale intensity")
plt.grid()
plt.show()

"""Mercurio"""
HgFile = pd.read_csv("Corrected_Mercury.csv",sep = ";")

DataHg = HgFile.to_numpy().transpose().tolist()
x_valuesHg = np.array(DataHg[0])
y_valuesHg = np.array(DataHg[1])


plt.scatter(x_valuesHg, y_valuesHg, c="springgreen", s = 10)
plt.title("Espectro experimental del mercurio")
plt.xlabel("Pixels")
plt.ylabel("Grayscale intensity")
plt.grid()
plt.show()

# %% Conversión pixeles a nanómetros

def pix_to_nm(pix)->int:
    
    Lambda_H_Alpha = 656.2797 # Valor teórico de longitud de onda de la línea H_alpha
    wavelenght = (Lambda_H_Alpha*pix)/562 # Regla de 3 simple con el pixel en el que se encontró 
                                          # H_alpha en IRIS astronomy
    return wavelenght

C = True

while C == True:
    
    print()
    print("Presione 1 para hacer conversión de pixeles a nanómetros.")
    
    Prompt = input("Presione cualquier otra tecla para continuar: ")
    
    
    if Prompt == "1":
       
       conv = int(input("Digite la cantidad de pixeles que desea convertir a nanómetros: "))
       print(conv,"pixeles es equivalente a:",pix_to_nm(conv),"nanómetros")
    
    else:
        C = False
        
print("\n---------------------------------------------------------------------------")
        
# %% Cálculo de la constante

Experimental_wavelenghts = [656.2797,476.4062,417.8894,393.5343] # Manualmente encontradas con el 
                                                                 # slice de IRIS Astronomy

ToMeters = Experimental_wavelenghts.copy()

i = 0
while i in range(len(ToMeters)):
    ToMeters[i] = (ToMeters[i])*10e-10 # Conversión de nanómetros a metros (el factor de conversión
                                       # es 10e-9 pero por error de truncamiento del intérprete fue
                                       # necesario sumar 1 para tener orden de magnitud correcto)
    i+=1
 
y = ToMeters.copy()

i = 0
while i in range(len(y)):
    y[i] = 1/(y[i])
    i+=1
    
x = [0.1388,0.1875,0.2100,0.2222] # De los niveles de energía de la fórmula de Balmer

x = np.array(x)
y = np.array(y)

regr = stats.linregress(x, y)
slope = regr[0]
intercept = regr[1]

def fit(x):
    return slope*x + intercept

# Barras de error
S_y = np.std(y)
S_x = np.std(x)
ybar = (S_y/np.sqrt(len(y))) # Error típico o estándar de un estadístico
xbar = (S_x/np.sqrt(len(x)))

# Plot de regresión
rydberg = list(x)
i = 0
while i in range(len(rydberg)):
     rydberg[i] = rydberg[i]*1.0973e+07 # Valor reportado en la literaturya de la constante de Rydberg
     i += 1

plt.scatter(x,y, c = "midnightblue")
plt.errorbar(x, y, fmt = " ", yerr = ybar,ecolor = "k")
plt.plot(x,fit(x), c = "green", label = "R Experimental")
plt.plot(x,rydberg, linestyle='dashdot',c = "slateblue",label = "R Reportado por Beyer et al.")
plt.legend()
plt.title(r"Regresión lineal de $ \frac{1}{\lambda} = \left ( \frac{1}{n^{2}}- \frac{1}{m^{2}}\right )$")
plt.grid()
plt.show()

# %% Reporte de parámetros

# Pendiente de regresión
print("\nConstante experimental de Rydberg para el hidrógeno =","{:e}".format(slope))

# Incertidumbre de la pendiente
Sigma = regr[4]
print("\nIncertidumbre de la pendiente =","{:e}".format(Sigma))

# Coeficiente de correlación lineal
print(u"\nCoeficiente de correlación lineal R\u00b2 =",regr[2])

# Tamaño de barras de error
print("\nTamaño de barras de error:")
print("\n - Distancia de dato a límite =","{:e}".format(ybar))
print("\n - Distancia de límite a límite =","{:e}".format(2*ybar))
