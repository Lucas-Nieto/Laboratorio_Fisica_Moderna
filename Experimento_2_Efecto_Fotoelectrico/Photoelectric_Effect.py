# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 12:52:05 2022

Experimento 2: Efecto fotoeléctrico

Objetivos:
    - Determinar el voltaje de frenado de fotoelectrones para diferentes colores de luz
    - Determinar la constante de Planck y la función de trabajo del material de una fotocelda
    por medio de regresión lineal a los valores de voltaje de frenado y frecuencia de luz.

@author: Lucas Nieto
"""

# %% Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Al abrir en un notebook de Jupyter, sacar la línea siguiente del comentario
#%matplotlib inline

# %% Regresión e incertidumbre

def fit(x,y):
    regr = stats.linregress(x, y)
    slope = regr[0]
    intercept = regr[1]
    return slope*x + intercept

def sigma_slope(x,y):
    regr = stats.linregress(x, y)
    sigma = regr[4]
    return sigma

def barras(x,y):
    S_y = np.std(y)
    S_x = np.std(x)
    ybar = (S_y/np.sqrt(len(y))) # Error típico o estándar de un estadístico
    xbar = (S_x/np.sqrt(len(x)))
    return ybar,xbar

def intercept_stderr(x,y):
    """Adaptado del repositorio GitHub de Scipy por indexación que no permite extraer la 
    posición 5 de la tupla de LinRegress que contiene la incertidumbre del intercepto de 
    regresión. URL del repositorio: https://github.com/scipy/scipy
    
    URL del archivo Python original: 
        https://github.com/scipy/scipy/blob/main/scipy/stats/_mstats_basic.py
    
    Referencia del repositorio: 
        @ARTICLE{2020SciPy-NMeth,
                 author  = {Virtanen, Pauli and Gommers, Ralf and Oliphant, Travis E. and
                            Haberland, Matt and Reddy, Tyler and Cournapeau, David and
                            Burovski, Evgeni and Peterson, Pearu and Weckesser, Warren and
                            Bright, Jonathan and {van der Walt}, St{\'e}fan J. and
                            Brett, Matthew and Wilson, Joshua and Millman, K. Jarrod and
                            Mayorov, Nikolay and Nelson, Andrew R. J. and Jones, Eric and
                            Kern, Robert and Larson, Eric and Carey, C J and
                            Polat, {\.I}lhan and Feng, Yu and Moore, Eric W. and
                            {VanderPlas}, Jake and Laxalde, Denis and Perktold, Josef and
                            Cimrman, Robert and Henriksen, Ian and Quintero, E. A. and
                            Harris, Charles R. and Archibald, Anne M. and
                            Ribeiro, Ant{\^o}nio H. and Pedregosa, Fabian and
                            {van Mulbregt}, Paul and {SciPy 1.0 Contributors}},
                 title   = {{{SciPy} 1.0: Fundamental Algorithms for Scientific
                             Computing in Python}},
                 journal = {Nature Methods},
                 year    = {2020},
                 volume  = {17},
                 pages   = {261--272},
                 adsurl  = {https://rdcu.be/b08Wh},
                 doi     = {10.1038/s41592-019-0686-2},
                 }
    """
    n = len(x)
    xmean = np.mean(x)
    ssxm, ssxym, _, ssym = np.cov(x, y, bias=1).flat
    r = ssxym / np.sqrt(ssxm * ssym)
    df = n - 2  # Number of degrees of freedom
        # n-2 degrees of freedom because 2 has been used up
        # to estimate the mean and standard deviation
    slope_stderr = np.sqrt((1 - r**2) * ssym / ssxm / df)
        # Also calculate the standard error of the intercept
        # The following relationship is used:
        #   ssxm = mean( (x-mean(x))^2 )
        #        = ssx - sx*sx
        #        = mean( x^2 ) - mean(x)^2
    intercept_stderr = slope_stderr * np.sqrt(ssxm + xmean**2)
    return intercept_stderr

# %% Datasets por color

RedFile = pd.read_csv("Red.csv",sep = ";")
DataRed = RedFile.to_numpy().transpose().tolist()
x_R = np.array(DataRed[0])
y_R = np.array(DataRed[1])

YellowFile = pd.read_csv("Yellow.csv",sep = ";")
DataYellow = YellowFile.to_numpy().transpose().tolist()
x_Y = np.array(DataYellow[0])
y_Y = np.array(DataYellow[1])

GreenFile = pd.read_csv("Green.csv",sep = ";")
DataGreen = GreenFile.to_numpy().transpose().tolist()
x_G = np.array(DataGreen[0])
y_G = np.array(DataGreen[1])

BlueFile = pd.read_csv("Blue.csv",sep = ";")
DataBlue = BlueFile.to_numpy().transpose().tolist()
x_B = np.array(DataBlue[0])
y_B = np.array(DataBlue[1])

print("\n--------------------------------DATASETS-----------------------------------")
print("\nV (volts) vs I (amps) for RED")
print(RedFile)
print("\nV (volts) vs I (amps) for YELLOW")
print(YellowFile)
print("\nV (volts) vs I (amps) for GREEN")
print(BlueFile)
print("\nV (volts) vs I (amps) for BLUE")
print(BlueFile)

# %% Plot corriente vs voltaje para rojo, amarillo, verde, azul

ybarR,xbarR = barras(x_R,y_R)
ybarY,xbarY = barras(x_Y,y_Y)
ybarG,xbarG = barras(x_G,y_G)
ybarB,xbarB = barras(x_B,y_B)

plt.errorbar(x_R, y_R, fmt = "o", color = "r", yerr = ybarR, ecolor = "k")
plt.plot(x_R,fit(x_R,y_R),linestyle='dashdot', c = "r", label = "Luz roja")

plt.errorbar(x_Y, y_Y, fmt = "o", color = "y", yerr = ybarY, ecolor = "k")
plt.plot(x_Y,fit(x_Y,y_Y),linestyle='dashdot', c = "y", label = "Luz amarilla")

plt.errorbar(x_G, y_G, fmt = "o", color = "g", yerr = ybarR,  ecolor = "k")
plt.plot(x_G,fit(x_G,y_G),linestyle='dashdot', c = "g", label = "Luz verde")

plt.errorbar(x_B, y_B, fmt = "o", color = "b", yerr = ybarB, ecolor = "k")
plt.plot(x_B,fit(x_B,y_B),linestyle='dashdot', c = "b", label = "Luz azul")

plt.legend()
plt.title(r"$Corriente\:(A\times10^{8})\:vs\:Voltaje\:({V})\:por\:colores$")
plt.grid()
plt.show()

# %% Reporte de voltajes de frenado

Frenado_R = (stats.linregress(x_R,y_R))[1]
Frenado_Y = (stats.linregress(x_Y,y_Y))[1]
Frenado_G = (stats.linregress(x_G,y_G))[1]
Frenado_B = (stats.linregress(x_B,y_B))[1]

S_Frenado_R = intercept_stderr(x_R,y_R)
S_Frenado_Y = intercept_stderr(x_Y,y_Y)
S_Frenado_G = intercept_stderr(x_G,y_G)
S_Frenado_B = intercept_stderr(x_B,y_B)

print("\n-----------------------Voltajes de frenado por color-----------------------")
print("\nRojo:")
print(Frenado_R,"±",S_Frenado_R,"Volts")
print("\nAmarillo:")
print(Frenado_Y,"±",S_Frenado_Y,"Volts")
print("\nVerde:")
print(Frenado_G,"±",S_Frenado_G,"Volts")
print("\nAzul:")
print(Frenado_B,"±",S_Frenado_B,"Volts")

# %% Plot hc + phi

c = 299792458 # Velocidad de la luz en el vacío en m/s
              # CODATA recommended values of the fundamental physical constants: 2018
              
e = 1.602176634e-19 # Carga elemental del electrón
                    # CODATA recommended values of the fundamental physical constants: 2018
                    
red_wavelength = 660        # Longitud de onda en nanómetros por color
yellow_wavelength = 590     # Broadbent, A. D. (2004). A critical review of the development of 
green_wavelength = 530      # the CIE1931 RGB color‐matching functions. (RGB)
blue_wavelength = 460       # VCG OPTOELECTRONICS. SUPERBRIGHT LED LAMP VAOL-5GCE4, url:
                            # https://www.farnell.com/datasheets/789166.pdf (amarillo)
                            

lambda_list = [red_wavelength,yellow_wavelength,green_wavelength,blue_wavelength]

Freq = np.array(lambda_list)
Freq = c*1e9/Freq               # Se convierte nanómetros a Hz

V_list = [Frenado_R,Frenado_Y,Frenado_G,Frenado_B]

V_list = np.array(V_list)
V_list = e*V_list               # Voltaje de frenado por carga elemental igual a K_máx
    
h = (stats.linregress(Freq,V_list))[0]      # Valor de h en Joules
phi = (stats.linregress(Freq,V_list))[1]    # Valor de función trabajo en Joules

phi = phi*6.242e18                          # Se vuelve a convertir a phi a eV
phi = abs(phi)

h = h*6.242e18                              # Se vuelve a convertir a h a eV

s_h = sigma_slope(Freq,V_list)              # Incertidumbre de h
s_phi = intercept_stderr(Freq,V_list)       # Incertidumbre de phi

s_h = s_h*6.242e18 
s_phi = s_phi*6.242e18

ybar = barras(Freq, V_list)[0]              # Barras de error


plt.errorbar(Freq, V_list, fmt = "o", color = "k", yerr = ybar, ecolor = "k")
plt.plot(Freq,fit(Freq,V_list),linestyle='dashdot', c = "b",)
#plt.legend()
plt.title(r"Regresión lineal de $ eV \: = \: \frac{1}{\lambda}$")
plt.grid()
plt.show()

# %% Reporte de constantes

print("\n---------------------------Constantes del ajuste---------------------------")
print("\nConstante de Planck:")
print(h,"±",s_h,"eV s")
print("\nFunción trabajo de la fotocelda:")
print(phi,"±",s_phi,"eV")

