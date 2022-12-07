#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms

# Errors with 2 parameters
training_values_2 = np.array([102, 504, 1002, 2004])
RMSE_2 = np.array([0.06854100047747748, 0.039513784059144894, 0.03479132590184217, 0.03477243051802245])
L_inf_2 = np.array([0.3370280258490137, 0.10888119691374909, 0.09368802186928205, 0.09252125927678225])

fig1 = plt.figure(figsize=(10, 6))
plt.plot(training_values_2, RMSE_2, '-o')
plt.xlabel("Number of training values", fontsize=12)
plt.ylabel("RMSE", fontsize=12)
fig1.savefig('RMSE_2_training_values.png')

fig2 = plt.figure(figsize=(10, 6))
plt.plot(training_values_2, L_inf_2, '-o')
plt.xlabel("Number of training values", fontsize=12)
plt.ylabel("$L^{\infty}$ error", fontsize=12)
fig2.savefig('L_inf_2_training_values.png')

# Errors with 4 parameters
training_values_4 = np.array([100, 500, 1000, 2000])
RMSE_4 = np.array([0.1902279701081096, 0.11138153345239732, 0.06101803061157883, 0.03510108603574594])
L_inf_4 = np.array([0.7558795121316383, 0.477500778610187, 0.3028109146441579, 0.16037507125354975])

fig3 = plt.figure(figsize=(10, 6))
plt.plot(training_values_4, RMSE_4, '-o')
plt.xlabel("Number of training values", fontsize=12)
plt.ylabel("RMSE", fontsize=12)
fig3.savefig('RMSE_4_training_values.png')

fig4 = plt.figure(figsize=(10, 6))
plt.plot(training_values_4, L_inf_4, '-o')
plt.xlabel("Number of training values", fontsize=12)
plt.ylabel("$L^{\infty}$ error", fontsize=12)
fig4.savefig('L_inf_4_training_values.png')

# Errors with 8 parameters
training_values_8 = np.array([108, 504, 1008, 2016])
RMSE_8 = np.array([0.20575551936394894, 0.11436378460620394, 0.0766614495021245, 0.04457386201167802])
L_inf_8 = np.array([0.6299016114371115, 0.4832260204983799, 0.3165354325415676, 0.26734336248097148])

fig5 = plt.figure(figsize=(10, 6))
plt.plot(training_values_8, RMSE_8, '-o')
plt.xlabel("Number of training values", fontsize=12)
plt.ylabel("RMSE", fontsize=12)
fig5.savefig('RMSE_8_training_values.png')

fig6 = plt.figure(figsize=(10, 6))
plt.plot(training_values_8, L_inf_8, '-o')
plt.xlabel("Number of training values", fontsize=12)
plt.ylabel("$L^{\infty}$ error", fontsize=12)
fig6.savefig('L_inf_8_training_values.png')

# Errors with 12 parameters
training_values_12 = np.array([104, 520, 1014, 2002])
RMSE_12 = np.array([0.2672132364676653, 0.1872134689963411, 0.11566561399238973, 0.10552059756203953])
L_inf_12 = np.array([0.8972870340285889, 0.7272775495233889, 0.5298389153552674, 0.4140849515107854])

fig7 = plt.figure(figsize=(10, 6))
plt.plot(training_values_12, RMSE_12, '-o')
plt.xlabel("Number of training values", fontsize=12)
plt.ylabel("RMSE", fontsize=12)
fig7.savefig('RMSE_12_training_values.png')

fig8 = plt.figure(figsize=(10, 6))
plt.plot(training_values_12, L_inf_12, '-o')
plt.xlabel("Number of training values", fontsize=12)
plt.ylabel("$L^{\infty}$ error", fontsize=12)
fig8.savefig('L_inf_12_training_values.png')

# Errors together
fig9 = plt.figure(figsize=(10, 6))
plt.plot(training_values_2, RMSE_2, '-o')
plt.plot(training_values_4, RMSE_4, '-o')
plt.plot(training_values_8, RMSE_8, '-o')
plt.plot(training_values_12, RMSE_12, '-o')
plt.yscale('log')
plt.xlabel("Number of training values", fontsize=12)
plt.ylabel("RMSE", fontsize=12)
plt.legend(['2 parameters', '4 parameters', '8 parameters', '12 parameters'], loc='best')
fig9.savefig('RMSE_training_values.png')

fig10 = plt.figure(figsize=(10, 6))
plt.plot(training_values_2, L_inf_2, '-o')
plt.plot(training_values_4, L_inf_4, '-o')
plt.plot(training_values_8, L_inf_8, '-o')
plt.plot(training_values_12, L_inf_12, '-o')
plt.yscale('log')
plt.xlabel("Number of training values", fontsize=12)
plt.ylabel("$L^{\infty}$ error", fontsize=12)
plt.legend(['2 parameters', '4 parameters', '8 parameters', '12 parameters'], loc='best')
fig10.savefig('L_inf_training_values.png')

parameters = np.array([2, 4, 8, 12])
RMSE_100 = np.array([0.06854100047747748, 0.1902279701081096, 0.20575551936394894, 0.2672132364676653])
L_inf_100 = np.array([0.3370280258490137, 0.7558795121316383, 0.6299016114371115, 0.8972870340285889])

RMSE_500 = np.array([0.039513784059144894, 0.11138153345239732, 0.11436378460620394, 0.1872134689963411])
L_inf_500 = np.array([0.10888119691374909, 0.477500778610187, 0.4832260204983799, 0.7272775495233889])

RMSE_1000 = np.array([0.03479132590184217, 0.06101803061157883, 0.0766614495021245, 0.11566561399238973])
L_inf_1000 = np.array([0.09368802186928205, 0.3028109146441579, 0.3165354325415676, 0.5298389153552674])

RMSE_2000 = np.array([0.03477243051802245, 0.03510108603574594, 0.04457386201167802, 0.10552059756203953])
L_inf_2000 = np.array([0.09252125927678225, 0.16037507125354975, 0.26734336248097148, 0.4140849515107854])

fig11 = plt.figure(figsize=(10, 6))
plt.plot(parameters, RMSE_100, '-o')
plt.plot(parameters, RMSE_500, '-o')
plt.plot(parameters, RMSE_1000, '-o')
plt.plot(parameters, RMSE_2000, '-o')
plt.yscale('log')
plt.xlabel("Number of parameters", fontsize=12)
plt.ylabel("RMSE", fontsize=12)
plt.legend(['100', '500', '1000', '2000'], loc='best')
fig11.savefig('RMSE_parameters.png')

fig12 = plt.figure(figsize=(10, 6))
plt.plot(parameters, L_inf_100, '-o')
plt.plot(parameters, L_inf_500, '-o')
plt.plot(parameters, L_inf_1000, '-o')
plt.plot(parameters, L_inf_2000, '-o')
plt.yscale('log')
plt.xlabel("Number of parameters", fontsize=12)
plt.ylabel("$L^{\infty}$ error", fontsize=12)
plt.legend(['100', '500', '1000', '2000'], loc='best')
fig12.savefig('L_inf_parameters.png')

plt.show()