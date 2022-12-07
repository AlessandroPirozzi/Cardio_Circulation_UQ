#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms

# Errors with 2 parameters
training_values_2 = np.array([102, 504, 1002, 2004])
RMSE_2 = np.array([0.04626644145980012, 0.02068404654964616, 0.014123392015951884, 0.013217517739680251])
L_inf_2 = np.array([0.0946078042662597, 0.03566893201804322, 0.027911761761092002, 0.024340386679165715])

fig1 = plt.figure(figsize=(10, 6))
plt.plot(training_values_2, RMSE_2, '-o')
plt.xlabel("Number of training values", fontsize=12)
plt.ylabel("RMSE", fontsize=12)
fig1.savefig('RMSE_2_training_values_circulation.png')

fig2 = plt.figure(figsize=(10, 6))
plt.plot(training_values_2, L_inf_2, '-o')
plt.xlabel("Number of training values", fontsize=12)
plt.ylabel("$L^{\infty}$ error", fontsize=12)
fig2.savefig('L_inf_2_training_values_circulation.png')

# Errors with 4 parameters
training_values_4 = np.array([100, 500, 1000, 2000])
RMSE_4 = np.array([0.15579055696027097, 0.1313369629621503, 0.0706311990679735, 0.017628040650113395])
L_inf_4 = np.array([0.760086810397875, 0.40652511861487994, 0.24364289779408388, 0.06901168457548512])

fig3 = plt.figure(figsize=(10, 6))
plt.plot(training_values_4, RMSE_4, '-o')
plt.xlabel("Number of training values", fontsize=12)
plt.ylabel("RMSE", fontsize=12)
fig3.savefig('RMSE_4_training_values_circulation.png')

fig4 = plt.figure(figsize=(10, 6))
plt.plot(training_values_4, L_inf_4, '-o')
plt.xlabel("Number of training values", fontsize=12)
plt.ylabel("$L^{\infty}$ error", fontsize=12)
fig4.savefig('L_inf_4_training_values_circulation.png')

# Errors with 8 parameters
training_values_8 = np.array([108, 504, 1008, 2016])
RMSE_8 = np.array([0.22849883244643318, 0.14602280410233412, 0.0718855949902265, 0.0697726054034505])
L_inf_8 = np.array([0.7625417703448948, 0.4295612282778199, 0.26192591688989797, 0.22002986634113126])

fig5 = plt.figure(figsize=(10, 6))
plt.plot(training_values_8, RMSE_8, '-o')
plt.xlabel("Number of training values", fontsize=12)
plt.ylabel("RMSE", fontsize=12)
fig5.savefig('RMSE_8_training_values_circulation.png')

fig6 = plt.figure(figsize=(10, 6))
plt.plot(training_values_8, L_inf_8, '-o')
plt.xlabel("Number of training values", fontsize=12)
plt.ylabel("$L^{\infty}$ error", fontsize=12)
fig6.savefig('L_inf_8_training_values_circulation.png')

# Errors with 12 parameters
training_values_12 = np.array([104, 520, 1014, 2002])
RMSE_12 = np.array([0.25074707085253865, 0.1448105223649905, 0.08226285145251008, 0.08209708082465887])
L_inf_12 = np.array([0.8019043302815625, 0.6473545602111763, 0.29016530805191604, 0.27825613842117236])

fig7 = plt.figure(figsize=(10, 6))
plt.plot(training_values_12, RMSE_12, '-o')
plt.xlabel("Number of training values", fontsize=12)
plt.ylabel("RMSE", fontsize=12)
fig7.savefig('RMSE_12_training_values_circulation.png')

fig8 = plt.figure(figsize=(10, 6))
plt.plot(training_values_12, L_inf_12, '-o')
plt.xlabel("Number of training values", fontsize=12)
plt.ylabel("$L^{\infty}$ error", fontsize=12)
fig8.savefig('L_inf_12_training_values_circulation.png')

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
fig9.savefig('RMSE_training_values_circulation.png')

fig10 = plt.figure(figsize=(10, 6))
plt.plot(training_values_2, L_inf_2, '-o')
plt.plot(training_values_4, L_inf_4, '-o')
plt.plot(training_values_8, L_inf_8, '-o')
plt.plot(training_values_12, L_inf_12, '-o')
plt.yscale('log')
plt.xlabel("Number of training values", fontsize=12)
plt.ylabel("$L^{\infty}$ error", fontsize=12)
plt.legend(['2 parameters', '4 parameters', '8 parameters', '12 parameters'], loc='best')
fig10.savefig('L_inf_training_values_circulation.png')

parameters = np.array([2, 4, 8, 12])
RMSE_100 = np.array([0.04626644145980012, 0.15579055696027097, 0.22849883244643318, 0.25074707085253865])
L_inf_100 = np.array([0.0946078042662597, 0.760086810397875, 0.7625417703448948, 0.8019043302815625])

RMSE_500 = np.array([0.02068404654964616, 0.1313369629621503, 0.14602280410233412, 0.1448105223649905])
L_inf_500 = np.array([0.03566893201804322, 0.40652511861487994, 0.4295612282778199, 0.6473545602111763])

RMSE_1000 = np.array([0.014123392015951884, 0.0706311990679735, 0.0718855949902265, 0.08226285145251008])
L_inf_1000 = np.array([0.027911761761092002, 0.24364289779408388, 0.26192591688989797, 0.29016530805191604])

RMSE_2000 = np.array([0.013217517739680251, 0.017628040650113395, 0.0697726054034505, 0.08209708082465887])
L_inf_2000 = np.array([0.024340386679165715, 0.06901168457548512, 0.22002986634113126, 0.27825613842117236])

fig11 = plt.figure(figsize=(10, 6))
plt.plot(parameters, RMSE_100, '-o')
plt.plot(parameters, RMSE_500, '-o')
plt.plot(parameters, RMSE_1000, '-o')
plt.plot(parameters, RMSE_2000, '-o')
plt.yscale('log')
plt.xlabel("Number of parameters", fontsize=12)
plt.ylabel("RMSE", fontsize=12)
plt.legend(['100', '500', '1000', '2000'], loc='best')
fig11.savefig('RMSE_parameters_circulation.png')

fig12 = plt.figure(figsize=(10, 6))
plt.plot(parameters, L_inf_100, '-o')
plt.plot(parameters, L_inf_500, '-o')
plt.plot(parameters, L_inf_1000, '-o')
plt.plot(parameters, L_inf_2000, '-o')
plt.yscale('log')
plt.xlabel("Number of parameters", fontsize=12)
plt.ylabel("$L^{\infty}$ error", fontsize=12)
plt.legend(['100', '500', '1000', '2000'], loc='best')
fig12.savefig('L_inf_parameters_circulation.png')

plt.show()