#!/usr/bin/env python3
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
from circulation_closed_loop import circulation_closed_loop
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from SALib.sample import saltelli
from SALib.analyze import sobol

tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels
tf.enable_v2_behavior()

initstate = {
                "V_LA" : 89.088809,
                "V_LV" : 130.277735,
                "V_RA" : 64.54748000000001,
                "V_RV" : 108.88405300000001,
                "p_AR_SYS" : 69.93770400000001,
                "p_VEN_SYS" : 31.972224,
                "p_AR_PUL" : 21.771397,
                "p_VEN_PUL" : 20.205695000000002,
                "Q_AR_SYS": 59.93303100000001,
                "Q_VEN_SYS": 79.550838,
                "Q_AR_PUL": 50.873861,
                "Q_VEN_PUL": 47.035134
            }

def generate_data(index_points_, observation_noise_variance):
    observations_ = np.zeros(shape = (len(index_points_), 1))
    i = 0
    while i < len(index_points_):
        params = {
            "BPM": 75,
            "LA": {
                "EA": 0.07,
                "EB": 0.18,
                "TC": 0.17,
                "TR": 0.17,
                "tC": 0.90,
                "V0": 4.0
            },
            "LV": {
                "EA": 3.35,
                "EB": 0.2,
                "TC": 0.25,
                "TR": 0.40,
                "tC": 0.1,
                "V0": 42.0
            },
            "RA": {
                "EA": 0.06,
                "EB": 0.07,
                "TC": 0.17,
                "TR": 0.17,
                "tC": 0.90,
                "V0": 4.0
            },
            "RV": {
                "EA": 0.55,
                "EB": 0.05,
                "TC": 0.25,
                "TR": 0.4,
                "tC": 0.1,
                "V0": 16.0
            },
            "valves": {
                "Rmin": 0.0075,
                "Rmax": 75006.2
            },
            "SYS": {
                "R_AR": 0.64,
                "C_AR": 1.2,
                "R_VEN": 0.32,
                "C_VEN": 60.0,
                "L_AR": 5e-3,
                "L_VEN": 5e-4
            },
            "PUL": {
                "R_AR": 0.032116,
                "C_AR": 10.0,
                "R_VEN": 0.035684,
                "C_VEN": 16.0,
                "L_AR": 5e-4,
                "L_VEN": 5e-4
            }
        }
        for j in range(num_parameters):
            params[dict[j][0]][dict[j][1]] = index_points_[i][j]
        circ = circulation_closed_loop(options=params)
        history = circ.solve(num_cycles=5, initial_state=initstate)
        observations_[i] = (np.array([(history.pLV[-1000:]).max()]) + np.random.normal(loc=0, scale=np.sqrt(observation_noise_variance), size=None))
        i = i + 1
    return observations_.T

def generate_data_noise_free(index_points_):
    observations_ = np.zeros(shape = (len(index_points_), 1))
    i = 0
    while i < len(index_points_):
        params = {
            "BPM": 75,
            "LA": {
                "EA": 0.07,
                "EB": 0.18,
                "TC": 0.17,
                "TR": 0.17,
                "tC": 0.90,
                "V0": 4.0
            },
            "LV": {
                "EA": 3.35,
                "EB": 0.2,
                "TC": 0.25,
                "TR": 0.40,
                "tC": 0.1,
                "V0": 42.0
            },
            "RA": {
                "EA": 0.06,
                "EB": 0.07,
                "TC": 0.17,
                "TR": 0.17,
                "tC": 0.90,
                "V0": 4.0
            },
            "RV": {
                "EA": 0.55,
                "EB": 0.05,
                "TC": 0.25,
                "TR": 0.4,
                "tC": 0.1,
                "V0": 16.0
            },
            "valves": {
                "Rmin": 0.0075,
                "Rmax": 75006.2
            },
            "SYS": {
                "R_AR": 0.64,
                "C_AR": 1.2,
                "R_VEN": 0.32,
                "C_VEN": 60.0,
                "L_AR": 5e-3,
                "L_VEN": 5e-4
            },
            "PUL": {
                "R_AR": 0.032116,
                "C_AR": 10.0,
                "R_VEN": 0.035684,
                "C_VEN": 16.0,
                "L_AR": 5e-4,
                "L_VEN": 5e-4
            }
        }
        for j in range(num_parameters):
            params[dict[j][0]][dict[j][1]] = index_points_[i][j]
        circ = circulation_closed_loop(options=params)
        history = circ.solve(num_cycles=5, initial_state=initstate)
        observations_[i] = (np.array([(history.pLV[-1000:]).max()]))
        i = i + 1
    return observations_.T

def generate_data_one_parameter(index_points_, dict, observation_noise_variance):
    observations_ = np.zeros(shape = (len(index_points_), 1))
    i = 0
    while i < len(index_points_):
        params = {
            "BPM": 75,
            "LA": {
                "EA": 0.07,
                "EB": 0.18,
                "TC": 0.17,
                "TR": 0.17,
                "tC": 0.90,
                "V0": 4.0
            },
            "LV": {
                "EA": 3.35,
                "EB": 0.2,
                "TC": 0.25,
                "TR": 0.40,
                "tC": 0.1,
                "V0": 42.0
            },
            "RA": {
                "EA": 0.06,
                "EB": 0.07,
                "TC": 0.17,
                "TR": 0.17,
                "tC": 0.90,
                "V0": 4.0
            },
            "RV": {
                "EA": 0.55,
                "EB": 0.05,
                "TC": 0.25,
                "TR": 0.4,
                "tC": 0.1,
                "V0": 16.0
            },
            "valves": {
                "Rmin": 0.0075,
                "Rmax": 75006.2
            },
            "SYS": {
                "R_AR": 0.64,
                "C_AR": 1.2,
                "R_VEN": 0.32,
                "C_VEN": 60.0,
                "L_AR": 5e-3,
                "L_VEN": 5e-4
            },
            "PUL": {
                "R_AR": 0.032116,
                "C_AR": 10.0,
                "R_VEN": 0.035684,
                "C_VEN": 16.0,
                "L_AR": 5e-4,
                "L_VEN": 5e-4
            }
        }
        params[dict[0]][dict[1]] = index_points_[i]
        circ = circulation_closed_loop(options=params)
        history = circ.solve(num_cycles=5, initial_state=initstate)
        observations_[i] = (np.array([(history.pLV[-1000:]).max()]) + np.random.normal(loc=0, scale=np.sqrt(observation_noise_variance), size=None))
        i = i + 1
    return observations_.T

# Here the bounds are from -50% to +50% of the baseline value
dict = np.array((["LV", "EA"], ["LV", "EB"], ["LV", "TC"], ["LV", "TR"],
                 ["LA", "tC"], ["RA", "tC"], ["SYS", "R_AR"], ["SYS", "C_AR"],
                 ["SYS", "R_VEN"], ["SYS", "C_VEN"], ["PUL", "R_VEN"], ["PUL", "C_VEN"]))
problem = {
    'num_vars': 12,
    'names': ['EA_LV', 'EB_LV', 'TC_LV', 'TR_LV',
              'tC_LA', 'tC_RA', 'R_AR_SYS', 'C_AR_SYS',
              'R_VEN_SYS', 'C_VEN_SYS', 'R_VEN_PUL', 'C_VEN_PUL'],
    'bounds': [[1.675, 5.025], [0.1, 0.3], [0.125, 0.375], [0.2, 0.6],
               [0.45, 1.35], [0.45, 1.35], [0.32, 0.96], [0.6, 1.8],
               [0.16, 0.48], [30.0, 90.0], [0.017842, 0.053526], [8.0, 24.0]]
}
num_parameters = problem['num_vars']
'''
# 10 means 260 training values
observation_index_points_ = saltelli.sample(problem, 10) # N*(2D+2) with D num_vars and N chosen
num_observations = len(observation_index_points_)

observations_one_parameter_ = np.zeros(shape = (num_parameters, num_observations))
observations_one_parameter_[0] = generate_data_one_parameter(observation_index_points_.T[0], dict[0], observation_noise_variance=.01)
observations_one_parameter_[1] = generate_data_one_parameter(observation_index_points_.T[1], dict[1], observation_noise_variance=.01)
observations_one_parameter_[2] = generate_data_one_parameter(observation_index_points_.T[2], dict[2], observation_noise_variance=.01)
observations_one_parameter_[3] = generate_data_one_parameter(observation_index_points_.T[3], dict[3], observation_noise_variance=.01)
observations_one_parameter_[4] = generate_data_one_parameter(observation_index_points_.T[4], dict[4], observation_noise_variance=.01)
observations_one_parameter_[5] = generate_data_one_parameter(observation_index_points_.T[5], dict[5], observation_noise_variance=.01)
observations_one_parameter_[6] = generate_data_one_parameter(observation_index_points_.T[6], dict[6], observation_noise_variance=.01)
observations_one_parameter_[7] = generate_data_one_parameter(observation_index_points_.T[7], dict[7], observation_noise_variance=.01)
observations_one_parameter_[8] = generate_data_one_parameter(observation_index_points_.T[8], dict[8], observation_noise_variance=.01)
observations_one_parameter_[9] = generate_data_one_parameter(observation_index_points_.T[9], dict[9], observation_noise_variance=.01)
observations_one_parameter_[10] = generate_data_one_parameter(observation_index_points_.T[10], dict[10], observation_noise_variance=.01)
observations_one_parameter_[11] = generate_data_one_parameter(observation_index_points_.T[11], dict[11], observation_noise_variance=.01)

# Let's optimize to find the parameter values with highest posterior probability. Notice that we constrain them to be positive
constrain_positive = tfb.Shift(np.finfo(np.float64).tiny)(tfb.Exp())

amplitude_var = tfp.util.TransformedVariable(
    initial_value=0.1209,
    bijector=constrain_positive,
    name='amplitude',
    dtype=np.float64)

length_scale_var = tfp.util.TransformedVariable(
    initial_value=1.,
    bijector=constrain_positive,
    name='length_scale',
    dtype=np.float64)

observation_noise_variance_var = tfp.util.TransformedVariable(
    initial_value=0.0146,
    bijector=constrain_positive,
    name='observation_noise_variance',
    dtype=np.float64)

length_scales = []
for i in range(num_parameters):
    length_scale_one_parameter = tfp.util.TransformedVariable(
        initial_value=0.0816,
        bijector=constrain_positive,
        name='length_scale_' + problem['names'][i],
        dtype=np.float64)
    length_scales.append(length_scale_one_parameter)

xr_oip = np.zeros(num_parameters)
xl_oip = np.zeros(num_parameters)
i = 0
while i < num_parameters:
    xr_oip[i] = max(observation_index_points_.T[i])
    xl_oip[i] = min(observation_index_points_.T[i])
    i = i + 1
observation_index_points_trans_ = (2*observation_index_points_ - (xl_oip + xr_oip))/(xr_oip - xl_oip)

observations_one_parameter_trans_ = np.zeros(shape = (num_parameters, num_observations))
for i in range(num_parameters):
    xr_o = max(observations_one_parameter_[i])
    xl_o = min(observations_one_parameter_[i])
    observations_one_parameter_trans_[i] = (2 * observations_one_parameter_[i] - (xl_o + xr_o)) / (xr_o - xl_o)

num_iters = 5000
for i in range(num_parameters):
    optimizer = tf.optimizers.Adam()

    def build_gp(amplitude, length_scale, observation_noise_variance):
        # Create the covariance kernel
        kernel = tfk.ExponentiatedQuadratic(amplitude, length_scale)
        # Create the GP prior distribution, which we will use to train the model parameters
        return tfd.GaussianProcess(kernel=kernel, index_points=np.array([observation_index_points_trans_.T[i]]).T,
                                   observation_noise_variance=observation_noise_variance)

    gp_joint_model = tfd.JointDistributionNamed({
        'amplitude': tfd.LogNormal(loc=0., scale=np.float64(1.)),
        'length_scale': tfd.LogNormal(loc=0., scale=np.float64(2.)),
        'observation_noise_variance': tfd.LogNormal(loc=0., scale=np.float64(10.)),
        'observations': build_gp})

    trainable_variables = length_scales[i].trainable_variables

    def target_log_prob(amplitude, length_scale, observation_noise_variance):
        return gp_joint_model.log_prob({
            'amplitude': amplitude,
            'length_scale': length_scale,
            'observation_noise_variance': observation_noise_variance,
            'observations': observations_one_parameter_trans_[i]})

    @tf.function(autograph=False, jit_compile=False)
    def train_model():
        with tf.GradientTape() as tape:
            loss = -target_log_prob(amplitude_var, length_scales[i], observation_noise_variance_var)
            grads = tape.gradient(loss, trainable_variables)
            optimizer.apply_gradients(zip(grads, trainable_variables))
            return loss

    lls_ = np.zeros(num_iters, np.float64)
    for j in range(num_iters):
        loss = train_model()
        lls_[j] = loss

    fig = plt.figure(figsize=(12, 4))
    plt.plot(lls_)
    plt.xlabel("Training iteration", fontsize=12)
    plt.ylabel("Log marginal likelihood", fontsize=12)
    plt.show()

print("Optimized specific length scales:")
for i in range(num_parameters):
    print(problem['names'][i], length_scales[i]._value().numpy())

# We create an offline file with the specific length_scales
L = np.zeros(shape = (num_parameters))
for i in range(num_parameters):
    L[i] = length_scales[i]._value().numpy()
np.savetxt('length_scales_12_parameters.csv', L, delimiter=',')
import pdb; pdb.set_trace()
'''
start = time.time()

# 4, 20, 39, 77
observation_index_points_ = saltelli.sample(problem, 4) # N*(2D+2) with D num_vars and N chosen
num_observations = len(observation_index_points_)
observations_ = generate_data(observation_index_points_, observation_noise_variance=.01)

xr_oip = np.zeros(num_parameters)
xl_oip = np.zeros(num_parameters)
i = 0
while i < num_parameters:
    xr_oip[i] = max(observation_index_points_.T[i])
    xl_oip[i] = min(observation_index_points_.T[i])
    i = i + 1
observation_index_points_trans_ = (2*observation_index_points_ - (xl_oip + xr_oip))/(xr_oip - xl_oip)

xr_o = max(observations_.T)
xl_o = min(observations_.T)
observations_trans_ = (2*observations_ - (xl_o + xr_o))/(xr_o - xl_o)

L = np.zeros(num_parameters)
with open("length_scales_12_parameters.csv") as file_name:
    X = np.loadtxt(file_name, delimiter=",")
for i in range(num_parameters):
    L[i] = X[i]

observation_index_points_mod_ = np.zeros(observation_index_points_trans_.shape)
for i in range(num_parameters):
    for j in range(num_observations):
        observation_index_points_mod_[j,i] = observation_index_points_trans_[j,i]*L[i]

print("Time used for the generation of the training values:")
end = time.time()
print(end - start)

start = time.time()

# Let's optimize to find the parameter values with highest posterior probability. Notice that we constrain them to be positive
constrain_positive = tfb.Shift(np.finfo(np.float64).tiny)(tfb.Exp())

amplitude_var = tfp.util.TransformedVariable(
    initial_value=0.1209,
    bijector=constrain_positive,
    name='amplitude',
    dtype=np.float64)

length_scale_var = tfp.util.TransformedVariable(
    initial_value=1.,
    bijector=constrain_positive,
    name='length_scale',
    dtype=np.float64)

observation_noise_variance_var = tfp.util.TransformedVariable(
    initial_value=0.0146,
    bijector=constrain_positive,
    name='observation_noise_variance',
    dtype=np.float64)

num_iters = 15000
# Amplitude and length scale are the kernel parameters and are used to construct the covariance/kernel function
def build_gp_general(amplitude, length_scale, observation_noise_variance):
    # Create the covariance kernel
    kernel = tfk.ExponentiatedQuadratic(amplitude, length_scale)
    # Create the GP prior distribution, which we will use to train the model parameters
    return tfd.GaussianProcess(kernel=kernel, index_points=observation_index_points_mod_, observation_noise_variance=observation_noise_variance)

gp_joint_model = tfd.JointDistributionNamed({
    'amplitude': tfd.LogNormal(loc=0., scale=np.float64(1.)),
    'length_scale': tfd.LogNormal(loc=0., scale=np.float64(2.)),
    'observation_noise_variance': tfd.LogNormal(loc=0., scale=np.float64(10.)),
    'observations': build_gp_general})

trainable_variables = amplitude_var.trainable_variables + observation_noise_variance_var.trainable_variables

def target_log_prob_general(amplitude, length_scale, observation_noise_variance):
    return gp_joint_model.log_prob({
        'amplitude': amplitude,
        'length_scale': length_scale,
        'observation_noise_variance': observation_noise_variance,
        'observations': observations_trans_})

# Now we optimize the model parameters.
optimizer = tf.optimizers.Adam()

# We also want to trace the loss for a more efficient evaluation
@tf.function(autograph=False, jit_compile=False)
def train_model_general():
    with tf.GradientTape() as tape:
        loss = -target_log_prob_general(amplitude_var, length_scale_var, observation_noise_variance_var)
        grads = tape.gradient(loss, trainable_variables)
        optimizer.apply_gradients(zip(grads, trainable_variables))
        return loss

# Store the likelihood values during training, so we can plot the progress
lls_ = np.zeros(num_iters, np.float64)
for i in range(num_iters):
    loss = train_model_general()
    lls_[i] = loss

print("Optimized general hyper-parameters")
print('amplitude: {}'.format(amplitude_var._value().numpy()))
print('length_scale: {}'.format(length_scale_var._value().numpy()))
print('observation_noise_variance: {}'.format(observation_noise_variance_var._value().numpy()))

# Plot the loss evolution
fig1 = plt.figure(figsize=(12, 5))
plt.plot(lls_)
plt.xlabel("Training iteration", fontsize=12)
plt.ylabel("Log marginal likelihood", fontsize=12)
#fig1.savefig('loss_ani_8_1008_benchmark.png')

print("Time used for the optimization of the hyperparameters:")
end = time.time()
print(end - start)

# Having trained the model, we'd like to sample from the posterior conditioned on observations. We'd like the samples
# to be at points other than the training inputs.
num_predictions = 8996
'''
predictive_index_points_ = saltelli.sample(problem, 346) # N*(2D+2) with D num_vars and N chosen
true_function_ = generate_data_noise_free(predictive_index_points_)

# We create an offline file with predictive_index_points_ and true_function_
Z = np.zeros(shape = (num_predictions, num_parameters + 1))
Z.T[0] = predictive_index_points_.T[0]
Z.T[1] = predictive_index_points_.T[1]
Z.T[2] = predictive_index_points_.T[2]
Z.T[3] = predictive_index_points_.T[3]
Z.T[4] = predictive_index_points_.T[4]
Z.T[5] = predictive_index_points_.T[5]
Z.T[6] = predictive_index_points_.T[6]
Z.T[7] = predictive_index_points_.T[7]
Z.T[8] = predictive_index_points_.T[8]
Z.T[9] = predictive_index_points_.T[9]
Z.T[10] = predictive_index_points_.T[10]
Z.T[11] = predictive_index_points_.T[11]
Z.T[12] = true_function_
np.savetxt('test_set_12_parameters.csv', Z, delimiter=',')
import pdb; pdb.set_trace()
'''
predictive_index_points_ = np.zeros(shape = (num_predictions, num_parameters))
true_function_ = np.zeros(shape = num_predictions)

with open("test_set_12_parameters.csv") as file_name:
    W = np.loadtxt(file_name, delimiter=",")
for i in range(num_predictions):
    predictive_index_points_.T[0][i] = W.T[0][i]
    predictive_index_points_.T[1][i] = W.T[1][i]
    predictive_index_points_.T[2][i] = W.T[2][i]
    predictive_index_points_.T[3][i] = W.T[3][i]
    predictive_index_points_.T[4][i] = W.T[4][i]
    predictive_index_points_.T[5][i] = W.T[5][i]
    predictive_index_points_.T[6][i] = W.T[6][i]
    predictive_index_points_.T[7][i] = W.T[7][i]
    predictive_index_points_.T[8][i] = W.T[8][i]
    predictive_index_points_.T[9][i] = W.T[9][i]
    predictive_index_points_.T[10][i] = W.T[10][i]
    predictive_index_points_.T[11][i] = W.T[11][i]
    true_function_[i] = W.T[12][i]

xr_pip = np.zeros(num_parameters)
xl_pip = np.zeros(num_parameters)
i = 0
while i < num_parameters:
    xr_pip[i] = max(predictive_index_points_.T[i])
    xl_pip[i] = min(predictive_index_points_.T[i])
    i = i + 1
predictive_index_points_trans_ = (2*predictive_index_points_ - (xl_pip + xr_pip))/(xr_pip - xl_pip)

xr_po = max(true_function_)
xl_po = min(true_function_)
true_function_trans_ = (2*true_function_ - (xl_po + xr_po))/(xr_po - xl_po)

predictive_index_points_mod_ = np.zeros(predictive_index_points_trans_.shape)
for i in range(num_parameters):
    for j in range(num_predictions):
        predictive_index_points_mod_[j,i] = predictive_index_points_trans_[j,i]*L[i]

start = time.time()
optimized_kernel = tfk.ExponentiatedQuadratic(amplitude_var, length_scale_var)

gprm = tfd.GaussianProcessRegressionModel(
    kernel=optimized_kernel,
    index_points=predictive_index_points_mod_,
    observation_index_points=observation_index_points_mod_,
    observations=observations_trans_,
    observation_noise_variance=observation_noise_variance_var,
    predictive_noise_variance=0.)

# We take 50 independent samples, each of which is a joint draw from the posterior at the predictive_index_points_.
num_samples = 50
samples = gprm.sample(num_samples)
mean_gp = gprm.mean()
mean_gp = np.reshape(mean_gp, num_predictions)
stddev_gp = gprm.stddev()
stddev_gp = np.reshape(stddev_gp, num_predictions)

print("Time used for the predictions:")
end = time.time()
print(end - start)

RMSE = np.sqrt(np.square(np.subtract(mean_gp, true_function_trans_)).mean())
err = np.linalg.norm(mean_gp - true_function_trans_, np.inf)
print("Root mean square error:", RMSE)
print("L-infinity error:", err)

fig2, ax = plt.subplots()
ax.scatter(mean_gp, true_function_trans_, marker='.', color='blue')
line = mlines.Line2D([0, 1], [0, 1], color='red')
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
plt.xlabel("Predictions", fontsize=12)
plt.ylabel("Real observations", fontsize=12)
plt.axis('square')
fig2.savefig('gp_12_104.png')

Si = sobol.analyze(problem, mean_gp) # calc_second_order=False
print("First-order Sobol indexes with respect to the predictions:")
for i in range(num_parameters):
    print(problem['names'][i], Si['S1'][i], Si['S1_conf'][i])

Si = sobol.analyze(problem, true_function_trans_) # calc_second_order=False
print("First-order Sobol indexes with respect to the real observations:")
for i in range(num_parameters):
    print(problem['names'][i], Si['S1'][i], Si['S1_conf'][i])

plt.show()