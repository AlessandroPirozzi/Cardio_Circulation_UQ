#!/usr/bin/env python3
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
import pandas as pd
import scipy.stats as stats
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from SALib.sample import saltelli
from SALib.analyze import sobol

tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels
tf.enable_v2_behavior()

def fun(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12):
    return np.sin(6*x2) + 0.1*np.cos(x1) + np.sin(x1)*np.cos(3*x2) + 0.25*np.sin(x3) + 0.6*np.cos(2*x4) + 0.5*np.cos(x3)*np.sin(x4) + \
           np.sin(x4)*np.cos(x8) + 0.3*np.cos(x10) + 0.7*np.sin(x11) + 0.1*np.cos(x5) + np.sin(x5)*np.cos(x12) + \
           0.25*np.sin(x9) + 0.1*np.sin(x6)*np.cos(x7)

def generate_data(index_points_, observation_noise_variance):
    observations_ = (fun(index_points_.T[0], index_points_.T[1], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) + np.random.normal(loc=0, scale=np.sqrt(observation_noise_variance), size=(num_observations)))
    return observations_.T

start = time.time()

# Generate training data with a known noise level
problem = {
    'num_vars': 2,
    'names': ['mu_1', 'mu_2'],
    'bounds': [[-1, 1], [-1, 1]]
}
# 17, 84, 167, 334
observation_index_points_ = saltelli.sample(problem, 17) # N*(2D+2) with D num_vars and N chosen
num_observations = len(observation_index_points_)
num_parameters = problem['num_vars']

observations_ = generate_data(observation_index_points_, observation_noise_variance=.01)

xr_oip = np.zeros(num_parameters)
xl_oip = np.zeros(num_parameters)
i = 0
while i < num_parameters:
    xr_oip[i] = max(observation_index_points_.T[i])
    xl_oip[i] = min(observation_index_points_.T[i])
    i = i + 1
observation_index_points_trans_ = (2*observation_index_points_ - (xl_oip + xr_oip))/(xr_oip - xl_oip)

xr_o = max(observations_)
xl_o = min(observations_)
observations_trans_ = (2*observations_ - (xl_o + xr_o))/(xr_o - xl_o)

print("Time used for the generation of the training values:")
end = time.time()
print(end - start)

start = time.time()

# Amplitude and length scale are the kernel parameters and are used to construct the covariance/kernel function
def build_gp(amplitude, length_scale, observation_noise_variance):
    # Create the covariance kernel
    kernel = tfk.ExponentiatedQuadratic(amplitude, length_scale)
    # Create the GP prior distribution, which we will use to train the model parameters
    return tfd.GaussianProcess(kernel=kernel, index_points=observation_index_points_trans_, observation_noise_variance=observation_noise_variance)

gp_joint_model = tfd.JointDistributionNamed({
    'amplitude': tfd.LogNormal(loc=0., scale=np.float64(1.)),
    'length_scale': tfd.LogNormal(loc=0., scale=np.float64(2.)),
    'observation_noise_variance': tfd.LogNormal(loc=0., scale=np.float64(10.)),
    'observations': build_gp})

# Let's optimize to find the parameter values with highest posterior probability. Notice that we constrain them to be positive
constrain_positive = tfb.Shift(np.finfo(np.float64).tiny)(tfb.Exp())
amplitude_var = tfp.util.TransformedVariable(
    initial_value=0.1209,
    bijector=constrain_positive,
    name='amplitude',
    dtype=np.float64)

length_scale_var = tfp.util.TransformedVariable(
    initial_value=0.0816,
    bijector=constrain_positive,
    name='length_scale',
    dtype=np.float64)

observation_noise_variance_var = tfp.util.TransformedVariable(
    initial_value=0.0146,
    bijector=constrain_positive,
    name='observation_noise_variance',
    dtype=np.float64)

trainable_variables = [v.trainable_variables[0] for v in
                       [amplitude_var,
                       length_scale_var,
                       observation_noise_variance_var]]

def target_log_prob(amplitude, length_scale, observation_noise_variance):
    return gp_joint_model.log_prob({
      'amplitude': amplitude,
      'length_scale': length_scale,
      'observation_noise_variance': observation_noise_variance,
      'observations': observations_trans_})

# Now we optimize the model parameters.
num_iters = 5000
optimizer = tf.optimizers.Adam(learning_rate=.01)

# We also want to trace the loss for a more efficient evaluation
@tf.function(autograph=False, jit_compile=False)
def train_model():
    with tf.GradientTape() as tape:
        loss = -target_log_prob(amplitude_var, length_scale_var, observation_noise_variance_var)
        grads = tape.gradient(loss, trainable_variables)
        optimizer.apply_gradients(zip(grads, trainable_variables))
        return loss

# Store the likelihood values during training, so we can plot the progress
lls_ = np.zeros(num_iters, np.float64)
for i in range(num_iters):
    loss = train_model()
    lls_[i] = loss

print('Trained parameters:')
print('amplitude: {}'.format(amplitude_var._value().numpy()))
print('length_scale: {}'.format(length_scale_var._value().numpy()))
print('observation_noise_variance: {}'.format(observation_noise_variance_var._value().numpy()))

# Plot the loss evolution
fig1 = plt.figure(figsize=(12, 5))
plt.plot(lls_)
plt.xlabel("Training iteration", fontsize=12)
plt.ylabel("Log marginal likelihood", fontsize=12)

print("Time used for the optimization of the hyperparameters:")
end = time.time()
print(end - start)

# Having trained the model, we'd like to sample from the posterior conditioned on observations. We'd like the samples
# to be at points other than the training inputs.
num_predictions = 9000
'''
predictive_index_points_ = saltelli.sample(problem, 1500) # N*(2D+2) with D num_vars and N chosen
true_function_ = fun(predictive_index_points_.T[0], predictive_index_points_.T[1], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

# We create an offline file with predictive_index_points_ and true_function_
Z = np.zeros(shape = (num_predictions, num_parameters + 1))
Z.T[0] = predictive_index_points_.T[0]
Z.T[1] = predictive_index_points_.T[1]
Z.T[2] = true_function_
np.savetxt('test_set_2_parameters.csv', Z, delimiter=',')
import pdb; pdb.set_trace()
'''
predictive_index_points_ = np.zeros(shape = (num_predictions, num_parameters))
true_function_ = np.zeros(shape = num_predictions)

with open("test_set_2_parameters.csv") as file_name:
    W = np.loadtxt(file_name, delimiter=",")
for i in range(num_predictions):
    predictive_index_points_.T[0][i] = W.T[0][i]
    predictive_index_points_.T[1][i] = W.T[1][i]
    true_function_[i] = W.T[2][i]

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

start = time.time()
optimized_kernel = tfk.ExponentiatedQuadratic(amplitude_var, length_scale_var)

gprm = tfd.GaussianProcessRegressionModel(
    kernel=optimized_kernel,
    index_points=predictive_index_points_trans_,
    observation_index_points=observation_index_points_trans_,
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

fig2 = plt.figure(figsize=(10, 6))
ax = plt.axes(projection='3d')
ax.xaxis.set_ticks(np.linspace(-1,1,5))
ax.yaxis.set_ticks(np.linspace(-1,1,5))
ax.zaxis.set_ticks(np.linspace(-1,1,5))
ax.scatter3D(predictive_index_points_trans_.T[0], predictive_index_points_trans_.T[1], mean_gp)
ax.set_xlabel('$\mu_1$', fontsize=12)
ax.set_ylabel('$\mu_2$', fontsize=12)
ax.set_zlabel('Predictions', fontsize=12)
#fig2.savefig('surface_iso_2_102_benchmark.png')

RMSE = np.sqrt(np.square(np.subtract(mean_gp, true_function_trans_)).mean())
err = np.linalg.norm(mean_gp - true_function_trans_, np.inf)
print("Root mean square error:", RMSE)
print("L-infinity error:", err)

plt.show()