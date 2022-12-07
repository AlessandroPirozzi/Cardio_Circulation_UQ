#!/usr/bin/env python3
import time
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
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

start = time.time()

def sinusoid(x):
    return np.sin(3 * np.pi * x[..., 0])

def generate_1d_data(num_training_points, observation_noise_variance):
    index_points_ = np.random.uniform(-1., 1., (num_training_points, 1))
    index_points_ = index_points_.astype(np.float64)
    # y = f(x) + noise
    observations_ = (sinusoid(index_points_) + np.random.normal(loc=0, scale=np.sqrt(observation_noise_variance), size=(num_training_points)))
    return index_points_, observations_

# Generate training data with a known noise level
num_observations = 100 #25, 50, 100
observation_index_points_, observations_ = generate_1d_data(num_training_points=num_observations, observation_noise_variance=0.001)

print("Time used for the generation of the training values:")
end = time.time()
print(end - start)

start = time.time()

# Amplitude and length scale are the kernel parameters and are used to construct the covariance/kernel function
def build_gp(amplitude, length_scale, observation_noise_variance):
    # Create the covariance kernel
    kernel = tfk.ExponentiatedQuadratic(amplitude, length_scale)
    # Create the GP prior distribution, which we will use to train the model parameters
    return tfd.GaussianProcess(kernel=kernel, index_points=observation_index_points_, observation_noise_variance=observation_noise_variance)

gp_joint_model = tfd.JointDistributionNamed({
    'amplitude': tfd.LogNormal(loc=0., scale=np.float64(1.)),
    'length_scale': tfd.LogNormal(loc=0., scale=np.float64(1.)),
    'observation_noise_variance': tfd.LogNormal(loc=0., scale=np.float64(1.)),
    'observations': build_gp})

# Let's optimize to find the parameter values with highest posterior probability. Notice that we constrain them to be positive
constrain_positive = tfb.Shift(np.finfo(np.float64).tiny)(tfb.Exp())
amplitude_var = tfp.util.TransformedVariable(
    initial_value=1.,
    bijector=constrain_positive,
    name='amplitude',
    dtype=np.float64)

length_scale_var = tfp.util.TransformedVariable(
    initial_value=1.,
    bijector=constrain_positive,
    name='length_scale',
    dtype=np.float64)

observation_noise_variance_var = tfp.util.TransformedVariable(
    initial_value=1.,
    bijector=constrain_positive,
    name='observation_noise_variance_var',
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
      'observations': observations_})

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
num_predictions = 10000
'''
predictive_index_points_ = np.linspace(-1, 1, num_predictions, dtype=np.float64)
predictive_index_points_ = predictive_index_points_[..., np.newaxis]
true_function_ = sinusoid(predictive_index_points_)

# We create an offline file with predictive_index_points_ and true_function_
Z = np.zeros(shape = (num_predictions, 2))
Z.T[0] = predictive_index_points_.T[0]
Z.T[1] = true_function_
np.savetxt('test_set.csv', Z, delimiter=',')
import pdb; pdb.set_trace()
'''
predictive_index_points_ = np.zeros(shape = (num_predictions, 1))
true_function_ = np.zeros(shape = num_predictions)

with open("test_set.csv") as file_name:
    W = np.loadtxt(file_name, delimiter=",")
for i in range(num_predictions):
    predictive_index_points_.T[0][i] = W.T[0][i]
    true_function_[i] = W.T[1][i]

start = time.time()

optimized_kernel = tfk.ExponentiatedQuadratic(amplitude_var, length_scale_var)

gprm = tfd.GaussianProcessRegressionModel(
    kernel=optimized_kernel,
    index_points=predictive_index_points_,
    observation_index_points=observation_index_points_,
    observations=observations_,
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

# Plot the true function, observations, and posterior samples.
fig2 = plt.figure(figsize=(7, 6))
plt.plot(predictive_index_points_, true_function_, label='True fn')
plt.scatter(observation_index_points_, observations_, label='Observations')
for i in range(num_samples):
    plt.plot(predictive_index_points_, tf.transpose(samples[i, :]), c='r', alpha=.1,
           label='Posterior Sample' if i == 0 else None)

leg = plt.legend(loc='lower right', fontsize=12)
for lh in leg.legendHandles:
    lh.set_alpha(1)
plt.xlabel("$\mu$", fontsize=15)
plt.ylabel("$f$", fontsize=15)
plt.tick_params(axis='y', labelsize=12)
plt.tick_params(axis='x', labelsize=12)

data = (true_function_ - mean_gp)/stddev_gp

num_bins = 10
x = np.linspace(-5, 5, 1000)
fig3 = plt.figure(figsize=(7, 6))
plt.hist(data, num_bins, density=True)
plt.plot(x, stats.norm.pdf(x, 0, 0.5))
plt.tick_params(axis='y', labelsize=12)
plt.tick_params(axis='x', labelsize=12)

sm.qqplot(data, line='45')

print('Quantiles:')
print('0.99:', np.quantile(data, q=0.99))
print('0.95:', np.quantile(data, q=0.95))
print('0.90:', np.quantile(data, q=0.9))
print('0.75:', np.quantile(data, q=0.75))
print('0.50:', np.quantile(data, q=0.5))

# 2.33, 1.64, 1.28, 0.67, 0.00
quantiles_ns = np.array([2.33, 1.64, 1.28, 0.67, 0.00])
percentage = np.zeros(len(quantiles_ns))
for i in range(len(quantiles_ns)):
    counter = 0
    for j in range(len(data)):
        if data[j] < quantiles_ns[i]:
            counter = counter + 1
    percentage[i] = counter/len(data)

print('Probabilities:')
print('0.99:', percentage[0])
print('0.95:', percentage[1])
print('0.90:', percentage[2])
print('0.75:', percentage[3])
print('0.50:', percentage[4])

plt.show()