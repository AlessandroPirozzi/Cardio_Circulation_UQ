# Cardiocirculation UQ

Repository of Python codes for the implementation of the Gaussian process (GP) for a benchmark problem. Codes for GPs, variance-based global sensitivity analysis and Bayesian parameter estimation by means of the Markov chain Monte Carlo (MCMC) method applied to the circulation model require the protected Python class circulation_closed_loop for the cardiocirculation outputs, thus they are not provided (for more information, contact the author).

## Getting Started

Gaussian processes are implemented by means of the open source Python library [TensorFlow](https://www.tensorflow.org/probability/examples/Gaussian_Process_Regression_In_TFP?hl=en), whereas for the sensitivity analysis and the Bayesian parameter estimation libraries [SALib](https://salib.readthedocs.io/en/latest/) and [UQpy](https://uqpyproject.readthedocs.io/en/latest/) are used respectively.

## Author

Alessandro Pirozzi, Politecnico di Milano (alessandro.pirozzi@mail.polimi.it)
