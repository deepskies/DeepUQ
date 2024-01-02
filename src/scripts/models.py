import numpyro
import numpyro.distributions as dist
import numpy as np
import jax
import jax.numpy as jnp # yes i know this is confusing
import torch.nn as nn
import torch

# tensorflow sucks
# build a similar thing in pytorch


class de_no_var(nn.Module):
    def __init__(self):
        super().__init__()
        drop_percent = 0.1
        self.ln_1 = nn.Linear(3, 100)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(drop_percent)
        self.ln_2 = nn.Linear(100, 100)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(drop_percent)
        self.ln_3 = nn.Linear(100, 100)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(drop_percent)
        self.ln_4 = nn.Linear(100,1) # needs to be 2 if using the GaussianNLLoss

    def forward(self, x):
        x = self.drop1(self.act1(self.ln_1(x)))
        x = self.drop2(self.act2(self.ln_2(x)))
        x = self.drop3(self.act3(self.ln_3(x)))
        x = self.ln_4(x)
        return x


class de_var(nn.Module):
    def __init__(self):
        super().__init__()
        drop_percent = 0.1
        self.ln_1 = nn.Linear(3, 100)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(drop_percent)
        self.ln_2 = nn.Linear(100, 100)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(drop_percent)
        self.ln_3 = nn.Linear(100, 100)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(drop_percent)
        self.ln_4 = nn.Linear(100,2) # needs to be 2 if using the GaussianNLLoss

    def forward(self, x):
        x = self.drop1(self.act1(self.ln_1(x)))
        x = self.drop2(self.act2(self.ln_2(x)))
        x = self.drop3(self.act3(self.ln_3(x)))
        x = self.ln_4(x)
        return x

## in numpyro, you must specify number of sampling chains you will use upfront

# words of wisdom from Tian Li and crew:
# on gpu, don't use conda, use pip install
# HMC after SBI to look at degeneracies between params
# different guides (some are slower but better at showing degeneracies)

## define the platform and number of cores (one chain per core)
numpyro.set_platform('cpu')
core_num = 4
numpyro.set_host_device_count(core_num)

def hierarchical_model(planet_code,
                       pendulum_code,
                       times,
                       exponential,
                       pos_obs=None):
    """
    """
    ## inputs to a numpyro model are rows from a dataframe:
    ## planet code - array of embedded numbers representing which planet {0...1}
    ## pendulum code - array of embedded numbers representing which pendulum {0...7}
    ## times - moments in time (s)
    ## pos_obs - this is optional, set to None but used to compare the model with data
    ## (when data, xpos, is defined)
    
    ## numpyro models function by drawing parameters from samples 
    ## first, we define the global parameters, mean and sigma of a normal from
    ## which the individual a_g values of each planet will be drawn
    

    #μ_a_g = numpyro.sample("μ_a_g", dist.LogUniform(5.0,15.0))
    μ_a_g = numpyro.sample("μ_a_g", dist.TruncatedNormal(12.5, 5, low=0.01))
    # scale parameters should be log uniform so that they don't go negative 
    # and so that they're not uniform
    # 1 / x in linear space
    σ_a_g = numpyro.sample("σ_a_g", dist.TruncatedNormal(0.1, 0.01, low=0.01))
    n_planets = len(np.unique(planet_code))
    n_pendulums = len(np.unique(pendulum_code))

    ## plates are a numpyro primitive or context manager for handing conditionally independence
    ## for instance, we wish to model a_g for each planet independently
    with numpyro.plate("planet_i", n_planets):
        a_g = numpyro.sample("a_g", dist.TruncatedNormal(μ_a_g, σ_a_g,
                                                         low=0.01))
        # helps because a_gs are being pulled from same normal dist
        # removes dependency of a_g on sigma_a_g on a prior level
        # removing one covariance from model, model is easier
        # to sample from
    
    ## we also wish to model L and theta for each pendulum independently
    ## here we draw from an uniform distribution
    with numpyro.plate("pend_i", n_pendulums):
        L = numpyro.sample("L", dist.TruncatedNormal(5, 2, low=0.01))
        theta = numpyro.sample("theta", dist.TruncatedNormal(jnp.pi/100,
                                                             jnp.pi/500,
                                                             low=0.00001))

    ## σ is the error on the position measurement for each moment in time
    ## we also model this
    ## eventually, we should also model the error on each parameter independently?
    ## draw from an exponential distribution parameterized by a rate parameter
    ## the mean of an exponential distribution is 1/r where r is the rate parameter
    ## exponential distributions are never negative. This is good for error.
    σ = numpyro.sample("σ", dist.Exponential(exponential))
    
    ## the moments in time are not independent, so we do not place the following in a plate
    ## instead, the brackets segment the model by pendulum and by planet,
    ## telling us how to conduct the inference
    modelx = L[pendulum_code] * jnp.sin(theta[pendulum_code] * jnp.cos(jnp.sqrt(a_g[planet_code] / L[pendulum_code]) * times))
    ## don't forget to use jnp instead of np so jax knows what to do
    ## A BIG QUESTION I STILL HAVE IS WHAT IS THE LIKELIHOOD? IS IT JUST SAMPLED FROM?
    ## again, for each pendulum we compare the observed to the modeled position:
    with numpyro.plate("data", len(pendulum_code)):
        pos = numpyro.sample("obs", dist.Normal(modelx, σ), obs=pos_obs)


def unpooled_model(planet_code,
                   pendulum_code,
                   times,
                   exponential,
                   pos_obs=None):
    n_planets = len(np.unique(planet_code))
    n_pendulums = len(np.unique(pendulum_code))
    with numpyro.plate("planet_i", n_planets):
        a_g = numpyro.sample("a_g", dist.TruncatedNormal(12.5, 5,
                                                         low=0, high=25))
    with numpyro.plate("pend_i", n_pendulums):
        L = numpyro.sample("L", dist.TruncatedNormal(5, 2, low = 0.01))
        theta = numpyro.sample("theta", dist.TruncatedNormal(jnp.pi/100,
                                                             jnp.pi/500,
                                                             low=0.00001))
    σ = numpyro.sample("σ", dist.Exponential(exponential))
    modelx = L[pendulum_code] * jnp.sin(theta[pendulum_code] *
                         jnp.cos(jnp.sqrt(a_g[planet_code] / L[pendulum_code]) * times))
    with numpyro.plate("data", len(pendulum_code)):
        pos = numpyro.sample("obs", dist.Normal(modelx, σ), obs=pos_obs)

# This is from PasteurLabs - 
# https://github.com/pasteurlabs/unreasonable_effective_der/blob/main/models.py


class Model(nn.Module):
    def __init__(self, n_output, n_hidden=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output),
        )

    def forward(self, x):
        return self.model(x)


class DERLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        gamma = x[:, 0]
        nu = nn.functional.softplus(x[:, 1])
        alpha = nn.functional.softplus(x[:, 2]) + 1.0
        beta = nn.functional.softplus(x[:, 3])
        return torch.stack((gamma, nu, alpha, beta), dim=1)


class SDERLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        gamma = x[:, 0]
        nu = nn.functional.softplus(x[:, 1])
        alpha = nu + 1.0
        beta = nn.functional.softplus(x[:, 3])
        return torch.stack((gamma, nu, alpha, beta), dim=1)


def loss_der(y, y_pred, coeff):
    gamma, nu, alpha, beta = y[:, 0], y[:, 1], y[:, 2], y[:, 3]
    error = gamma - y_pred
    omega = 2.0 * beta * (1.0 + nu)

    return torch.mean(
        0.5 * torch.log(math.pi / nu)
        - alpha * torch.log(omega)
        + (alpha + 0.5) * torch.log(error**2 * nu + omega)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
        + coeff * torch.abs(error) * (2.0 * nu + alpha)
    )


def loss_sder(y, y_pred, coeff):
    gamma, nu, _, beta = y[:, 0], y[:, 1], y[:, 2], y[:, 3]
    error = gamma - y_pred
    var = beta / nu

    return torch.mean(torch.log(var) + (1. + coeff * nu) * error**2 / var)
