import numpyro
import numpyro.distributions as dist
import numpy as np
import jax
import jax.numpy as jnp # yes i know this is confusing




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
    μ_a_g = numpyro.sample("μ_a_g", dist.TruncatedNormal(12.5, 2, low=0.01))
    # scale parameters should be log uniform so that they don't go negative 
    # and so that they're not uniform
    # 1 / x in linear space
    σ_a_g = numpyro.sample("σ_a_g", dist.TruncatedNormal(2, 0.5, low=0.01))
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
        L = 5#numpyro.sample("L", dist.TruncatedNormal(5, 2, low = 0.01))
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
    modelx = L * jnp.sin(theta[pendulum_code] * jnp.cos(jnp.sqrt(a_g[planet_code] / L) * times))
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
        a_g = numpyro.sample("a_g", dist.TruncatedNormal(10, 5,
                                                         low=0, high=25))
    with numpyro.plate("pend_i", n_pendulums):
        L = 5#numpyro.sample("L", dist.TruncatedNormal(5, 2, low = 0.01))
        theta = numpyro.sample("theta", dist.TruncatedNormal(jnp.pi/100,
                                                             jnp.pi/500,
                                                             low=0.00001))
    σ = numpyro.sample("σ", dist.Exponential(exponential))
    modelx = L * jnp.sin(theta[pendulum_code] *
                         jnp.cos(jnp.sqrt(a_g[planet_code] / L) * times))
    with numpyro.plate("data", len(pendulum_code)):
        pos = numpyro.sample("obs", dist.Normal(modelx, σ), obs=pos_obs)
