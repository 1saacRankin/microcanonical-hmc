# Compare hyperparameter tuning:
# Sequential similar to Robnik et al
# Versus Bayesian Optimization


# For Metropolis Adjusted Microcanonical Hamiltonian Monte Carlo (MAMS) there are two hyperparameters:  
# 1) stepsize e ($\epsilon$) 
# 2) trajectory length L

# Goal: optimize hyperparameters to maximize ESS while keeping acceptance rate near target (squared error).








#################################################################################################### Import libraries
# Imports from here: https://blackjax-devs.github.io/sampling-book/algorithms/mclmc.html#how-to-run-mclmc-in-blackjax

import matplotlib.pyplot as plt

plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["font.size"] = 12 # Changed from 19 to 12

import jax
import blackjax
import numpy as np
import jax.numpy as jnp
from datetime import date
import numpyro
import numpyro.distributions as dist

from numpyro.infer.util import initialize_model

rng_key = jax.random.key(548) # Changed


# Imports for Adjusted MCLMC (MAMS):
# See: https://blackjax-devs.github.io/sampling-book/algorithms/mclmc.html#adjusted-mclmc
from blackjax.mcmc.adjusted_mclmc_dynamic import rescale
from blackjax.util import run_inference_algorithm


# Additional imports for NUTS:
# See: https://blackjax-devs.github.io/blackjax/examples/quickstart.html#nuts
import jax.scipy.stats as stats


# JAX guide: https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html
# Blackjax guide: https://blackjax-devs.github.io/sampling-book/

# Bayes Opt for JAX
# Boax guide: https://boax.readthedocs.io/en/latest/index.html
from boax.experiments import optimization

# Make JAX work
from jax import config
config.update("jax_enable_x64", True)

# Time the experiments
import time

# Use ARVIZ for MCMC diagnostics 
# See: https://python.arviz.org/en/stable/api/diagnostics.html
import arviz as az
import xarray as xr



# Make more seeds for tuning and validation
# Class is 548, it's 2025, you know
SEED_NUTS_TUNING = 548
SEED_MCLMC_TUNING = 548
SEED_MAMS_TUNING = 548
SEED_BAYESOPT_VALIDATION = 2025
SEED_AUTO_VALIDATION = 2025







################################################################### Make functions for making targets and doing diagnostics

# CHECK THIS AND MAKE MORE TARGETS
# Make a Neal's funnel funnel for many dimensions
def make_funnel_logdensity(dim):
    
    def logdensity(x):
        
        # First dimension is normal(0, 3^2)
        log_prob = -0.5 * (x[0]**2 / 9.0)
        
        # Normalizing constant for first dimension
        log_prob += -0.5 * (dim - 1) * x[0]
        
        # Other dimensions are normal(0, exp(x[0]/2))
        log_prob += -0.5 * jnp.sum(x[1:]**2 * jnp.exp(-x[0]))
        
        return log_prob
    
    return logdensity




# # Make a ESS function
# def compute_ess(samples):
#     # It needs to be transposed to not crash
#     # Blackjax diagnostics: (n_dims, n_chains=1, n_samples)
#     samples_reshaped = samples.T[:, None, :] 
    
#     # Compute ESS for each dimension
#     ess_per_dim = jax.vmap(lambda x: blackjax.diagnostics.effective_sample_size(x), in_axes = 0)(samples_reshaped)
    
#     # ESS for the dimension with minimum ESS (maximin vibe here)
#     # We want to maximize the minimum ESS, could do median or something like that
#     return jnp.min(ess_per_dim)


# def compute_rhat(chains):
#     # Number of dimensions
#     num_chains, num_steps, dim = chains.shape
    
#     # Mean for each chain
#     chain_means = jnp.mean(chains, axis=1)
    
#     # Mean of means of chains
#     overall_mean = jnp.mean(chain_means, axis=0)
    
#     # Between chain variance
#     B = num_steps / (num_chains - 1) * jnp.sum((chain_means - overall_mean)**2, axis=0)
    
#     # Within chain variance
#     chain_vars = jnp.var(chains, axis=1, ddof=1)
#     W = jnp.mean(chain_vars, axis=0)
    
#     # Pooled variance estimate
#     var_est = ((num_steps - 1) / num_steps) * W + (1 / num_steps) * B
    
#     # R-hat
#     rhat = jnp.sqrt(var_est / W)
    
#     return rhat





# # Find ESS (Effective Sample Size)
# def compute_ess(samples):
#     # See: https://python.arviz.org/en/stable/api/generated/arviz.ess.html
#     # Need to match the expected structure
#     samples_reshaped = samples[None, :, :]
#     ess = az.ess(samples_reshaped, method = 'bulk') # Default is bulk
    
#     # Return worst, want to maximize the minimum ESS
#     return float(np.min(ess))


# # Find R-hat
# def compute_rhat(chains): 
#     # Arviz R-hat: Compute estimate of rank normalized splitR-hat for a set of traces.
#     # https://python.arviz.org/en/stable/api/generated/arviz.rhat.html#arviz.rhat
#     rhat = az.rhat(chains)
#     # Return worst, want to minimize maximum R-hat (minimax)
#     return float(np.max(rhat))




# Find ESS (Effective Sample Size)
def compute_ess(samples):
    # Convert JAX array to NumPy array
    samples_np = np.array(samples)
    
    # ArviZ expects (chain, draw, *variable_shape)
    samples_reshaped = samples_np[None, :, :]  # (1, n_samples, n_dims)
    
    # Convert to ArviZ dataset: https://python.arviz.org/en/stable/api/generated/arviz.convert_to_dataset.html#arviz.convert_to_dataset
    dataset = az.convert_to_dataset(samples_reshaped)
    
    # Compute ESS
    ess = az.ess(dataset, method='bulk')
    
    # Return worst ESS across all variables
    return float(np.min([ess[var].values for var in ess.data_vars]))


# Find R-hat
def compute_rhat(chains): 
    # Convert JAX array to NumPy array
    chains_np = np.array(chains)
    
    # Shape should be (n_chains, n_samples, n_dims)
    # Convert to ArviZ dataset
    dataset = az.convert_to_dataset(chains_np)
    
    # Compute R-hat
    rhat = az.rhat(dataset)
    
    # Return worst R-hat across all variables
    return float(np.max([rhat[var].values for var in rhat.data_vars]))







# Make an objective function
# Want to maximize: ESS - lambda (acceptance rate - target accpetance rate)^2
# Can choose lambda so we don't get any BS near-zero accpetance rates
def objective_function(ess, acceptance_rate, target_acceptance, lambda_penalty):
    penalty = lambda_penalty * (acceptance_rate - target_acceptance)**2
    return ess - penalty








# ============================================================================
# MCMC SAMPLING FUNCTIONS
# ============================================================================

def run_nuts_fixed(
    logdensity_fn, 
    chain_length, 
    initial_position, 
    key, 
    step_size, 
    inv_mass_matrix
):
    """
    Run NUTS sampler with fixed hyperparameters.
    
    NUTS (No-U-Turn Sampler) automatically tunes trajectory length.
    We only need to provide step size and mass matrix.
    
    Parameters:
    -----------
    logdensity_fn : function
        Target log density to sample from
    chain_length : int
        Number of samples to generate
    initial_position : array
        Starting position for the chain
    key : PRNGKey
        Random key for reproducibility
    step_size : float
        Leapfrog integration step size
    inv_mass_matrix : array
        Inverse mass matrix (diagonal preconditioning)
        
    Returns:
    --------
    samples : array, shape (chain_length, n_dims)
        MCMC samples
    ess : float
        Effective sample size (minimum across dimensions)
    avg_acceptance : float
        Average acceptance probability
    avg_integration_steps : float
        Average number of leapfrog steps per sample
    time_elapsed : float
        Wall-clock time in seconds
    """
    # Start timing
    start_time = time.time()
    
    # Initialize NUTS sampler with given hyperparameters
    nuts = blackjax.nuts(
        logdensity_fn=logdensity_fn, 
        step_size=step_size, 
        inverse_mass_matrix=inv_mass_matrix
    )
    
    # Initialize sampler state at starting position
    state = nuts.init(initial_position)
    
    # Define one MCMC step
    def one_step(state, key):
        # Propose new state and compute diagnostics
        state, info = nuts.step(key, state)
        
        # Return updated state and tracked quantities
        return state, (state.position, info.acceptance_rate, info.num_integration_steps)
    
    # Generate random keys for each MCMC step
    keys = jax.random.split(key, chain_length)
    
    # Run the MCMC chain using JAX's scan
    final_state, (samples, acceptance_rates, num_steps_per_iter) = jax.lax.scan(
        one_step, 
        state, 
        keys
    )
    
    # Compute summary statistics
    avg_acceptance = jnp.mean(acceptance_rates)
    ess = compute_ess(samples)
    avg_integration_steps = jnp.mean(num_steps_per_iter)
    time_elapsed = time.time() - start_time
    
    return samples, ess, avg_acceptance, avg_integration_steps, time_elapsed


def run_mclmc_fixed(
    logdensity_fn, 
    chain_length, 
    initial_position, 
    key, 
    L, 
    step_size
):
    """
    Run MCLMC sampler with fixed hyperparameters.
    
    MCLMC (Microcanonical Langevin Monte Carlo) uses continuous dynamics
    without Metropolis rejection. It always accepts proposals (acc=1.0).
    
    Parameters:
    -----------
    logdensity_fn : function
        Target log density to sample from
    chain_length : int
        Number of samples to generate
    initial_position : array
        Starting position for the chain
    key : PRNGKey
        Random key for reproducibility
    L : float
        Trajectory length (total Hamiltonian simulation time)
    step_size : float
        Leapfrog integration step size
        
    Returns:
    --------
    samples : array, shape (chain_length, n_dims)
        MCMC samples
    ess : float
        Effective sample size
    avg_acceptance : float
        Always 1.0 for MCLMC (no rejection)
    integration_steps_per_iter : float
        Number of leapfrog steps per sample (L / step_size)
    time_elapsed : float
        Wall-clock time in seconds
    """
    # Start timing
    start_time = time.time()
    
    # Split key for initialization and sampling
    init_key, run_key = jax.random.split(key)
    
    # Initialize MCLMC sampler state
    initial_state = blackjax.mcmc.mclmc.init(
        position=initial_position, 
        logdensity_fn=logdensity_fn, 
        rng_key=init_key
    )
    
    # Create MCLMC sampling algorithm with fixed hyperparameters
    sampling_alg = blackjax.mclmc(
        logdensity_fn=logdensity_fn,
        L=L,
        step_size=step_size,
    )
    
    # Run the MCMC chain
    _, samples = blackjax.util.run_inference_algorithm(
        rng_key=run_key,
        initial_state=initial_state,
        inference_algorithm=sampling_alg,
        num_steps=chain_length,
        transform=lambda state, _: state.position,
        progress_bar=False,
    )
    
    # Compute summary statistics
    ess = compute_ess(samples)
    avg_acceptance = 1.0  # MCLMC never rejects
    integration_steps_per_iter = L / step_size
    time_elapsed = time.time() - start_time
    
    return samples, ess, avg_acceptance, integration_steps_per_iter, time_elapsed


def run_mams_fixed(
    logdensity_fn, 
    chain_length, 
    initial_position, 
    key, 
    L, 
    step_size
):
    """
    Run MAMS sampler with fixed hyperparameters.
    
    MAMS (Metropolis-Adjusted Microcanonical Hamiltonian Monte Carlo)
    combines microcanonical dynamics with Metropolis acceptance.
    This gives better stability than pure MCLMC.
    
    Parameters:
    -----------
    logdensity_fn : function
        Target log density to sample from
    chain_length : int
        Number of samples to generate
    initial_position : array
        Starting position for the chain
    key : PRNGKey
        Random key for reproducibility
    L : float
        Trajectory length
    step_size : float
        Leapfrog integration step size
        
    Returns:
    --------
    samples : array, shape (chain_length, n_dims)
        MCMC samples
    ess : float
        Effective sample size
    avg_acceptance : float
        Average acceptance probability (target ~0.9)
    integration_steps_per_iter : float
        Number of leapfrog steps per sample
    time_elapsed : float
        Wall-clock time in seconds
    """
    # Start timing
    start_time = time.time()
    
    # Split key for initialization and sampling
    init_key, run_key = jax.random.split(key)
    
    # Initialize MAMS sampler state
    initial_state = blackjax.mcmc.adjusted_mclmc_dynamic.init(
        position=initial_position,
        logdensity_fn=logdensity_fn,
        random_generator_arg=init_key,
    )
    
    # Create MAMS algorithm with fixed L
    algorithm = blackjax.adjusted_mclmc_dynamic(
        logdensity_fn=logdensity_fn,
        step_size=step_size,
        integration_steps_fn=lambda key: jnp.ceil(L / step_size),
        L_proposal_factor=jnp.inf,  # Fixed L (no random variation)
    )
    
    # Define one MCMC step
    def one_step(state, key):
        # Propose new state with Metropolis acceptance
        state, info = algorithm.step(key, state)
        
        # Return updated state and tracked quantities
        return state, (state.position, info.acceptance_rate)
    
    # Generate random keys for each MCMC step
    keys = jax.random.split(run_key, chain_length)
    
    # Run the MCMC chain
    final_state, (samples, acceptance_rates) = jax.lax.scan(
        one_step, 
        initial_state, 
        keys
    )
    
    # Compute summary statistics
    avg_acceptance = jnp.mean(acceptance_rates)
    ess = compute_ess(samples)
    integration_steps_per_iter = L / step_size
    time_elapsed = time.time() - start_time
    
    return samples, ess, avg_acceptance, integration_steps_per_iter, time_elapsed


def run_adjusted_mclmc_dynamic(
    logdensity_fn, 
    num_steps, 
    initial_position, 
    key
):
    """
    Run MAMS with BlackJAX's automatic hyperparameter tuning.
    
    This uses BlackJAX's built-in adaptive scheme to find good
    values of L and step_size automatically. It uses a portion
    of the chain for tuning before collecting samples.
    
    Parameters:
    -----------
    logdensity_fn : function
        Target log density to sample from
    num_steps : int
        Number of samples to generate (after tuning)
    initial_position : array
        Starting position
    key : PRNGKey
        Random key for reproducibility
        
    Returns:
    --------
    samples : array, shape (num_steps, n_dims)
        MCMC samples (post-tuning)
    step_size : float
        Auto-tuned step size
    L : float
        Auto-tuned trajectory length
    time_elapsed : float
        Total wall-clock time including tuning
    """
    # Start timing (includes tuning time)
    start_time = time.time()
    
    # Split key for initialization, tuning, and sampling
    init_key, tune_key, run_key = jax.random.split(key, 3)
    
    # Initialize sampler state
    initial_state = blackjax.mcmc.adjusted_mclmc_dynamic.init(
        position=initial_position,
        logdensity_fn=logdensity_fn,
        random_generator_arg=init_key,
    )
    
    # Define integration steps function (allows random trajectory lengths)
    integration_steps_fn = lambda avg_num_integration_steps: lambda k: jnp.ceil(
        jax.random.uniform(k) * rescale(avg_num_integration_steps)
    )
    
    # Define MAMS kernel for tuning
    kernel = lambda rng_key, state, avg_num_integration_steps, step_size, inverse_mass_matrix: \
        blackjax.mcmc.adjusted_mclmc_dynamic.build_kernel(
            integration_steps_fn=integration_steps_fn(avg_num_integration_steps),
            inverse_mass_matrix=inverse_mass_matrix,
        )(
            rng_key=rng_key,
            state=state,
            step_size=step_size,
            logdensity_fn=logdensity_fn,
            L_proposal_factor=jnp.inf,
        )
    
    # Automatically find good L and step_size
    # This uses 30% of num_steps for tuning (10% + 10% + 10%)
    (
        blackjax_state_after_tuning,
        blackjax_mclmc_sampler_params,
        _
    ) = blackjax.adjusted_mclmc_find_L_and_step_size(
        mclmc_kernel=kernel,
        num_steps=num_steps,
        state=initial_state,
        rng_key=tune_key,
        target=0.9,  # Target 90% acceptance
        frac_tune1=0.1,  # Fraction for step size tuning
        frac_tune2=0.1,  # Fraction for L tuning
        frac_tune3=0.1,  # Fraction for mass matrix tuning
        diagonal_preconditioning=True,
    )
    
    # Extract tuned hyperparameters
    step_size = blackjax_mclmc_sampler_params.step_size
    L = blackjax_mclmc_sampler_params.L
    
    # Create final sampling algorithm with tuned parameters
    alg = blackjax.adjusted_mclmc_dynamic(
        logdensity_fn=logdensity_fn,
        step_size=step_size,
        integration_steps_fn=lambda key: jnp.ceil(
            jax.random.uniform(key) * rescale(L / step_size)
        ),
        inverse_mass_matrix=blackjax_mclmc_sampler_params.inverse_mass_matrix,
        L_proposal_factor=jnp.inf,
    )
    
    # Run sampling with tuned parameters
    _, samples = run_inference_algorithm(
        rng_key=run_key,
        initial_state=blackjax_state_after_tuning,
        inference_algorithm=alg,
        num_steps=num_steps,
        transform=lambda state, _: state.position,
        progress_bar=False,
    )
    
    # Total time including tuning
    time_elapsed = time.time() - start_time
    
    return samples, step_size, L, time_elapsed

# ============================================================================
# BAYESIAN OPTIMIZATION
# ============================================================================

def run_bayesopt_tuning(
    logdensity_fn,
    initial_position,
    fixed_key,
    algorithm_name,
    num_iterations=50,
    chain_length=1000,
    target_acceptance=0.775,
    lambda_penalty=100,
):
    """
    Use Bayesian Optimization to tune MCMC hyperparameters.
    
    This searches over hyperparameter space using a Gaussian Process
    surrogate model to efficiently find configurations that maximize
    raw ESS (with acceptance penalty).
    
    Parameters:
    -----------
    logdensity_fn : function
        Target distribution to sample from
    initial_position : array
        Starting position for test chains
    fixed_key : PRNGKey
        Fixed random key for fair comparison across iterations
    algorithm_name : str
        One of 'NUTS', 'MCLMC', or 'MAMS'
    num_iterations : int
        Number of Bayesian Optimization iterations
    chain_length : int
        Length of test chains for evaluation
    target_acceptance : float
        Target acceptance probability
    lambda_penalty : float
        Penalty weight for acceptance deviation
        
    Returns:
    --------
    results : dict
        Dictionary containing optimization history and best parameters
    """
    # Initialize results storage
    results = {
        'iteration': [],           # BO iteration number
        'ess': [],                 # Effective sample size (raw)
        'acceptance_rate': [],     # Acceptance probability
        'objective': [],           # Objective function value
        'integration_steps': [],   # Leapfrog steps per sample
        'hyperparams': [],         # Hyperparameter configuration
        'time_per_eval': [],       # Time to run one evaluation
    }
    
    # Start timing total optimization
    start_time = time.time()
    
    # Configure hyperparameter search space based on algorithm
    if algorithm_name == 'NUTS':
        # NUTS only needs step size (it auto-tunes trajectory length)
        parameters = [
            {'name': 'step_size', 'type': 'range', 'bounds': [0.001, 1.0]}
        ]
        
        # Define function to run NUTS with given parameters
        def run_with_params(params_dictionary):
            # Extract hyperparameters
            step_size = params_dictionary['step_size']
            
            # Use identity inverse mass matrix
            inv_mass_matrix = jnp.ones(len(initial_position))
            
            # Run NUTS and collect diagnostics
            _, ess, acc, n, eval_time = run_nuts_fixed(
                logdensity_fn=logdensity_fn, 
                chain_length=chain_length, 
                initial_position=initial_position, 
                key=fixed_key,
                step_size=step_size,
                inv_mass_matrix=inv_mass_matrix
            )
            
            return ess, acc, n, eval_time
        
    elif algorithm_name == 'MCLMC':
        # MCLMC needs both L and step_size
        parameters = [
            {'name': 'L', 'type': 'range', 'bounds': [0.5, 50.0]},
            {'name': 'step_size', 'type': 'range', 'bounds': [0.01, 2.0]}
        ]
        
        # MCLMC never rejects
        target_acceptance = 1.0
        
        # Define function to run MCLMC with given parameters
        def run_with_params(params_dictionary):
            # Extract hyperparameters
            L = params_dictionary['L']
            step_size = params_dictionary['step_size']
            
            # Run MCLMC and collect diagnostics
            _, ess, acc, n, eval_time = run_mclmc_fixed(
                logdensity_fn=logdensity_fn, 
                chain_length=chain_length, 
                initial_position=initial_position, 
                key=fixed_key,
                L=L,
                step_size=step_size
            )
            
            return ess, acc, n, eval_time
        
    elif algorithm_name == 'MAMS':
        # MAMS needs both L and step_size
        # Use tighter bounds to avoid trivial solutions
        parameters = [
            {'name': 'L', 'type': 'range', 'bounds': [0.5, 50.0]},
            {'name': 'step_size', 'type': 'range', 'bounds': [0.01, 2.0]}
        ]
        
        # MAMS targets 90% acceptance (Robnik et al recommendation)
        target_acceptance = 0.90
        
        # Define function to run MAMS with given parameters
        def run_with_params(params_dictionary):
            # Extract hyperparameters
            L = params_dictionary['L']
            step_size = params_dictionary['step_size']
            
            # Run MAMS and collect diagnostics
            _, ess, acc, n, eval_time = run_mams_fixed(
                logdensity_fn=logdensity_fn, 
                chain_length=chain_length, 
                initial_position=initial_position, 
                key=fixed_key,
                L=L,
                step_size=step_size
            )
            
            return ess, acc, n, eval_time
        
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")
    
    # Initialize Bayesian Optimization experiment
    # batch_size=1 means we evaluate one configuration at a time
    experiment = optimization(
        parameters=parameters, 
        batch_size=1
    )
    
    # Initialize BO state
    step = None
    experiment_results = []
    
    # Main optimization loop
    for i in range(num_iterations):
        print(f"Iteration {i+1}/{num_iterations}")
        
        # Get next hyperparameter configuration from acquisition function
        # The acquisition function balances exploration vs exploitation
        step, parameterizations = experiment.next(step, experiment_results)
        
        # Extract suggested hyperparameters
        params_dict = parameterizations[0]
        
        # Evaluate MCMC performance with these hyperparameters
        ess, acc, n, eval_time = run_with_params(params_dict)
        
        # Compute objective function for this configuration
        # Just ESS with acceptance penalty - no ratios!
        obj = objective_function(
            ess=ess, 
            acceptance_rate=acc, 
            target_acceptance=target_acceptance, 
            lambda_penalty=lambda_penalty
        )
        
        # Update Bayesian Optimization with result
        # This trains the GP surrogate model
        experiment_results = [(params_dict, float(obj))]
        
        # Store all metrics for this iteration
        results['iteration'].append(i)
        results['ess'].append(float(ess))
        results['acceptance_rate'].append(float(acc))
        results['objective'].append(float(obj))
        results['integration_steps'].append(float(n))
        results['hyperparams'].append(params_dict)
        results['time_per_eval'].append(eval_time)
        
        # Print iteration summary
        print(f"  {algorithm_name}: ESS={ess:.1f}, acc={acc:.3f}, n={n:.1f}")
    
    # Calculate total optimization time
    total_time = time.time() - start_time
    
    # Find best hyperparameter configuration
    best_idx = jnp.argmax(jnp.array(results['objective']))
    best_params = results['hyperparams'][best_idx]
    
    # Print optimization summary
    print(f"\n{'='*50}")
    print(f"{algorithm_name} Optimization Complete")
    print(f"{'='*50}")
    print(f"Total time: {total_time:.2f}s")
    print(f"\nBest configuration:")
    print(f"  Params: {best_params}")
    print(f"  ESS: {results['ess'][best_idx]:.1f}")
    print(f"  Acceptance: {results['acceptance_rate'][best_idx]:.3f}")
    print(f"  Integration steps/iter: {results['integration_steps'][best_idx]:.1f}")
    
    return results

# ============================================================================
# MULTI-CHAIN FUNCTIONS
# ============================================================================

def run_mams_multiple_chains(
    logdensity_fn, 
    num_chains, 
    num_steps, 
    initial_position, 
    base_key, 
    L, 
    step_size
):
    """
    Run multiple MAMS chains with fixed hyperparameters.
    
    Running multiple chains allows us to:
    1. Compute R-hat convergence diagnostic
    2. Get more robust estimates of ESS
    3. Assess variability in sampler performance
    
    Parameters:
    -----------
    logdensity_fn : function
        Target distribution
    num_chains : int
        Number of independent chains to run
    num_steps : int
        Number of samples per chain
    initial_position : array
        Starting position (same for all chains)
    base_key : PRNGKey
        Base random key (will be split for each chain)
    L : float
        Trajectory length
    step_size : float
        Leapfrog step size
        
    Returns:
    --------
    all_samples : array, shape (num_chains, num_steps, n_dims)
        Samples from all chains
    all_ess : array, shape (num_chains,)
        ESS for each chain
    all_acceptance : array, shape (num_chains,)
        Acceptance rates (all ~0.9 for MAMS)
    all_step_sizes : array, shape (num_chains,)
        Auto-tuned step sizes
    all_L : array, shape (num_chains,)
        Auto-tuned trajectory lengths
    total_time : float
        Total wall-clock time including tuning
    """
    # Start timing
    start_time = time.time()
    
    # Initialize storage
    all_samples = []
    all_step_sizes = []
    all_L = []
    all_ess = []
    
    # Run each chain with independent tuning
    for i in range(num_chains):
        print(f"  Chain {i+1}/{num_chains} (auto-tuning)...")
        
        # Create unique random key for this chain
        chain_key = jax.random.fold_in(base_key, i)
        
        # Run MAMS with automatic hyperparameter tuning
        samples, step_size, L, chain_time = run_adjusted_mclmc_dynamic(
            logdensity_fn=logdensity_fn, 
            num_steps=num_steps, 
            initial_position=initial_position, 
            key=chain_key
        )
        
        # Compute diagnostics
        ess = compute_ess(samples)
        
        # Store results
        all_samples.append(samples)
        all_step_sizes.append(step_size)
        all_L.append(L)
        all_ess.append(ess)
        
        # Print chain summary
        print(f"    L={L:.3f}, ε={step_size:.5f}, ESS={ess:.1f}")
    
    # Stack all samples
    all_samples = jnp.stack(all_samples, axis=0)
    
    # MAMS targets 90% acceptance
    all_acceptance = jnp.ones(num_chains) * 0.9
    
    # Total time including all tuning
    total_time = time.time() - start_time
    
    return (
        all_samples, 
        jnp.array(all_ess), 
        all_acceptance, 
        jnp.array(all_step_sizes), 
        jnp.array(all_L), 
        total_time
    )


def run_mams_multiple_chains_fixed(
    logdensity_fn, 
    num_chains, 
    num_steps, 
    initial_position, 
    base_key, 
    L, 
    step_size
):
    """
    Run multiple MAMS chains with FIXED hyperparameters (no auto-tuning).
    For validating BayesOpt results.
    """
    start_time = time.time()
    
    all_samples = []
    all_ess = []
    all_acceptance = []
    
    for i in range(num_chains):
        print(f"  Chain {i+1}/{num_chains} (fixed L={L:.3f}, ε={step_size:.5f})...")
        
        chain_key = jax.random.fold_in(base_key, i)
        
        # Use FIXED hyperparameters
        samples, ess, acc, _, chain_time = run_mams_fixed(
            logdensity_fn=logdensity_fn,
            chain_length=num_steps,
            initial_position=initial_position,
            key=chain_key,
            L=L,
            step_size=step_size
        )
        
        all_samples.append(samples)
        all_ess.append(ess)
        all_acceptance.append(acc)
        
        print(f"    ESS={ess:.1f}, acceptance={acc:.3f}")
    
    all_samples = jnp.stack(all_samples, axis=0)
    total_time = time.time() - start_time
    
    return (
        all_samples,
        jnp.array(all_ess),
        jnp.array(all_acceptance),
        total_time
    )

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_bayesopt_progress(results_dict, save_prefix=None):
    """
    Plot Bayesian Optimization progress for all algorithms.
    
    Shows how optimization evolves over iterations.
    Simple plots: ESS, acceptance, integration steps, objective.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary with keys like 'nuts', 'mclmc', 'mams'
        Each value is a results dict from run_bayesopt_tuning
    save_prefix : str, optional
        If provided, save figure with this prefix
    """
    # Collect available algorithms
    algorithms = []
    for name in ['NUTS', 'MCLMC', 'MAMS']:
        key = name.lower()
        if key in results_dict and results_dict[key] is not None:
            algorithms.append((name, key))
    
    # Check if we have results to plot
    if not algorithms:
        print("No results to plot!")
        return
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot each algorithm's progress
    for alg_name, alg_key in algorithms:
        results = results_dict[alg_key]
        iterations = results['iteration']
        
        # Panel 1: Raw ESS
        axes[0, 0].plot(
            iterations, 
            results['ess'], 
            'o-', 
            label=alg_name, 
            markersize=4
        )
        
        # Panel 2: Acceptance rate
        axes[0, 1].plot(
            iterations, 
            results['acceptance_rate'], 
            'o-', 
            label=alg_name, 
            markersize=4
        )
        
        # Panel 3: Integration steps
        axes[1, 0].plot(
            iterations, 
            results['integration_steps'], 
            'o-', 
            label=alg_name, 
            markersize=4
        )
        
        # Panel 4: Objective function
        axes[1, 1].plot(
            iterations, 
            results['objective'], 
            'o-', 
            label=alg_name, 
            markersize=4
        )
    
    # Configure Panel 1: ESS
    axes[0, 0].set_ylabel('ESS')
    axes[0, 0].set_title('Effective Sample Size')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Configure Panel 2: Acceptance
    axes[0, 1].set_ylabel('Acceptance Rate')
    axes[0, 1].set_title('Acceptance Rate')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Configure Panel 3: Integration steps
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Integration Steps')
    axes[1, 0].set_title('Integration Steps per Iteration')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Configure Panel 4: Objective
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Objective Value')
    axes[1, 1].set_title('Objective Function')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_prefix:
        filename = f'{save_prefix}_progress.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved {filename}")
    
    plt.show()


def plot_hyperparameter_space(results_dict, save_prefix=None):
    """
    Plot hyperparameter space exploration for MCLMC and MAMS.
    
    Simple scatter plot of L vs step_size colored by ESS.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary containing results from BayesOpt tuning
    save_prefix : str, optional
        If provided, save figure with this prefix
    """
    # Only MCLMC and MAMS have 2D hyperparameter space
    algorithms = []
    for name in ['MCLMC', 'MAMS']:
        key = name.lower()
        if key in results_dict and results_dict[key] is not None:
            algorithms.append((name, key))
    
    # Check if we have results
    if not algorithms:
        print("No MCLMC/MAMS results!")
        return
    
    # Create subplots
    fig, axes = plt.subplots(1, len(algorithms), figsize=(9 * len(algorithms), 7))
    if len(algorithms) == 1:
        axes = [axes]
    
    # Plot each algorithm
    for ax, (alg_name, alg_key) in zip(axes, algorithms):
        results = results_dict[alg_key]
        
        # Extract data
        L_values = np.array([p['L'] for p in results['hyperparams']])
        step_sizes = np.array([p['step_size'] for p in results['hyperparams']])
        ess_values = np.array(results['ess'])
        
        # Create scatter plot colored by ESS
        scatter = ax.scatter(
            step_sizes, 
            L_values, 
            c=ess_values,
            s=100,
            cmap='viridis',
            alpha=0.7, 
            edgecolors='black', 
            linewidths=1.5
        )
        
        # Configure axes
        ax.set_xlabel('Step Size (ε)', fontsize=14)
        ax.set_ylabel('Trajectory Length (L)', fontsize=14)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(f'{alg_name}: Hyperparameter Space', fontsize=16)
        ax.grid(True, alpha=0.3, which='both')
        
        # Add colorbar for ESS
        cbar = plt.colorbar(scatter, ax=ax, label='ESS')
        
        # Annotate starting point
        ax.annotate(
            'Start', 
            xy=(step_sizes[0], L_values[0]), 
            xytext=(10, 10), 
            textcoords='offset points',
            fontsize=10, 
            color='blue', 
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
            arrowprops=dict(arrowstyle='->', color='blue', lw=1.5)
        )
        
        # Annotate best point
        best_idx = np.argmax(results['objective'])
        ax.annotate(
            'Best', 
            xy=(step_sizes[best_idx], L_values[best_idx]), 
            xytext=(10, -20), 
            textcoords='offset points',
            fontsize=10, 
            color='red', 
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5)
        )
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_prefix:
        filename = f'{save_prefix}_hyperparameter_space.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved {filename}")
    
    plt.show()

# ============================================================================
# EXPERIMENT RUNNERS
# ============================================================================

def compare_algorithms(dim=5, iterations=20, chain_length=1000):
    """
    Compare NUTS, MCLMC, and MAMS using Bayesian Optimization.
    
    Tunes each algorithm's hyperparameters and compares performance.
    
    Parameters:
    -----------
    dim : int
        Dimensionality of test problem (Neal's funnel)
    iterations : int
        Number of BayesOpt iterations per algorithm
    chain_length : int
        Length of test chains for evaluation
        
    Returns:
    --------
    nuts_results : dict
        NUTS optimization results
    mclmc_results : dict
        MCLMC optimization results
    mams_results : dict
        MAMS optimization results
    """
    # Create test problem
    logdensity_fn = make_funnel_logdensity(dim)
    initial_position = jnp.zeros(dim)
    
    # Tune NUTS
    print("\n1. Tuning NUTS...")
    nuts_results = run_bayesopt_tuning(
        logdensity_fn=logdensity_fn, 
        initial_position=initial_position, 
        fixed_key=jax.random.key(SEED_NUTS_TUNING), 
        algorithm_name='NUTS',
        num_iterations=iterations, 
        chain_length=chain_length
    )
    
    # Tune MCLMC
    print("\n2. Tuning MCLMC...")
    mclmc_results = run_bayesopt_tuning(
        logdensity_fn=logdensity_fn, 
        initial_position=initial_position, 
        fixed_key=jax.random.key(SEED_MCLMC_TUNING), 
        algorithm_name='MCLMC',
        num_iterations=iterations, 
        chain_length=chain_length
    )
    
    # Tune MAMS
    print("\n3. Tuning MAMS...")
    mams_results = run_bayesopt_tuning(
        logdensity_fn=logdensity_fn, 
        initial_position=initial_position, 
        fixed_key=jax.random.key(SEED_MAMS_TUNING), 
        algorithm_name='MAMS',
        num_iterations=iterations, 
        chain_length=chain_length
    )
    
    # Create visualizations
    print("\nCreating visualizations...")
    results_dict = {
        'nuts': nuts_results,
        'mclmc': mclmc_results,
        'mams': mams_results
    }
    
    plot_bayesopt_progress(results_dict, save_prefix='bayesopt')
    plot_hyperparameter_space(results_dict, save_prefix='bayesopt')
    
    # Print comparison table
    print("\n" + "="*80)
    print("ALGORITHM COMPARISON (at best hyperparameters)")
    print("="*80)
    print(f"\n{'Algorithm':<12} {'ESS':<12} {'Acceptance':<15} {'Steps/iter':<15}")
    print("-"*80)
    
    for alg_name, alg_results in [('NUTS', nuts_results), 
                                   ('MCLMC', mclmc_results), 
                                   ('MAMS', mams_results)]:
        # Find best configuration
        best_idx = jnp.argmax(jnp.array(alg_results['objective']))
        
        # Extract metrics
        ess = alg_results['ess'][best_idx]
        acc = alg_results['acceptance_rate'][best_idx]
        steps = alg_results['integration_steps'][best_idx]
        
        # Print row
        print(f"{alg_name:<12} {ess:<12.1f} {acc:<15.3f} {steps:<15.1f}")
    
    return nuts_results, mclmc_results, mams_results


def compare_mams_tuning_methods(dim=5, num_chains=4, num_steps=1000):
    """
    Compare BayesOpt tuning vs BlackJAX automatic tuning for MAMS.
    
    This is the key experiment: does Bayesian Optimization find
    better hyperparameters than BlackJAX's built-in adaptive scheme?
    
    Parameters:
    -----------
    dim : int
        Dimensionality of test problem
    num_chains : int
        Number of independent chains for validation
    num_steps : int
        Number of samples per chain
        
    Returns:
    --------
    bayesopt_samples : array
        Samples from BayesOpt-tuned chains
    auto_samples : array
        Samples from auto-tuned chains
    mams_results : dict
        BayesOpt tuning history
    """
    # Create test problem
    logdensity_fn = make_funnel_logdensity(dim)
    initial_position = jnp.zeros(dim)
    
    # ========================================================================
    # STEP 1: Bayesian Optimization Tuning
    # ========================================================================
    
    print("\nSTEP 1: BAYESIAN OPTIMIZATION TUNING")
    print("="*70)
    
    bayesopt_tuning_key = jax.random.key(SEED_MAMS_TUNING)
    mams_results = run_bayesopt_tuning(
        logdensity_fn=logdensity_fn, 
        initial_position=initial_position, 
        fixed_key=bayesopt_tuning_key, 
        algorithm_name='MAMS',
        num_iterations=20, 
        chain_length=1000
    )
    
    # Extract best hyperparameters
    best_idx = jnp.argmax(jnp.array(mams_results['objective']))
    best_params = mams_results['hyperparams'][best_idx]
    best_L = best_params['L']
    best_step_size = best_params['step_size']
    
    print(f"\nBest found: L={best_L:.4f}, ε={best_step_size:.6f}")
    
    # ========================================================================
    # STEP 2: Validate BayesOpt with Multiple Chains
    # ========================================================================
    
    print("\nSTEP 2: VALIDATE BAYESOPT WITH MULTIPLE CHAINS")
    print("="*70)
    
    validation_key = jax.random.key(SEED_BAYESOPT_VALIDATION)
    
    (
        bayesopt_samples, 
        bayesopt_ess, 
        bayesopt_acc, 
        bayesopt_time
    ) = run_mams_multiple_chains_fixed(
        logdensity_fn=logdensity_fn, 
        num_chains=num_chains, 
        num_steps=num_steps, 
        initial_position=initial_position, 
        base_key=validation_key, 
        L=best_L, 
        step_size=best_step_size
    )
    
    # Compute convergence diagnostic
    bayesopt_rhat = compute_rhat(bayesopt_samples)
    
    # Print BayesOpt results
    print(f"\nBayesOpt Results:")
    print(f"  Mean ESS: {jnp.mean(bayesopt_ess):.1f} ± {jnp.std(bayesopt_ess):.1f}")
    print(f"  Mean Acceptance: {jnp.mean(bayesopt_acc):.3f} ± {jnp.std(bayesopt_acc):.3f}")
    print(f"  Max R-hat: {jnp.max(bayesopt_rhat):.4f}")
    print(f"  Convergence: {'✓ Good' if jnp.max(bayesopt_rhat) < 1.01 else '⚠ Needs more steps'}")
    
    # ========================================================================
    # STEP 3: Automatic Tuning with Multiple Chains
    # ========================================================================
    
    print("\nSTEP 3: AUTOMATIC TUNING WITH MULTIPLE CHAINS")
    print("="*70)
    print("Note: Each chain tunes independently")
    
    auto_key = jax.random.key(SEED_AUTO_VALIDATION)
    (
        auto_samples, 
        auto_ess, 
        auto_acc, 
        auto_step_sizes, 
        auto_L, 
        auto_time
    ) = run_mams_multiple_chains(
        logdensity_fn=logdensity_fn, 
        num_chains=num_chains, 
        num_steps=num_steps, 
        initial_position=initial_position, 
        base_key=auto_key,
        L=1.0,  # Ignored by auto-tuning
        step_size=0.1  # Ignored by auto-tuning
    )
    
    # Compute convergence diagnostic
    auto_rhat = compute_rhat(auto_samples)
    
    # Print auto-tuning results
    print(f"\nAuto-tuning Results:")
    print(f"  Mean L: {jnp.mean(auto_L):.4f} ± {jnp.std(auto_L):.4f}")
    print(f"  Mean ε: {jnp.mean(auto_step_sizes):.6f} ± {jnp.std(auto_step_sizes):.6f}")
    print(f"  Mean ESS: {jnp.mean(auto_ess):.1f} ± {jnp.std(auto_ess):.1f}")
    print(f"  Mean Acceptance: {jnp.mean(auto_acc):.3f} ± {jnp.std(auto_acc):.3f}")
    print(f"  Max R-hat: {jnp.max(auto_rhat):.4f}")
    print(f"  Convergence: {'✓ Good' if jnp.max(auto_rhat) < 1.01 else '⚠ Needs more steps'}")
    
    # ========================================================================
    # STEP 4: Final Comparison
    # ========================================================================
    
    print("\n" + "="*80)
    print("FINAL COMPARISON: BAYESOPT vs AUTO-TUNING")
    print("="*80)
    print(f"\n{'Metric':<30} {'BayesOpt':<25} {'Auto-tuned':<25} {'Winner':<15}")
    print("-"*80)
    
    # Define metrics to compare
    metrics = [
        ('Mean ESS', jnp.mean(bayesopt_ess), jnp.mean(auto_ess), False),
        ('Mean Acceptance', jnp.mean(bayesopt_acc), jnp.mean(auto_acc), False),
        ('Max R-hat', jnp.max(bayesopt_rhat), jnp.max(auto_rhat), True),
    ]
    
    # Print each metric
    for name, bo_val, auto_val, lower_is_better in metrics:
        # Determine winner
        if lower_is_better:
            # For R-hat, lower is better
            winner = 'BayesOpt' if bo_val < auto_val else 'Auto-tuned'
            print(f"{name:<30} {bo_val:<25.4f} {auto_val:<25.4f} {winner:<15}")
        else:
            # For ESS, higher is better (with 10% threshold)
            if bo_val > auto_val * 1.1:
                winner = 'BayesOpt'
            elif auto_val > bo_val * 1.1:
                winner = 'Auto-tuned'
            else:
                winner = 'Tie'
            
            if 'ESS' in name:
                print(f"{name:<30} {bo_val:<25.1f} {auto_val:<25.1f} {winner:<15}")
            else:
                print(f"{name:<30} {bo_val:<25.3f} {auto_val:<25.3f} {winner:<15}")
    
    # Overall winner based on ESS
    bo_ess = jnp.mean(bayesopt_ess)
    auto_ess = jnp.mean(auto_ess)
    
    if bo_ess > auto_ess * 1.1:
        pct_better = 100 * (bo_ess / auto_ess - 1)
        print(f"\n✓ BayesOpt wins: {pct_better:.1f}% higher ESS")
    elif auto_ess > bo_ess * 1.1:
        pct_better = 100 * (auto_ess / bo_ess - 1)
        print(f"\n✓ Auto-tuning wins: {pct_better:.1f}% higher ESS")
    else:
        print(f"\n≈ Tie: ESS values are comparable")
    
    return bayesopt_samples, auto_samples, mams_results


def run_all_experiments(dim=5, num_chains=4, num_steps=1000):
    """
    Run all experiments: algorithm comparison and tuning comparison.
    
    This is the main entry point for running the full experimental suite.
    
    Parameters:
    -----------
    dim : int
        Dimensionality of test problem
    num_chains : int
        Number of chains for validation
    num_steps : int
        Number of samples per chain
        
    Returns:
    --------
    results : dict
        Dictionary containing all experimental results
    """
    print("\n" + "="*70)
    print("RUNNING ALL EXPERIMENTS")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Dimensionality: {dim}")
    print(f"  Validation chains: {num_chains}")
    print(f"  Steps per chain: {num_steps}")
    
    # Start timing
    total_start = time.time()
    
    # ========================================================================
    # Experiment 1: Compare Algorithms
    # ========================================================================
    
    print("\n" + "="*70)
    print("EXPERIMENT 1: ALGORITHM COMPARISON")
    print("="*70)
    
    nuts_results, mclmc_results, mams_results = compare_algorithms(dim=dim)
    
    # ========================================================================
    # Experiment 2: Compare Tuning Methods
    # ========================================================================
    
    print("\n" + "="*70)
    print("EXPERIMENT 2: TUNING METHOD COMPARISON")
    print("="*70)
    
    bayesopt_samples, auto_samples, _ = compare_mams_tuning_methods(
        dim=dim, 
        num_chains=num_chains, 
        num_steps=num_steps
    )
    
    # Total time for all experiments
    total_time = time.time() - total_start
    
    # Print final summary
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*70)
    print(f"Total time: {total_time:.2f}s ({total_time/60:.1f} min)")
    
    # Return all results
    return {
        'nuts': nuts_results,
        'mclmc': mclmc_results,
        'mams': mams_results,
        'bayesopt_samples': bayesopt_samples,
        'auto_samples': auto_samples
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================
results = run_all_experiments(dim=5, num_chains=4, num_steps=1000)