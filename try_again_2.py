# Compare hyperparameter tuning:
# Sequential similar to Robnik et al
# Versus Bayesian Optimization


# For Metropolis Adjusted Microcanonical Hamiltonian Monte Carlo (MAMS) there are two hyperparameters:  
# 1) stepsize ($\epsilon$) 
# 2) trajectory length L

# Goal: optimize hyperparameters to maximize ESS while keeping acceptance rate near target (squared error).





# Code here is adapted from the project directory for Robnik et al's paper, Boax for BayesOpt, and BlackJax for NUTS, MCLMC, and MAMS (adjusted MCLMC in blackjax)
# For Github for the Metropolis adjusted Microcanonical HMC paper https://github.com/reubenharry/sampler-benchmarks


#################################################################################################### Import libraries
# Imports from here: https://blackjax-devs.github.io/sampling-book/algorithms/mclmc.html#how-to-run-mclmc-in-blackjax

import matplotlib.pyplot as plt 
# I always get my matplotlib and ggplot syntax mixed up
# https://matplotlib.org/stable/users/explain/quick_start.html

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

rng_key = jax.random.key(548) # Changed from time based to course number


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






# Find ESS (Effective Sample Size)
def compute_ess(samples):
    # See: https://python.arviz.org/en/stable/api/generated/arviz.ess.html
    # Need to match the expected structure
    samples_np = np.array(samples)
    
    # Arviz expects (chain, draw, *variable_shape)
    samples_reshaped = samples_np[None, :, :]
    
    # Convert to Arviz dataset: https://python.arviz.org/en/stable/api/generated/arviz.convert_to_dataset.html#arviz.convert_to_dataset
    dataset = az.convert_to_dataset(samples_reshaped)
    
    # Compute ESS
    ess = az.ess(dataset, method = 'bulk')
    
     # Return worst, want to maximize the minimum ESS (maximin)
    return float(np.min([ess[var].values for var in ess.data_vars]))


# Find R-hat
def compute_rhat(chains): 
    # Arviz R-hat: Compute estimate of rank normalized splitR-hat for a set of traces.
    # https://python.arviz.org/en/stable/api/generated/arviz.rhat.html#arviz.rhat
    chains_np = np.array(chains)
    dataset = az.convert_to_dataset(chains_np)
    
    # Compute R-hat
    rhat = az.rhat(dataset)
    
    # Return worst, want to minimize maximum R-hat (minimax)
    return float(np.max([rhat[var].values for var in rhat.data_vars]))







# Make an objective function
# Want to maximize: ESS - lambda (acceptance rate - target accpetance rate)^2
# Can choose lambda so we don't get any BS near-zero accpetance rates
def objective_function(ess, acceptance_rate, target_acceptance, lambda_penalty):
    penalty = lambda_penalty * (acceptance_rate - target_acceptance)**2
    return ess - penalty








############################################################## MCMC Algorithms with fixed parameters for BayesOpt

def run_nuts_fixed(
    logdensity_fn, # Give it a target density
    chain_length,  # Number of samples desired
    initial_position, # Give it a starting poristion
    key,              # And a key/seed
    step_size, # Test a stepsize, maybe let it auto-tune since NUTS is a benchmark
    inv_mass_matrix # Preconditioner, use I
    ):
    
    # Start timing
    start_time = time.time()
    
    # Give NUTS hyperparameters
    nuts = blackjax.nuts(
        logdensity_fn = logdensity_fn, 
        step_size = step_size, 
        inverse_mass_matrix = inv_mass_matrix
    )
    
    # Initialize NUTS at starting position
    state = nuts.init(initial_position)
    
    # Define one NUTS step/iteration
    def one_step(state, key):
        # Propose new state and compute diagnostics
        state, info = nuts.step(key, state)
        
        # Return updated state and stats
        return state, (state.position, info.acceptance_rate, info.num_integration_steps)
    
    
    # Keys for each NUTS step/iteration
    keys = jax.random.split(key, chain_length)
    
    # Run the NUTS chain
    final_state, (samples, acceptance_rates, num_steps_per_iter) = jax.lax.scan(
        one_step, 
        state, 
        keys
    )
    
    # Summary stats
    avg_acceptance = jnp.mean(acceptance_rates)
    ess = compute_ess(samples)
    avg_integration_steps = jnp.mean(num_steps_per_iter)
    time_elapsed = time.time() - start_time
    
    # Return the chain and stats
    return samples, ess, avg_acceptance, avg_integration_steps, time_elapsed


# Run MCLMC with fixed step size and L for optimizing with BayesOpt
# https://blackjax-devs.github.io/sampling-book/algorithms/mclmc.html#how-to-run-mclmc-in-blackjax
def run_mclmc_fixed(
    logdensity_fn, 
    chain_length, 
    initial_position, 
    key, 
    L, 
    step_size
    ):
    
    # Start timing
    start_time = time.time()
    
    # initialization and sampling key/seed
    init_key, run_key = jax.random.split(key)
    
    # Initialize MCLMC
    initial_state = blackjax.mcmc.mclmc.init(
        position = initial_position, 
        logdensity_fn = logdensity_fn, 
        rng_key = init_key
    )
    
    # Use the given hyperparameters
    sampling_alg = blackjax.mclmc(
        logdensity_fn = logdensity_fn,
        L = L,
        step_size = step_size,
    )
    
    # Run the chain
    _, samples = blackjax.util.run_inference_algorithm(
        rng_key = run_key,
        initial_state = initial_state,
        inference_algorithm = sampling_alg,
        num_steps = chain_length,
        transform = lambda state, _: state.position,
        progress_bar = False
        )
    
    # Summary stats
    ess = compute_ess(samples)
    avg_acceptance = 1.0  # MCLMC never rejects
    integration_steps_per_iter = L / step_size
    time_elapsed = time.time() - start_time
    
    return samples, ess, avg_acceptance, integration_steps_per_iter, time_elapsed


# Same as MCLMC but now with the Metropolis step so it can reject
# https://blackjax-devs.github.io/sampling-book/algorithms/mclmc.html#adjusted-mclmc
def run_mams_fixed(
    logdensity_fn, 
    chain_length, 
    initial_position, 
    key, 
    L, 
    step_size
    ):
    
    # Start timing
    start_time = time.time()
    
    # Initialization and iteration seeds
    init_key, run_key = jax.random.split(key)
    
    # Initialize MAMS
    initial_state = blackjax.mcmc.adjusted_mclmc_dynamic.init(
        position = initial_position,
        logdensity_fn = logdensity_fn,
        random_generator_arg=init_key,
    )
    
    # Create MAMS algorithm with fixed L
    algorithm = blackjax.adjusted_mclmc_dynamic(
        logdensity_fn = logdensity_fn,
        step_size = step_size,
        integration_steps_fn=lambda key: jnp.ceil(L / step_size),
        L_proposal_factor = jnp.inf,
    )
    
    # One iteration
    def one_step(state, key):
        
        # Propose new state with Metropolis accpetance probability
        state, info = algorithm.step(key, state)
        return state, (state.position, info.acceptance_rate)
    
    
    # Seed/key for each sample
    keys = jax.random.split(run_key, chain_length)
    
    # Run MAMS
    final_state, (samples, acceptance_rates) = jax.lax.scan(
        one_step, 
        initial_state, 
        keys
    )
    
    # Summary stats
    avg_acceptance = jnp.mean(acceptance_rates)
    ess = compute_ess(samples)
    integration_steps_per_iter = L / step_size
    time_elapsed = time.time() - start_time
    
    return samples, ess, avg_acceptance, integration_steps_per_iter, time_elapsed




############################################################## MCMC Algorithms with auto-tuned parameters (regular blackjax)


# Metropolis Adjusted MCLMC
# Copied from here: https://blackjax-devs.github.io/sampling-book/algorithms/mclmc.html#adjusted-mclmc
def run_mams_auto(
    logdensity_fn, 
    num_steps, 
    initial_position, 
    key
    ):
    
    # Start timing
    start_time = time.time()
    
    # initialization, tuning, running seeds
    init_key, tune_key, run_key = jax.random.split(key, 3)
    
    # Initial state of MAMS
    initial_state = blackjax.mcmc.adjusted_mclmc_dynamic.init(
        position=initial_position,
        logdensity_fn=logdensity_fn,
        random_generator_arg=init_key,
    )
    
    integration_steps_fn = lambda avg_num_integration_steps: lambda k: jnp.ceil(
        jax.random.uniform(k) * rescale(avg_num_integration_steps)
    )
    
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
    
    (
        blackjax_state_after_tuning,
        blackjax_mclmc_sampler_params,
        _
    ) = blackjax.adjusted_mclmc_find_L_and_step_size(
        mclmc_kernel=kernel,
        num_steps=num_steps,
        state=initial_state,
        rng_key=tune_key,
        target = 0.90,      # Target 90% acceptance
        frac_tune1 = 0.1,  # Fraction for step size tuning
        frac_tune2 = 0.1,  # Fraction for L tuning
        frac_tune3 = 0.1,  # Fraction for mass matrix tuning
        diagonal_preconditioning = False, # Turn off diagonal preconditioner for fair comparison against BayesOpt
    )
    
    # Extract hyperparameters from tuning phase
    step_size = blackjax_mclmc_sampler_params.step_size
    L = blackjax_mclmc_sampler_params.L
    
    # The algorithm with the tuned hyperparameters
    alg = blackjax.adjusted_mclmc_dynamic(
        logdensity_fn=logdensity_fn,
        step_size=step_size,
        integration_steps_fn=lambda key: jnp.ceil(
            jax.random.uniform(key) * rescale(L / step_size)
        ),
        inverse_mass_matrix=blackjax_mclmc_sampler_params.inverse_mass_matrix,
        L_proposal_factor=jnp.inf,
    )
    
    # Run it
    _, samples = run_inference_algorithm(
        rng_key=run_key,
        initial_state=blackjax_state_after_tuning,
        inference_algorithm=alg,
        num_steps=num_steps,
        transform=lambda state, _: state.position,
        progress_bar=False,
    )
    
    # Time including tuning and discarding tuning phase
    time_elapsed = time.time() - start_time
    
    return samples, step_size, L, time_elapsed



# MAKE NUTS AND MCLMC 




########################################################################## BayesOpt them

def run_bayesopt_tuning(
    logdensity_fn, # Target density
    initial_position, # Consitent starting position
    fixed_key, # Reproducible
    algorithm_name, # MAMS/MCLMC/NUTS
    num_iterations = 50, # Number of BayesOpt explore/exploit
    chain_length = 1000,
    target_acceptance = 0.775, # NUTS target is 0.65, MAMS target is 0.90, 0.775 is between
    lambda_penalty = 100, # Penalize difference between target and actual accpetance rate to stop BS never accept high to get high ESS
):
    
    # Store history and results
    results = {
        'iteration': [],           # BayesOpt iteration number
        'ess': [],  
        'acceptance_rate': [],    
        'objective': [],           # Value of Objective function fo rthis iteration
        'integration_steps': [],   # Leapfrog steps per sample
        'hyperparams': [],         # Hyperparameter pair
        'time_per_eval': [],       # Time to run one evaluation
    }
    
    # Start timing the full optimization procedure
    start_time = time.time()
    
    
    #  Hyperparameter search space per algorithm
    
    if algorithm_name == 'NUTS':
        # NUTS only needs step size since it does the tree thing for L
        parameters = [
            {'name': 'step_size', 'type': 'range', 'bounds': [0.001, 2.0]}
        ]
        
        # Run NUTS with given stepsize
        def run_with_params(params_dictionary):
            
            step_size = params_dictionary['step_size']
            
            # Use identity inverse mass matrix
            inv_mass_matrix = jnp.ones(len(initial_position))
            
            # Run NUTS and get the results and stats
            _, ess, acc, n, eval_time = run_nuts_fixed(
                logdensity_fn = logdensity_fn, 
                chain_length = chain_length, 
                initial_position = initial_position, 
                key = fixed_key,
                step_size = step_size,
                inv_mass_matrix = inv_mass_matrix
            )
            
            return ess, acc, n, eval_time
        
        
    # MCLMC needs L and step size tuned
    elif algorithm_name == 'MCLMC':
        
        
        parameters = [
            {'name': 'L', 'type': 'range', 'bounds': [0.5, 50.0]},
            {'name': 'step_size', 'type': 'range', 'bounds': [0.01, 5.0]}
        ]
        
        # MCLMC never rejects
        # target_acceptance = 1.0
        
        # Run MCLMC with L and step size
        def run_with_params(params_dictionary):
            
            
            L = params_dictionary['L']
            step_size = params_dictionary['step_size']
            
            # Run MCLMC
            _, ess, acc, n, eval_time = run_mclmc_fixed(
                logdensity_fn = logdensity_fn, 
                chain_length = chain_length, 
                initial_position = initial_position, 
                key = fixed_key,
                L = L,
                step_size = step_size
            )
            
            return ess, acc, n, eval_time
        
        
    # MAMS needs L and step size
    elif algorithm_name == 'MAMS':
        
        
        parameters = [
            {'name': 'L', 'type': 'range', 'bounds': [0.5, 50.0]},
            {'name': 'step_size', 'type': 'range', 'bounds': [0.01, 5.0]}
        ]
        
        # MAMS target acceptance is 90% (Robnik et al recommendation)
        # target_acceptance = 0.90
        
        def run_with_params(params_dictionary):
            L = params_dictionary['L']
            step_size = params_dictionary['step_size']
            
            _, ess, acc, n, eval_time = run_mams_fixed(
                logdensity_fn = logdensity_fn, 
                chain_length = chain_length, 
                initial_position = initial_position, 
                key = fixed_key,
                L = L,
                step_size = step_size
            )
            
            return ess, acc, n, eval_time
    
    
    
    # Initialize BayesOpt experiment
    # batch_size = 1 means try one hyperparameter set at a time
    experiment = optimization(
        parameters = parameters, 
        batch_size = 1
    )
    
    step = None
    experiment_results = []
    
    
    # Loop over BayesOpt iteration
    for i in range(num_iterations):
        
        # Tell me where the optimization is at
        print(f"Iteration {i+1}/{num_iterations}")
        
        # Acquisition function choose next L stepsize pair (step size only for NUTS)
        step, parameterizations = experiment.next(step, experiment_results)
        
        # Extract next hyperparameters to try
        params_dict = parameterizations[0]
        
        # Stats for these hyperparameters
        ess, acc, n, eval_time = run_with_params(params_dict)
        
        # Compute objective for these hyperparameters
        obj = objective_function(
            ess = ess, 
            acceptance_rate = acc, 
            target_acceptance = target_acceptance, 
            lambda_penalty = lambda_penalty
        )
        
        # Update results history
        
        # These hyperparametrs get this objective function value
        experiment_results = [(params_dict, float(obj))]
        
        # Add all these stats to the history
        results['iteration'].append(i)
        results['ess'].append(float(ess))
        results['acceptance_rate'].append(float(acc))
        results['objective'].append(float(obj))
        results['integration_steps'].append(float(n))
        results['hyperparams'].append(params_dict)
        results['time_per_eval'].append(eval_time)
        
        # Print iteration summary
        print(f"{algorithm_name}: ESS is {ess:.1f} and acceptance rate  is {acc:.3f}")
    
    # Total optimization time
    total_time = time.time() - start_time
    
    # Which hyperparameters maximized the objective?
    best_id = jnp.argmax(jnp.array(results['objective']))
    best_params = results['hyperparams'][best_id]
    
    # Print summary
    print(f"{algorithm_name} BayesOpt done")
    print(f"Time: {total_time:.2f}s")
    print(f"Best hyperparamwetrs: {best_params}")
    print(f"ESS: {results['ess'][best_id]:.1f}")
    print(f"Acceptance rate: {results['acceptance_rate'][best_id]:.3f}")
    
    return results





############################################################ Run many chains for an algorithm 

# Run many MAMS chains with the Robnik et al auto tuned L and step size
def run_mams_multiple_chains(
    logdensity_fn, 
    num_chains, 
    num_steps, 
    initial_position, 
    base_key, 
    L, 
    step_size
    ):
    
    # Start timing
    start_time = time.time()
    
    # Initialize storage
    all_samples = []
    all_step_sizes = []
    all_L = []
    all_ess = []
    
    # Run each chain with independent tuning
    for i in range(num_chains):
        print(f"  Chain {i+1}/{num_chains} (auto-tuning)")
        
        # Random key for this chain
        chain_key = jax.random.fold_in(base_key, i)
        
        # Run MAMS with automatic hyperparameter tuning
        samples, step_size, L, chain_time = run_mams_auto(
            logdensity_fn = logdensity_fn, 
            num_steps = num_steps, 
            initial_position = initial_position, 
            key = chain_key
        )
        
        # Get ESS
        ess = compute_ess(samples)
        
        # Store results
        all_samples.append(samples)
        all_step_sizes.append(step_size)
        all_L.append(L)
        all_ess.append(ess)
        
        # Print chain summary
        print(f"L is {L:.3f}, step size is {step_size:.5f}, and ESS is {ess:.1f}")
    
    # Samples from all the chains
    all_samples = jnp.stack(all_samples, axis=0)
    
    # NEED TO GET THE REAL ACCPETANCES
    all_acceptance = jnp.ones(num_chains) * 0.9
    
    # Total time since teh begining
    total_time = time.time() - start_time
    
    # Return the results form teh multiple chains
    return (all_samples, jnp.array(all_ess), all_acceptance, jnp.array(all_step_sizes), jnp.array(all_L), total_time)


# Run multiple chains of MAMS with the BayesOpt hyperparameters
def run_mams_multiple_chains_fixed(
    logdensity_fn, 
    num_chains, 
    num_steps, 
    initial_position, 
    base_key, 
    L, 
    step_size
    ):
    
    # Time the procedure
    start_time = time.time()
    
    # Store teh results
    all_samples = []
    all_ess = []
    all_acceptance = []
    
    # Loop over teh chains (could paralelize but this is fine)
    for i in range(num_chains):
        
        # Tell me which chain and the hyperparameters
        print(f"Chain {i+1}/{num_chains} with L {L:.3f}, and step size {step_size:.5f}")
        
        # The ith seed
        chain_key = jax.random.fold_in(base_key, i)
        
        # Use the BayesOpt-ed hyperparameters
        samples, ess, acc, _, chain_time = run_mams_fixed(
            logdensity_fn = logdensity_fn,
            chain_length = num_steps,
            initial_position = initial_position,
            key = chain_key,
            L = L,
            step_size = step_size
        )
        
        # Store the resulst and samples
        all_samples.append(samples)
        all_ess.append(ess)
        all_acceptance.append(acc)
        
        print(f"Chains {i+1} has ESS {ess:.1f} and acceptance rate is {acc:.3f}")
    
    # Keep all the samples
    all_samples = jnp.stack(all_samples, axis=0)
    
    # The time to do all this
    total_time = time.time() - start_time
    
    return (all_samples, jnp.array(all_ess), jnp.array(all_acceptance), total_time)




############################################### Plot the iterations of BayesOpt

def plot_bayesopt_progress(results_dict, save_plot_name = None):
    
    # Check for available algorithm results to plot
    algorithms = []
    for name in ['NUTS', 'MCLMC', 'MAMS']:
        key = name.lower()
        if key in results_dict and results_dict[key] is not None:
            algorithms.append((name, key))
    if not algorithms:
        print("No results to plot")
        return
    
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize = (14, 10))
    
    # Plot each algorithm's progress as BayesOpt goes on
    for alg_name, alg_key in algorithms:
        results = results_dict[alg_key]
        iterations = results['iteration']
        
        # Plot #1: ESS
        axes[0, 0].plot(iterations, results['ess'], 'o-', label = alg_name, markersize = 4)
        
        # Plot #2: acceptance rate
        axes[0, 1].plot(iterations, results['acceptance_rate'], 'o-', label = alg_name, markersize = 4)
        
        # Plot #3: number of leapfrog steps
        axes[1, 0].plot(iterations, results['integration_steps'], 'o-', label = alg_name, markersize = 4)
        
        # Plot #4: value of objective function
        axes[1, 1].plot(iterations, results['objective'], 'o-', label = alg_name, markersize = 4)
    
    # Plot 1: ESS
    axes[0, 0].set_ylabel('ESS')
    axes[0, 0].set_title('Effective Sample Size')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Acceptance
    axes[0, 1].set_ylabel('Acceptance Rate')
    axes[0, 1].set_title('Acceptance Rate')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Integration steps
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Leapfrog Steps')
    axes[1, 0].set_title('Leapfrog Steps per Iteration')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Objective
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Objective Value')
    axes[1, 1].set_title('Objective Function')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot if name is given
    if save_plot_name:
        filename = f'{save_plot_name}.png'
        plt.savefig(filename, dpi = 150, bbox_inches = 'tight')
        print(f"Saved {filename}")
    
    plt.show()



############################################### Plot the hyperparameter space (L, epsilon) that BayesOpt explores and exploits coloured by ESS

def plot_hyperparameter_space(results_dict, save_plot_name = None):
    
    # Only MCLMC and MAMS have 2 dimensional hyperparameter space
    algorithms = []
    for name in ['MCLMC', 'MAMS']:
        key = name.lower()
        if key in results_dict and results_dict[key] is not None:
            algorithms.append((name, key))
    
    if not algorithms:
        print("No MCLMC/MAMS results!")
        return
    
    # Subplots
    fig, axes = plt.subplots(1, len(algorithms), figsize=(9, 7))
    if len(algorithms) == 1:
        axes = [axes]
    
    # Plot for each algorithm
    for ax, (alg_name, alg_key) in zip(axes, algorithms):
        results = results_dict[alg_key]
        
        L_values = np.array([p['L'] for p in results['hyperparams']])
        step_sizes = np.array([p['step_size'] for p in results['hyperparams']])
        ess_values = np.array(results['ess'])
        
        # Scatter plot coloured by ESS
        scatter = ax.scatter(step_sizes, L_values, c = ess_values, s = 100, cmap = 'viridis', alpha = 0.7, edgecolors = 'black', linewidths = 1)
        
        # Mess around with plot axes
        ax.set_xlabel('Step Size', fontsize=14)
        ax.set_ylabel('Trajectory Length', fontsize=14)
        #ax.set_xscale('log')
        #ax.set_yscale('log')
        ax.set_title(f'{alg_name}: Hyperparameter Space', fontsize = 16)
        ax.grid(True, alpha=0.3, which = 'both')
        
        # Add colour legend for ESS
        cbar = plt.colorbar(scatter, ax = ax, label = 'ESS')
        
        # Show starting point
        ax.annotate(
            'Start', 
            xy=(step_sizes[0], L_values[0]), xytext=(10, 10), textcoords='offset points', fontsize=10, color='blue', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7), arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
        
        # Annotate best point
        best_idx = np.argmax(results['objective'])
        ax.annotate('Best', xy=(step_sizes[best_idx], L_values[best_idx]), xytext=(10, -20), 
            textcoords='offset points', fontsize=10, color='red', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5)
        )
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot if name given
    if save_plot_name:
        filename = f'{save_plot_name}_hyperparameter_space.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved {filename}")
    
    plt.show()




####################################### Do experiments


# BayesOpt MAMS, NUTS, and MCLMC
def compare_algorithms(dim = 5, iterations = 20, chain_length = 1000):
    
    # Create Funnel target with dimension dim
    # NEED TO MAKE OTHER TARGETS
    logdensity_fn = make_funnel_logdensity(dim)
    
    # Start at origin
    initial_position = jnp.zeros(dim)
    
    # Tune NUTS
    print("########################################### BayesOpt-ing NUTS")
    nuts_results = run_bayesopt_tuning(
        logdensity_fn = logdensity_fn, 
        initial_position = initial_position, 
        fixed_key = jax.random.key(SEED_NUTS_TUNING), 
        algorithm_name = 'NUTS',
        num_iterations = iterations, 
        chain_length = chain_length
    )
    
    # Tune MCLMC
    print("########################################### BayesOpt-ing MCLMC")
    mclmc_results = run_bayesopt_tuning(
        logdensity_fn = logdensity_fn, 
        initial_position = initial_position, 
        fixed_key = jax.random.key(SEED_MCLMC_TUNING), 
        algorithm_name = 'MCLMC',
        num_iterations = iterations, 
        chain_length = chain_length
    )
    
    # Tune MAMS
    print("########################################## BayesOpt-ing MAMS")
    mams_results = run_bayesopt_tuning(
        logdensity_fn = logdensity_fn, 
        initial_position = initial_position, 
        fixed_key = jax.random.key(SEED_MAMS_TUNING), 
        algorithm_name = 'MAMS',
        num_iterations = iterations, 
        chain_length = chain_length
    )
    
    
    # Plot the results
    results_dict = {
        'nuts': nuts_results,
        'mclmc': mclmc_results,
        'mams': mams_results
    }
    plot_bayesopt_progress(results_dict, save_plot_name = 'BayesOpt_sequence')
    plot_hyperparameter_space(results_dict, save_plot_name ='BayesOpt_hyperparameter_space')
    
    
    # Print results
    print("Best hyperparameters for each of NUTS, MCLMC, and MAMS")    
    for alg_name, alg_results in [('NUTS', nuts_results), ('MCLMC', mclmc_results), ('MAMS', mams_results)]:
        
        # Which iteration
        best_idx = jnp.argmax(jnp.array(alg_results['objective']))
        
        # Stats for that iteration
        ess = alg_results['ess'][best_idx]
        acc = alg_results['acceptance_rate'][best_idx]
        steps = alg_results['integration_steps'][best_idx]
        
        # Print stats for the best hyperparameter settings
        print("Algorithm       ESS        Acceptance Rate          Steps = ceilling(L/epsilon)")
        print(f"{alg_name:<12} {ess:<12.1f} {acc:<15.3f} {steps:<15.1f}")
    
    return nuts_results, mclmc_results, mams_results


# Compare Robnik et al auto tuning to BayesOpt parameters
def compare_mams_tuning_methods(dim = 5, num_chains = 4, num_steps = 1000):
    
    # Make funnel or another target density
    logdensity_fn = make_funnel_logdensity(dim)
    
    # Start at the origin
    initial_position = jnp.zeros(dim)
    
    
    print("BayesOpt for best step size and L")    
    bayesopt_tuning_key = jax.random.key(SEED_MAMS_TUNING)
    mams_results = run_bayesopt_tuning(
        logdensity_fn = logdensity_fn, 
        initial_position = initial_position, 
        fixed_key = bayesopt_tuning_key, 
        algorithm_name = 'MAMS',
        num_iterations = 20, ############# Make this bigger to try more hyperparameter pairs
        chain_length = 1000
    )
    
    # Get the best settings
    best_idx = jnp.argmax(jnp.array(mams_results['objective']))
    best_params = mams_results['hyperparams'][best_idx]
    best_L = best_params['L']
    best_step_size = best_params['step_size']
    print(f"Best hyperparametrs: L = {best_L:.5f}, epsilon = {best_step_size:.5f}")
    
    
    # Do a few chains with those hyperparameters
    print("Testing BayesOpt hyperparameters on a few chains")
    
    validation_key = jax.random.key(SEED_BAYESOPT_VALIDATION)
    (
        bayesopt_samples, 
        bayesopt_ess, 
        bayesopt_acc, 
        bayesopt_time
    ) = run_mams_multiple_chains_fixed(
        logdensity_fn = logdensity_fn, 
        num_chains = num_chains, 
        num_steps = num_steps, 
        initial_position = initial_position, 
        base_key = validation_key, ############### Inside run_mams_multiple_chains_fixed it splits the key, make sure that gives different chians
        L = best_L, 
        step_size = best_step_size
    )
    
    # Compute R hat to check convergence
    bayesopt_rhat = compute_rhat(bayesopt_samples)
    
    # Print BayesOpt results
    print(f"BayesOpt Results:")
    print(f"Mean ESS: {jnp.mean(bayesopt_ess):.1f} ± {jnp.std(bayesopt_ess):.1f}")
    print(f"Mean Acceptance: {jnp.mean(bayesopt_acc):.3f} ± {jnp.std(bayesopt_acc):.3f}")
    print(f"Max R-hat: {jnp.max(bayesopt_rhat):.4f}")
    
    
    
    
    # Run auto-tune
    print("Using Robnik et al auto tuning where each chain does its own auto tuning")
    
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
        L = 1.0,  # Ignored by auto-tuning but it seems to crash if nothing is given
        step_size = 0.1  # Same as L
    )
    
    # Compute R hat
    auto_rhat = compute_rhat(auto_samples)
    
    # Print auto-tuning results
    print(f"Auto-tuning Results:")
    print(f"Mean L: {jnp.mean(auto_L):.4f} ± {jnp.std(auto_L):.4f}")
    print(f"Mean step size: {jnp.mean(auto_step_sizes):.6f} ± {jnp.std(auto_step_sizes):.6f}")
    print(f"Mean ESS: {jnp.mean(auto_ess):.1f} ± {jnp.std(auto_ess):.1f}")
    print(f"Mean Acceptance: {jnp.mean(auto_acc):.3f} ± {jnp.std(auto_acc):.3f}")
    print(f"Max R-hat: {jnp.max(auto_rhat):.4f}")
    
    
    return bayesopt_samples, auto_samples, mams_results





def run_all_experiments(dim=5, num_chains=4, num_steps=1000):
    
    print(f"Number of dimensions of target density: {dim}")
    print(f"Number of chains for validation: {num_chains}")
    print(f"Number of sampels per chain: {num_steps}")
    
    # Start timing
    total_start = time.time()
    
    # Compare NUTS, MCLMC, and MAMS
    nuts_results, mclmc_results, mams_results = compare_algorithms(dim=dim)
    
    # Compare MAMS tuning
    bayesopt_samples, auto_samples, _ = compare_mams_tuning_methods(
        dim=dim, 
        num_chains=num_chains, 
        num_steps=num_steps
    )
    
    # Total time for both experiments
    total_time = time.time() - total_start
    print(f"Total time: {total_time:.2f}s ({total_time/60:.1f} min)")
    
    # Return all results
    return {
        'nuts': nuts_results,
        'mclmc': mclmc_results,
        'mams': mams_results,
        'bayesopt_samples': bayesopt_samples,
        'auto_samples': auto_samples
    }




# results = run_all_experiments(dim=5, num_chains=4, num_steps=1000)


nuts_results, mclmc_results, mams_results = compare_algorithms(dim=5)


bayesopt_samples, auto_samples, _ = compare_mams_tuning_methods(dim=5, num_chains = 4, num_steps = 1000)
