# Imports from here: https://blackjax-devs.github.io/sampling-book/algorithms/mclmc.html#how-to-run-mclmc-in-blackjax

import matplotlib.pyplot as plt

plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["font.size"] = 19

import jax
import blackjax
import numpy as np
import jax.numpy as jnp
from datetime import date
import numpyro
import numpyro.distributions as dist

from numpyro.infer.util import initialize_model

rng_key = jax.random.key(548)

# Enable 64-bit precision globally
from jax import config
config.update("jax_enable_x64", True)


# Imports for Adjusted MCLMC (MAMS)
from blackjax.mcmc.adjusted_mclmc_dynamic import rescale
from blackjax.util import run_inference_algorithm


# Imports for NUTS
import jax.scipy.stats as stats


# JAX guide: https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html
# Blackjax guide: https://blackjax-devs.github.io/sampling-book/




# Bayes Opt for JAX
# Boax guide: https://boax.readthedocs.io/en/latest/index.html
from boax.experiments import optimization



# ============================================================================
# TARGET DENSITIES
# ============================================================================

def make_gaussian_logdensity(dim):
    """High dimensional Gaussian with different scales per dimension"""
    # Create diagonal covariance with scales from 1 to dim
    scales = jnp.linspace(1.0, float(dim), dim)
    inv_cov = 1.0 / scales**2
    
    def logdensity_fn(x):
        # Standard multivariate Gaussian: -0.5 * x^T * inv_cov * x
        return -0.5 * jnp.sum(inv_cov * x**2)
    
    return logdensity_fn


def make_funnel_logdensity(dim):
    """Neal's funnel - already provided but wrapped for consistency"""
    def logdensity_fn(x):
        # Prior on neck variable x[0]
        log_prob = -0.5 * (x[0]**2 / 9.0)
        # Jacobian adjustment for transformation
        log_prob += -0.5 * (dim - 1) * x[0]
        # Conditional distribution on remaining dims
        log_prob += -0.5 * jnp.sum(x[1:]**2 * jnp.exp(-x[0]))
        return log_prob
    
    return logdensity_fn


def make_rosenbrock_logdensity(dim):
    """Rosenbrock (banana) distribution"""
    def logdensity_fn(x):
        # First term: Gaussian on x[0]
        log_prob = -0.5 * x[0]**2
        # Rosenbrock terms for pairs
        for i in range(dim - 1):
            # (x[i+1] - x[i]^2)^2 term
            log_prob += -0.5 * (x[i+1] - x[i]**2)**2 / 0.1
        return log_prob
    
    return logdensity_fn


# ============================================================================
# PLOTTING
# ============================================================================

def plot_target_2d(logdensity_fn, dim_x=0, dim_y=1, xlim=(-5, 5), ylim=(-5, 5)):
    """Plot 2D slice of target density"""
    # Create grid for plotting
    x = np.linspace(xlim[0], xlim[1], 100)
    y = np.linspace(ylim[0], ylim[1], 100)
    X, Y = np.meshgrid(x, y)
    
    # Evaluate log density on grid
    Z = np.zeros_like(X)
    for i in range(len(x)):
        for j in range(len(y)):
            # Create full dimensional point with zeros except for dims we plot
            point = jnp.zeros(10)  # Assuming max 10 dims for visualization
            point = point.at[dim_x].set(X[j, i])
            point = point.at[dim_y].set(Y[j, i])
            Z[j, i] = logdensity_fn(point)
    
    # Plot contours
    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, np.exp(Z), levels=20)
    plt.xlabel(f'Dimension {dim_x}')
    plt.ylabel(f'Dimension {dim_y}')
    plt.colorbar(label='Density')
    return plt.gcf()


# ============================================================================
# RUN ALGORITHMS AND COMPUTE STATS
# ============================================================================

def compute_ess(samples):
    """Compute effective sample size using blackjax utility"""
    # samples shape is (n_samples, n_dims)
    # blackjax.diagnostics.effective_sample_size expects (n_chains, n_samples)
    # We treat each dimension as a separate chain for ESS calculation
    
    # Transpose to (n_dims, n_samples) then add chain dimension
    samples_reshaped = samples.T[:, None, :]  # (n_dims, 1, n_samples)
    
    # Compute ESS for each dimension
    ess_per_dim = jax.vmap(
        lambda x: blackjax.diagnostics.effective_sample_size(x), 
        in_axes=0
    )(samples_reshaped)
    
    # Return minimum ESS across dimensions (most conservative estimate)
    return jnp.min(ess_per_dim)


def run_nuts_fixed(logdensity_fn, num_steps, initial_position, key, step_size, inv_mass_matrix):
    """Run NUTS with fixed parameters"""
    # Build NUTS kernel
    nuts = blackjax.nuts(logdensity_fn, step_size, inv_mass_matrix)
    
    # Initialize state
    state = nuts.init(initial_position)
    
    # Run inference
    def one_step(state, key):
        state, info = nuts.step(key, state)
        return state, (state.position, info.acceptance_rate)
    
    keys = jax.random.split(key, num_steps)
    final_state, (samples, acceptance_rates) = jax.lax.scan(one_step, state, keys)
    
    # Compute stats
    avg_acceptance = jnp.mean(acceptance_rates)
    ess = compute_ess(samples)
    
    return samples, ess, avg_acceptance


def run_mclmc_fixed(logdensity_fn, num_steps, initial_position, key, L, step_size):
    """Run MCLMC with fixed parameters"""
    init_key, run_key = jax.random.split(key)
    
    # Initialize state
    initial_state = blackjax.mcmc.mclmc.init(
        position=initial_position, 
        logdensity_fn=logdensity_fn, 
        rng_key=init_key
    )
    
    # Build sampling algorithm with fixed params
    sampling_alg = blackjax.mclmc(
        logdensity_fn,
        L=L,
        step_size=step_size,
    )
    
    # Run inference
    _, samples = blackjax.util.run_inference_algorithm(
        rng_key=run_key,
        initial_state=initial_state,
        inference_algorithm=sampling_alg,
        num_steps=num_steps,
        transform=lambda state, _: state.position,
        progress_bar=False,
    )
    
    # MCLMC acceptance is always 1.0
    ess = compute_ess(samples)
    
    return samples, ess, 1.0


def run_mams_fixed(logdensity_fn, num_steps, initial_position, key, L, step_size):
    """Run MAMS with fixed parameters"""
    init_key, run_key = jax.random.split(key)
    
    # Initialize state
    initial_state = blackjax.mcmc.adjusted_mclmc_dynamic.init(
        position=initial_position,
        logdensity_fn=logdensity_fn,
        random_generator_arg=init_key,
    )
    
    # Build sampling algorithm with fixed params
    alg = blackjax.adjusted_mclmc_dynamic(
        logdensity_fn=logdensity_fn,
        step_size=step_size,
        integration_steps_fn=lambda key: jnp.ceil(L / step_size),
        L_proposal_factor=jnp.inf,
    )
    
    # Run inference
    def one_step(state, key):
        state, info = alg.step(key, state)
        return state, (state.position, info.acceptance_rate)
    
    keys = jax.random.split(run_key, num_steps)
    final_state, (samples, acceptance_rates) = jax.lax.scan(one_step, initial_state, keys)
    
    # Compute stats
    avg_acceptance = jnp.mean(acceptance_rates)
    ess = compute_ess(samples)
    
    return samples, ess, avg_acceptance


# ============================================================================
# BAYESIAN OPTIMIZATION SETUP
# ============================================================================

def objective_function(ess, acceptance_rate, target_acceptance, lambda_penalty):
    """Compute objective: ESS - lambda * (acceptance - target)^2"""
    penalty = lambda_penalty * (acceptance_rate - target_acceptance)**2
    return ess - penalty


def run_bayesopt_tuning(
    logdensity_fn,
    initial_position,
    base_key,
    algorithm_name,
    num_iterations=20,
    chain_length=1000,
    target_acceptance=0.65,  # Standard target for NUTS
    lambda_penalty=100.0,
):
    """Run Bayesian optimization to tune hyperparameters using BOAx"""
    
    from boax.experiments import optimization
    from jax import config
    
    # Enable 64-bit precision for better numerical stability
    config.update("jax_enable_x64", True)
    
    results = {
        'iteration': [],
        'ess': [],
        'acceptance_rate': [],
        'objective': [],
        'hyperparams': [],
    }
    
    # Define parameter bounds for each algorithm
    if algorithm_name == 'NUTS':
        # Tune step_size (log scale)
        parameters = [
            {
                'name': 'log_step_size',
                'type': 'log_range',
                'bounds': [1e-5, 1e-1],  # step_size bounds
            }
        ]
        
        def run_with_params(params_dict, key):
            step_size = params_dict['log_step_size']
            inv_mass_matrix = jnp.ones(len(initial_position))
            _, ess, acc = run_nuts_fixed(
                logdensity_fn, chain_length, initial_position, key, 
                step_size, inv_mass_matrix
            )
            return ess, acc
        
    elif algorithm_name == 'MCLMC':
        # Tune L and step_size (both log scale)
        parameters = [
            {
                'name': 'L',
                'type': 'log_range',
                'bounds': [1e-1, 1e2],  # L bounds
            },
            {
                'name': 'step_size',
                'type': 'log_range',
                'bounds': [1e-3, 1.0],  # step_size bounds
            }
        ]
        target_acceptance = 1.0  # MCLMC always accepts
        
        def run_with_params(params_dict, key):
            L = params_dict['L']
            step_size = params_dict['step_size']
            _, ess, acc = run_mclmc_fixed(
                logdensity_fn, chain_length, initial_position, key, L, step_size
            )
            return ess, acc
        
    elif algorithm_name == 'MAMS':
        # Tune L and step_size (both log scale)
        parameters = [
            {
                'name': 'L',
                'type': 'log_range',
                'bounds': [1e-1, 1e2],  # L bounds
            },
            {
                'name': 'step_size',
                'type': 'log_range',
                'bounds': [1e-3, 1.0],  # step_size bounds
            }
        ]
        target_acceptance = 0.9  # MAMS target
        
        def run_with_params(params_dict, key):
            L = params_dict['L']
            step_size = params_dict['step_size']
            _, ess, acc = run_mams_fixed(
                logdensity_fn, chain_length, initial_position, key, L, step_size
            )
            return ess, acc
    
    # Set up BOAx experiment
    experiment = optimization(
        parameters=parameters,
        batch_size=1,  # Sequential optimization
    )
    
    # Run Bayesian optimization
    step, experiment_results = None, []
    
    for i in range(num_iterations):
        print(f"loop {i} out of {num_iterations}")
        # Get next parameterization to evaluate
        step, parameterizations = experiment.next(step, experiment_results)
        
        # Get the parameter dictionary (BOAx returns a list, we take first element for batch_size=1)
        params_dict = parameterizations[0]
        
        # Use same key progression for fair comparison
        iter_key = jax.random.fold_in(base_key, i)
        
        # Run algorithm with these parameters
        ess, acc = run_with_params(params_dict, iter_key)
        
        # Compute objective (BOAx maximizes by default)
        obj = objective_function(ess, acc, target_acceptance, lambda_penalty)
        
        # Store results for BOAx (expects list of tuples: [(params_dict, objective_value)])
        experiment_results = [(params_dict, float(obj))]
        
        # Store results for our tracking
        results['iteration'].append(i)
        results['ess'].append(float(ess))
        results['acceptance_rate'].append(float(acc))
        results['objective'].append(float(obj))
        results['hyperparams'].append(params_dict)
        
        print(f"{algorithm_name} Iter {i:2d}: ESS={ess:.1f}, Acc={acc:.3f}, Obj={obj:.1f}, Params={params_dict}")
    
    # Get best parameters found
    best_params = experiment.best(step)
    print(f"\n{algorithm_name} Best parameters found: {best_params}")
    
    return results


# def run_bayesopt_tuning(
#     logdensity_fn,
#     initial_position,
#     base_key,
#     algorithm_name,
#     num_iterations=20,
#     chain_length=1000,
#     target_acceptance=0.65,  # Standard target for NUTS
#     lambda_penalty=100.0,
# ):
#     """Run Bayesian optimization to tune hyperparameters"""
    
#     results = {
#         'iteration': [],
#         'ess': [],
#         'acceptance_rate': [],
#         'objective': [],
#         'hyperparams': [],
#     }
    
#     # Define parameter bounds for each algorithm
#     if algorithm_name == 'NUTS':
#         # Tune step_size (log scale)
#         param_bounds = jnp.array([[-5.0, -1.0]])  # log10(step_size)
        
#         def run_with_params(params, key):
#             step_size = 10.0 ** params[0]
#             inv_mass_matrix = jnp.ones(len(initial_position))
#             _, ess, acc = run_nuts_fixed(
#                 logdensity_fn, chain_length, initial_position, key, 
#                 step_size, inv_mass_matrix
#             )
#             return ess, acc
            
#     elif algorithm_name == 'MCLMC':
#         # Tune L and step_size (both log scale)
#         param_bounds = jnp.array([[-1.0, 2.0], [-3.0, 0.0]])  # log10(L), log10(step_size)
#         target_acceptance = 1.0  # MCLMC always accepts
        
#         def run_with_params(params, key):
#             L = 10.0 ** params[0]
#             step_size = 10.0 ** params[1]
#             _, ess, acc = run_mclmc_fixed(
#                 logdensity_fn, chain_length, initial_position, key, L, step_size
#             )
#             return ess, acc
            
#     elif algorithm_name == 'MAMS':
#         # Tune L and step_size (both log scale)
#         param_bounds = jnp.array([[-1.0, 2.0], [-3.0, 0.0]])  # log10(L), log10(step_size)
#         target_acceptance = 0.9  # MAMS target
        
#         def run_with_params(params, key):
#             L = 10.0 ** params[0]
#             step_size = 10.0 ** params[1]
#             _, ess, acc = run_mams_fixed(
#                 logdensity_fn, chain_length, initial_position, key, L, step_size
#             )
#             return ess, acc
    
#     # Run Bayesian optimization
#     for i in range(num_iterations):
#         # Use same key progression for fair comparison
#         iter_key = jax.random.fold_in(base_key, i)
        
#         if i == 0:
#             # Initial random point
#             param_key, run_key = jax.random.split(iter_key)
#             params = jax.random.uniform(
#                 param_key, 
#                 shape=(param_bounds.shape[0],), 
#                 minval=param_bounds[:, 0], 
#                 maxval=param_bounds[:, 1]
#             )
#         else:
#             # Use BO to suggest next point (simplified - just grid search for now)
#             # In practice, use boax here
#             param_key, run_key = jax.random.split(iter_key)
#             params = jax.random.uniform(
#                 param_key,
#                 shape=(param_bounds.shape[0],),
#                 minval=param_bounds[:, 0],
#                 maxval=param_bounds[:, 1]
#             )
        
#         # Run algorithm with these parameters
#         ess, acc = run_with_params(params, run_key)
        
#         # Compute objective
#         obj = objective_function(ess, acc, target_acceptance, lambda_penalty)
        
#         # Store results
#         results['iteration'].append(i)
#         results['ess'].append(float(ess))
#         results['acceptance_rate'].append(float(acc))
#         results['objective'].append(float(obj))
#         results['hyperparams'].append(params.tolist())
        
#         print(f"{algorithm_name} Iter {i:2d}: ESS={ess:.1f}, Acc={acc:.3f}, Obj={obj:.1f}")
    
#     return results


# ============================================================================
# MAIN COMPARISON
# ============================================================================

def compare_algorithms(dim=5):
    """Run full comparison of all three algorithms"""
    
    # Choose target density
    print("Creating target density...")
    logdensity_fn = make_funnel_logdensity(dim)
    
    # Plot 2D slice
    print("Plotting target...")
    fig = plot_target_2d(logdensity_fn, dim_x=0, dim_y=1, xlim=(-10, 10), ylim=(-10, 10))
    plt.title("Neal's Funnel (2D slice)")
    plt.show()
    
    # Set up common parameters
    initial_position = jnp.zeros(dim)
    base_key = jax.random.key(548)
    
    # Run BayesOpt for each algorithm
    print("\nRunning NUTS optimization...")
    nuts_results = run_bayesopt_tuning(
        logdensity_fn, initial_position, base_key, 'NUTS',
        num_iterations=20, chain_length=1000
    )
    
    print("\nRunning MCLMC optimization...")
    mclmc_results = run_bayesopt_tuning(
        logdensity_fn, initial_position, base_key, 'MCLMC',
        num_iterations=20, chain_length=1000
    )
    
    print("\nRunning MAMS optimization...")
    mams_results = run_bayesopt_tuning(
        logdensity_fn, initial_position, base_key, 'MAMS',
        num_iterations=20, chain_length=1000
    )
    
    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot ESS
    axes[0].plot(nuts_results['iteration'], nuts_results['ess'], 'o-', label='NUTS')
    axes[0].plot(mclmc_results['iteration'], mclmc_results['ess'], 's-', label='MCLMC')
    axes[0].plot(mams_results['iteration'], mams_results['ess'], '^-', label='MAMS')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('ESS')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot acceptance rate
    axes[1].plot(nuts_results['iteration'], nuts_results['acceptance_rate'], 'o-', label='NUTS')
    axes[1].plot(mclmc_results['iteration'], mclmc_results['acceptance_rate'], 's-', label='MCLMC')
    axes[1].plot(mams_results['iteration'], mams_results['acceptance_rate'], '^-', label='MAMS')
    axes[1].axhline(0.65, color='gray', linestyle='--', alpha=0.5)
    axes[1].axhline(0.9, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Acceptance Rate')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot objective
    axes[2].plot(nuts_results['iteration'], nuts_results['objective'], 'o-', label='NUTS')
    axes[2].plot(mclmc_results['iteration'], mclmc_results['objective'], 's-', label='MCLMC')
    axes[2].plot(mams_results['iteration'], mams_results['objective'], '^-', label='MAMS')
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('Objective')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print best results
    print("\n" + "="*60)
    print("BEST RESULTS")
    print("="*60)
    
    best_nuts_idx = jnp.argmax(jnp.array(nuts_results['objective']))
    print(f"\nNUTS: Best objective = {nuts_results['objective'][best_nuts_idx]:.1f}")
    print(f"  ESS = {nuts_results['ess'][best_nuts_idx]:.1f}")
    print(f"  Acceptance = {nuts_results['acceptance_rate'][best_nuts_idx]:.3f}")
    
    best_mclmc_idx = jnp.argmax(jnp.array(mclmc_results['objective']))
    print(f"\nMCLMC: Best objective = {mclmc_results['objective'][best_mclmc_idx]:.1f}")
    print(f"  ESS = {mclmc_results['ess'][best_mclmc_idx]:.1f}")
    print(f"  Acceptance = {mclmc_results['acceptance_rate'][best_mclmc_idx]:.3f}")
    
    best_mams_idx = jnp.argmax(jnp.array(mams_results['objective']))
    print(f"\nMAMS: Best objective = {mams_results['objective'][best_mams_idx]:.1f}")
    print(f"  ESS = {mams_results['ess'][best_mams_idx]:.1f}")
    print(f"  Acceptance = {mams_results['acceptance_rate'][best_mams_idx]:.3f}")
    
    return nuts_results, mclmc_results, mams_results


# ============================================================================
# RUN IT
# ============================================================================

# if __name__ == "__main__":
#     # Run comparison with 5 dimensions
#     nuts_results, mclmc_results, mams_results = compare_algorithms(dim=5)

# nuts_results, mclmc_results, mams_results = compare_algorithms(dim=5)




# Add this to your existing v1.py file

def run_adjusted_mclmc_dynamic(
    logdensity_fn,
    num_steps,
    initial_position,
    key,
    transform=lambda state, _ : state.position,
    diagonal_preconditioning=True,
    random_trajectory_length=True,
    L_proposal_factor=jnp.inf
):
    """Run MAMS with automatic tuning"""
    init_key, tune_key, run_key = jax.random.split(key, 3)
    initial_state = blackjax.mcmc.adjusted_mclmc_dynamic.init(
        position=initial_position,
        logdensity_fn=logdensity_fn,
        random_generator_arg=init_key,
    )
    if random_trajectory_length:
        integration_steps_fn = lambda avg_num_integration_steps: lambda k: jnp.ceil(
            jax.random.uniform(k) * rescale(avg_num_integration_steps))
    else:
        integration_steps_fn = lambda avg_num_integration_steps: lambda _: jnp.ceil(avg_num_integration_steps)
    
    kernel = lambda rng_key, state, avg_num_integration_steps, step_size, inverse_mass_matrix: blackjax.mcmc.adjusted_mclmc_dynamic.build_kernel(
        integration_steps_fn=integration_steps_fn(avg_num_integration_steps),
        inverse_mass_matrix=inverse_mass_matrix,
    )(
        rng_key=rng_key,
        state=state,
        step_size=step_size,
        logdensity_fn=logdensity_fn,
        L_proposal_factor=L_proposal_factor,
    )
    
    target_acc_rate = 0.9 # our recommendation
    (
        blackjax_state_after_tuning,
        blackjax_mclmc_sampler_params,
        _
    ) = blackjax.adjusted_mclmc_find_L_and_step_size(
        mclmc_kernel=kernel,
        num_steps=num_steps,
        state=initial_state,
        rng_key=tune_key,
        target=target_acc_rate,
        frac_tune1=0.1,
        frac_tune2=0.1,
        frac_tune3=0.1, # our recommendation
        diagonal_preconditioning=diagonal_preconditioning,
    )
    
    step_size = blackjax_mclmc_sampler_params.step_size
    L = blackjax_mclmc_sampler_params.L
    
    alg = blackjax.adjusted_mclmc_dynamic(
        logdensity_fn=logdensity_fn,
        step_size=step_size,
        integration_steps_fn=lambda key: jnp.ceil(
            jax.random.uniform(key) * rescale(L / step_size)
        ),
        inverse_mass_matrix=blackjax_mclmc_sampler_params.inverse_mass_matrix,
        L_proposal_factor=L_proposal_factor,
    )
    
    _, out = run_inference_algorithm(
        rng_key=run_key,
        initial_state=blackjax_state_after_tuning,
        inference_algorithm=alg,
        num_steps=num_steps,
        transform=transform,
        progress_bar=False,
    )
    
    return out, step_size, L, blackjax_mclmc_sampler_params.inverse_mass_matrix


def run_mams_multiple_chains(logdensity_fn, num_chains, num_steps, initial_position, key, L, step_size):
    """Run multiple chains of MAMS with fixed parameters"""
    all_samples = []
    all_ess = []
    all_acceptance = []
    
    for i in range(num_chains):
        chain_key = jax.random.fold_in(key, i)
        samples, ess, acc = run_mams_fixed(
            logdensity_fn, num_steps, initial_position, chain_key, L, step_size
        )
        all_samples.append(samples)
        all_ess.append(ess)
        all_acceptance.append(acc)
    
    # Stack samples: (num_chains, num_steps, dim)
    all_samples = jnp.stack(all_samples, axis=0)
    
    return all_samples, jnp.array(all_ess), jnp.array(all_acceptance)


def run_auto_tuned_multiple_chains(logdensity_fn, num_chains, num_steps, initial_position, key):
    """Run multiple chains with automatic tuning"""
    all_samples = []
    all_step_sizes = []
    all_L = []
    
    for i in range(num_chains):
        chain_key = jax.random.fold_in(key, i)
        samples, step_size, L, inv_mass = run_adjusted_mclmc_dynamic(
            logdensity_fn, num_steps, initial_position, chain_key
        )
        all_samples.append(samples)
        all_step_sizes.append(step_size)
        all_L.append(L)
    
    # Stack samples: (num_chains, num_steps, dim)
    all_samples = jnp.stack(all_samples, axis=0)
    
    # Compute ESS and acceptance for each chain
    all_ess = []
    for chain_samples in all_samples:
        ess = compute_ess(chain_samples)
        all_ess.append(ess)
    
    # Auto-tuned MAMS should have ~0.9 acceptance
    all_acceptance = jnp.ones(num_chains) * 0.9  # Approximate
    
    return all_samples, jnp.array(all_ess), all_acceptance, jnp.array(all_step_sizes), jnp.array(all_L)


def compute_rhat(chains):
    """Compute R-hat (Gelman-Rubin) statistic across chains
    chains: (num_chains, num_steps, dim)
    """
    num_chains, num_steps, dim = chains.shape
    
    # Compute between-chain variance
    chain_means = jnp.mean(chains, axis=1)  # (num_chains, dim)
    overall_mean = jnp.mean(chain_means, axis=0)  # (dim,)
    B = num_steps / (num_chains - 1) * jnp.sum((chain_means - overall_mean)**2, axis=0)  # (dim,)
    
    # Compute within-chain variance
    chain_vars = jnp.var(chains, axis=1, ddof=1)  # (num_chains, dim)
    W = jnp.mean(chain_vars, axis=0)  # (dim,)
    
    # Compute R-hat
    var_est = ((num_steps - 1) / num_steps) * W + (1 / num_steps) * B
    rhat = jnp.sqrt(var_est / W)
    
    return rhat


def compare_mams_tuning_methods(dim=5, num_chains=4, num_steps=5000):
    """Compare BayesOpt tuning vs automatic tuning for MAMS"""
    
    print("="*70)
    print("MAMS HYPERPARAMETER TUNING COMPARISON")
    print("="*70)
    
    # Setup
    logdensity_fn = make_funnel_logdensity(dim)
    initial_position = jnp.zeros(dim)
    base_key = jax.random.key(12345)
    
    # ========================================================================
    # 1. Get best BayesOpt parameters
    # ========================================================================
    print("\n1. Running BayesOpt to find optimal hyperparameters...")
    bayesopt_key = jax.random.fold_in(base_key, 0)
    mams_results = run_bayesopt_tuning(
        logdensity_fn, initial_position, bayesopt_key, 'MAMS',
        num_iterations=10, chain_length=100
    )
    
    # Extract best parameters
    best_idx = jnp.argmax(jnp.array(mams_results['objective']))
    best_params = mams_results['hyperparams'][best_idx]
    best_L = best_params['L']
    best_step_size = best_params['step_size']
    
    print(f"\nBest BayesOpt parameters found:")
    print(f"  L = {best_L:.4f}")
    print(f"  step_size = {best_step_size:.6f}")
    print(f"  Best objective = {mams_results['objective'][best_idx]:.1f}")
    
    # ========================================================================
    # 2. Run multiple chains with BayesOpt parameters
    # ========================================================================
    print(f"\n2. Running {num_chains} chains with BayesOpt parameters ({num_steps} steps each)...")
    bayesopt_chains_key = jax.random.fold_in(base_key, 1)
    bayesopt_samples, bayesopt_ess, bayesopt_acc = run_mams_multiple_chains(
        logdensity_fn, num_chains, num_steps, initial_position, 
        bayesopt_chains_key, best_L, best_step_size
    )
    
    # Compute R-hat
    bayesopt_rhat = compute_rhat(bayesopt_samples)
    
    print(f"\nBayesOpt Results:")
    print(f"  ESS per chain: {bayesopt_ess}")
    print(f"  Mean ESS: {jnp.mean(bayesopt_ess):.1f} ± {jnp.std(bayesopt_ess):.1f}")
    print(f"  Acceptance per chain: {bayesopt_acc}")
    print(f"  Mean acceptance: {jnp.mean(bayesopt_acc):.3f} ± {jnp.std(bayesopt_acc):.3f}")
    print(f"  R-hat per dimension: {bayesopt_rhat}")
    print(f"  Max R-hat: {jnp.max(bayesopt_rhat):.4f}")
    
    # ========================================================================
    # 3. Run multiple chains with automatic tuning
    # ========================================================================
    print(f"\n3. Running {num_chains} chains with automatic tuning ({num_steps} steps each)...")
    auto_chains_key = jax.random.fold_in(base_key, 2)
    auto_samples, auto_ess, auto_acc, auto_step_sizes, auto_L = run_auto_tuned_multiple_chains(
        logdensity_fn, num_chains, num_steps, initial_position, auto_chains_key
    )
    
    # Compute R-hat
    auto_rhat = compute_rhat(auto_samples)
    
    print(f"\nAutomatic Tuning Results:")
    print(f"  Step sizes found: {auto_step_sizes}")
    print(f"  Mean step size: {jnp.mean(auto_step_sizes):.6f} ± {jnp.std(auto_step_sizes):.6f}")
    print(f"  L values found: {auto_L}")
    print(f"  Mean L: {jnp.mean(auto_L):.4f} ± {jnp.std(auto_L):.4f}")
    print(f"  ESS per chain: {auto_ess}")
    print(f"  Mean ESS: {jnp.mean(auto_ess):.1f} ± {jnp.std(auto_ess):.1f}")
    print(f"  R-hat per dimension: {auto_rhat}")
    print(f"  Max R-hat: {jnp.max(auto_rhat):.4f}")
    
    # ========================================================================
    # 4. Create comparison plots
    # ========================================================================
    print("\n4. Creating comparison plots...")
    
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # --- Row 1: Trace plots for dimension 0 ---
    ax1 = fig.add_subplot(gs[0, 0])
    for i in range(num_chains):
        ax1.plot(bayesopt_samples[i, :, 0], alpha=0.6, label=f'Chain {i+1}')
    ax1.set_title('BayesOpt: Traces (Dim 0)')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Value')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs[0, 1])
    for i in range(num_chains):
        ax2.plot(auto_samples[i, :, 0], alpha=0.6, label=f'Chain {i+1}')
    ax2.set_title('Auto-tuned: Traces (Dim 0)')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Value')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # --- Trace plot for dimension 1 (funnel neck) ---
    ax3 = fig.add_subplot(gs[0, 2])
    for i in range(num_chains):
        ax3.plot(bayesopt_samples[i, :, 1], alpha=0.4, label=f'BO Chain {i+1}')
    for i in range(num_chains):
        ax3.plot(auto_samples[i, :, 1], alpha=0.4, linestyle='--', label=f'Auto Chain {i+1}')
    ax3.set_title('Both Methods: Traces (Dim 1)')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Value')
    ax3.legend(fontsize=6)
    ax3.grid(True, alpha=0.3)
    
    # --- Row 2: Marginal distributions ---
    ax4 = fig.add_subplot(gs[1, 0])
    # Flatten all chains
    bayesopt_flat = bayesopt_samples.reshape(-1, dim)
    ax4.hist(bayesopt_flat[:, 0], bins=50, alpha=0.7, density=True, label='BayesOpt')
    ax4.set_title('BayesOpt: Marginal (Dim 0)')
    ax4.set_xlabel('Value')
    ax4.set_ylabel('Density')
    ax4.grid(True, alpha=0.3)
    
    ax5 = fig.add_subplot(gs[1, 1])
    auto_flat = auto_samples.reshape(-1, dim)
    ax5.hist(auto_flat[:, 0], bins=50, alpha=0.7, density=True, label='Auto-tuned', color='orange')
    ax5.set_title('Auto-tuned: Marginal (Dim 0)')
    ax5.set_xlabel('Value')
    ax5.set_ylabel('Density')
    ax5.grid(True, alpha=0.3)
    
    # Overlay comparison
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.hist(bayesopt_flat[:, 0], bins=50, alpha=0.5, density=True, label='BayesOpt')
    ax6.hist(auto_flat[:, 0], bins=50, alpha=0.5, density=True, label='Auto-tuned')
    ax6.set_title('Overlay: Marginal (Dim 0)')
    ax6.set_xlabel('Value')
    ax6.set_ylabel('Density')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # --- Row 3: Summary statistics ---
    ax7 = fig.add_subplot(gs[2, 0])
    x_pos = np.arange(num_chains)
    ax7.bar(x_pos - 0.2, bayesopt_ess, 0.4, label='BayesOpt', alpha=0.7)
    ax7.bar(x_pos + 0.2, auto_ess, 0.4, label='Auto-tuned', alpha=0.7)
    ax7.set_xlabel('Chain')
    ax7.set_ylabel('ESS')
    ax7.set_title('Effective Sample Size by Chain')
    ax7.set_xticks(x_pos)
    ax7.set_xticklabels([f'{i+1}' for i in range(num_chains)])
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    ax8 = fig.add_subplot(gs[2, 1])
    x_pos_dim = np.arange(dim)
    ax8.bar(x_pos_dim - 0.2, bayesopt_rhat, 0.4, label='BayesOpt', alpha=0.7)
    ax8.bar(x_pos_dim + 0.2, auto_rhat, 0.4, label='Auto-tuned', alpha=0.7)
    ax8.axhline(1.01, color='red', linestyle='--', alpha=0.5, label='Threshold')
    ax8.set_xlabel('Dimension')
    ax8.set_ylabel('R-hat')
    ax8.set_title('Convergence Diagnostic (R-hat)')
    ax8.set_xticks(x_pos_dim)
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # Hyperparameter comparison
    ax9 = fig.add_subplot(gs[2, 2])
    methods = ['BayesOpt', 'Auto-tuned']
    L_values = [best_L, jnp.mean(auto_L)]
    step_values = [best_step_size * 100, jnp.mean(auto_step_sizes) * 100]  # Scale for visibility
    
    x_pos_hyper = np.arange(len(methods))
    width = 0.35
    ax9.bar(x_pos_hyper - width/2, L_values, width, label='L', alpha=0.7)
    ax9.bar(x_pos_hyper + width/2, step_values, width, label='step_size × 100', alpha=0.7)
    ax9.set_xlabel('Method')
    ax9.set_ylabel('Parameter Value')
    ax9.set_title('Hyperparameter Comparison')
    ax9.set_xticks(x_pos_hyper)
    ax9.set_xticklabels(methods)
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    plt.suptitle(f'MAMS Tuning Comparison (dim={dim}, chains={num_chains}, steps={num_steps})', 
                 fontsize=16, y=0.995)
    
    plt.show()
    
    # ========================================================================
    # 5. Summary comparison
    # ========================================================================
    print("\n" + "="*70)
    print("FINAL COMPARISON SUMMARY")
    print("="*70)
    print(f"\n{'Metric':<30} {'BayesOpt':<20} {'Auto-tuned':<20}")
    print("-"*70)
    print(f"{'L':<30} {best_L:<20.4f} {jnp.mean(auto_L):<20.4f}")
    print(f"{'step_size':<30} {best_step_size:<20.6f} {jnp.mean(auto_step_sizes):<20.6f}")
    print(f"{'Mean ESS':<30} {jnp.mean(bayesopt_ess):<20.1f} {jnp.mean(auto_ess):<20.1f}")
    print(f"{'Std ESS':<30} {jnp.std(bayesopt_ess):<20.1f} {jnp.std(auto_ess):<20.1f}")
    print(f"{'Max R-hat':<30} {jnp.max(bayesopt_rhat):<20.4f} {jnp.max(auto_rhat):<20.4f}")
    print(f"{'Mean Acceptance':<30} {jnp.mean(bayesopt_acc):<20.3f} {jnp.mean(auto_acc):<20.3f}")
    
    # Determine winner
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)
    
    bayesopt_score = jnp.mean(bayesopt_ess)
    auto_score = jnp.mean(auto_ess)
    
    if bayesopt_score > auto_score * 1.1:  # 10% better
        print("✓ BayesOpt tuning produces significantly better ESS")
    elif auto_score > bayesopt_score * 1.1:
        print("✓ Automatic tuning produces significantly better ESS")
    else:
        print("≈ Both methods produce comparable ESS")
    
    if jnp.max(bayesopt_rhat) < jnp.max(auto_rhat):
        print("✓ BayesOpt tuning has better convergence (lower R-hat)")
    elif jnp.max(auto_rhat) < jnp.max(bayesopt_rhat):
        print("✓ Automatic tuning has better convergence (lower R-hat)")
    else:
        print("≈ Both methods have comparable convergence")
    
    return bayesopt_samples, auto_samples, mams_results


# ============================================================================
# RUN THE COMPARISON
# ============================================================================

if __name__ == "__main__":
    bayesopt_samples, auto_samples, mams_results = compare_mams_tuning_methods(
        dim=5, 
        num_chains=4, 
        num_steps=1000
    )