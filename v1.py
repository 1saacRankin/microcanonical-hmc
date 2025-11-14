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
    """Run Bayesian optimization to tune hyperparameters"""
    
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
        param_bounds = jnp.array([[-5.0, -1.0]])  # log10(step_size)
        
        def run_with_params(params, key):
            step_size = 10.0 ** params[0]
            inv_mass_matrix = jnp.ones(len(initial_position))
            _, ess, acc = run_nuts_fixed(
                logdensity_fn, chain_length, initial_position, key, 
                step_size, inv_mass_matrix
            )
            return ess, acc
            
    elif algorithm_name == 'MCLMC':
        # Tune L and step_size (both log scale)
        param_bounds = jnp.array([[-1.0, 2.0], [-3.0, 0.0]])  # log10(L), log10(step_size)
        target_acceptance = 1.0  # MCLMC always accepts
        
        def run_with_params(params, key):
            L = 10.0 ** params[0]
            step_size = 10.0 ** params[1]
            _, ess, acc = run_mclmc_fixed(
                logdensity_fn, chain_length, initial_position, key, L, step_size
            )
            return ess, acc
            
    elif algorithm_name == 'MAMS':
        # Tune L and step_size (both log scale)
        param_bounds = jnp.array([[-1.0, 2.0], [-3.0, 0.0]])  # log10(L), log10(step_size)
        target_acceptance = 0.9  # MAMS target
        
        def run_with_params(params, key):
            L = 10.0 ** params[0]
            step_size = 10.0 ** params[1]
            _, ess, acc = run_mams_fixed(
                logdensity_fn, chain_length, initial_position, key, L, step_size
            )
            return ess, acc
    
    # Run Bayesian optimization
    for i in range(num_iterations):
        # Use same key progression for fair comparison
        iter_key = jax.random.fold_in(base_key, i)
        
        if i == 0:
            # Initial random point
            param_key, run_key = jax.random.split(iter_key)
            params = jax.random.uniform(
                param_key, 
                shape=(param_bounds.shape[0],), 
                minval=param_bounds[:, 0], 
                maxval=param_bounds[:, 1]
            )
        else:
            # Use BO to suggest next point (simplified - just grid search for now)
            # In practice, use boax here
            param_key, run_key = jax.random.split(iter_key)
            params = jax.random.uniform(
                param_key,
                shape=(param_bounds.shape[0],),
                minval=param_bounds[:, 0],
                maxval=param_bounds[:, 1]
            )
        
        # Run algorithm with these parameters
        ess, acc = run_with_params(params, run_key)
        
        # Compute objective
        obj = objective_function(ess, acc, target_acceptance, lambda_penalty)
        
        # Store results
        results['iteration'].append(i)
        results['ess'].append(float(ess))
        results['acceptance_rate'].append(float(acc))
        results['objective'].append(float(obj))
        results['hyperparams'].append(params.tolist())
        
        print(f"{algorithm_name} Iter {i:2d}: ESS={ess:.1f}, Acc={acc:.3f}, Obj={obj:.1f}")
    
    return results


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

nuts_results, mclmc_results, mams_results = compare_algorithms(dim=5)
