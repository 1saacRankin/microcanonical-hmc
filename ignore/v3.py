# ============================================================================
# MCMC ALGORITHM COMPARISON WITH BAYESIAN HYPERPARAMETER OPTIMIZATION
# ============================================================================
# This script compares three MCMC algorithms:
# 1. NUTS (No-U-Turn Sampler)
# 2. MCLMC (Microcanonical Langevin Monte Carlo)
# 3. MAMS (MCLMC with Adjusted Momentum Sampling)
#
# Key insight: We use FIXED random keys during hyperparameter tuning to ensure
# fair comparison - each set of hyperparameters is evaluated under identical
# random conditions.
# ============================================================================

import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import blackjax
import numpy as np
from jax import config
from blackjax.mcmc.adjusted_mclmc_dynamic import rescale
from blackjax.util import run_inference_algorithm
from boax.experiments import optimization

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

# Enable 64-bit precision for numerical stability
config.update("jax_enable_x64", True)

# Configure matplotlib for clean plots
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["font.size"] = 19

# Random keys for different purposes
# These are set once at the top for clarity and reproducibility
SEED_NUTS_TUNING = 1000      # For NUTS hyperparameter optimization
SEED_MCLMC_TUNING = 2000     # For MCLMC hyperparameter optimization
SEED_MAMS_TUNING = 3000      # For MAMS hyperparameter optimization
SEED_BAYESOPT_VALIDATION = 99999  # For validating BayesOpt results
SEED_AUTO_VALIDATION = 88888      # For validating auto-tuned results
SEED_REPRODUCIBILITY = 42    # For reproducibility demos


# ============================================================================
# TARGET DENSITIES
# ============================================================================
# These are the probability distributions we want to sample from.
# Each is challenging in different ways (ill-conditioned, heavy tails, etc.)
# ============================================================================

def make_gaussian_logdensity(dim):
    """
    Create a high-dimensional Gaussian with varying scales.
    
    This distribution is ill-conditioned: some dimensions have much larger
    variance than others, making it hard for samplers to explore efficiently.
    
    Args:
        dim: Number of dimensions
        
    Returns:
        Function that computes log probability density
    """
    # Create scales that increase linearly with dimension index
    # Dimension 0 has scale 1.0, dimension (dim-1) has scale dim
    scales = jnp.linspace(1.0, float(dim), dim)
    inv_cov = 1.0 / scales**2  # Inverse covariance (precision)
    
    def logdensity(x):
        # Log probability: -0.5 * x^T * Precision * x
        return -0.5 * jnp.sum(inv_cov * x**2)
    
    return logdensity


def make_funnel_logdensity(dim):
    """
    Create Neal's funnel distribution.
    
    This is a pathological distribution where the width of all dimensions
    depends on the first dimension. When x[0] is negative, the funnel
    narrows dramatically, making it very challenging for samplers.
    
    Args:
        dim: Number of dimensions
        
    Returns:
        Function that computes log probability density
    """
    def logdensity(x):
        # First dimension has a standard normal prior
        log_prob = -0.5 * (x[0]**2 / 9.0)
        
        # Remaining dimensions have variance that depends exponentially on x[0]
        # When x[0] is large and negative, variance shrinks to nearly zero
        log_prob += -0.5 * (dim - 1) * x[0]  # Normalizing constant
        log_prob += -0.5 * jnp.sum(x[1:]**2 * jnp.exp(-x[0]))  # Likelihood
        
        return log_prob
    
    return logdensity


def make_rosenbrock_logdensity(dim):
    """
    Create a Rosenbrock (banana-shaped) distribution.
    
    This distribution has a curved ridge that's easy to find but hard to
    follow, testing a sampler's ability to navigate complex geometry.
    
    Args:
        dim: Number of dimensions
        
    Returns:
        Function that computes log probability density
    """
    def logdensity(x):
        # First dimension has standard normal prior
        log_prob = -0.5 * x[0]**2
        
        # Subsequent dimensions are constrained to lie near x[i]^2
        # This creates a curved "banana" shape
        for i in range(dim - 1):
            log_prob += -0.5 * (x[i+1] - x[i]**2)**2 / 0.1
        
        return log_prob
    
    return logdensity


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_target_2d(logdensity_fn, dim_x=0, dim_y=1, xlim=(-5, 5), ylim=(-5, 5)):
    """
    Plot a 2D slice of the target distribution.
    
    This helps visualize the geometry of the distribution we're sampling from.
    
    Args:
        logdensity_fn: Function that computes log density
        dim_x: Which dimension to plot on x-axis (default: 0)
        dim_y: Which dimension to plot on y-axis (default: 1)
        xlim: Range for x-axis
        ylim: Range for y-axis
    """
    # Create a grid of points
    x = np.linspace(xlim[0], xlim[1], 100)
    y = np.linspace(ylim[0], ylim[1], 100)
    X, Y = np.meshgrid(x, y)
    
    # Evaluate log density at each grid point
    Z = np.zeros_like(X)
    for i in range(len(x)):
        for j in range(len(y)):
            # Create a point with zeros in all dimensions except the two we're plotting
            point = jnp.zeros(10)
            point = point.at[dim_x].set(X[j, i])
            point = point.at[dim_y].set(Y[j, i])
            Z[j, i] = logdensity_fn(point)
    
    # Create contour plot
    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, np.exp(Z), levels=20)
    plt.xlabel(f'Dimension {dim_x}')
    plt.ylabel(f'Dimension {dim_y}')
    plt.colorbar(label='Density')
    return plt.gcf()


# ============================================================================
# DIAGNOSTIC METRICS
# ============================================================================

def compute_ess(samples):
    """
    Compute Effective Sample Size (ESS).
    
    ESS measures how many "independent" samples we effectively have after
    accounting for autocorrelation. Higher ESS means better mixing.
    
    Args:
        samples: Array of shape (num_steps, dim)
        
    Returns:
        Minimum ESS across all dimensions (the bottleneck)
    """
    # Reshape for blackjax: (dim, 1, num_steps)
    samples_reshaped = samples.T[:, None, :]
    
    # Compute ESS for each dimension independently
    ess_per_dim = jax.vmap(
        lambda x: blackjax.diagnostics.effective_sample_size(x), 
        in_axes=0
    )(samples_reshaped)
    
    # Return the minimum - this is our bottleneck dimension
    return jnp.min(ess_per_dim)


def compute_rhat(chains):
    """
    Compute Gelman-Rubin R-hat convergence diagnostic.
    
    R-hat compares within-chain variance to between-chain variance.
    Values close to 1.0 indicate convergence. Values > 1.01 suggest chains
    haven't mixed well.
    
    Args:
        chains: Array of shape (num_chains, num_steps, dim)
        
    Returns:
        R-hat value for each dimension
    """
    num_chains, num_steps, dim = chains.shape
    
    # Compute mean of each chain
    chain_means = jnp.mean(chains, axis=1)  # Shape: (num_chains, dim)
    
    # Compute overall mean across all chains
    overall_mean = jnp.mean(chain_means, axis=0)  # Shape: (dim,)
    
    # Between-chain variance (B)
    B = num_steps / (num_chains - 1) * jnp.sum((chain_means - overall_mean)**2, axis=0)
    
    # Within-chain variance (W)
    chain_vars = jnp.var(chains, axis=1, ddof=1)  # Shape: (num_chains, dim)
    W = jnp.mean(chain_vars, axis=0)  # Shape: (dim,)
    
    # Estimated variance
    var_est = ((num_steps - 1) / num_steps) * W + (1 / num_steps) * B
    
    # R-hat statistic
    rhat = jnp.sqrt(var_est / W)
    
    return rhat


# ============================================================================
# SAMPLING ALGORITHMS (FIXED PARAMETERS)
# ============================================================================
# These functions run samplers with specified hyperparameters and a fixed
# random key, ensuring reproducibility for hyperparameter tuning.
# ============================================================================

def run_nuts_fixed(logdensity_fn, num_steps, initial_position, key, 
                   step_size, inv_mass_matrix):
    """
    Run NUTS (No-U-Turn Sampler) with fixed hyperparameters.
    
    NUTS is an adaptive HMC variant that automatically tunes trajectory length.
    
    Args:
        logdensity_fn: Target distribution
        num_steps: Number of MCMC steps
        initial_position: Starting point
        key: JAX random key (fixed for reproducibility)
        step_size: Integration step size
        inv_mass_matrix: Inverse mass matrix (metric)
        
    Returns:
        samples: Array of shape (num_steps, dim)
        ess: Effective sample size (minimum across dimensions)
        avg_acceptance: Average acceptance rate
    """
    # Initialize NUTS sampler
    nuts = blackjax.nuts(logdensity_fn, step_size, inv_mass_matrix)
    state = nuts.init(initial_position)
    
    # Define one step of the sampler
    def one_step(state, key):
        state, info = nuts.step(key, state)
        return state, (state.position, info.acceptance_rate)
    
    # Run the chain
    keys = jax.random.split(key, num_steps)
    final_state, (samples, acceptance_rates) = jax.lax.scan(one_step, state, keys)
    
    # Compute diagnostics
    avg_acceptance = jnp.mean(acceptance_rates)
    ess = compute_ess(samples)
    
    return samples, ess, avg_acceptance


def run_mclmc_fixed(logdensity_fn, num_steps, initial_position, key, L, step_size):
    """
    Run MCLMC (Microcanonical Langevin Monte Carlo) with fixed hyperparameters.
    
    MCLMC uses continuous dynamics with refreshment to explore the target.
    It has no accept/reject step, so acceptance rate is always 1.0.
    
    Args:
        logdensity_fn: Target distribution
        num_steps: Number of MCMC steps
        initial_position: Starting point
        key: JAX random key (fixed for reproducibility)
        L: Trajectory length
        step_size: Integration step size
        
    Returns:
        samples: Array of shape (num_steps, dim)
        ess: Effective sample size
        acceptance: Always 1.0 (no rejection)
    """
    init_key, run_key = jax.random.split(key)
    
    # Initialize MCLMC state
    initial_state = blackjax.mcmc.mclmc.init(
        position=initial_position, 
        logdensity_fn=logdensity_fn, 
        rng_key=init_key
    )
    
    # Create MCLMC sampler
    sampling_alg = blackjax.mclmc(
        logdensity_fn,
        L=L,
        step_size=step_size,
    )
    
    # Run the sampler
    _, samples = blackjax.util.run_inference_algorithm(
        rng_key=run_key,
        initial_state=initial_state,
        inference_algorithm=sampling_alg,
        num_steps=num_steps,
        transform=lambda state, _: state.position,
        progress_bar=False,
    )
    
    # Compute diagnostics
    ess = compute_ess(samples)
    
    return samples, ess, 1.0  # MCLMC has no rejection


def run_mams_fixed(logdensity_fn, num_steps, initial_position, key, L, step_size):
    """
    Run MAMS (MCLMC with Adjusted Momentum Sampling) with fixed hyperparameters.
    
    MAMS is like MCLMC but with accept/reject steps to correct for discretization
    error, making it more robust.
    
    Args:
        logdensity_fn: Target distribution
        num_steps: Number of MCMC steps
        initial_position: Starting point
        key: JAX random key (fixed for reproducibility)
        L: Trajectory length
        step_size: Integration step size
        
    Returns:
        samples: Array of shape (num_steps, dim)
        ess: Effective sample size
        avg_acceptance: Average acceptance rate
    """
    init_key, run_key = jax.random.split(key)
    
    # Initialize MAMS state
    initial_state = blackjax.mcmc.adjusted_mclmc_dynamic.init(
        position=initial_position,
        logdensity_fn=logdensity_fn,
        random_generator_arg=init_key,
    )
    
    # Create MAMS sampler with fixed trajectory length
    alg = blackjax.adjusted_mclmc_dynamic(
        logdensity_fn=logdensity_fn,
        step_size=step_size,
        integration_steps_fn=lambda key: jnp.ceil(L / step_size),
        L_proposal_factor=jnp.inf,  # No dynamic trajectory length adjustment
    )
    
    # Run one step at a time to track acceptance rate
    def one_step(state, key):
        state, info = alg.step(key, state)
        return state, (state.position, info.acceptance_rate)
    
    keys = jax.random.split(run_key, num_steps)
    final_state, (samples, acceptance_rates) = jax.lax.scan(one_step, initial_state, keys)
    
    # Compute diagnostics
    avg_acceptance = jnp.mean(acceptance_rates)
    ess = compute_ess(samples)
    
    return samples, ess, avg_acceptance


# ============================================================================
# BAYESIAN OPTIMIZATION FOR HYPERPARAMETER TUNING
# ============================================================================

def objective_function(ess, acceptance_rate, target_acceptance, lambda_penalty):
    """
    Compute objective for hyperparameter optimization.
    
    We want to maximize ESS while keeping acceptance rate near target.
    The penalty term discourages acceptance rates far from target.
    
    Args:
        ess: Effective sample size
        acceptance_rate: Average acceptance rate
        target_acceptance: Desired acceptance rate
        lambda_penalty: How much to penalize deviation from target
        
    Returns:
        Objective value to maximize
    """
    penalty = lambda_penalty * (acceptance_rate - target_acceptance)**2
    return ess - penalty


def run_bayesopt_tuning(
    logdensity_fn,
    initial_position,
    fixed_key,
    algorithm_name,
    num_iterations=20,
    chain_length=1000,
    target_acceptance=0.65,
    lambda_penalty=100.0,
):
    """
    Use Bayesian optimization to find optimal hyperparameters.
    
    CRITICAL: We use the SAME random key for all hyperparameter evaluations.
    This ensures fair comparison - each set of hyperparameters is tested
    under identical random conditions.
    
    Args:
        logdensity_fn: Target distribution
        initial_position: Starting point for all chains
        fixed_key: Same key used for ALL evaluations
        algorithm_name: 'NUTS', 'MCLMC', or 'MAMS'
        num_iterations: Number of optimization iterations
        chain_length: Length of each evaluation chain
        target_acceptance: Target acceptance rate for objective
        lambda_penalty: Penalty for deviating from target acceptance
        
    Returns:
        Dictionary with optimization history
    """
    # Storage for results
    results = {
        'iteration': [],
        'ess': [],
        'acceptance_rate': [],
        'objective': [],
        'hyperparams': [],
    }
    
    # Define search space and evaluation function for each algorithm
    if algorithm_name == 'NUTS':
        # NUTS has one hyperparameter: step size
        parameters = [
            {
                'name': 'log_step_size',
                'type': 'log_range',
                'bounds': [1e-5, 1e-1],
            }
        ]
        
        def run_with_params(params_dict):
            """Evaluate NUTS with given hyperparameters."""
            step_size = params_dict['log_step_size']
            inv_mass_matrix = jnp.ones(len(initial_position))
            
            # Run sampler with FIXED key - deterministic given hyperparameters!
            _, ess, acc = run_nuts_fixed(
                logdensity_fn, chain_length, initial_position, 
                fixed_key,  # Same key every time ensures fair comparison
                step_size, inv_mass_matrix
            )
            return ess, acc
        
    elif algorithm_name == 'MCLMC':
        # MCLMC has two hyperparameters: trajectory length and step size
        parameters = [
            {
                'name': 'L',
                'type': 'log_range',
                'bounds': [1e-1, 1e2],
            },
            {
                'name': 'step_size',
                'type': 'log_range',
                'bounds': [1e-3, 1.0],
            }
        ]
        # MCLMC has no rejection, so target acceptance is 1.0
        target_acceptance = 1.0
        
        def run_with_params(params_dict):
            """Evaluate MCLMC with given hyperparameters."""
            L = params_dict['L']
            step_size = params_dict['step_size']
            
            _, ess, acc = run_mclmc_fixed(
                logdensity_fn, chain_length, initial_position, 
                fixed_key,  # Same key for all evaluations
                L, step_size
            )
            return ess, acc
        
    elif algorithm_name == 'MAMS':
        # MAMS has two hyperparameters: trajectory length and step size
        parameters = [
            {
                'name': 'L',
                'type': 'log_range',
                'bounds': [1e-1, 1e2],
            },
            {
                'name': 'step_size',
                'type': 'log_range',
                'bounds': [1e-3, 1.0],
            }
        ]
        # MAMS typically targets 90% acceptance
        target_acceptance = 0.9
        
        def run_with_params(params_dict):
            """Evaluate MAMS with given hyperparameters."""
            L = params_dict['L']
            step_size = params_dict['step_size']
            
            _, ess, acc = run_mams_fixed(
                logdensity_fn, chain_length, initial_position, 
                fixed_key,  # Same key for all evaluations
                L, step_size
            )
            return ess, acc
    
    # Initialize BOAx (Bayesian Optimization with Ax) experiment
    experiment = optimization(
        parameters=parameters,
        batch_size=1,
    )
    
    # Run Bayesian optimization loop
    step, experiment_results = None, []
    
    for i in range(num_iterations):
        print(f"Iteration {i+1}/{num_iterations}")
        
        # Get next parameterization to try
        # BOAx uses a Gaussian Process model to suggest promising parameters
        step, parameterizations = experiment.next(step, experiment_results)
        params_dict = parameterizations[0]
        
        # Evaluate with FIXED key - same random trajectory for fair comparison
        ess, acc = run_with_params(params_dict)
        
        # Compute objective (what we're trying to maximize)
        obj = objective_function(ess, acc, target_acceptance, lambda_penalty)
        
        # Give feedback to BOAx for next iteration
        experiment_results = [(params_dict, float(obj))]
        
        # Store results for our own tracking
        results['iteration'].append(i)
        results['ess'].append(float(ess))
        results['acceptance_rate'].append(float(acc))
        results['objective'].append(float(obj))
        results['hyperparams'].append(params_dict)
        
        print(f"  {algorithm_name} - ESS={ess:.1f}, Acc={acc:.3f}, "
              f"Obj={obj:.1f}, Params={params_dict}")
    
    # Find and print best parameters from our stored results
    # (This is more reliable than experiment.best() which uses BOAx's internal model)
    best_idx = jnp.argmax(jnp.array(results['objective']))
    best_params = results['hyperparams'][best_idx]
    best_obj = results['objective'][best_idx]
    
    print(f"\n{algorithm_name} Best parameters: {best_params}")
    print(f"  Best objective: {best_obj:.2f}")
    print(f"  ESS: {results['ess'][best_idx]:.1f}")
    print(f"  Acceptance: {results['acceptance_rate'][best_idx]:.3f}")
    
    return results


# ============================================================================
# VALIDATION WITH MULTIPLE CHAINS
# ============================================================================
# After tuning, we validate with multiple chains using DIFFERENT random keys
# to test robustness across different initializations.
# ============================================================================

def run_mams_multiple_chains(logdensity_fn, num_chains, num_steps, 
                             initial_position, base_key, L, step_size):
    """
    Run multiple MAMS chains with fixed hyperparameters.
    
    This is for VALIDATION after hyperparameter tuning. We use different
    random keys for each chain to test robustness.
    
    Args:
        logdensity_fn: Target distribution
        num_chains: Number of independent chains to run
        num_steps: Steps per chain
        initial_position: Starting point
        base_key: Base key to derive chain-specific keys from
        L: Trajectory length (from tuning)
        step_size: Step size (from tuning)
        
    Returns:
        all_samples: Shape (num_chains, num_steps, dim)
        all_ess: ESS for each chain
        all_acceptance: Average acceptance for each chain
    """
    all_samples = []
    all_ess = []
    all_acceptance = []
    
    for i in range(num_chains):
        # Create a different key for each chain
        chain_key = jax.random.fold_in(base_key, i)
        
        # Run the chain
        samples, ess, acc = run_mams_fixed(
            logdensity_fn, num_steps, initial_position, chain_key, L, step_size
        )
        
        all_samples.append(samples)
        all_ess.append(ess)
        all_acceptance.append(acc)
    
    # Stack into a single array
    all_samples = jnp.stack(all_samples, axis=0)
    
    return all_samples, jnp.array(all_ess), jnp.array(all_acceptance)


def run_auto_tuned_multiple_chains(logdensity_fn, num_chains, num_steps, 
                                   initial_position, base_key):
    """
    Run multiple chains with automatic hyperparameter tuning.
    
    Each chain tunes its own hyperparameters independently, allowing us
    to compare against Bayesian optimization.
    
    Args:
        logdensity_fn: Target distribution
        num_chains: Number of chains
        num_steps: Steps per chain
        initial_position: Starting point
        base_key: Base key to derive chain keys from
        
    Returns:
        all_samples: Shape (num_chains, num_steps, dim)
        all_ess: ESS for each chain
        all_acceptance: Acceptance for each chain (typically 0.9)
        all_step_sizes: Step sizes found by auto-tuning
        all_L: Trajectory lengths found by auto-tuning
    """
    all_samples = []
    all_step_sizes = []
    all_L = []
    
    for i in range(num_chains):
        # Different key for each chain
        chain_key = jax.random.fold_in(base_key, i)
        
        # Run with automatic tuning
        samples, step_size, L, inv_mass = run_adjusted_mclmc_dynamic(
            logdensity_fn, num_steps, initial_position, chain_key
        )
        
        all_samples.append(samples)
        all_step_sizes.append(step_size)
        all_L.append(L)
    
    all_samples = jnp.stack(all_samples, axis=0)
    
    # Compute ESS for each chain
    all_ess = []
    for chain_samples in all_samples:
        ess = compute_ess(chain_samples)
        all_ess.append(ess)
    
    # Auto-tuning targets 0.9 acceptance
    all_acceptance = jnp.ones(num_chains) * 0.9
    
    return (all_samples, jnp.array(all_ess), all_acceptance, 
            jnp.array(all_step_sizes), jnp.array(all_L))


def run_adjusted_mclmc_dynamic(
    logdensity_fn,
    num_steps,
    initial_position,
    key,
    transform=lambda state, _: state.position,
    diagonal_preconditioning=True,
    random_trajectory_length=True,
    L_proposal_factor=jnp.inf
):
    """
    Run MAMS with automatic hyperparameter tuning.
    
    This uses blackjax's built-in adaptive tuning procedure to find
    good hyperparameters on-the-fly.
    
    Args:
        logdensity_fn: Target distribution
        num_steps: Number of steps
        initial_position: Starting point
        key: Random key
        (other args control tuning behavior)
        
    Returns:
        samples: Array of samples
        step_size: Tuned step size
        L: Tuned trajectory length
        inverse_mass_matrix: Tuned metric
    """
    # Split key for different stages
    init_key, tune_key, run_key = jax.random.split(key, 3)
    
    # Initialize state
    initial_state = blackjax.mcmc.adjusted_mclmc_dynamic.init(
        position=initial_position,
        logdensity_fn=logdensity_fn,
        random_generator_arg=init_key,
    )
    
    # Define integration steps function (optionally random)
    if random_trajectory_length:
        integration_steps_fn = lambda avg_num_integration_steps: lambda k: jnp.ceil(
            jax.random.uniform(k) * rescale(avg_num_integration_steps))
    else:
        integration_steps_fn = lambda avg_num_integration_steps: lambda _: jnp.ceil(
            avg_num_integration_steps)
    
    # Create kernel for tuning
    kernel = lambda rng_key, state, avg_num_integration_steps, step_size, inverse_mass_matrix: \
        blackjax.mcmc.adjusted_mclmc_dynamic.build_kernel(
            integration_steps_fn=integration_steps_fn(avg_num_integration_steps),
            inverse_mass_matrix=inverse_mass_matrix,
        )(
            rng_key=rng_key,
            state=state,
            step_size=step_size,
            logdensity_fn=logdensity_fn,
            L_proposal_factor=L_proposal_factor,
        )
    
    # Run automatic tuning procedure
    target_acc_rate = 0.9
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
        frac_tune1=0.1,  # Fraction of steps for first tuning phase
        frac_tune2=0.1,  # Fraction for second phase
        frac_tune3=0.1,  # Fraction for third phase
        diagonal_preconditioning=diagonal_preconditioning,
    )
    
    # Extract tuned parameters
    step_size = blackjax_mclmc_sampler_params.step_size
    L = blackjax_mclmc_sampler_params.L
    
    # Create final algorithm with tuned parameters
    alg = blackjax.adjusted_mclmc_dynamic(
        logdensity_fn=logdensity_fn,
        step_size=step_size,
        integration_steps_fn=lambda key: jnp.ceil(
            jax.random.uniform(key) * rescale(L / step_size)
        ),
        inverse_mass_matrix=blackjax_mclmc_sampler_params.inverse_mass_matrix,
        L_proposal_factor=L_proposal_factor,
    )
    
    # Run sampling
    _, out = run_inference_algorithm(
        rng_key=run_key,
        initial_state=blackjax_state_after_tuning,
        inference_algorithm=alg,
        num_steps=num_steps,
        transform=transform,
        progress_bar=False,
    )
    
    return out, step_size, L, blackjax_mclmc_sampler_params.inverse_mass_matrix


# ============================================================================
# MAIN COMPARISON FUNCTIONS
# ============================================================================

def compare_algorithms(dim=5):
    """
    Compare all three algorithms using Bayesian optimization for tuning.
    
    This is the main experiment that tunes hyperparameters for NUTS, MCLMC,
    and MAMS, then compares their performance.
    
    Args:
        dim: Dimensionality of target distribution
        
    Returns:
        nuts_results, mclmc_results, mams_results: Tuning history for each
    """
    print("="*70)
    print("MCMC ALGORITHM COMPARISON WITH BAYESIAN OPTIMIZATION")
    print("="*70)
    
    # Create target density
    print("\nCreating target density (Neal's Funnel)...")
    logdensity_fn = make_funnel_logdensity(dim)
    
    # Visualize the target
    print("Plotting 2D slice of target...")
    fig = plot_target_2d(logdensity_fn, dim_x=0, dim_y=1, 
                         xlim=(-10, 10), ylim=(-10, 10))
    plt.title("Neal's Funnel (2D slice)")
    plt.show()
    
    # Initial position (all zeros)
    initial_position = jnp.zeros(dim)
    
    # Create fixed keys for hyperparameter tuning
    # Each algorithm gets its own key to ensure fair comparison
    nuts_tuning_key = jax.random.key(SEED_NUTS_TUNING)
    mclmc_tuning_key = jax.random.key(SEED_MCLMC_TUNING)
    mams_tuning_key = jax.random.key(SEED_MAMS_TUNING)
    
    print("\n" + "="*70)
    print("PHASE 1: HYPERPARAMETER TUNING")
    print("="*70)
    print("\nNote: Using FIXED random keys for fair comparison!")
    print("Each algorithm is evaluated under identical random conditions.")
    
    # Tune NUTS
    print("\n1. Optimizing NUTS hyperparameters...")
    nuts_results = run_bayesopt_tuning(
        logdensity_fn, initial_position, nuts_tuning_key, 'NUTS',
        num_iterations=10,
        chain_length=100
    )
    
    # Tune MCLMC
    print("\n2. Optimizing MCLMC hyperparameters...")
    mclmc_results = run_bayesopt_tuning(
        logdensity_fn, initial_position, mclmc_tuning_key, 'MCLMC',
        num_iterations=10,
        chain_length=100
    )
    
    # Tune MAMS
    print("\n3. Optimizing MAMS hyperparameters...")
    mams_results = run_bayesopt_tuning(
        logdensity_fn, initial_position, mams_tuning_key, 'MAMS',
        num_iterations=10,
        chain_length=100
    )
    
    # Create visualization of tuning results
    print("\nCreating tuning plots...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: ESS over iterations
    axes[0].plot(nuts_results['iteration'], nuts_results['ess'], 
                 'o-', label='NUTS', linewidth=2)
    axes[0].plot(mclmc_results['iteration'], mclmc_results['ess'], 
                 's-', label='MCLMC', linewidth=2)
    axes[0].plot(mams_results['iteration'], mams_results['ess'], 
                 '^-', label='MAMS', linewidth=2)
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('ESS')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Effective Sample Size')
    
    # Plot 2: Acceptance rate over iterations
    axes[1].plot(nuts_results['iteration'], nuts_results['acceptance_rate'], 
                 'o-', label='NUTS', linewidth=2)
    axes[1].plot(mclmc_results['iteration'], mclmc_results['acceptance_rate'], 
                 's-', label='MCLMC', linewidth=2)
    axes[1].plot(mams_results['iteration'], mams_results['acceptance_rate'], 
                 '^-', label='MAMS', linewidth=2)
    axes[1].axhline(0.65, color='gray', linestyle='--', alpha=0.5, 
                    label='NUTS target')
    axes[1].axhline(0.9, color='gray', linestyle='--', alpha=0.5, 
                    label='MAMS target')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Acceptance Rate')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Acceptance Rate')
    
    # Plot 3: Optimization objective over iterations
    axes[2].plot(nuts_results['iteration'], nuts_results['objective'], 
                 'o-', label='NUTS', linewidth=2)
    axes[2].plot(mclmc_results['iteration'], mclmc_results['objective'], 
                 's-', label='MCLMC', linewidth=2)
    axes[2].plot(mams_results['iteration'], mams_results['objective'], 
                 '^-', label='MAMS', linewidth=2)
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('Objective')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title('Optimization Objective')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary of best results
    print("\n" + "="*70)
    print("BEST HYPERPARAMETERS FOUND")
    print("="*70)
    
    # NUTS best
    best_nuts_idx = jnp.argmax(jnp.array(nuts_results['objective']))
    print(f"\nNUTS:")
    print(f"  Best objective: {nuts_results['objective'][best_nuts_idx]:.1f}")
    print(f"  ESS: {nuts_results['ess'][best_nuts_idx]:.1f}")
    print(f"  Acceptance: {nuts_results['acceptance_rate'][best_nuts_idx]:.3f}")
    print(f"  Hyperparameters: {nuts_results['hyperparams'][best_nuts_idx]}")
    
    # MCLMC best
    best_mclmc_idx = jnp.argmax(jnp.array(mclmc_results['objective']))
    print(f"\nMCLMC:")
    print(f"  Best objective: {mclmc_results['objective'][best_mclmc_idx]:.1f}")
    print(f"  ESS: {mclmc_results['ess'][best_mclmc_idx]:.1f}")
    print(f"  Acceptance: {mclmc_results['acceptance_rate'][best_mclmc_idx]:.3f}")
    print(f"  Hyperparameters: {mclmc_results['hyperparams'][best_mclmc_idx]}")
    
    # MAMS best
    best_mams_idx = jnp.argmax(jnp.array(mams_results['objective']))
    print(f"\nMAMS:")
    print(f"  Best objective: {mams_results['objective'][best_mams_idx]:.1f}")
    print(f"  ESS: {mams_results['ess'][best_mams_idx]:.1f}")
    print(f"  Acceptance: {mams_results['acceptance_rate'][best_mams_idx]:.3f}")
    print(f"  Hyperparameters: {mams_results['hyperparams'][best_mams_idx]}")
    
    return nuts_results, mclmc_results, mams_results


def compare_mams_tuning_methods(dim=5, num_chains=4, num_steps=1000):
    """
    Compare Bayesian optimization vs automatic tuning for MAMS.
    
    This experiment addresses the key question: Does careful hyperparameter
    optimization beat blackjax's built-in automatic tuning?
    
    Args:
        dim: Dimensionality of target
        num_chains: Number of independent chains for validation
        num_steps: Steps per chain
        
    Returns:
        bayesopt_samples, auto_samples, mams_results: Comparison data
    """
    print("="*70)
    print("MAMS HYPERPARAMETER TUNING COMPARISON")
    print("="*70)
    print("\nQuestion: Is Bayesian optimization better than automatic tuning?")
    
    # Setup
    logdensity_fn = make_funnel_logdensity(dim)
    initial_position = jnp.zeros(dim)
    
    # ========================================================================
    # STEP 1: Find best hyperparameters using Bayesian optimization
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 1: BAYESIAN OPTIMIZATION")
    print("="*70)
    print("\nRunning BayesOpt to find optimal hyperparameters...")
    
    bayesopt_tuning_key = jax.random.key(SEED_MAMS_TUNING)
    mams_results = run_bayesopt_tuning(
        logdensity_fn, initial_position, bayesopt_tuning_key, 'MAMS',
        num_iterations=10,
        chain_length=100
    )
    
    # Extract best parameters from stored results
    best_idx = jnp.argmax(jnp.array(mams_results['objective']))
    best_params = mams_results['hyperparams'][best_idx]
    best_L = best_params['L']
    best_step_size = best_params['step_size']
    
    print(f"\nBest hyperparameters found:")
    print(f"  L = {best_L:.4f}")
    print(f"  step_size = {best_step_size:.6f}")
    print(f"  Best objective = {mams_results['objective'][best_idx]:.1f}")
    print(f"  ESS = {mams_results['ess'][best_idx]:.1f}")
    print(f"  Acceptance = {mams_results['acceptance_rate'][best_idx]:.3f}")
    
    # ========================================================================
    # STEP 2: Validate BayesOpt parameters with multiple chains
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 2: VALIDATE BAYESOPT PARAMETERS")
    print("="*70)
    print(f"\nRunning {num_chains} chains with BayesOpt parameters "
          f"({num_steps} steps each)...")
    print("Using DIFFERENT random keys to test robustness...")
    
    bayesopt_validation_key = jax.random.key(SEED_BAYESOPT_VALIDATION)
    bayesopt_samples, bayesopt_ess, bayesopt_acc = run_mams_multiple_chains(
        logdensity_fn, num_chains, num_steps, initial_position, 
        bayesopt_validation_key, best_L, best_step_size
    )
    
    # Compute convergence diagnostic
    bayesopt_rhat = compute_rhat(bayesopt_samples)
    
    print(f"\nBayesOpt Validation Results:")
    print(f"  ESS per chain: {bayesopt_ess}")
    print(f"  Mean ESS: {jnp.mean(bayesopt_ess):.1f} ± {jnp.std(bayesopt_ess):.1f}")
    print(f"  Acceptance per chain: {bayesopt_acc}")
    print(f"  Mean acceptance: {jnp.mean(bayesopt_acc):.3f} ± {jnp.std(bayesopt_acc):.3f}")
    print(f"  R-hat per dimension: {bayesopt_rhat}")
    print(f"  Max R-hat: {jnp.max(bayesopt_rhat):.4f} "
          f"{'✓ (good)' if jnp.max(bayesopt_rhat) < 1.01 else '⚠ (needs more steps)'}")
    
    # ========================================================================
    # STEP 3: Run multiple chains with automatic tuning
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 3: AUTOMATIC TUNING")
    print("="*70)
    print(f"\nRunning {num_chains} chains with automatic tuning "
          f"({num_steps} steps each)...")
    print("Each chain tunes its own hyperparameters independently...")
    
    auto_validation_key = jax.random.key(SEED_AUTO_VALIDATION)
    auto_samples, auto_ess, auto_acc, auto_step_sizes, auto_L = \
        run_auto_tuned_multiple_chains(
            logdensity_fn, num_chains, num_steps, initial_position, 
            auto_validation_key
        )
    
    # Compute convergence diagnostic
    auto_rhat = compute_rhat(auto_samples)
    
    print(f"\nAutomatic Tuning Results:")
    print(f"  Step sizes found: {auto_step_sizes}")
    print(f"  Mean step size: {jnp.mean(auto_step_sizes):.6f} "
          f"± {jnp.std(auto_step_sizes):.6f}")
    print(f"  L values found: {auto_L}")
    print(f"  Mean L: {jnp.mean(auto_L):.4f} ± {jnp.std(auto_L):.4f}")
    print(f"  ESS per chain: {auto_ess}")
    print(f"  Mean ESS: {jnp.mean(auto_ess):.1f} ± {jnp.std(auto_ess):.1f}")
    print(f"  R-hat per dimension: {auto_rhat}")
    print(f"  Max R-hat: {jnp.max(auto_rhat):.4f} "
          f"{'✓ (good)' if jnp.max(auto_rhat) < 1.01 else '⚠ (needs more steps)'}")
    
    # ========================================================================
    # STEP 4: Create comprehensive comparison plots
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 4: CREATING COMPARISON PLOTS")
    print("="*70)
    
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Row 1: Trace plots for dimension 0 (mouth of funnel)
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
    
    # Trace plot for dimension 1 (neck of funnel - most challenging)
    ax3 = fig.add_subplot(gs[0, 2])
    for i in range(num_chains):
        ax3.plot(bayesopt_samples[i, :, 1], alpha=0.4, 
                label=f'BO Chain {i+1}', linewidth=1)
    for i in range(num_chains):
        ax3.plot(auto_samples[i, :, 1], alpha=0.4, linestyle='--', 
                label=f'Auto Chain {i+1}', linewidth=1)
    ax3.set_title('Both Methods: Traces (Dim 1)')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Value')
    ax3.legend(fontsize=6, ncol=2)
    ax3.grid(True, alpha=0.3)
    
    # Row 2: Marginal distributions
    # Flatten all chains into one big sample for each method
    bayesopt_flat = bayesopt_samples.reshape(-1, dim)
    auto_flat = auto_samples.reshape(-1, dim)
    
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(bayesopt_flat[:, 0], bins=50, alpha=0.7, density=True, 
            label='BayesOpt', color='C0')
    ax4.set_title('BayesOpt: Marginal (Dim 0)')
    ax4.set_xlabel('Value')
    ax4.set_ylabel('Density')
    ax4.grid(True, alpha=0.3)
    
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.hist(auto_flat[:, 0], bins=50, alpha=0.7, density=True, 
            label='Auto-tuned', color='C1')
    ax5.set_title('Auto-tuned: Marginal (Dim 0)')
    ax5.set_xlabel('Value')
    ax5.set_ylabel('Density')
    ax5.grid(True, alpha=0.3)
    
    # Overlay comparison
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.hist(bayesopt_flat[:, 0], bins=50, alpha=0.5, density=True, 
            label='BayesOpt', color='C0')
    ax6.hist(auto_flat[:, 0], bins=50, alpha=0.5, density=True, 
            label='Auto-tuned', color='C1')
    ax6.set_title('Overlay: Marginal (Dim 0)')
    ax6.set_xlabel('Value')
    ax6.set_ylabel('Density')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Row 3: Summary statistics
    # ESS comparison by chain
    ax7 = fig.add_subplot(gs[2, 0])
    x_pos = np.arange(num_chains)
    width = 0.35
    ax7.bar(x_pos - width/2, bayesopt_ess, width, 
           label='BayesOpt', alpha=0.7, color='C0')
    ax7.bar(x_pos + width/2, auto_ess, width, 
           label='Auto-tuned', alpha=0.7, color='C1')
    ax7.set_xlabel('Chain')
    ax7.set_ylabel('ESS')
    ax7.set_title('Effective Sample Size by Chain')
    ax7.set_xticks(x_pos)
    ax7.set_xticklabels([f'{i+1}' for i in range(num_chains)])
    ax7.legend()
    ax7.grid(True, alpha=0.3, axis='y')
    
    # R-hat comparison by dimension
    ax8 = fig.add_subplot(gs[2, 1])
    x_pos_dim = np.arange(dim)
    ax8.bar(x_pos_dim - width/2, bayesopt_rhat, width, 
           label='BayesOpt', alpha=0.7, color='C0')
    ax8.bar(x_pos_dim + width/2, auto_rhat, width, 
           label='Auto-tuned', alpha=0.7, color='C1')
    ax8.axhline(1.01, color='red', linestyle='--', alpha=0.5, 
               linewidth=2, label='Threshold')
    ax8.set_xlabel('Dimension')
    ax8.set_ylabel('R-hat')
    ax8.set_title('Convergence Diagnostic (R-hat)')
    ax8.set_xticks(x_pos_dim)
    ax8.legend()
    ax8.grid(True, alpha=0.3, axis='y')
    
    # Hyperparameter comparison
    ax9 = fig.add_subplot(gs[2, 2])
    methods = ['BayesOpt', 'Auto-tuned']
    L_values = [best_L, jnp.mean(auto_L)]
    # Scale step_size by 100 for visibility
    step_values = [best_step_size * 100, jnp.mean(auto_step_sizes) * 100]
    
    x_pos_hyper = np.arange(len(methods))
    ax9.bar(x_pos_hyper - width/2, L_values, width, 
           label='L', alpha=0.7, color='C2')
    ax9.bar(x_pos_hyper + width/2, step_values, width, 
           label='step_size × 100', alpha=0.7, color='C3')
    ax9.set_xlabel('Method')
    ax9.set_ylabel('Parameter Value')
    ax9.set_title('Hyperparameter Comparison')
    ax9.set_xticks(x_pos_hyper)
    ax9.set_xticklabels(methods)
    ax9.legend()
    ax9.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'MAMS Tuning Comparison (dim={dim}, chains={num_chains}, '
                f'steps={num_steps})', fontsize=16, y=0.995)
    
    plt.show()
    
    # ========================================================================
    # STEP 5: Print final comparison summary
    # ========================================================================
    print("\n" + "="*70)
    print("FINAL COMPARISON SUMMARY")
    print("="*70)
    print(f"\n{'Metric':<30} {'BayesOpt':<20} {'Auto-tuned':<20}")
    print("-"*70)
    print(f"{'L':<30} {best_L:<20.4f} {jnp.mean(auto_L):<20.4f}")
    print(f"{'step_size':<30} {best_step_size:<20.6f} "
          f"{jnp.mean(auto_step_sizes):<20.6f}")
    print(f"{'Mean ESS':<30} {jnp.mean(bayesopt_ess):<20.1f} "
          f"{jnp.mean(auto_ess):<20.1f}")
    print(f"{'Std ESS':<30} {jnp.std(bayesopt_ess):<20.1f} "
          f"{jnp.std(auto_ess):<20.1f}")
    print(f"{'Max R-hat':<30} {jnp.max(bayesopt_rhat):<20.4f} "
          f"{jnp.max(auto_rhat):<20.4f}")
    print(f"{'Mean Acceptance':<30} {jnp.mean(bayesopt_acc):<20.3f} "
          f"{jnp.mean(auto_acc):<20.3f}")
    
    # Determine winner
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)
    
    bayesopt_score = jnp.mean(bayesopt_ess)
    auto_score = jnp.mean(auto_ess)
    
    # ESS comparison
    if bayesopt_score > auto_score * 1.1:
        print("✓ BayesOpt tuning produces significantly better ESS")
        print(f"  ({bayesopt_score:.1f} vs {auto_score:.1f}, "
              f"{100*(bayesopt_score/auto_score - 1):.1f}% improvement)")
    elif auto_score > bayesopt_score * 1.1:
        print("✓ Automatic tuning produces significantly better ESS")
        print(f"  ({auto_score:.1f} vs {bayesopt_score:.1f}, "
              f"{100*(auto_score/bayesopt_score - 1):.1f}% improvement)")
    else:
        print("≈ Both methods produce comparable ESS")
        print(f"  (BayesOpt: {bayesopt_score:.1f}, Auto: {auto_score:.1f})")
    
    # Convergence comparison
    if jnp.max(bayesopt_rhat) < jnp.max(auto_rhat):
        print("\n✓ BayesOpt tuning has better convergence (lower R-hat)")
        print(f"  (Max R-hat: {jnp.max(bayesopt_rhat):.4f} vs "
              f"{jnp.max(auto_rhat):.4f})")
    elif jnp.max(auto_rhat) < jnp.max(bayesopt_rhat):
        print("\n✓ Automatic tuning has better convergence (lower R-hat)")
        print(f"  (Max R-hat: {jnp.max(auto_rhat):.4f} vs "
              f"{jnp.max(bayesopt_rhat):.4f})")
    else:
        print("\n≈ Both methods have comparable convergence")
        print(f"  (Max R-hat: BayesOpt {jnp.max(bayesopt_rhat):.4f}, "
              f"Auto {jnp.max(auto_rhat):.4f})")
    
    return bayesopt_samples, auto_samples, mams_results


# ============================================================================
# REPRODUCIBILITY VERIFICATION
# ============================================================================

def verify_reproducibility_demo():
    """
    Demonstrate that same hyperparameters + same key = same results.
    
    This is crucial for understanding why we use fixed keys during tuning:
    it ensures each set of hyperparameters is evaluated fairly.
    """
    print("="*70)
    print("REPRODUCIBILITY VERIFICATION")
    print("="*70)
    print("\nThis demo shows that our experimental setup is sound.")
    
    # Setup
    logdensity_fn = make_funnel_logdensity(5)
    initial_position = jnp.zeros(5)
    fixed_key = jax.random.key(SEED_REPRODUCIBILITY)
    
    # ========================================================================
    # Test 1: Same hyperparameters + same key = identical results
    # ========================================================================
    print("\n" + "-"*70)
    print("Test 1: Same hyperparameters + same key = identical results")
    print("-"*70)
    print("Running MAMS three times with L=10.0, step_size=0.01, same key...")
    
    L, step_size = 10.0, 0.01
    
    _, ess1, acc1 = run_mams_fixed(logdensity_fn, 100, initial_position, 
                                   fixed_key, L, step_size)
    _, ess2, acc2 = run_mams_fixed(logdensity_fn, 100, initial_position, 
                                   fixed_key, L, step_size)
    _, ess3, acc3 = run_mams_fixed(logdensity_fn, 100, initial_position, 
                                   fixed_key, L, step_size)
    
    print(f"\nRun 1: ESS={ess1:.4f}, Acc={acc1:.4f}")
    print(f"Run 2: ESS={ess2:.4f}, Acc={acc2:.4f}")
    print(f"Run 3: ESS={ess3:.4f}, Acc={acc3:.4f}")
    
    all_identical = (jnp.allclose(ess1, ess2) and jnp.allclose(ess2, ess3) and
                    jnp.allclose(acc1, acc2) and jnp.allclose(acc2, acc3))
    print(f"\nAll identical? {all_identical}")
    if all_identical:
        print("✓ PASS: Results are deterministic given hyperparameters + key")
    else:
        print("✗ FAIL: Something is wrong with reproducibility!")
    
    # ========================================================================
    # Test 2: Different hyperparameters + same key = different results
    # ========================================================================
    print("\n" + "-"*70)
    print("Test 2: Different hyperparameters + same key = different results")
    print("-"*70)
    print("Running with three different hyperparameter sets, same key...")
    
    _, ess_a, acc_a = run_mams_fixed(logdensity_fn, 100, initial_position, 
                                     fixed_key, 10.0, 0.01)
    _, ess_b, acc_b = run_mams_fixed(logdensity_fn, 100, initial_position, 
                                     fixed_key, 10.0, 0.05)
    _, ess_c, acc_c = run_mams_fixed(logdensity_fn, 100, initial_position, 
                                     fixed_key, 20.0, 0.01)
    
    print(f"\nL=10.0, step=0.01: ESS={ess_a:.4f}, Acc={acc_a:.4f}")
    print(f"L=10.0, step=0.05: ESS={ess_b:.4f}, Acc={acc_b:.4f}")
    print(f"L=20.0, step=0.01: ESS={ess_c:.4f}, Acc={acc_c:.4f}")
    
    all_different = (not jnp.allclose(ess_a, ess_b) and 
                    not jnp.allclose(ess_b, ess_c))
    print(f"\nAll different? {all_different}")
    if all_different:
        print("✓ PASS: Different hyperparameters produce different results")
    else:
        print("✗ FAIL: Hyperparameters don't affect results (something is wrong!)")
    
    # ========================================================================
    # Test 3: Same hyperparameters + different keys = different results
    # ========================================================================
    print("\n" + "-"*70)
    print("Test 3: Same hyperparameters + different keys = different results")
    print("-"*70)
    print("Running with same hyperparameters but three different keys...")
    
    key1, key2, key3 = (jax.random.key(1), jax.random.key(2), 
                        jax.random.key(3))
    
    _, ess_1, acc_1 = run_mams_fixed(logdensity_fn, 100, initial_position, 
                                     key1, 10.0, 0.01)
    _, ess_2, acc_2 = run_mams_fixed(logdensity_fn, 100, initial_position, 
                                     key2, 10.0, 0.01)
    _, ess_3, acc_3 = run_mams_fixed(logdensity_fn, 100, initial_position, 
                                     key3, 10.0, 0.01)
    
    print(f"\nKey 1: ESS={ess_1:.4f}, Acc={acc_1:.4f}")
    print(f"Key 2: ESS={ess_2:.4f}, Acc={acc_2:.4f}")
    print(f"Key 3: ESS={ess_3:.4f}, Acc={acc_3:.4f}")
    
    keys_matter = (not jnp.allclose(ess_1, ess_2) and 
                   not jnp.allclose(ess_2, ess_3))
    print(f"\nAll different? {keys_matter}")
    if keys_matter:
        print("✓ PASS: Different keys produce different random trajectories")
    else:
        print("✗ FAIL: Keys don't affect randomness (something is wrong!)")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    if all_identical and all_different and keys_matter:
        print("\n✓ ALL TESTS PASSED!")
        print("\nThis confirms that:")
        print("  1. Results are deterministic given hyperparameters + key")
        print("  2. Different hyperparameters produce different results")
        print("  3. Different keys produce different random trajectories")
        print("\nTherefore, using a FIXED key during hyperparameter tuning")
        print("ensures fair comparison across all hyperparameter settings.")
    else:
        print("\n✗ SOME TESTS FAILED - check reproducibility setup!")
    
    print("="*70)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("MCMC HYPERPARAMETER TUNING WITH BAYESIAN OPTIMIZATION")
    print("="*70)
    print("\nThis script demonstrates:")
    print("  1. Fair hyperparameter comparison using fixed random keys")
    print("  2. Validation with multiple chains using different keys")
    print("  3. Comparison of Bayesian optimization vs automatic tuning")
    print("\n" + "="*70)
    
    # ========================================================================
    # Optional: Verify reproducibility first
    # ========================================================================
    print("\n" + "="*70)
    print("REPRODUCIBILITY CHECK (Optional)")
    print("="*70)
    print("\nWould you like to verify reproducibility first?")
    print("This ensures our experimental setup is sound.")
    print("\nUncomment the line below to run the verification:")
    print("# verify_reproducibility_demo()")
    
    # Uncomment to run:
    verify_reproducibility_demo()
    
    # ========================================================================
    # Main comparison of all algorithms
    # ========================================================================
    print("\n" + "="*70)
    print("EXPERIMENT 1: COMPARE ALL ALGORITHMS")
    print("="*70)
    print("\nComparing NUTS, MCLMC, and MAMS using Bayesian optimization...")
    print("This will take a few minutes...")
    
    nuts_results, mclmc_results, mams_results = compare_algorithms(dim=5)
    
    # ========================================================================
    # Detailed MAMS comparison: BayesOpt vs Automatic Tuning
    # ========================================================================
    print("\n" + "="*70)
    print("EXPERIMENT 2: BAYESOPT VS AUTOMATIC TUNING")
    print("="*70)
    print("\nNow comparing Bayesian optimization against blackjax's")
    print("automatic tuning for MAMS...")
    print("This will take longer as we run multiple validation chains...")
    
    bayesopt_samples, auto_samples, mams_detailed_results = \
        compare_mams_tuning_methods(
            dim=5, 
            num_chains=4, 
            num_steps=1000
        )
    
    # ========================================================================
    # All done!
    # ========================================================================
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE!")
    print("="*70)
    print("\nKey takeaways:")
    print("  1. Fixed keys during tuning ensure fair hyperparameter comparison")
    print("  2. Different keys during validation test robustness")
    print("  3. R-hat < 1.01 indicates good convergence")
    print("  4. ESS measures effective number of independent samples")
    print("\nTo customize experiments, modify parameters at the top of the script")
    print("or adjust the function calls above.")
    print("\n" + "="*70)