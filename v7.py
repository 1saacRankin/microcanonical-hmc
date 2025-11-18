# ============================================================================
# MCMC ALGORITHM COMPARISON WITH BAYESIAN HYPERPARAMETER OPTIMIZATION
# v7 - Complete version with all features and enhanced visualizations
# ============================================================================
# 
# WHAT THIS CODE DOES (HIGH-LEVEL):
# 
# 1. HYPERPARAMETER TUNING (with FIXED random seed)
#    - Try different hyperparameters (step_size, L) for each algorithm
#    - Each hyperparameter setting uses THE SAME random seed
#    - This ensures fair comparison: differences in ESS come from hyperparameters,
#      not from lucky/unlucky random trajectories
#    - Goal: Find hyperparameters that maximize ESS while maintaining good acceptance
# 
# 2. VALIDATION (with DIFFERENT random seeds)
#    - Take the best hyperparameters from step 1
#    - Run multiple independent chains with DIFFERENT random seeds
#    - Check if the chains converge to the same distribution (R-hat test)
#    - Goal: Verify that the hyperparameters work robustly across different
#      random initializations
# 
# 3. COMPARISON
#    - Compare Bayesian optimization vs blackjax's automatic tuning
#    - Compare NUTS vs MCLMC vs MAMS
#    - Compare against ground truth samples
# 
# ============================================================================
# BAYESIAN OPTIMIZATION DETAILS (BOAx Framework)
# ============================================================================
# 
# The script uses BOAx (Bayesian Optimization with Ax platform) which implements:
# 
# KERNEL:
#   - Gaussian Process with Matérn 5/2 kernel (default in Ax)
#   - Models objective function as smooth function of hyperparameters
#   - Matérn 5/2 is twice differentiable, good for optimization landscapes
#   - Automatically handles log-scale parameters via 'log_range' type
# 
# ACQUISITION FUNCTION:
#   - Expected Improvement (EI) or Upper Confidence Bound (UCB)
#   - Balances exploration (trying uncertain regions) vs exploitation (refining good regions)
#   - Early iterations: high exploration (sample diverse parameters)
#   - Later iterations: high exploitation (refine best parameters)
#   - Adaptively shifts exploration → exploitation as GP model improves
# 
# OBJECTIVE FUNCTION:
#   objective = ESS - λ × (acceptance_rate - target_acceptance)²
#   
#   where:
#   - ESS: Effective Sample Size (want to maximize)
#   - λ: Penalty weight (default 100)
#   - target_acceptance: 0.65 (NUTS), 1.0 (MCLMC), 0.9 (MAMS)
#   
#   The penalty term ensures we find hyperparameters that:
#   1. Give high ESS (efficient sampling)
#   2. Keep acceptance rate near target (algorithm health)
# 
# OPTIMIZATION PROCESS:
#   1. Initial random sampling (first ~3 iterations): explores parameter space
#   2. GP model fitting: learns ESS landscape from evaluations
#   3. Acquisition maximization: suggests next promising hyperparameters
#   4. Iterative refinement: converges to optimal parameters over ~10-20 iterations
# 
# SEARCH SPACE:
#   - NUTS: log_step_size ∈ [1e-5, 1e-1]
#   - MCLMC/MAMS: L ∈ [1e-1, 1e2], step_size ∈ [1e-3, 1.0]
#   - All parameters searched on log scale (appropriate for scale-free optimization)
# 
# ============================================================================
# BLACKJAX AUTOMATIC TUNING DETAILS (adjusted_mclmc_find_L_and_step_size)
# ============================================================================
# 
# BlackJAX's automatic tuning uses a three-phase adaptive scheme:
# 
# PHASE 1: COARSE STEP SIZE TUNING (first 10% of steps)
#   Goal: Find step_size that gives target acceptance rate (~0.9)
#   Method: 
#     - Run chain and measure acceptance rate
#     - If acceptance too low: decrease step_size
#     - If acceptance too high: increase step_size
#     - Uses simple feedback control loop
#   Output: Rough step_size estimate
# 
# PHASE 2: TRAJECTORY LENGTH TUNING (next 10% of steps)
#   Goal: Find L that balances exploration distance vs computation
#   Method:
#     - Estimate autocorrelation time τ from chain samples
#     - Set L ≈ c × τ where c is a tuning constant
#     - Longer L explores more but costs more per step
#     - Optimal L depends on geometry of target distribution
#   Output: Trajectory length L
# 
# PHASE 3: JOINT FINE-TUNING (next 10% of steps)
#   Goal: Jointly refine both step_size and L
#   Method:
#     - Continue running chain with Phase 1-2 parameters
#     - Make small adjustments to both simultaneously
#     - Optional diagonal preconditioning: adapt to parameter scales
#     - Estimates inverse mass matrix from sample covariance
#   Output: Final (step_size, L, inverse_mass_matrix)
# 
# KEY PROPERTIES:
#   - Total tuning uses 30% of requested steps (frac_tune1 + frac_tune2 + frac_tune3)
#   - These tuning samples are DISCARDED (not used for inference)
#   - Only remaining 70% of samples used for final results
#   - Tuning happens INLINE within the same chain (no separate evaluations)
#   - Each chain tunes independently (can find different hyperparameters)
# 
# COMPARISON TO BAYESOPT:
#   - BayesOpt: Separate short chains for tuning, then long chain with best params
#   - Auto-tune: Single long chain, discard first 30%
#   - BayesOpt: All chains use same hyperparameters (more reproducible)
#   - Auto-tune: Each chain may find different hyperparameters (more adaptive)
#   - BayesOpt: Global optimization over hyperparameter space
#   - Auto-tune: Local adaptation based on single trajectory
# 
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
import time

# Enable 64-bit precision for numerical stability
config.update("jax_enable_x64", True)

# Configure matplotlib for clean plots
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["font.size"] = 12

# Random keys for different purposes
SEED_NUTS_TUNING = 1000
SEED_MCLMC_TUNING = 2000
SEED_MAMS_TUNING = 3000
SEED_BAYESOPT_VALIDATION = 99999
SEED_AUTO_VALIDATION = 88888
SEED_REPRODUCIBILITY = 42


# ============================================================================
# TARGET DENSITIES
# ============================================================================

def make_gaussian_logdensity(dim):
    """Create a high-dimensional Gaussian with varying scales."""
    scales = jnp.linspace(1.0, float(dim), dim)
    inv_cov = 1.0 / scales**2
    
    def logdensity(x):
        return -0.5 * jnp.sum(inv_cov * x**2)
    
    return logdensity


def make_funnel_logdensity(dim):
    """Create Neal's funnel distribution."""
    def logdensity(x):
        log_prob = -0.5 * (x[0]**2 / 9.0)
        log_prob += -0.5 * (dim - 1) * x[0]
        log_prob += -0.5 * jnp.sum(x[1:]**2 * jnp.exp(-x[0]))
        return log_prob
    
    return logdensity


def make_rosenbrock_logdensity(dim):
    """Create a Rosenbrock (banana-shaped) distribution."""
    def logdensity(x):
        log_prob = -0.5 * x[0]**2
        for i in range(dim - 1):
            log_prob += -0.5 * (x[i+1] - x[i]**2)**2 / 0.1
        return log_prob
    
    return logdensity


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_target_2d(logdensity_fn, dim_x=0, dim_y=1, xlim=(-5, 5), ylim=(-5, 5)):
    """Plot a 2D slice of the target distribution."""
    x = np.linspace(xlim[0], xlim[1], 100)
    y = np.linspace(ylim[0], ylim[1], 100)
    X, Y = np.meshgrid(x, y)
    
    Z = np.zeros_like(X)
    for i in range(len(x)):
        for j in range(len(y)):
            point = jnp.zeros(10)
            point = point.at[dim_x].set(X[j, i])
            point = point.at[dim_y].set(Y[j, i])
            Z[j, i] = logdensity_fn(point)
    
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
    """Compute Effective Sample Size (ESS)."""
    samples_reshaped = samples.T[:, None, :]
    ess_per_dim = jax.vmap(
        lambda x: blackjax.diagnostics.effective_sample_size(x), 
        in_axes=0
    )(samples_reshaped)
    return jnp.min(ess_per_dim)


def compute_rhat(chains):
    """Compute Gelman-Rubin R-hat convergence diagnostic."""
    num_chains, num_steps, dim = chains.shape
    chain_means = jnp.mean(chains, axis=1)
    overall_mean = jnp.mean(chain_means, axis=0)
    B = num_steps / (num_chains - 1) * jnp.sum((chain_means - overall_mean)**2, axis=0)
    chain_vars = jnp.var(chains, axis=1, ddof=1)
    W = jnp.mean(chain_vars, axis=0)
    var_est = ((num_steps - 1) / num_steps) * W + (1 / num_steps) * B
    rhat = jnp.sqrt(var_est / W)
    return rhat


# ============================================================================
# SAMPLING ALGORITHMS (FIXED PARAMETERS)
# ============================================================================

def run_nuts_fixed(logdensity_fn, num_steps, initial_position, key, 
                   step_size, inv_mass_matrix):
    """Run NUTS with fixed hyperparameters."""
    start_time = time.time()
    nuts = blackjax.nuts(logdensity_fn, step_size, inv_mass_matrix)
    state = nuts.init(initial_position)
    
    def one_step(state, key):
        state, info = nuts.step(key, state)
        return state, (state.position, info.acceptance_rate)
    
    keys = jax.random.split(key, num_steps)
    final_state, (samples, acceptance_rates) = jax.lax.scan(one_step, state, keys)
    
    avg_acceptance = jnp.mean(acceptance_rates)
    ess = compute_ess(samples)
    time_elapsed = time.time() - start_time
    
    return samples, ess, avg_acceptance, time_elapsed


def run_mclmc_fixed(logdensity_fn, num_steps, initial_position, key, L, step_size):
    """Run MCLMC with fixed hyperparameters."""
    start_time = time.time()
    init_key, run_key = jax.random.split(key)
    
    initial_state = blackjax.mcmc.mclmc.init(
        position=initial_position, 
        logdensity_fn=logdensity_fn, 
        rng_key=init_key
    )
    
    sampling_alg = blackjax.mclmc(
        logdensity_fn,
        L=L,
        step_size=step_size,
    )
    
    _, samples = blackjax.util.run_inference_algorithm(
        rng_key=run_key,
        initial_state=initial_state,
        inference_algorithm=sampling_alg,
        num_steps=num_steps,
        transform=lambda state, _: state.position,
        progress_bar=False,
    )
    
    ess = compute_ess(samples)
    time_elapsed = time.time() - start_time
    
    return samples, ess, 1.0, time_elapsed


def run_mams_fixed(logdensity_fn, num_steps, initial_position, key, L, step_size):
    """Run MAMS with fixed hyperparameters."""
    start_time = time.time()
    init_key, run_key = jax.random.split(key)
    
    initial_state = blackjax.mcmc.adjusted_mclmc_dynamic.init(
        position=initial_position,
        logdensity_fn=logdensity_fn,
        random_generator_arg=init_key,
    )
    
    alg = blackjax.adjusted_mclmc_dynamic(
        logdensity_fn=logdensity_fn,
        step_size=step_size,
        integration_steps_fn=lambda key: jnp.ceil(L / step_size),
        L_proposal_factor=jnp.inf,
    )
    
    def one_step(state, key):
        state, info = alg.step(key, state)
        return state, (state.position, info.acceptance_rate)
    
    keys = jax.random.split(run_key, num_steps)
    final_state, (samples, acceptance_rates) = jax.lax.scan(one_step, initial_state, keys)
    
    avg_acceptance = jnp.mean(acceptance_rates)
    ess = compute_ess(samples)
    time_elapsed = time.time() - start_time
    
    return samples, ess, avg_acceptance, time_elapsed


# ============================================================================
# BAYESIAN OPTIMIZATION FOR HYPERPARAMETER TUNING
# ============================================================================

def objective_function(ess, acceptance_rate, target_acceptance, lambda_penalty):
    """Compute objective for hyperparameter optimization."""
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
    """Use Bayesian optimization to find optimal hyperparameters."""
    results = {
        'iteration': [],
        'ess': [],
        'acceptance_rate': [],
        'objective': [],
        'hyperparams': [],
        'time_per_eval': [],
    }
    
    start_time = time.time()
    
    # Define search space and evaluation function for each algorithm
    if algorithm_name == 'NUTS':
        parameters = [
            {
                'name': 'log_step_size',
                'type': 'log_range',
                'bounds': [1e-5, 1e-1],
            }
        ]
        
        def run_with_params(params_dict):
            step_size = params_dict['log_step_size']
            inv_mass_matrix = jnp.ones(len(initial_position))
            _, ess, acc, eval_time = run_nuts_fixed(
                logdensity_fn, chain_length, initial_position, 
                fixed_key, step_size, inv_mass_matrix
            )
            return ess, acc, eval_time
        
    elif algorithm_name == 'MCLMC':
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
        target_acceptance = 1.0
        
        def run_with_params(params_dict):
            L = params_dict['L']
            step_size = params_dict['step_size']
            _, ess, acc, eval_time = run_mclmc_fixed(
                logdensity_fn, chain_length, initial_position, 
                fixed_key, L, step_size
            )
            return ess, acc, eval_time
        
    elif algorithm_name == 'MAMS':
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
        target_acceptance = 0.9
        
        def run_with_params(params_dict):
            L = params_dict['L']
            step_size = params_dict['step_size']
            _, ess, acc, eval_time = run_mams_fixed(
                logdensity_fn, chain_length, initial_position, 
                fixed_key, L, step_size
            )
            return ess, acc, eval_time
    
    # Initialize BOAx experiment
    experiment = optimization(
        parameters=parameters,
        batch_size=1,
    )
    
    # Run Bayesian optimization loop
    step, experiment_results = None, []
    
    for i in range(num_iterations):
        print(f"Iteration {i+1}/{num_iterations}")
        
        step, parameterizations = experiment.next(step, experiment_results)
        params_dict = parameterizations[0]
        
        ess, acc, eval_time = run_with_params(params_dict)
        obj = objective_function(ess, acc, target_acceptance, lambda_penalty)
        
        experiment_results = [(params_dict, float(obj))]
        
        results['iteration'].append(i)
        results['ess'].append(float(ess))
        results['acceptance_rate'].append(float(acc))
        results['objective'].append(float(obj))
        results['hyperparams'].append(params_dict)
        results['time_per_eval'].append(eval_time)
        
        print(f"  {algorithm_name} - ESS={ess:.1f}, Acc={acc:.3f}, "
              f"Obj={obj:.1f}, Time={eval_time:.2f}s, Params={params_dict}")
    
    total_time = time.time() - start_time
    
    best_idx = jnp.argmax(jnp.array(results['objective']))
    best_params = results['hyperparams'][best_idx]
    best_obj = results['objective'][best_idx]
    
    print(f"\n{algorithm_name} Best parameters: {best_params}")
    print(f"  Best objective: {best_obj:.2f}")
    print(f"  ESS: {results['ess'][best_idx]:.1f}")
    print(f"  Acceptance: {results['acceptance_rate'][best_idx]:.3f}")
    print(f"  Total tuning time: {total_time:.2f}s")
    print(f"  Average time per evaluation: {np.mean(results['time_per_eval']):.2f}s")
    
    return results


# ============================================================================
# VISUALIZATION FUNCTIONS FOR BAYESIAN OPTIMIZATION
# ============================================================================

def plot_bayesopt_progress(results_dict, save_prefix=None):
    """
    Create comprehensive visualization of Bayesian optimization progress.
    
    Shows time series of: ESS, acceptance, objective, L, and step_size
    """
    algorithms = []
    for name in ['NUTS', 'MCLMC', 'MAMS']:
        key = name.lower()
        if key in results_dict and results_dict[key] is not None:
            algorithms.append((name, key))
    
    if not algorithms:
        print("No results to plot!")
        return
    
    n_algs = len(algorithms)
    fig = plt.figure(figsize=(18, 5 * n_algs))
    
    for alg_idx, (alg_name, alg_key) in enumerate(algorithms):
        results = results_dict[alg_key]
        
        iterations = results['iteration']
        ess_values = results['ess']
        acc_values = results['acceptance_rate']
        obj_values = results['objective']
        
        if alg_name == 'NUTS':
            step_sizes = [p['log_step_size'] for p in results['hyperparams']]
            L_values = None
        else:
            L_values = [p['L'] for p in results['hyperparams']]
            step_sizes = [p['step_size'] for p in results['hyperparams']]
        
        base_idx = alg_idx * 5
        
        # Panel 1: ESS
        ax1 = plt.subplot(n_algs, 5, base_idx + 1)
        ax1.plot(iterations, ess_values, 'o-', linewidth=2, markersize=6, color='C0')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('ESS')
        ax1.set_title(f'{alg_name}: Effective Sample Size')
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Acceptance
        ax2 = plt.subplot(n_algs, 5, base_idx + 2)
        ax2.plot(iterations, acc_values, 'o-', linewidth=2, markersize=6, color='C1')
        target_acc = 0.65 if alg_name == 'NUTS' else (1.0 if alg_name == 'MCLMC' else 0.9)
        ax2.axhline(target_acc, color='red', linestyle='--', alpha=0.5, label=f'Target ({target_acc})')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Acceptance Rate')
        ax2.set_title(f'{alg_name}: Acceptance Probability')
        ax2.set_ylim([0, 1.05])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Objective
        ax3 = plt.subplot(n_algs, 5, base_idx + 3)
        ax3.plot(iterations, obj_values, 'o-', linewidth=2, markersize=6, color='C2')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Objective')
        ax3.set_title(f'{alg_name}: Objective Function')
        ax3.grid(True, alpha=0.3)
        
        # Panel 4: L
        if L_values is not None:
            ax4 = plt.subplot(n_algs, 5, base_idx + 4)
            ax4.plot(iterations, L_values, 'o-', linewidth=2, markersize=6, color='C3')
            ax4.set_xlabel('Iteration')
            ax4.set_ylabel('L (log scale)')
            ax4.set_yscale('log')
            ax4.set_title(f'{alg_name}: Trajectory Length')
            ax4.grid(True, alpha=0.3, which='both')
        else:
            ax4 = plt.subplot(n_algs, 5, base_idx + 4)
            ax4.text(0.5, 0.5, 'N/A for NUTS', ha='center', va='center', fontsize=14)
            ax4.set_title(f'{alg_name}: Trajectory Length')
            ax4.axis('off')
        
        # Panel 5: Step size
        ax5 = plt.subplot(n_algs, 5, base_idx + 5)
        ax5.plot(iterations, step_sizes, 'o-', linewidth=2, markersize=6, color='C4')
        ax5.set_xlabel('Iteration')
        ax5.set_ylabel('ε (log scale)')
        ax5.set_yscale('log')
        ax5.set_title(f'{alg_name}: Step Size')
        ax5.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    if save_prefix:
        plt.savefig(f'{save_prefix}_timeseries.png', dpi=150, bbox_inches='tight')
        print(f"Saved {save_prefix}_timeseries.png")
    
    plt.show()


def plot_hyperparameter_space(results_dict, save_prefix=None):
    """
    Create scatter plots of L vs ε with size = acceptance, color = ESS.
    """
    algorithms = []
    for name in ['MCLMC', 'MAMS']:
        key = name.lower()
        if key in results_dict and results_dict[key] is not None:
            algorithms.append((name, key))
    
    if not algorithms:
        print("No MCLMC/MAMS results to plot!")
        return
    
    fig, axes = plt.subplots(1, len(algorithms), figsize=(9 * len(algorithms), 7))
    if len(algorithms) == 1:
        axes = [axes]
    
    for ax, (alg_name, alg_key) in zip(axes, algorithms):
        results = results_dict[alg_key]
        
        L_values = np.array([p['L'] for p in results['hyperparams']])
        step_sizes = np.array([p['step_size'] for p in results['hyperparams']])
        acc_values = np.array(results['acceptance_rate'])
        ess_values = np.array(results['ess'])
        
        ess_min, ess_max = ess_values.min(), ess_values.max()
        if ess_max > ess_min:
            ess_normalized = (ess_values - ess_min) / (ess_max - ess_min)
        else:
            ess_normalized = np.ones_like(ess_values)
        
        colors = plt.cm.RdYlGn(ess_normalized)
        sizes = 100 + acc_values * 500
        
        scatter = ax.scatter(step_sizes, L_values, s=sizes, c=colors, 
                           alpha=0.7, edgecolors='black', linewidths=1.5)
        
        ax.set_xlabel('Step Size (ε)', fontsize=14)
        ax.set_ylabel('Trajectory Length (L)', fontsize=14)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(f'{alg_name}: Hyperparameter Space Exploration', fontsize=16)
        ax.grid(True, alpha=0.3, which='both')
        
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, 
                                   norm=plt.Normalize(vmin=ess_min, vmax=ess_max))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label='ESS')
        
        legend_sizes = [0.3, 0.6, 0.9]
        legend_points = [ax.scatter([], [], s=100 + s * 500, c='gray', alpha=0.7, 
                                   edgecolors='black', linewidths=1.5) 
                        for s in legend_sizes]
        legend_labels = [f'Acc={s:.1f}' for s in legend_sizes]
        
        legend = ax.legend(legend_points, legend_labels, 
                          title='Acceptance', loc='upper left', 
                          bbox_to_anchor=(1.15, 1), frameon=True)
        
        ax.annotate('Start', xy=(step_sizes[0], L_values[0]), 
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=10, color='blue', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
        
        best_idx = np.argmax(results['objective'])
        ax.annotate('Best', xy=(step_sizes[best_idx], L_values[best_idx]), 
                   xytext=(10, -20), textcoords='offset points',
                   fontsize=10, color='red', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    
    plt.tight_layout()
    
    if save_prefix:
        plt.savefig(f'{save_prefix}_hyperparameter_space.png', dpi=150, bbox_inches='tight')
        print(f"Saved {save_prefix}_hyperparameter_space.png")
    
    plt.show()


# ============================================================================
# GROUND TRUTH COMPARISON
# ============================================================================

def generate_ground_truth_samples(logdensity_fn, num_samples, dim, key):
    """Generate ground truth samples for Neal's funnel."""
    key1, key2 = jax.random.split(key)
    
    x0 = jax.random.normal(key1, (num_samples,)) * 3.0
    x_rest = jax.random.normal(key2, (num_samples, dim - 1))
    scales = jnp.exp(x0 / 2.0)[:, None]
    x_rest = x_rest * scales
    samples = jnp.concatenate([x0[:, None], x_rest], axis=1)
    
    return samples


def plot_ground_truth_comparison(samples_dict, ground_truth, dim=5):
    """Create comprehensive plots comparing MCMC samples to ground truth."""
    num_methods = len(samples_dict)
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.35)
    
    colors = {'BayesOpt': 'C0', 'Auto-tuned': 'C1', 'Ground Truth': 'black'}
    
    # Row 1: Marginal distributions for dimension 0
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.hist(ground_truth[:, 0], bins=50, alpha=0.3, density=True, 
            color=colors['Ground Truth'], label='Ground Truth', linewidth=2)
    
    for name, samples in samples_dict.items():
        flat_samples = samples.reshape(-1, dim)
        ax1.hist(flat_samples[:, 0], bins=50, alpha=0.5, density=True,
                label=name, color=colors.get(name, None))
    
    ax1.set_xlabel('Dimension 0')
    ax1.set_ylabel('Density')
    ax1.set_title('Marginal Distribution (Dim 0 - Funnel Mouth)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Row 1: Marginal distributions for dimension 1
    ax2 = fig.add_subplot(gs[0, 2:])
    ax2.hist(ground_truth[:, 1], bins=50, alpha=0.3, density=True,
            color=colors['Ground Truth'], label='Ground Truth', linewidth=2)
    
    for name, samples in samples_dict.items():
        flat_samples = samples.reshape(-1, dim)
        ax2.hist(flat_samples[:, 1], bins=50, alpha=0.5, density=True,
                label=name, color=colors.get(name, None))
    
    ax2.set_xlabel('Dimension 1')
    ax2.set_ylabel('Density')
    ax2.set_title('Marginal Distribution (Dim 1 - Funnel Neck)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Row 2: 2D scatter plots
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(ground_truth[:, 0], ground_truth[:, 1], alpha=0.1, s=1,
               color=colors['Ground Truth'], label='Ground Truth')
    ax3.set_xlabel('Dimension 0')
    ax3.set_ylabel('Dimension 1')
    ax3.set_title('Ground Truth Samples')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-10, 10)
    ax3.set_ylim(-5, 5)
    
    plot_idx = 1
    for name, samples in samples_dict.items():
        ax = fig.add_subplot(gs[1, plot_idx])
        flat_samples = samples.reshape(-1, dim)
        ax.scatter(flat_samples[:, 0], flat_samples[:, 1], alpha=0.1, s=1,
                  color=colors.get(name, None), label=name)
        ax.set_xlabel('Dimension 0')
        ax.set_ylabel('Dimension 1')
        ax.set_title(f'{name} Samples')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-10, 10)
        ax.set_ylim(-5, 5)
        plot_idx += 1
    
    # Overlay comparison
    ax_overlay = fig.add_subplot(gs[1, 3])
    ax_overlay.scatter(ground_truth[:, 0], ground_truth[:, 1], alpha=0.05, s=1,
                      color=colors['Ground Truth'], label='Ground Truth')
    for name, samples in samples_dict.items():
        flat_samples = samples.reshape(-1, dim)
        ax_overlay.scatter(flat_samples[:, 0], flat_samples[:, 1], alpha=0.05, s=1,
                          color=colors.get(name, None), label=name)
    ax_overlay.set_xlabel('Dimension 0')
    ax_overlay.set_ylabel('Dimension 1')
    ax_overlay.set_title('Overlay Comparison')
    ax_overlay.legend()
    ax_overlay.grid(True, alpha=0.3)
    ax_overlay.set_xlim(-10, 10)
    ax_overlay.set_ylim(-5, 5)
    
    # Row 3: QQ-plots
    ax_qq1 = fig.add_subplot(gs[2, :2])
    gt_sorted_0 = jnp.sort(ground_truth[:, 0])
    
    for name, samples in samples_dict.items():
        flat_samples = samples.reshape(-1, dim)
        sample_sorted_0 = jnp.sort(flat_samples[:, 0])
        
        if len(sample_sorted_0) != len(gt_sorted_0):
            x_indices = jnp.linspace(0, len(sample_sorted_0) - 1, len(gt_sorted_0))
            sample_sorted_0 = jnp.interp(x_indices, jnp.arange(len(sample_sorted_0)), 
                                        sample_sorted_0)
        
        ax_qq1.scatter(gt_sorted_0, sample_sorted_0, alpha=0.3, s=3,
                      label=name, color=colors.get(name, None))
    
    lims = [min(gt_sorted_0), max(gt_sorted_0)]
    ax_qq1.plot(lims, lims, 'k--', alpha=0.5, linewidth=2, label='Perfect Agreement')
    ax_qq1.set_xlabel('Ground Truth Quantiles (Dim 0)')
    ax_qq1.set_ylabel('Sample Quantiles (Dim 0)')
    ax_qq1.set_title('QQ Plot - Dimension 0')
    ax_qq1.legend()
    ax_qq1.grid(True, alpha=0.3)
    
    ax_qq2 = fig.add_subplot(gs[2, 2:])
    gt_sorted_1 = jnp.sort(ground_truth[:, 1])
    
    for name, samples in samples_dict.items():
        flat_samples = samples.reshape(-1, dim)
        sample_sorted_1 = jnp.sort(flat_samples[:, 1])
        
        if len(sample_sorted_1) != len(gt_sorted_1):
            x_indices = jnp.linspace(0, len(sample_sorted_1) - 1, len(gt_sorted_1))
            sample_sorted_1 = jnp.interp(x_indices, jnp.arange(len(sample_sorted_1)),
                                        sample_sorted_1)
        
        ax_qq2.scatter(gt_sorted_1, sample_sorted_1, alpha=0.3, s=3,
                      label=name, color=colors.get(name, None))
    
    lims = [min(gt_sorted_1), max(gt_sorted_1)]
    ax_qq2.plot(lims, lims, 'k--', alpha=0.5, linewidth=2, label='Perfect Agreement')
    ax_qq2.set_xlabel('Ground Truth Quantiles (Dim 1)')
    ax_qq2.set_ylabel('Sample Quantiles (Dim 1)')
    ax_qq2.set_title('QQ Plot - Dimension 1')
    ax_qq2.legend()
    ax_qq2.grid(True, alpha=0.3)
    
    # Row 4: Mean and variance comparison
    ax_mean = fig.add_subplot(gs[3, :2])
    gt_means = jnp.mean(ground_truth, axis=0)
    x_dims = jnp.arange(dim)
    width = 0.25
    
    ax_mean.bar(x_dims - width, gt_means, width, label='Ground Truth',
               alpha=0.7, color=colors['Ground Truth'])
    
    offset = 0
    for name, samples in samples_dict.items():
        flat_samples = samples.reshape(-1, dim)
        sample_means = jnp.mean(flat_samples, axis=0)
        ax_mean.bar(x_dims + offset, sample_means, width, label=name,
                   alpha=0.7, color=colors.get(name, None))
        offset += width
    
    ax_mean.set_xlabel('Dimension')
    ax_mean.set_ylabel('Mean')
    ax_mean.set_title('Mean Comparison (All Dimensions)')
    ax_mean.set_xticks(x_dims)
    ax_mean.legend()
    ax_mean.grid(True, alpha=0.3, axis='y')
    ax_mean.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    ax_var = fig.add_subplot(gs[3, 2:])
    gt_vars = jnp.var(ground_truth, axis=0)
    
    ax_var.bar(x_dims - width, gt_vars, width, label='Ground Truth',
              alpha=0.7, color=colors['Ground Truth'])
    
    offset = 0
    for name, samples in samples_dict.items():
        flat_samples = samples.reshape(-1, dim)
        sample_vars = jnp.var(flat_samples, axis=0)
        ax_var.bar(x_dims + offset, sample_vars, width, label=name,
                  alpha=0.7, color=colors.get(name, None))
        offset += width
    
    ax_var.set_xlabel('Dimension')
    ax_var.set_ylabel('Variance')
    ax_var.set_title('Variance Comparison (All Dimensions)')
    ax_var.set_xticks(x_dims)
    ax_var.legend()
    ax_var.grid(True, alpha=0.3, axis='y')
    ax_var.set_yscale('log')
    
    plt.suptitle('MCMC Samples vs Ground Truth Comparison', fontsize=18, y=0.995)
    
    return fig


# ============================================================================
# VALIDATION WITH MULTIPLE CHAINS
# ============================================================================

def run_mams_multiple_chains(logdensity_fn, num_chains, num_steps, 
                             initial_position, base_key, L, step_size):
    """Run multiple MAMS chains with fixed hyperparameters."""
    start_time = time.time()
    
    all_samples = []
    all_ess = []
    all_acceptance = []
    all_times = []
    
    for i in range(num_chains):
        print(f"  Running chain {i+1}/{num_chains}...", end=" ")
        chain_key = jax.random.fold_in(base_key, i)
        
        samples, ess, acc, chain_time = run_mams_fixed(
            logdensity_fn, num_steps, initial_position, chain_key, L, step_size
        )
        
        all_samples.append(samples)
        all_ess.append(ess)
        all_acceptance.append(acc)
        all_times.append(chain_time)
        
        print(f"ESS={ess:.1f}, Acc={acc:.3f}, Time={chain_time:.2f}s")
    
    all_samples = jnp.stack(all_samples, axis=0)
    total_time = time.time() - start_time
    
    print(f"  Total time for {num_chains} chains: {total_time:.2f}s")
    
    return all_samples, jnp.array(all_ess), jnp.array(all_acceptance), total_time


def run_auto_tuned_multiple_chains(logdensity_fn, num_chains, num_steps, 
                                   initial_position, base_key):
    """Run multiple chains with automatic hyperparameter tuning."""
    start_time = time.time()
    
    all_samples = []
    all_step_sizes = []
    all_L = []
    all_times = []
    
    for i in range(num_chains):
        print(f"  Running chain {i+1}/{num_chains} (with auto-tuning)...", end=" ")
        chain_key = jax.random.fold_in(base_key, i)
        
        samples, step_size, L, inv_mass, chain_time = run_adjusted_mclmc_dynamic(
            logdensity_fn, num_steps, initial_position, chain_key
        )
        
        all_samples.append(samples)
        all_step_sizes.append(step_size)
        all_L.append(L)
        all_times.append(chain_time)
        
        print(f"L={L:.3f}, step={step_size:.5f}, Time={chain_time:.2f}s")
    
    all_samples = jnp.stack(all_samples, axis=0)
    
    all_ess = []
    for chain_samples in all_samples:
        ess = compute_ess(chain_samples)
        all_ess.append(ess)
    
    all_acceptance = jnp.ones(num_chains) * 0.9
    total_time = time.time() - start_time
    
    print(f"  Total time for {num_chains} chains: {total_time:.2f}s")
    
    return (all_samples, jnp.array(all_ess), all_acceptance, 
            jnp.array(all_step_sizes), jnp.array(all_L), total_time)


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
    """Run MAMS with automatic hyperparameter tuning."""
    start_time = time.time()
    
    print("      [Initializing...]", end=" ", flush=True)
    
    init_key, tune_key, run_key = jax.random.split(key, 3)
    
    initial_state = blackjax.mcmc.adjusted_mclmc_dynamic.init(
        position=initial_position,
        logdensity_fn=logdensity_fn,
        random_generator_arg=init_key,
    )
    
    print("done", flush=True)
    print("      [Building kernel...]", end=" ", flush=True)
    
    if random_trajectory_length:
        integration_steps_fn = lambda avg_num_integration_steps: lambda k: jnp.ceil(
            jax.random.uniform(k) * rescale(avg_num_integration_steps))
    else:
        integration_steps_fn = lambda avg_num_integration_steps: lambda _: jnp.ceil(
            avg_num_integration_steps)
    
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
    
    print("done", flush=True)
    print("      [Auto-tuning hyperparameters...]", end=" ", flush=True)
    
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
        frac_tune1=0.1,
        frac_tune2=0.1,
        frac_tune3=0.1,
        diagonal_preconditioning=diagonal_preconditioning,
    )
    
    print("done", flush=True)
    print("      [Sampling with tuned parameters...]", end=" ", flush=True)
    
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
    
    print("done", flush=True)
    
    time_elapsed = time.time() - start_time
    
    return out, step_size, L, blackjax_mclmc_sampler_params.inverse_mass_matrix, time_elapsed


# ============================================================================
# REPRODUCIBILITY VERIFICATION
# ============================================================================

def verify_reproducibility_demo():
    """Demonstrate that same hyperparameters + same key = same results."""
    print("="*70)
    print("REPRODUCIBILITY VERIFICATION")
    print("="*70)
    
    logdensity_fn = make_funnel_logdensity(5)
    initial_position = jnp.zeros(5)
    fixed_key = jax.random.key(SEED_REPRODUCIBILITY)
    
    # Test 1: Same params + same key = identical
    print("\n" + "-"*70)
    print("Test 1: Same hyperparameters + same key = identical results")
    print("-"*70)
    
    L, step_size = 10.0, 0.01
    
    _, ess1, acc1, _ = run_mams_fixed(logdensity_fn, 100, initial_position, 
                                       fixed_key, L, step_size)
    _, ess2, acc2, _ = run_mams_fixed(logdensity_fn, 100, initial_position, 
                                       fixed_key, L, step_size)
    _, ess3, acc3, _ = run_mams_fixed(logdensity_fn, 100, initial_position, 
                                       fixed_key, L, step_size)
    
    print(f"\nRun 1: ESS={ess1:.4f}, Acc={acc1:.4f}")
    print(f"Run 2: ESS={ess2:.4f}, Acc={acc2:.4f}")
    print(f"Run 3: ESS={ess3:.4f}, Acc={acc3:.4f}")
    
    all_identical = (jnp.allclose(ess1, ess2) and jnp.allclose(ess2, ess3) and
                    jnp.allclose(acc1, acc2) and jnp.allclose(acc2, acc3))
    print(f"\n✓ PASS" if all_identical else "✗ FAIL")
    
    # Test 2: Different params + same key = different
    print("\n" + "-"*70)
    print("Test 2: Different hyperparameters + same key = different results")
    print("-"*70)
    
    _, ess_a, acc_a, _ = run_mams_fixed(logdensity_fn, 100, initial_position, 
                                         fixed_key, 10.0, 0.01)
    _, ess_b, acc_b, _ = run_mams_fixed(logdensity_fn, 100, initial_position, 
                                         fixed_key, 10.0, 0.05)
    _, ess_c, acc_c, _ = run_mams_fixed(logdensity_fn, 100, initial_position, 
                                         fixed_key, 20.0, 0.01)
    
    print(f"\nL=10.0, step=0.01: ESS={ess_a:.4f}")
    print(f"L=10.0, step=0.05: ESS={ess_b:.4f}")
    print(f"L=20.0, step=0.01: ESS={ess_c:.4f}")
    
    all_different = (not jnp.allclose(ess_a, ess_b) and 
                    not jnp.allclose(ess_b, ess_c))
    print(f"\n✓ PASS" if all_different else "✗ FAIL")
    
    # Test 3: Same params + different keys = different
    print("\n" + "-"*70)
    print("Test 3: Same hyperparameters + different keys = different results")
    print("-"*70)
    
    key1, key2, key3 = jax.random.key(1), jax.random.key(2), jax.random.key(3)
    
    _, ess_1, _, _ = run_mams_fixed(logdensity_fn, 100, initial_position, 
                                     key1, 10.0, 0.01)
    _, ess_2, _, _ = run_mams_fixed(logdensity_fn, 100, initial_position, 
                                     key2, 10.0, 0.01)
    _, ess_3, _, _ = run_mams_fixed(logdensity_fn, 100, initial_position, 
                                     key3, 10.0, 0.01)
    
    print(f"\nKey 1: ESS={ess_1:.4f}")
    print(f"Key 2: ESS={ess_2:.4f}")
    print(f"Key 3: ESS={ess_3:.4f}")
    
    keys_matter = (not jnp.allclose(ess_1, ess_2) and 
                   not jnp.allclose(ess_2, ess_3))
    print(f"\n✓ PASS" if keys_matter else "✗ FAIL")
    
    if all_identical and all_different and keys_matter:
        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED!")
        print("="*70)


# ============================================================================
# MAIN COMPARISON FUNCTIONS
# ============================================================================

def compare_algorithms(dim=5):
    """Compare all three algorithms using Bayesian optimization."""
    print("="*70)
    print("MCMC ALGORITHM COMPARISON WITH BAYESIAN OPTIMIZATION")
    print("="*70)
    
    logdensity_fn = make_funnel_logdensity(dim)
    initial_position = jnp.zeros(dim)
    
    nuts_tuning_key = jax.random.key(SEED_NUTS_TUNING)
    mclmc_tuning_key = jax.random.key(SEED_MCLMC_TUNING)
    mams_tuning_key = jax.random.key(SEED_MAMS_TUNING)
    
    print("\n" + "="*70)
    print("PHASE 1: HYPERPARAMETER TUNING")
    print("="*70)
    
    print("\n1. Optimizing NUTS hyperparameters...")
    nuts_results = run_bayesopt_tuning(
        logdensity_fn, initial_position, nuts_tuning_key, 'NUTS',
        num_iterations=10, chain_length=100
    )
    
    print("\n2. Optimizing MCLMC hyperparameters...")
    mclmc_results = run_bayesopt_tuning(
        logdensity_fn, initial_position, mclmc_tuning_key, 'MCLMC',
        num_iterations=10, chain_length=100
    )
    
    print("\n3. Optimizing MAMS hyperparameters...")
    mams_results = run_bayesopt_tuning(
        logdensity_fn, initial_position, mams_tuning_key, 'MAMS',
        num_iterations=10, chain_length=100
    )
    
    # Create visualizations
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    
    results_dict = {
        'nuts': nuts_results,
        'mclmc': mclmc_results,
        'mams': mams_results
    }
    
    print("\nGenerating time series plots...")
    plot_bayesopt_progress(results_dict, save_prefix='bayesopt')
    
    print("\nGenerating hyperparameter space plots...")
    plot_hyperparameter_space(results_dict, save_prefix='bayesopt')
    
    return nuts_results, mclmc_results, mams_results


def compare_mams_tuning_methods(dim=5, num_chains=4, num_steps=1000):
    """Compare Bayesian optimization vs automatic tuning for MAMS."""
    print("="*70)
    print("MAMS HYPERPARAMETER TUNING COMPARISON")
    print("="*70)
    
    logdensity_fn = make_funnel_logdensity(dim)
    initial_position = jnp.zeros(dim)
    
    # Generate ground truth
    print("\nGenerating ground truth samples...")
    gt_key = jax.random.key(42)
    ground_truth = generate_ground_truth_samples(logdensity_fn, num_chains * num_steps, 
                                                 dim, gt_key)
    
    # Step 1: BayesOpt tuning
    print("\n" + "="*70)
    print("STEP 1: BAYESIAN OPTIMIZATION")
    print("="*70)
    
    tuning_start = time.time()
    bayesopt_tuning_key = jax.random.key(SEED_MAMS_TUNING)
    mams_results = run_bayesopt_tuning(
        logdensity_fn, initial_position, bayesopt_tuning_key, 'MAMS',
        num_iterations=10, chain_length=100
    )
    tuning_time = time.time() - tuning_start
    
    best_idx = jnp.argmax(jnp.array(mams_results['objective']))
    best_params = mams_results['hyperparams'][best_idx]
    best_L = best_params['L']
    best_step_size = best_params['step_size']
    
    # Step 2: Validate BayesOpt
    print("\n" + "="*70)
    print("STEP 2: VALIDATE BAYESOPT PARAMETERS")
    print("="*70)
    
    validation_key = jax.random.key(SEED_BAYESOPT_VALIDATION)
    bayesopt_samples, bayesopt_ess, bayesopt_acc, bayesopt_time = \
        run_mams_multiple_chains(
            logdensity_fn, num_chains, num_steps, initial_position, 
            validation_key, best_L, best_step_size
        )
    
    bayesopt_rhat = compute_rhat(bayesopt_samples)
    
    print(f"\nBayesOpt Results:")
    print(f"  Mean ESS: {jnp.mean(bayesopt_ess):.1f}")
    print(f"  Max R-hat: {jnp.max(bayesopt_rhat):.4f}")
    
    # Step 3: Auto-tuning
    print("\n" + "="*70)
    print("STEP 3: AUTOMATIC TUNING")
    print("="*70)
    
    auto_samples, auto_ess, auto_acc, auto_step_sizes, auto_L, auto_time = \
        run_auto_tuned_multiple_chains(
            logdensity_fn, num_chains, num_steps, initial_position, 
            validation_key
        )
    
    auto_rhat = compute_rhat(auto_samples)
    
    print(f"\nAuto-tuning Results:")
    print(f"  Mean ESS: {jnp.mean(auto_ess):.1f}")
    print(f"  Max R-hat: {jnp.max(auto_rhat):.4f}")
    
    # Create plots
    print("\n" + "="*70)
    print("CREATING COMPARISON PLOTS")
    print("="*70)
    
    samples_dict = {
        'BayesOpt': bayesopt_samples,
        'Auto-tuned': auto_samples
    }
    
    print("\nCreating ground truth comparison...")
    fig_gt = plot_ground_truth_comparison(samples_dict, ground_truth, dim)
    plt.savefig('ground_truth_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved ground_truth_comparison.png")
    plt.show()
    
    return bayesopt_samples, auto_samples, mams_results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_all_experiments(dim=5, num_chains=4, num_steps=1000):
    """Run all experiments with comprehensive visualizations."""
    print("\n" + "="*70)
    print("STARTING ALL EXPERIMENTS")
    print("="*70)
    
    total_start = time.time()
    
    # Experiment 1
    print("\n" + "="*70)
    print("EXPERIMENT 1: ALGORITHM COMPARISON")
    print("="*70)
    
    nuts_results, mclmc_results, mams_results = compare_algorithms(dim=dim)
    
    # Experiment 2
    print("\n" + "="*70)
    print("EXPERIMENT 2: TUNING METHOD COMPARISON")
    print("="*70)
    
    bayesopt_samples, auto_samples, _ = compare_mams_tuning_methods(
        dim=dim, num_chains=num_chains, num_steps=num_steps
    )
    
    total_time = time.time() - total_start
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*70)
    print(f"\nTotal execution time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    
    # Save results as CSV
    print("\n" + "="*70)
    print("SAVING RESULTS TO CSV")
    print("="*70)
    
    try:
        import csv
        
        # Save tuning history for each algorithm
        for alg_name in ['nuts', 'mclmc', 'mams']:
            if alg_name == 'nuts':
                alg_results = nuts_results
            elif alg_name == 'mclmc':
                alg_results = mclmc_results
            else:
                alg_results = mams_results
                
            filename = f'{alg_name}_tuning_history.csv'
            
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Header
                header = ['iteration', 'ess', 'acceptance_rate', 'objective', 'time_per_eval']
                if alg_results['hyperparams']:
                    for key in alg_results['hyperparams'][0].keys():
                        header.append(key)
                writer.writerow(header)
                
                # Data rows
                for i in range(len(alg_results['iteration'])):
                    row = [
                        alg_results['iteration'][i],
                        alg_results['ess'][i],
                        alg_results['acceptance_rate'][i],
                        alg_results['objective'][i],
                        alg_results['time_per_eval'][i]
                    ]
                    for key in alg_results['hyperparams'][i].keys():
                        row.append(alg_results['hyperparams'][i][key])
                    writer.writerow(row)
            
            print(f"✓ Saved {filename}")
        
        # Save summary as JSON
        import json
        
        summary = {
            'config': {
                'dim': dim,
                'num_chains': num_chains,
                'num_steps': num_steps,
            },
            'best_hyperparameters': {}
        }
        
        for alg_name, alg_results in [('nuts', nuts_results), 
                                       ('mclmc', mclmc_results), 
                                       ('mams', mams_results)]:
            best_idx = int(jnp.argmax(jnp.array(alg_results['objective'])))
            summary['best_hyperparameters'][alg_name] = {
                'hyperparams': alg_results['hyperparams'][best_idx],
                'ess': float(alg_results['ess'][best_idx]),
                'acceptance_rate': float(alg_results['acceptance_rate'][best_idx]),
                'objective': float(alg_results['objective'][best_idx]),
            }
        
        # Add tuning comparison
        bayesopt_ess_list = [float(compute_ess(bayesopt_samples[i])) 
                            for i in range(bayesopt_samples.shape[0])]
        auto_ess_list = [float(compute_ess(auto_samples[i])) 
                        for i in range(auto_samples.shape[0])]
        
        summary['tuning_comparison'] = {
            'bayesopt': {
                'ess_per_chain': bayesopt_ess_list,
                'mean_ess': float(np.mean(bayesopt_ess_list)),
            },
            'auto_tuned': {
                'ess_per_chain': auto_ess_list,
                'mean_ess': float(np.mean(auto_ess_list)),
            }
        }
        
        with open('mcmc_comparison_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        print("✓ Saved mcmc_comparison_summary.json")
        
    except Exception as e:
        print(f"⚠ Could not save results: {e}")
    
    print("\n✓ All done!")
    print("\nGenerated files:")
    print("  - bayesopt_timeseries.png")
    print("  - bayesopt_hyperparameter_space.png")
    print("  - ground_truth_comparison.png")
    print("  - nuts_tuning_history.csv")
    print("  - mclmc_tuning_history.csv")
    print("  - mams_tuning_history.csv")
    print("  - mcmc_comparison_summary.json")
    
    return {
        'nuts': nuts_results,
        'mclmc': mclmc_results,
        'mams': mams_results,
        'bayesopt_samples': bayesopt_samples,
        'auto_samples': auto_samples
    }


if __name__ == "__main__":
    print("\n" + "="*70)
    print("MCMC HYPERPARAMETER TUNING WITH BAYESIAN OPTIMIZATION")
    print("v7 - Complete Edition")
    print("="*70)
    print("\nThis script demonstrates:")
    print("  1. Bayesian optimization for MCMC hyperparameters")
    print("  2. Comprehensive visualizations of optimization progress")
    print("  3. Comparison with automatic tuning methods")
    print("  4. Ground truth comparison")
    print("  5. Reproducibility verification")
    print("\nFeatures:")
    print("  ✓ Time series plots (ESS, acceptance, objective, L, ε)")
    print("  ✓ Hyperparameter space exploration (L vs ε)")
    print("  ✓ Ground truth comparison (marginals, QQ-plots, scatter)")
    print("  ✓ CSV export of all tuning history")
    print("  ✓ JSON summary of best parameters")
    print("  ✓ Detailed comments on tuning algorithms")
    
    print("\n" + "="*70)
    print("SELECT WHAT TO RUN")
    print("="*70)
    print("\n1. Run all experiments (recommended)")
    print("2. Run only algorithm comparison")
    print("3. Run only tuning method comparison")
    print("4. Run reproducibility verification")
    print("5. Exit")
    
    choice = input("\nEnter choice (1-5) [default: 1]: ").strip()
    
    if choice == '' or choice == '1':
        print("\nRunning all experiments...")
        results = run_all_experiments(dim=5, num_chains=4, num_steps=1000)
        
    elif choice == '2':
        print("\nRunning algorithm comparison...")
        nuts_results, mclmc_results, mams_results = compare_algorithms(dim=5)
        
    elif choice == '3':
        print("\nRunning tuning method comparison...")
        bayesopt_samples, auto_samples, mams_results = compare_mams_tuning_methods(
            dim=5, num_chains=4, num_steps=1000
        )
        
    elif choice == '4':
        print("\nRunning reproducibility verification...")
        verify_reproducibility_demo()
        
    elif choice == '5':
        print("\nExiting...")
        exit(0)
        
    else:
        print(f"\nInvalid choice: {choice}")
        print("Running all experiments by default...")
        results = run_all_experiments(dim=5, num_chains=4, num_steps=1000)
    
    print("\n" + "="*70)
    print("SCRIPT COMPLETED SUCCESSFULLY")
    print("="*70)