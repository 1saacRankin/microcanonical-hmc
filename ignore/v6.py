# ============================================================================
# MCMC ALGORITHM COMPARISON WITH BAYESIAN HYPERPARAMETER OPTIMIZATION
# Enhanced version with detailed tuning algorithm explanations and visualizations
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
    
    Args:
        results_dict: Dictionary with keys 'nuts', 'mclmc', 'mams' containing tuning results
        save_prefix: If provided, saves plots with this prefix
    """
    algorithms = []
    for name in ['NUTS', 'MCLMC', 'MAMS']:
        key = name.lower()
        if key in results_dict and results_dict[key] is not None:
            algorithms.append((name, key))
    
    if not algorithms:
        print("No results to plot!")
        return
    
    # Create figure with subplots for each algorithm
    n_algs = len(algorithms)
    fig = plt.figure(figsize=(18, 5 * n_algs))
    
    for alg_idx, (alg_name, alg_key) in enumerate(algorithms):
        results = results_dict[alg_key]
        
        # Extract data
        iterations = results['iteration']
        ess_values = results['ess']
        acc_values = results['acceptance_rate']
        obj_values = results['objective']
        
        # Extract hyperparameters
        if alg_name == 'NUTS':
            step_sizes = [p['log_step_size'] for p in results['hyperparams']]
            L_values = None
        else:
            L_values = [p['L'] for p in results['hyperparams']]
            step_sizes = [p['step_size'] for p in results['hyperparams']]
        
        # Create 5-panel plot for this algorithm
        base_idx = alg_idx * 5
        
        # Panel 1: ESS over iterations
        ax1 = plt.subplot(n_algs, 5, base_idx + 1)
        ax1.plot(iterations, ess_values, 'o-', linewidth=2, markersize=6, color='C0')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('ESS')
        ax1.set_title(f'{alg_name}: Effective Sample Size')
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Acceptance probability
        ax2 = plt.subplot(n_algs, 5, base_idx + 2)
        ax2.plot(iterations, acc_values, 'o-', linewidth=2, markersize=6, color='C1')
        # Add target line
        target_acc = 0.65 if alg_name == 'NUTS' else (1.0 if alg_name == 'MCLMC' else 0.9)
        ax2.axhline(target_acc, color='red', linestyle='--', alpha=0.5, label=f'Target ({target_acc})')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Acceptance Rate')
        ax2.set_title(f'{alg_name}: Acceptance Probability')
        ax2.set_ylim([0, 1.05])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Objective function
        ax3 = plt.subplot(n_algs, 5, base_idx + 3)
        ax3.plot(iterations, obj_values, 'o-', linewidth=2, markersize=6, color='C2')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Objective')
        ax3.set_title(f'{alg_name}: Objective Function')
        ax3.grid(True, alpha=0.3)
        
        # Panel 4: Trajectory length L (if applicable)
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
        
        # Panel 5: Step size ε
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
    
    Args:
        results_dict: Dictionary with keys 'mclmc', 'mams' containing tuning results
        save_prefix: If provided, saves plots with this prefix
    """
    # Only plot for MCLMC and MAMS (NUTS doesn't have L)
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
        
        # Extract data
        L_values = np.array([p['L'] for p in results['hyperparams']])
        step_sizes = np.array([p['step_size'] for p in results['hyperparams']])
        acc_values = np.array(results['acceptance_rate'])
        ess_values = np.array(results['ess'])
        
        # Normalize ESS for color mapping (0 to 1)
        ess_min, ess_max = ess_values.min(), ess_values.max()
        if ess_max > ess_min:
            ess_normalized = (ess_values - ess_min) / (ess_max - ess_min)
        else:
            ess_normalized = np.ones_like(ess_values)
        
        # Create color map: red (low ESS) to green (high ESS)
        colors = plt.cm.RdYlGn(ess_normalized)
        
        # Size proportional to acceptance (scale for visibility)
        sizes = 100 + acc_values * 500
        
        # Create scatter plot
        scatter = ax.scatter(step_sizes, L_values, s=sizes, c=colors, 
                           alpha=0.7, edgecolors='black', linewidths=1.5)
        
        ax.set_xlabel('Step Size (ε)', fontsize=14)
        ax.set_ylabel('Trajectory Length (L)', fontsize=14)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(f'{alg_name}: Hyperparameter Space Exploration', fontsize=16)
        ax.grid(True, alpha=0.3, which='both')
        
        # Add colorbar for ESS
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, 
                                   norm=plt.Normalize(vmin=ess_min, vmax=ess_max))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label='ESS')
        
        # Add legend for size (acceptance)
        # Create dummy points for legend
        legend_sizes = [0.3, 0.6, 0.9]
        legend_points = [ax.scatter([], [], s=100 + s * 500, c='gray', alpha=0.7, 
                                   edgecolors='black', linewidths=1.5) 
                        for s in legend_sizes]
        legend_labels = [f'Acc={s:.1f}' for s in legend_sizes]
        
        # Position legend outside plot
        legend = ax.legend(legend_points, legend_labels, 
                          title='Acceptance', loc='upper left', 
                          bbox_to_anchor=(1.15, 1), frameon=True)
        
        # Add text annotations for first and last points
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
        chain_start = time.time()
        
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
    print(f"  Average time per chain: {np.mean(all_times):.2f}s")
    
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
        chain_start = time.time()
        
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
    
    # Compute ESS for each chain
    all_ess = []
    for chain_samples in all_samples:
        ess = compute_ess(chain_samples)
        all_ess.append(ess)
    
    all_acceptance = jnp.ones(num_chains) * 0.9
    
    total_time = time.time() - start_time
    
    print(f"  Total time for {num_chains} chains: {total_time:.2f}s")
    print(f"  Average time per chain: {np.mean(all_times):.2f}s")
    
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
    print("      [Auto-tuning hyperparameters (this may take a while)...]", end=" ", flush=True)
    
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
# MAIN COMPARISON FUNCTIONS
# ============================================================================

def compare_algorithms(dim=5):
    """Compare all three algorithms using Bayesian optimization for tuning."""
    print("="*70)
    print("MCMC ALGORITHM COMPARISON WITH BAYESIAN OPTIMIZATION")
    print("="*70)
    
    print("\nCreating target density (Neal's Funnel)...")
    logdensity_fn = make_funnel_logdensity(dim)
    
    initial_position = jnp.zeros(dim)
    
    nuts_tuning_key = jax.random.key(SEED_NUTS_TUNING)
    mclmc_tuning_key = jax.random.key(SEED_MCLMC_TUNING)
    mams_tuning_key = jax.random.key(SEED_MAMS_TUNING)
    
    print("\n" + "="*70)
    print("PHASE 1: HYPERPARAMETER TUNING")
    print("="*70)
    print("\nNote: Using FIXED random keys for fair comparison!")
    
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
    
    # Print summary
    print("\n" + "="*70)
    print("BEST HYPERPARAMETERS FOUND")
    print("="*70)
    
    for alg_name, alg_results in [('NUTS', nuts_results), 
                                   ('MCLMC', mclmc_results), 
                                   ('MAMS', mams_results)]:
        best_idx = jnp.argmax(jnp.array(alg_results['objective']))
        print(f"\n{alg_name}:")
        print(f"  Best objective: {alg_results['objective'][best_idx]:.1f}")
        print(f"  ESS: {alg_results['ess'][best_idx]:.1f}")
        print(f"  Acceptance: {alg_results['acceptance_rate'][best_idx]:.3f}")
        print(f"  Hyperparameters: {alg_results['hyperparams'][best_idx]}")
    
    return nuts_results, mclmc_results, mams_results


def compare_mams_tuning_methods(dim=5, num_chains=4, num_steps=1000):
    """Compare Bayesian optimization vs automatic tuning for MAMS."""
    print("="*70)
    print("MAMS HYPERPARAMETER TUNING COMPARISON")
    print("="*70)
    print("\nQuestion: Is Bayesian optimization better than automatic tuning?")
    
    logdensity_fn = make_funnel_logdensity(dim)
    initial_position = jnp.zeros(dim)
    
    # STEP 1: Bayesian Optimization
    print("\n" + "="*70)
    print("STEP 1: BAYESIAN OPTIMIZATION")
    print("="*70)
    
    tuning_start_time = time.time()
    
    bayesopt_tuning_key = jax.random.key(SEED_MAMS_TUNING)
    mams_results = run_bayesopt_tuning(
        logdensity_fn, initial_position, bayesopt_tuning_key, 'MAMS',
        num_iterations=10,
        chain_length=100
    )
    
    tuning_total_time = time.time() - tuning_start_time
    
    best_idx = jnp.argmax(jnp.array(mams_results['objective']))
    best_params = mams_results['hyperparams'][best_idx]
    best_L = best_params['L']
    best_step_size = best_params['step_size']
    
    print(f"\nBest hyperparameters found:")
    print(f"  L = {best_L:.4f}")
    print(f"  step_size = {best_step_size:.6f}")
    
    # STEP 2: Validate BayesOpt
    print("\n" + "="*70)
    print("STEP 2: VALIDATE BAYESOPT PARAMETERS")
    print("="*70)
    
    validation_base_key = jax.random.key(SEED_BAYESOPT_VALIDATION)
    
    bayesopt_samples, bayesopt_ess, bayesopt_acc, bayesopt_time = \
        run_mams_multiple_chains(
            logdensity_fn, num_chains, num_steps, initial_position, 
            validation_base_key, best_L, best_step_size
        )
    
    bayesopt_rhat = compute_rhat(bayesopt_samples)
    
    print(f"\nBayesOpt Validation Results:")
    print(f"  Mean ESS: {jnp.mean(bayesopt_ess):.1f} ± {jnp.std(bayesopt_ess):.1f}")
    print(f"  Max R-hat: {jnp.max(bayesopt_rhat):.4f}")
    
    # STEP 3: Auto-tuning
    print("\n" + "="*70)
    print("STEP 3: AUTOMATIC TUNING")
    print("="*70)
    
    auto_samples, auto_ess, auto_acc, auto_step_sizes, auto_L, auto_time = \
        run_auto_tuned_multiple_chains(
            logdensity_fn, num_chains, num_steps, initial_position, 
            validation_base_key
        )
    
    auto_rhat = compute_rhat(auto_samples)
    
    print(f"\nAutomatic Tuning Results:")
    print(f"  Mean L: {jnp.mean(auto_L):.4f} ± {jnp.std(auto_L):.4f}")
    print(f"  Mean step size: {jnp.mean(auto_step_sizes):.6f}")
    print(f"  Mean ESS: {jnp.mean(auto_ess):.1f} ± {jnp.std(auto_ess):.1f}")
    print(f"  Max R-hat: {jnp.max(auto_rhat):.4f}")
    
    # STEP 4: Create comparison plots
    print("\n" + "="*70)
    print("STEP 4: CREATING COMPARISON PLOTS")
    print("="*70)
    
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Flatten samples
    bayesopt_flat = bayesopt_samples.reshape(-1, dim)
    auto_flat = auto_samples.reshape(-1, dim)
    
    # Row 1: Trace plots
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
    
    # Row 2: Marginals
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
    
    ax9 = fig.add_subplot(gs[2, 2])
    methods = ['BayesOpt', 'Auto-tuned']
    L_values = [best_L, jnp.mean(auto_L)]
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
    
    plt.savefig('mams_tuning_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved mams_tuning_comparison.png")
    plt.show()
    
    # Print final summary
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
    print(f"{'Max R-hat':<30} {jnp.max(bayesopt_rhat):<20.4f} "
          f"{jnp.max(auto_rhat):<20.4f}")
    
    return bayesopt_samples, auto_samples, mams_results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_all_experiments(dim=5, num_chains=4, num_steps=1000):
    """Run all experiments with comprehensive visualizations."""
    print("\n" + "="*70)
    print("STARTING ALL EXPERIMENTS")
    print("="*70)
    
    total_start_time = time.time()
    
    # Experiment 1: Compare algorithms
    print("\n" + "="*70)
    print("EXPERIMENT 1: ALGORITHM COMPARISON")
    print("="*70)
    
    nuts_results, mclmc_results, mams_results = compare_algorithms(dim=dim)
    
    # Experiment 2: Compare tuning methods
    print("\n" + "="*70)
    print("EXPERIMENT 2: TUNING METHOD COMPARISON")
    print("="*70)
    
    bayesopt_samples, auto_samples, _ = compare_mams_tuning_methods(
        dim=dim, num_chains=num_chains, num_steps=num_steps
    )
    
    total_time = time.time() - total_start_time
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*70)
    print(f"\nTotal execution time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    print("\n✓ All done!")
    
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
    print("="*70)
    print("\nThis script demonstrates:")
    print("  1. Bayesian optimization for MCMC hyperparameters")
    print("  2. Detailed visualizations of optimization progress")
    print("  3. Comparison with automatic tuning methods")
    
    # Run all experiments
    results = run_all_experiments(dim=5, num_chains=4, num_steps=1000)
    
    print("\n✓ Script completed successfully!")
    print("  - Time series plots saved as 'bayesopt_timeseries.png'")
    print("  - Hyperparameter space plots saved as 'bayesopt_hyperparameter_space.png'")
    print("  - Tuning comparison saved as 'mams_tuning_comparison.png'")