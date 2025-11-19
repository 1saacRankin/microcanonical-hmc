# ============================================================================
# MCMC ALGORITHM COMPARISON WITH BAYESIAN HYPERPARAMETER OPTIMIZATION
# v8 - With normalized ESS metrics for fair comparison
# ============================================================================
# 
# CRITICAL INSIGHT: FAIR COMPARISON REQUIRES NORMALIZATION
# 
# Different hyperparameters → different computational cost per MCMC step:
#   - L=10, ε=0.01 → n=1000 integration steps per MCMC iteration
#   - L=5,  ε=0.05 → n=100 integration steps per MCMC iteration
# 
# Raw ESS comparison is UNFAIR because one algorithm does 10x more work!
# 
# SOLUTION: Normalize by computational cost
#   1. ESS/n (ESS per integration step) - primary metric for fair comparison
#   2. ESS per second (wall-clock efficiency) - practical metric
#   3. ESS per gradient evaluation (same as ESS/n since 1 step = 1 gradient)
# 
# OBJECTIVE FUNCTION:
#   - MCLMC/MAMS: Maximize ESS/n - λ × (acceptance - target)²
#   - NUTS: Maximize ESS - λ × (acceptance - target)²  (L is adaptive)
# 
# ============================================================================
# BAYESIAN OPTIMIZATION DETAILS (BOAx Framework)
# ============================================================================
# 
# KERNEL: Gaussian Process with Matérn 5/2 kernel (default in Ax)
# ACQUISITION: Expected Improvement (EI) or Upper Confidence Bound (UCB)
# OBJECTIVE: ESS/n - λ × (acceptance_rate - target)²  [for MCLMC/MAMS]
#            ESS - λ × (acceptance_rate - target)²      [for NUTS]
# 
# SEARCH SPACE:
#   - NUTS: log_step_size ∈ [1e-5, 1e-1]
#   - MCLMC/MAMS: L ∈ [1e-1, 1e2], step_size ∈ [1e-3, 1.0]
# 
# ============================================================================
# BLACKJAX AUTOMATIC TUNING (adjusted_mclmc_find_L_and_step_size)
# ============================================================================
# 
# Three-phase adaptive scheme:
# Phase 1 (10%): Coarse step size tuning → target acceptance ~0.9
# Phase 2 (10%): Trajectory length tuning → estimate from autocorrelation
# Phase 3 (10%): Joint fine-tuning with optional preconditioning
# 
# Total: 30% of steps used for tuning (discarded), 70% kept for inference
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

config.update("jax_enable_x64", True)

plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["font.size"] = 12

# Random seeds
SEED_NUTS_TUNING = 1000
SEED_MCLMC_TUNING = 2000
SEED_MAMS_TUNING = 3000
SEED_BAYESOPT_VALIDATION = 99999
SEED_AUTO_VALIDATION = 88888


# ============================================================================
# TARGET DENSITIES
# ============================================================================

def make_funnel_logdensity(dim):
    """Create Neal's funnel distribution."""
    def logdensity(x):
        log_prob = -0.5 * (x[0]**2 / 9.0)
        log_prob += -0.5 * (dim - 1) * x[0]
        log_prob += -0.5 * jnp.sum(x[1:]**2 * jnp.exp(-x[0]))
        return log_prob
    return logdensity


# ============================================================================
# DIAGNOSTIC METRICS
# ============================================================================

def compute_ess(samples):
    """Compute minimum Effective Sample Size across all dimensions."""
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
        return state, (state.position, info.acceptance_rate, info.num_integration_steps)
    
    keys = jax.random.split(key, num_steps)
    final_state, (samples, acceptance_rates, num_steps_per_iter) = jax.lax.scan(one_step, state, keys)
    
    avg_acceptance = jnp.mean(acceptance_rates)
    ess = compute_ess(samples)
    avg_integration_steps = jnp.mean(num_steps_per_iter)
    time_elapsed = time.time() - start_time
    
    return samples, ess, avg_acceptance, avg_integration_steps, time_elapsed


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
    integration_steps_per_iter = L / step_size
    time_elapsed = time.time() - start_time
    
    return samples, ess, 1.0, integration_steps_per_iter, time_elapsed


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
    integration_steps_per_iter = L / step_size
    time_elapsed = time.time() - start_time
    
    return samples, ess, avg_acceptance, integration_steps_per_iter, time_elapsed


# ============================================================================
# BAYESIAN OPTIMIZATION FOR HYPERPARAMETER TUNING
# ============================================================================

def objective_function(ess, acceptance_rate, target_acceptance, lambda_penalty, 
                       integration_steps_per_iter=None, normalize_by_cost=False):
    """
    Compute objective for hyperparameter optimization.
    
    Args:
        ess: Effective Sample Size
        acceptance_rate: Average acceptance rate
        target_acceptance: Desired acceptance rate
        lambda_penalty: Penalty weight for acceptance deviation
        integration_steps_per_iter: Number of integration steps per MCMC iteration
        normalize_by_cost: If True, use ESS/n; if False, use raw ESS
    
    Returns:
        Objective value to maximize
    """
    if normalize_by_cost and integration_steps_per_iter is not None:
        # ESS per integration step (fair comparison across different L, ε)
        ess_normalized = ess / integration_steps_per_iter
    else:
        # Raw ESS (for NUTS where L is adaptive)
        ess_normalized = ess
    
    penalty = lambda_penalty * (acceptance_rate - target_acceptance)**2
    return ess_normalized - penalty


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
        'ess_per_step': [],
        'ess_per_second': [],
        'acceptance_rate': [],
        'objective': [],
        'integration_steps': [],
        'hyperparams': [],
        'time_per_eval': [],
    }
    
    start_time = time.time()
    
    # Define search space and evaluation function
    if algorithm_name == 'NUTS':
        parameters = [{'name': 'log_step_size', 'type': 'log_range', 'bounds': [1e-5, 1e-1]}]
        
        def run_with_params(params_dict):
            step_size = params_dict['log_step_size']
            inv_mass_matrix = jnp.ones(len(initial_position))
            _, ess, acc, avg_n, eval_time = run_nuts_fixed(
                logdensity_fn, chain_length, initial_position, 
                fixed_key, step_size, inv_mass_matrix
            )
            return ess, acc, avg_n, eval_time
        
        normalize_by_cost = False  # NUTS has adaptive L
        
    elif algorithm_name == 'MCLMC':
        parameters = [
            {'name': 'L', 'type': 'log_range', 'bounds': [1e-1, 1e2]},
            {'name': 'step_size', 'type': 'log_range', 'bounds': [1e-3, 1.0]}
        ]
        target_acceptance = 1.0
        
        def run_with_params(params_dict):
            L, step_size = params_dict['L'], params_dict['step_size']
            _, ess, acc, n, eval_time = run_mclmc_fixed(
                logdensity_fn, chain_length, initial_position, 
                fixed_key, L, step_size
            )
            return ess, acc, n, eval_time
        
        normalize_by_cost = True  # Fair comparison needed
        
    elif algorithm_name == 'MAMS':
        parameters = [
            {'name': 'L', 'type': 'log_range', 'bounds': [1e-1, 1e2]},
            {'name': 'step_size', 'type': 'log_range', 'bounds': [1e-3, 1.0]}
        ]
        target_acceptance = 0.9
        
        def run_with_params(params_dict):
            L, step_size = params_dict['L'], params_dict['step_size']
            _, ess, acc, n, eval_time = run_mams_fixed(
                logdensity_fn, chain_length, initial_position, 
                fixed_key, L, step_size
            )
            return ess, acc, n, eval_time
        
        normalize_by_cost = True  # Fair comparison needed
    
    # Initialize BOAx experiment
    experiment = optimization(parameters=parameters, batch_size=1)
    
    # Run optimization loop
    step, experiment_results = None, []
    
    for i in range(num_iterations):
        print(f"Iteration {i+1}/{num_iterations}")
        
        step, parameterizations = experiment.next(step, experiment_results)
        params_dict = parameterizations[0]
        
        ess, acc, n, eval_time = run_with_params(params_dict)
        
        # Compute normalized metrics
        ess_per_step = ess / n if n > 0 else 0
        ess_per_second = ess / eval_time if eval_time > 0 else 0
        
        # Compute objective (normalized or not)
        obj = objective_function(ess, acc, target_acceptance, lambda_penalty, 
                                n, normalize_by_cost)
        
        experiment_results = [(params_dict, float(obj))]
        
        results['iteration'].append(i)
        results['ess'].append(float(ess))
        results['ess_per_step'].append(float(ess_per_step))
        results['ess_per_second'].append(float(ess_per_second))
        results['acceptance_rate'].append(float(acc))
        results['objective'].append(float(obj))
        results['integration_steps'].append(float(n))
        results['hyperparams'].append(params_dict)
        results['time_per_eval'].append(eval_time)
        
        print(f"  {algorithm_name}: ESS={ess:.1f}, ESS/n={ess_per_step:.4f}, "
              f"ESS/s={ess_per_second:.1f}, n={n:.1f}")
    
    total_time = time.time() - start_time
    
    best_idx = jnp.argmax(jnp.array(results['objective']))
    best_params = results['hyperparams'][best_idx]
    
    print(f"\n{algorithm_name} Best:")
    print(f"  Params: {best_params}")
    print(f"  ESS: {results['ess'][best_idx]:.1f}")
    print(f"  ESS/n: {results['ess_per_step'][best_idx]:.4f}")
    print(f"  ESS/s: {results['ess_per_second'][best_idx]:.1f}")
    
    return results


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_bayesopt_progress(results_dict, save_prefix=None):
    """Create time series plots showing ESS, ESS/n, ESS/s, acceptance, L, ε."""
    algorithms = []
    for name in ['NUTS', 'MCLMC', 'MAMS']:
        key = name.lower()
        if key in results_dict and results_dict[key] is not None:
            algorithms.append((name, key))
    
    if not algorithms:
        print("No results to plot!")
        return
    
    n_algs = len(algorithms)
    fig = plt.figure(figsize=(20, 5 * n_algs))
    
    for alg_idx, (alg_name, alg_key) in enumerate(algorithms):
        results = results_dict[alg_key]
        
        iterations = results['iteration']
        ess_values = results['ess']
        ess_per_step_values = results['ess_per_step']
        ess_per_second_values = results['ess_per_second']
        acc_values = results['acceptance_rate']
        n_values = results['integration_steps']
        
        if alg_name == 'NUTS':
            step_sizes = [p['log_step_size'] for p in results['hyperparams']]
            L_values = None
        else:
            L_values = [p['L'] for p in results['hyperparams']]
            step_sizes = [p['step_size'] for p in results['hyperparams']]
        
        base_idx = alg_idx * 6
        
        # Panel 1: Raw ESS
        ax1 = plt.subplot(n_algs, 6, base_idx + 1)
        ax1.plot(iterations, ess_values, 'o-', linewidth=2, markersize=6, color='C0')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('ESS')
        ax1.set_title(f'{alg_name}: Raw ESS')
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: ESS/n (FAIR COMPARISON METRIC)
        ax2 = plt.subplot(n_algs, 6, base_idx + 2)
        ax2.plot(iterations, ess_per_step_values, 'o-', linewidth=2, markersize=6, color='C5')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('ESS/n')
        ax2.set_title(f'{alg_name}: ESS per Integration Step ⭐')
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: ESS/s
        ax3 = plt.subplot(n_algs, 6, base_idx + 3)
        ax3.plot(iterations, ess_per_second_values, 'o-', linewidth=2, markersize=6, color='C6')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('ESS/s')
        ax3.set_title(f'{alg_name}: ESS per Second')
        ax3.grid(True, alpha=0.3)
        
        # Panel 4: Acceptance
        ax4 = plt.subplot(n_algs, 6, base_idx + 4)
        ax4.plot(iterations, acc_values, 'o-', linewidth=2, markersize=6, color='C1')
        target_acc = 0.65 if alg_name == 'NUTS' else (1.0 if alg_name == 'MCLMC' else 0.9)
        ax4.axhline(target_acc, color='red', linestyle='--', alpha=0.5, label=f'Target')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Acceptance')
        ax4.set_title(f'{alg_name}: Acceptance Rate')
        ax4.set_ylim([0, 1.05])
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Panel 5: L and n
        if L_values is not None:
            ax5 = plt.subplot(n_algs, 6, base_idx + 5)
            ax5_twin = ax5.twinx()
            l1 = ax5.plot(iterations, L_values, 'o-', linewidth=2, markersize=6, 
                         color='C3', label='L')
            l2 = ax5_twin.plot(iterations, n_values, 's-', linewidth=2, markersize=6, 
                              color='C7', label='n')
            ax5.set_xlabel('Iteration')
            ax5.set_ylabel('L', color='C3')
            ax5_twin.set_ylabel('n (steps/iter)', color='C7')
            ax5.set_title(f'{alg_name}: L and n')
            ax5.tick_params(axis='y', labelcolor='C3')
            ax5_twin.tick_params(axis='y', labelcolor='C7')
            ax5.grid(True, alpha=0.3)
            lns = l1 + l2
            labs = [l.get_label() for l in lns]
            ax5.legend(lns, labs, loc='upper left')
        else:
            ax5 = plt.subplot(n_algs, 6, base_idx + 5)
            ax5.plot(iterations, n_values, 'o-', linewidth=2, markersize=6, color='C7')
            ax5.set_xlabel('Iteration')
            ax5.set_ylabel('n (avg steps/iter)')
            ax5.set_title(f'{alg_name}: Integration Steps')
            ax5.grid(True, alpha=0.3)
        
        # Panel 6: Step size
        ax6 = plt.subplot(n_algs, 6, base_idx + 6)
        ax6.plot(iterations, step_sizes, 'o-', linewidth=2, markersize=6, color='C4')
        ax6.set_xlabel('Iteration')
        ax6.set_ylabel('ε (log scale)')
        ax6.set_yscale('log')
        ax6.set_title(f'{alg_name}: Step Size')
        ax6.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    if save_prefix:
        plt.savefig(f'{save_prefix}_timeseries.png', dpi=150, bbox_inches='tight')
        print(f"Saved {save_prefix}_timeseries.png")
    
    plt.show()


def plot_hyperparameter_space(results_dict, save_prefix=None):
    """Create L vs ε scatter with size=acceptance, color=ESS/n."""
    algorithms = []
    for name in ['MCLMC', 'MAMS']:
        key = name.lower()
        if key in results_dict and results_dict[key] is not None:
            algorithms.append((name, key))
    
    if not algorithms:
        print("No MCLMC/MAMS results!")
        return
    
    fig, axes = plt.subplots(1, len(algorithms), figsize=(9 * len(algorithms), 7))
    if len(algorithms) == 1:
        axes = [axes]
    
    for ax, (alg_name, alg_key) in zip(axes, algorithms):
        results = results_dict[alg_key]
        
        L_values = np.array([p['L'] for p in results['hyperparams']])
        step_sizes = np.array([p['step_size'] for p in results['hyperparams']])
        acc_values = np.array(results['acceptance_rate'])
        ess_per_step_values = np.array(results['ess_per_step'])
        
        # Normalize ESS/n for color mapping
        ess_min, ess_max = ess_per_step_values.min(), ess_per_step_values.max()
        if ess_max > ess_min:
            ess_normalized = (ess_per_step_values - ess_min) / (ess_max - ess_min)
        else:
            ess_normalized = np.ones_like(ess_per_step_values)
        
        colors = plt.cm.RdYlGn(ess_normalized)
        sizes = 100 + acc_values * 500
        
        scatter = ax.scatter(step_sizes, L_values, s=sizes, c=colors, 
                           alpha=0.7, edgecolors='black', linewidths=1.5)
        
        ax.set_xlabel('Step Size (ε)', fontsize=14)
        ax.set_ylabel('Trajectory Length (L)', fontsize=14)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(f'{alg_name}: Hyperparameter Space (color=ESS/n)', fontsize=16)
        ax.grid(True, alpha=0.3, which='both')
        
        # Colorbar for ESS/n
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, 
                                   norm=plt.Normalize(vmin=ess_min, vmax=ess_max))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label='ESS/n')
        
        # Legend for acceptance (size)
        legend_sizes = [0.3, 0.6, 0.9]
        legend_points = [ax.scatter([], [], s=100 + s * 500, c='gray', alpha=0.7, 
                                   edgecolors='black', linewidths=1.5) 
                        for s in legend_sizes]
        legend_labels = [f'Acc={s:.1f}' for s in legend_sizes]
        
        legend = ax.legend(legend_points, legend_labels, 
                          title='Acceptance', loc='upper left', 
                          bbox_to_anchor=(1.15, 1), frameon=True)
        
        # Annotate start and best
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
    all_ess_per_step = []
    all_ess_per_second = []
    all_acceptance = []
    
    n = L / step_size
    
    for i in range(num_chains):
        print(f"  Chain {i+1}/{num_chains}...", end=" ")
        chain_key = jax.random.fold_in(base_key, i)
        
        samples, ess, acc, _, chain_time = run_mams_fixed(
            logdensity_fn, num_steps, initial_position, chain_key, L, step_size
        )
        
        ess_per_step = ess / n
        ess_per_second = ess / chain_time
        
        all_samples.append(samples)
        all_ess.append(ess)
        all_ess_per_step.append(ess_per_step)
        all_ess_per_second.append(ess_per_second)
        all_acceptance.append(acc)
        
        print(f"ESS={ess:.1f}, ESS/n={ess_per_step:.4f}, ESS/s={ess_per_second:.1f}")
    
    all_samples = jnp.stack(all_samples, axis=0)
    total_time = time.time() - start_time
    
    return (all_samples, jnp.array(all_ess), jnp.array(all_ess_per_step), 
            jnp.array(all_ess_per_second), jnp.array(all_acceptance), total_time)


def run_auto_tuned_multiple_chains(logdensity_fn, num_chains, num_steps, 
                                   initial_position, base_key):
    """Run multiple chains with BlackJAX automatic tuning."""
    start_time = time.time()
    
    all_samples = []
    all_step_sizes = []
    all_L = []
    all_ess = []
    all_ess_per_step = []
    all_ess_per_second = []
    
    for i in range(num_chains):
        print(f"  Chain {i+1}/{num_chains} (auto-tuning)...", end=" ")
        chain_key = jax.random.fold_in(base_key, i)
        
        samples, step_size, L, chain_time = run_adjusted_mclmc_dynamic(
            logdensity_fn, num_steps, initial_position, chain_key
        )
        
        ess = compute_ess(samples)
        n = L / step_size
        ess_per_step = ess / n
        ess_per_second = ess / chain_time
        
        all_samples.append(samples)
        all_step_sizes.append(step_size)
        all_L.append(L)
        all_ess.append(ess)
        all_ess_per_step.append(ess_per_step)
        all_ess_per_second.append(ess_per_second)
        
        print(f"L={L:.3f}, ε={step_size:.5f}, ESS/n={ess_per_step:.4f}")
    
    all_samples = jnp.stack(all_samples, axis=0)
    all_acceptance = jnp.ones(num_chains) * 0.9
    total_time = time.time() - start_time
    
    return (all_samples, jnp.array(all_ess), jnp.array(all_ess_per_step),
            jnp.array(all_ess_per_second), all_acceptance, 
            jnp.array(all_step_sizes), jnp.array(all_L), total_time)


def run_adjusted_mclmc_dynamic(logdensity_fn, num_steps, initial_position, key):
    """Run MAMS with BlackJAX automatic hyperparameter tuning."""
    start_time = time.time()
    
    print("      [Init...]", end=" ", flush=True)
    
    init_key, tune_key, run_key = jax.random.split(key, 3)
    
    initial_state = blackjax.mcmc.adjusted_mclmc_dynamic.init(
        position=initial_position,
        logdensity_fn=logdensity_fn,
        random_generator_arg=init_key,
    )
    
    print("done", flush=True)
    print("      [Kernel...]", end=" ", flush=True)
    
    integration_steps_fn = lambda avg_num_integration_steps: lambda k: jnp.ceil(
        jax.random.uniform(k) * rescale(avg_num_integration_steps))
    
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
    
    print("done", flush=True)
    print("      [Auto-tune...]", end=" ", flush=True)
    
    (
        blackjax_state_after_tuning,
        blackjax_mclmc_sampler_params,
        _
    ) = blackjax.adjusted_mclmc_find_L_and_step_size(
        mclmc_kernel=kernel,
        num_steps=num_steps,
        state=initial_state,
        rng_key=tune_key,
        target=0.9,
        frac_tune1=0.1,
        frac_tune2=0.1,
        frac_tune3=0.1,
        diagonal_preconditioning=True,
    )
    
    print("done", flush=True)
    print("      [Sample...]", end=" ", flush=True)
    
    step_size = blackjax_mclmc_sampler_params.step_size
    L = blackjax_mclmc_sampler_params.L
    
    alg = blackjax.adjusted_mclmc_dynamic(
        logdensity_fn=logdensity_fn,
        step_size=step_size,
        integration_steps_fn=lambda key: jnp.ceil(
            jax.random.uniform(key) * rescale(L / step_size)
        ),
        inverse_mass_matrix=blackjax_mclmc_sampler_params.inverse_mass_matrix,
        L_proposal_factor=jnp.inf,
    )
    
    _, out = run_inference_algorithm(
        rng_key=run_key,
        initial_state=blackjax_state_after_tuning,
        inference_algorithm=alg,
        num_steps=num_steps,
        transform=lambda state, _: state.position,
        progress_bar=False,
    )
    
    print("done", flush=True)
    
    time_elapsed = time.time() - start_time
    
    return out, step_size, L, time_elapsed


# ============================================================================
# MAIN COMPARISON FUNCTIONS
# ============================================================================

def compare_algorithms(dim=5):
    """Compare NUTS, MCLMC, MAMS using Bayesian optimization."""
    print("="*70)
    print("ALGORITHM COMPARISON WITH BAYESIAN OPTIMIZATION")
    print("="*70)
    
    logdensity_fn = make_funnel_logdensity(dim)
    initial_position = jnp.zeros(dim)
    
    print("\n1. Tuning NUTS...")
    nuts_results = run_bayesopt_tuning(
        logdensity_fn, initial_position, jax.random.key(SEED_NUTS_TUNING), 'NUTS',
        num_iterations=10, chain_length=100
    )
    
    print("\n2. Tuning MCLMC...")
    mclmc_results = run_bayesopt_tuning(
        logdensity_fn, initial_position, jax.random.key(SEED_MCLMC_TUNING), 'MCLMC',
        num_iterations=10, chain_length=100
    )
    
    print("\n3. Tuning MAMS...")
    mams_results = run_bayesopt_tuning(
        logdensity_fn, initial_position, jax.random.key(SEED_MAMS_TUNING), 'MAMS',
        num_iterations=10, chain_length=100
    )
    
    print("\nCreating visualizations...")
    results_dict = {
        'nuts': nuts_results,
        'mclmc': mclmc_results,
        'mams': mams_results
    }
    
    plot_bayesopt_progress(results_dict, save_prefix='bayesopt')
    plot_hyperparameter_space(results_dict, save_prefix='bayesopt')
    
    # Print comparison table
    print("\n" + "="*70)
    print("ALGORITHM COMPARISON (at best hyperparameters)")
    print("="*70)
    print(f"\n{'Algorithm':<12} {'ESS':<10} {'ESS/n':<12} {'ESS/s':<12} {'Acceptance':<12}")
    print("-"*70)
    
    for alg_name, alg_results in [('NUTS', nuts_results), 
                                   ('MCLMC', mclmc_results), 
                                   ('MAMS', mams_results)]:
        best_idx = jnp.argmax(jnp.array(alg_results['objective']))
        ess = alg_results['ess'][best_idx]
        ess_per_step = alg_results['ess_per_step'][best_idx]
        ess_per_second = alg_results['ess_per_second'][best_idx]
        acc = alg_results['acceptance_rate'][best_idx]
        print(f"{alg_name:<12} {ess:<10.1f} {ess_per_step:<12.4f} {ess_per_second:<12.1f} {acc:<12.3f}")
    
    return nuts_results, mclmc_results, mams_results


def compare_mams_tuning_methods(dim=5, num_chains=4, num_steps=1000):
    """Compare Bayesian optimization vs BlackJAX automatic tuning for MAMS."""
    print("="*70)
    print("MAMS TUNING COMPARISON: BAYESOPT vs AUTO-TUNING")
    print("="*70)
    
    logdensity_fn = make_funnel_logdensity(dim)
    initial_position = jnp.zeros(dim)
    
    # Step 1: BayesOpt tuning
    print("\nSTEP 1: BAYESIAN OPTIMIZATION TUNING")
    print("="*70)
    
    bayesopt_tuning_key = jax.random.key(SEED_MAMS_TUNING)
    mams_results = run_bayesopt_tuning(
        logdensity_fn, initial_position, bayesopt_tuning_key, 'MAMS',
        num_iterations=10, chain_length=100
    )
    
    best_idx = jnp.argmax(jnp.array(mams_results['objective']))
    best_params = mams_results['hyperparams'][best_idx]
    best_L = best_params['L']
    best_step_size = best_params['step_size']
    
    print(f"\nBest found: L={best_L:.4f}, ε={best_step_size:.6f}")
    
    # Step 2: Validate BayesOpt with multiple chains
    print("\nSTEP 2: VALIDATE BAYESOPT WITH MULTIPLE CHAINS")
    print("="*70)
    
    validation_key = jax.random.key(SEED_BAYESOPT_VALIDATION)
    (bayesopt_samples, bayesopt_ess, bayesopt_ess_per_step, 
     bayesopt_ess_per_second, bayesopt_acc, bayesopt_time) = \
        run_mams_multiple_chains(
            logdensity_fn, num_chains, num_steps, initial_position, 
            validation_key, best_L, best_step_size
        )
    
    bayesopt_rhat = compute_rhat(bayesopt_samples)
    
    print(f"\nBayesOpt Results:")
    print(f"  Mean ESS: {jnp.mean(bayesopt_ess):.1f} ± {jnp.std(bayesopt_ess):.1f}")
    print(f"  Mean ESS/n: {jnp.mean(bayesopt_ess_per_step):.4f} ± {jnp.std(bayesopt_ess_per_step):.4f}")
    print(f"  Mean ESS/s: {jnp.mean(bayesopt_ess_per_second):.1f} ± {jnp.std(bayesopt_ess_per_second):.1f}")
    print(f"  Max R-hat: {jnp.max(bayesopt_rhat):.4f}")
    print(f"  Convergence: {'✓ Good' if jnp.max(bayesopt_rhat) < 1.01 else '⚠ Needs more steps'}")
    
    # Step 3: Auto-tuning with multiple chains
    print("\nSTEP 3: AUTOMATIC TUNING WITH MULTIPLE CHAINS")
    print("="*70)
    print("Note: Each chain tunes independently")
    
    (auto_samples, auto_ess, auto_ess_per_step, auto_ess_per_second,
     auto_acc, auto_step_sizes, auto_L, auto_time) = \
        run_auto_tuned_multiple_chains(
            logdensity_fn, num_chains, num_steps, initial_position, 
            validation_key  # Same key for fair comparison
        )
    
    auto_rhat = compute_rhat(auto_samples)
    
    print(f"\nAuto-tuning Results:")
    print(f"  Mean L: {jnp.mean(auto_L):.4f} ± {jnp.std(auto_L):.4f}")
    print(f"  Mean ε: {jnp.mean(auto_step_sizes):.6f} ± {jnp.std(auto_step_sizes):.6f}")
    print(f"  Mean ESS: {jnp.mean(auto_ess):.1f} ± {jnp.std(auto_ess):.1f}")
    print(f"  Mean ESS/n: {jnp.mean(auto_ess_per_step):.4f} ± {jnp.std(auto_ess_per_step):.4f}")
    print(f"  Mean ESS/s: {jnp.mean(auto_ess_per_second):.1f} ± {jnp.std(auto_ess_per_second):.1f}")
    print(f"  Max R-hat: {jnp.max(auto_rhat):.4f}")
    print(f"  Convergence: {'✓ Good' if jnp.max(auto_rhat) < 1.01 else '⚠ Needs more steps'}")
    
    # Step 4: Print final comparison
    print("\n" + "="*70)
    print("FINAL COMPARISON: BAYESOPT vs AUTO-TUNING")
    print("="*70)
    print(f"\n{'Metric':<25} {'BayesOpt':<20} {'Auto-tuned':<20} {'Winner':<15}")
    print("-"*80)
    
    metrics = [
        ('Mean ESS', jnp.mean(bayesopt_ess), jnp.mean(auto_ess)),
        ('Mean ESS/n', jnp.mean(bayesopt_ess_per_step), jnp.mean(auto_ess_per_step)),
        ('Mean ESS/s', jnp.mean(bayesopt_ess_per_second), jnp.mean(auto_ess_per_second)),
        ('Max R-hat', jnp.max(bayesopt_rhat), jnp.max(auto_rhat)),
    ]
    
    for name, bo_val, auto_val in metrics:
        if 'R-hat' in name:
            winner = 'BayesOpt' if bo_val < auto_val else 'Auto-tuned'
            print(f"{name:<25} {bo_val:<20.4f} {auto_val:<20.4f} {winner:<15}")
        else:
            winner = 'BayesOpt' if bo_val > auto_val * 1.05 else ('Auto-tuned' if auto_val > bo_val * 1.05 else 'Tie')
            if 'ESS/n' in name:
                print(f"{name:<25} {bo_val:<20.4f} {auto_val:<20.4f} {winner:<15} ⭐")
            else:
                print(f"{name:<25} {bo_val:<20.1f} {auto_val:<20.1f} {winner:<15}")
    
    print("\n⭐ = Primary metric for fair comparison")
    
    # Determine overall winner based on ESS/n
    if jnp.mean(bayesopt_ess_per_step) > jnp.mean(auto_ess_per_step) * 1.1:
        print(f"\n✓ BayesOpt wins: {100*(jnp.mean(bayesopt_ess_per_step)/jnp.mean(auto_ess_per_step) - 1):.1f}% better ESS/n")
    elif jnp.mean(auto_ess_per_step) > jnp.mean(bayesopt_ess_per_step) * 1.1:
        print(f"\n✓ Auto-tuning wins: {100*(jnp.mean(auto_ess_per_step)/jnp.mean(bayesopt_ess_per_step) - 1):.1f}% better ESS/n")
    else:
        print(f"\n≈ Tie: ESS/n values are comparable")
    
    return bayesopt_samples, auto_samples, mams_results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_all_experiments(dim=5, num_chains=4, num_steps=1000):
    """Run all experiments with normalized ESS metrics."""
    print("\n" + "="*70)
    print("MCMC COMPARISON v8 - WITH NORMALIZED ESS METRICS")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Dimensionality: {dim}")
    print(f"  Validation chains: {num_chains}")
    print(f"  Steps per chain: {num_steps}")
    
    total_start = time.time()
    
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
    
    total_time = time.time() - total_start
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*70)
    print(f"Total time: {total_time:.2f}s ({total_time/60:.1f} min)")
    print("\n✓ All done!")
    print("\nKey takeaways:")
    print("  • ESS/n is the FAIR metric for comparing different (L, ε)")
    print("  • ESS/s shows practical wall-clock efficiency")
    print("  • Raw ESS can be misleading when n varies")
    print("  • Always normalize by computational cost for fair comparison")
    print("\nGenerated files:")
    print("  - bayesopt_timeseries.png (6 panels including ESS/n)")
    print("  - bayesopt_hyperparameter_space.png (colored by ESS/n)")
    
    return {
        'nuts': nuts_results,
        'mclmc': mclmc_results,
        'mams': mams_results,
        'bayesopt_samples': bayesopt_samples,
        'auto_samples': auto_samples
    }


if __name__ == "__main__":
    print("\n" + "="*70)
    print("MCMC HYPERPARAMETER TUNING v8")
    print("Fair Comparison with Normalized ESS Metrics")
    print("="*70)
    print("\nKey innovation:")
    print("  ✓ ESS/n (ESS per integration step) - fair comparison")
    print("  ✓ ESS/s (ESS per second) - wall-clock efficiency")
    print("  ✓ BayesOpt optimizes ESS/n for MCLMC/MAMS")
    print("  ✓ Hyperparameter space colored by ESS/n")
    
    print("\n" + "="*70)
    print("SELECT EXPERIMENT")
    print("="*70)
    print("\n1. Run all experiments (recommended)")
    print("2. Run only algorithm comparison")
    print("3. Run only tuning method comparison")
    print("4. Exit")
    
    choice = input("\nEnter choice (1-4) [default: 1]: ").strip()
    
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
        print("\nExiting...")
        exit(0)
        
    else:
        print(f"\nInvalid choice: {choice}")
        print("Running all experiments by default...")
        results = run_all_experiments(dim=5, num_chains=4, num_steps=1000)
    
    print("\n" + "="*70)
    print("SCRIPT COMPLETED")
    print("="*70)