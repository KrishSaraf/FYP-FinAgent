"""
Portfolio weights projection utilities for constraint enforcement.
"""

import numpy as np
from typing import Tuple


def project_weights(
    w_raw: np.ndarray, 
    w_max: float = 0.10, 
    target_net: float = 1.0, 
    gross_cap: float = 1.5
) -> np.ndarray:
    """
    Project raw weights to satisfy portfolio constraints.
    
    Constraints:
    1. Individual position limit: |w_i| <= w_max
    2. Gross exposure limit: sum(|w_i|) <= gross_cap  
    3. Net exposure target: sum(w_i) = target_net
    
    Args:
        w_raw: Raw weight vector (n_stocks,)
        w_max: Maximum absolute weight per stock
        target_net: Target net exposure (sum of weights)
        gross_cap: Maximum gross exposure (sum of absolute weights)
        
    Returns:
        Projected weights satisfying all constraints
    """
    w_raw = np.asarray(w_raw)
    n_stocks = len(w_raw)
    
    # Step 1: Clip individual positions to w_max
    w_clipped = np.clip(w_raw, -w_max, w_max)
    
    # Step 2: Check if gross exposure constraint is satisfied
    gross_exposure = np.sum(np.abs(w_clipped))
    
    if gross_exposure <= gross_cap:
        # Gross constraint satisfied, now adjust for net target
        current_net = np.sum(w_clipped)
        net_adjustment = target_net - current_net
        
        # Distribute net adjustment proportionally
        if gross_exposure > 0:
            # Proportional adjustment based on current weights
            adjustment_weights = w_clipped / gross_exposure
        else:
            # Equal adjustment if no current exposure
            adjustment_weights = np.ones(n_stocks) / n_stocks
            
        w_adjusted = w_clipped + net_adjustment * adjustment_weights
        
        # Re-clip to ensure individual constraints still satisfied
        w_final = np.clip(w_adjusted, -w_max, w_max)
        
    else:
        # Gross constraint violated, need to scale down
        scale_factor = gross_cap / gross_exposure
        w_scaled = w_clipped * scale_factor
        
        # Now adjust for net target
        current_net = np.sum(w_scaled)
        net_adjustment = target_net - current_net
        
        # Distribute net adjustment proportionally
        if gross_cap > 0:
            adjustment_weights = w_scaled / gross_cap
        else:
            adjustment_weights = np.ones(n_stocks) / n_stocks
            
        w_adjusted = w_scaled + net_adjustment * adjustment_weights
        
        # Final clipping to ensure all constraints satisfied
        w_final = np.clip(w_adjusted, -w_max, w_max)
    
    # Final validation
    assert np.all(np.abs(w_final) <= w_max + 1e-8), "Individual constraint violated"
    assert np.sum(np.abs(w_final)) <= gross_cap + 1e-8, "Gross constraint violated"
    
    return w_final


def validate_weights(
    w: np.ndarray, 
    w_max: float = 0.10, 
    target_net: float = 1.0, 
    gross_cap: float = 1.5
) -> Tuple[bool, str]:
    """
    Validate that weights satisfy all constraints.
    
    Args:
        w: Weight vector to validate
        w_max: Maximum absolute weight per stock
        target_net: Target net exposure
        gross_cap: Maximum gross exposure
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    w = np.asarray(w)
    
    # Check individual constraints
    if np.any(np.abs(w) > w_max + 1e-8):
        return False, f"Individual constraint violated: max weight {np.max(np.abs(w)):.4f} > {w_max}"
    
    # Check gross constraint
    gross_exposure = np.sum(np.abs(w))
    if gross_exposure > gross_cap + 1e-8:
        return False, f"Gross constraint violated: {gross_exposure:.4f} > {gross_cap}"
    
    # Check net constraint (allow small tolerance)
    net_exposure = np.sum(w)
    if abs(net_exposure - target_net) > 1e-6:
        return False, f"Net constraint violated: {net_exposure:.6f} != {target_net}"
    
    return True, "All constraints satisfied"


def calculate_exposure_metrics(w: np.ndarray) -> dict:
    """
    Calculate exposure metrics for a weight vector.
    
    Args:
        w: Weight vector
        
    Returns:
        Dictionary with exposure metrics
    """
    w = np.asarray(w)
    
    return {
        'net_exposure': np.sum(w),
        'gross_exposure': np.sum(np.abs(w)),
        'long_exposure': np.sum(w[w > 0]),
        'short_exposure': np.sum(np.abs(w[w < 0])),
        'num_long': np.sum(w > 0),
        'num_short': np.sum(w < 0),
        'num_neutral': np.sum(w == 0),
        'max_weight': np.max(np.abs(w)),
        'concentration': np.sum(w**2),  # Herfindahl index
    }
