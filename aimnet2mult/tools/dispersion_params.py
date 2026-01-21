"""
Unified dispersion parameter interface for D3 and D4 corrections.

This module provides a consistent interface to retrieve dispersion parameters
for various DFT functionals from the dftd3 and dftd4 packages.

Usage:
    from aimnet2mult.tools.dispersion_params import get_dispersion_params, list_functionals

    # Get D3-BJ parameters for B3LYP
    params = get_dispersion_params('b3lyp', method='d3bj')
    # Returns: {'s6': 1.0, 's8': 1.9889, 'a1': 0.3981, 'a2': 4.4211}

    # Get D4 parameters for PBE
    params = get_dispersion_params('pbe', method='d4')
    # Returns: {'s6': 1.0, 's8': 0.9595, 's9': 1.0, 'a1': 0.3857, 'a2': 4.8069}

    # List available functionals
    d3_funcs = list_functionals('d3bj')
    d4_funcs = list_functionals('d4')

References:
    - DFT-D3: https://dftd3.readthedocs.io/
    - DFT-D4: https://dftd4.readthedocs.io/
    - D3-BJ: J. Chem. Phys. 2011, 132, 154104 (DOI: 10.1063/1.3382344)
    - D4: J. Chem. Phys. 2019, 150, 154122 (DOI: 10.1063/1.5090222)
"""

import os
from typing import Dict, List, Optional, Any, Tuple


# Try to import dftd3 and dftd4 packages
_HAS_DFTD3 = False
_HAS_DFTD4 = False
_D3_DATABASE = None
_D4_DATABASE = None

try:
    from dftd3 import parameters as d3_params
    _HAS_DFTD3 = True
except ImportError:
    d3_params = None

try:
    from dftd4 import parameters as d4_params
    _HAS_DFTD4 = True
except ImportError:
    d4_params = None


def _load_d3_database() -> dict:
    """Load and cache the D3 parameter database."""
    global _D3_DATABASE
    if _D3_DATABASE is None and _HAS_DFTD3:
        data_file = d3_params.get_data_file_name()
        _D3_DATABASE = d3_params.load_data_base(data_file)
    return _D3_DATABASE


def _load_d4_database() -> dict:
    """Load and cache the D4 parameter database."""
    global _D4_DATABASE
    if _D4_DATABASE is None and _HAS_DFTD4:
        _D4_DATABASE = d4_params.get_all_damping_params()
    return _D4_DATABASE


def _normalize_functional_name(name: str) -> str:
    """Normalize functional name for lookup."""
    # Convert to lowercase, remove hyphens and underscores
    normalized = name.lower().replace('-', '').replace('_', '')
    return normalized


def get_d3_params(functional: str, damping: str = 'bj') -> Dict[str, float]:
    """
    Get DFT-D3 parameters for a functional.

    Args:
        functional: Name of the DFT functional (e.g., 'b3lyp', 'pbe', 'wb97m')
        damping: Damping type - 'bj' (Becke-Johnson), 'zero', 'bjm', 'zerom', 'op'

    Returns:
        Dictionary with keys: s6, s8, a1, a2 (for BJ damping)
        or: s6, s8, rs6 (for zero damping)

    Raises:
        ImportError: If dftd3 package is not installed
        ValueError: If functional or damping type not found
    """
    if not _HAS_DFTD3:
        raise ImportError("dftd3 package not installed. Install with: pip install dftd3")

    db = _load_d3_database()
    if db is None:
        raise RuntimeError("Failed to load D3 parameter database")

    func_norm = _normalize_functional_name(functional)

    # Find functional in database
    param_db = db.get('parameter', {})
    defaults = db.get('default', {}).get('parameter', {}).get(f'd3.{damping}', {})

    found_func = None
    for key in param_db.keys():
        if _normalize_functional_name(key) == func_norm:
            found_func = key
            break

    if found_func is None:
        available = list(param_db.keys())
        raise ValueError(f"Functional '{functional}' not found. Available: {available[:20]}...")

    # Get parameters for the damping type
    func_params = param_db[found_func].get('d3', {})
    if damping not in func_params:
        available_dampings = list(func_params.keys())
        raise ValueError(f"Damping '{damping}' not available for {functional}. "
                        f"Available: {available_dampings}")

    # Merge with defaults
    params = {**defaults}
    params.update(func_params[damping])

    # Remove non-numeric fields
    result = {}
    for key, value in params.items():
        if isinstance(value, (int, float)) and key not in ('doi',):
            result[key] = float(value)

    # Ensure s6 has default value if not specified
    if 's6' not in result:
        result['s6'] = 1.0

    return result


def get_d4_params(functional: str) -> Dict[str, float]:
    """
    Get DFT-D4 parameters for a functional.

    Args:
        functional: Name of the DFT functional (e.g., 'b3lyp', 'pbe', 'wb97x')

    Returns:
        Dictionary with keys: s6, s8, s9, a1, a2, alp

    Raises:
        ImportError: If dftd4 package is not installed
        ValueError: If functional not found
    """
    if not _HAS_DFTD4:
        raise ImportError("dftd4 package not installed. Install with: pip install dftd4")

    try:
        params = d4_params.get_damping_param(functional)
        return {k: float(v) for k, v in params.items() if isinstance(v, (int, float))}
    except Exception as e:
        if 'not found' in str(e).lower() or functional in str(e):
            db = _load_d4_database()
            available = list(db.keys()) if db else []
            raise ValueError(f"Functional '{functional}' not found in D4 database. "
                           f"Available: {available[:20]}...")
        raise


def get_dispersion_params(
    functional: str,
    method: str = 'd3bj',
    damping: str = 'bj'
) -> Dict[str, float]:
    """
    Get dispersion parameters for a functional using unified interface.

    Args:
        functional: Name of the DFT functional
        method: Dispersion method - 'd3bj', 'd3zero', 'd4'
        damping: For D3, the damping type ('bj', 'zero', 'bjm', 'zerom', 'op')

    Returns:
        Dictionary with dispersion parameters

    Examples:
        >>> get_dispersion_params('b3lyp', 'd3bj')
        {'s6': 1.0, 's8': 1.9889, 'a1': 0.3981, 'a2': 4.4211}

        >>> get_dispersion_params('pbe', 'd4')
        {'s6': 1.0, 's8': 0.9595, 's9': 1.0, 'a1': 0.3857, 'a2': 4.8069, 'alp': 16.0}
    """
    method = method.lower()

    if method in ('d3bj', 'd3-bj', 'd3'):
        return get_d3_params(functional, 'bj')
    elif method in ('d3zero', 'd3-zero'):
        return get_d3_params(functional, 'zero')
    elif method in ('d3bjm', 'd3-bjm'):
        return get_d3_params(functional, 'bjm')
    elif method in ('d3zerom', 'd3-zerom'):
        return get_d3_params(functional, 'zerom')
    elif method in ('d3op', 'd3-op'):
        return get_d3_params(functional, 'op')
    elif method in ('d4', 'dftd4'):
        return get_d4_params(functional)
    else:
        raise ValueError(f"Unknown dispersion method: {method}. "
                        f"Available: d3bj, d3zero, d3bjm, d3zerom, d3op, d4")


def list_functionals(method: str = 'd3bj') -> List[str]:
    """
    List all available functionals for a dispersion method.

    Args:
        method: Dispersion method - 'd3bj', 'd3zero', 'd4'

    Returns:
        List of functional names
    """
    method = method.lower()

    if method.startswith('d3'):
        if not _HAS_DFTD3:
            raise ImportError("dftd3 package not installed")
        db = _load_d3_database()
        return sorted(db.get('parameter', {}).keys())
    elif method in ('d4', 'dftd4'):
        if not _HAS_DFTD4:
            raise ImportError("dftd4 package not installed")
        db = _load_d4_database()
        return sorted(db.keys()) if db else []
    else:
        raise ValueError(f"Unknown method: {method}")


def get_dispersion_config(
    functional: str,
    method: str = 'd3bj'
) -> Dict[str, Any]:
    """
    Get a complete dispersion configuration dict for compile_jit.

    Args:
        functional: DFT functional name
        method: Dispersion method ('d3bj', 'd4', 'none')

    Returns:
        Configuration dict suitable for FidelityModelWithSAEAndDispersion

    Example:
        >>> config = get_dispersion_config('b3lyp', 'd3bj')
        >>> # Returns: {'type': 'd3bj', 'params': {'s8': 1.9889, 'a1': 0.3981, 'a2': 4.4211, 's6': 1.0}}
    """
    if method.lower() == 'none':
        return {'type': 'none'}

    params = get_dispersion_params(functional, method)

    # Map method name to compile_jit type
    if method.lower().startswith('d3'):
        disp_type = 'd3bj'  # All D3 variants use the same module
    elif method.lower() in ('d4', 'dftd4'):
        disp_type = 'd4'
    else:
        disp_type = method.lower()

    return {
        'type': disp_type,
        'params': params,
        'functional': functional,
        'method': method,
    }


def print_params_table(functionals: List[str] = None, method: str = 'd3bj'):
    """
    Print a formatted table of dispersion parameters.

    Args:
        functionals: List of functionals to display (default: common ones)
        method: Dispersion method
    """
    if functionals is None:
        functionals = ['b3lyp', 'pbe', 'pbe0', 'tpss', 'bp', 'blyp', 'wb97x', 'b97d']

    print(f"\n{method.upper()} Dispersion Parameters")
    print("=" * 70)

    if method.startswith('d3'):
        print(f"{'Functional':<15} {'s6':>8} {'s8':>10} {'a1':>10} {'a2':>10}")
        print("-" * 70)
        for func in functionals:
            try:
                p = get_dispersion_params(func, method)
                print(f"{func:<15} {p.get('s6', 1.0):>8.4f} {p.get('s8', 0):>10.4f} "
                      f"{p.get('a1', 0):>10.4f} {p.get('a2', 0):>10.4f}")
            except ValueError:
                print(f"{func:<15} {'N/A':>8}")
    elif method == 'd4':
        print(f"{'Functional':<15} {'s6':>8} {'s8':>10} {'s9':>8} {'a1':>10} {'a2':>10}")
        print("-" * 70)
        for func in functionals:
            try:
                p = get_dispersion_params(func, method)
                print(f"{func:<15} {p.get('s6', 1.0):>8.4f} {p.get('s8', 0):>10.4f} "
                      f"{p.get('s9', 0):>8.4f} {p.get('a1', 0):>10.4f} {p.get('a2', 0):>10.4f}")
            except ValueError:
                print(f"{func:<15} {'N/A':>8}")

    print()


# Convenience aliases for common functionals
COMMON_FUNCTIONALS = {
    # Hybrid GGA
    'b3lyp': 'B3LYP hybrid functional',
    'pbe0': 'PBE0 (PBE1PBE) hybrid',
    # GGA
    'pbe': 'PBE GGA',
    'bp': 'BP86 GGA',
    'blyp': 'BLYP GGA',
    # Meta-GGA
    'tpss': 'TPSS meta-GGA',
    # Range-separated
    'wb97x': 'wB97X range-separated hybrid',
    'wb97m': 'wB97M range-separated',
    'camb3lyp': 'CAM-B3LYP range-separated',
    # Double hybrid
    'b2plyp': 'B2PLYP double hybrid',
}


if __name__ == '__main__':
    # Demo: print parameters for common functionals
    print("=" * 70)
    print("Dispersion Parameter Database")
    print("=" * 70)

    print(f"\nInstalled packages:")
    print(f"  dftd3: {'Yes' if _HAS_DFTD3 else 'No'}")
    print(f"  dftd4: {'Yes' if _HAS_DFTD4 else 'No'}")

    if _HAS_DFTD3:
        funcs = list_functionals('d3bj')
        print(f"\nD3-BJ database: {len(funcs)} functionals")
        print_params_table(method='d3bj')

    if _HAS_DFTD4:
        funcs = list_functionals('d4')
        print(f"\nD4 database: {len(funcs)} functionals")
        print_params_table(method='d4')

    # Show example usage
    print("\nExample: Getting B3LYP parameters")
    print("-" * 40)
    if _HAS_DFTD3:
        p = get_dispersion_params('b3lyp', 'd3bj')
        print(f"D3-BJ: {p}")
    if _HAS_DFTD4:
        p = get_dispersion_params('b3lyp', 'd4')
        print(f"D4:    {p}")

    print("\nExample: Getting dispersion config for compilation")
    print("-" * 40)
    config = get_dispersion_config('pbe', 'd3bj')
    print(f"PBE D3-BJ config: {config}")
