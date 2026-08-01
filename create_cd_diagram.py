"""
Create Critical Difference Diagram for JMLR Paper
==================================================
Based on Demšar (2006) methodology
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def create_cd_diagram(ranks, cd, output_file='cd_diagram.pdf'):
    """
    Create a Critical Difference diagram.

    Parameters:
    -----------
    ranks : dict - Method names to average ranks
    cd : float - Critical difference value
    output_file : str - Output file path
    """
    methods = list(ranks.keys())
    rank_values = [ranks[m] for m in methods]

    # Sort by rank
    sorted_idx = np.argsort(rank_values)
    methods = [methods[i] for i in sorted_idx]
    rank_values = [rank_values[i] for i in sorted_idx]

    n_methods = len(methods)

    fig, ax = plt.subplots(figsize=(10, 4))

    # Draw the axis
    ax.set_xlim(0.5, n_methods + 0.5)
    ax.set_ylim(0, 1)

    # Draw rank axis at top
    for i in range(1, n_methods + 1):
        ax.axvline(x=i, ymin=0.85, ymax=0.95, color='black', linewidth=1)
        ax.text(i, 0.97, str(i), ha='center', va='bottom', fontsize=10)

    ax.plot([1, n_methods], [0.9, 0.9], 'k-', linewidth=2)

    # Draw CD bar
    cd_start = 0.5
    ax.plot([cd_start, cd_start + cd], [0.75, 0.75], 'k-', linewidth=3)
    ax.text((cd_start + cd_start + cd) / 2, 0.78, f'CD = {cd:.2f}', ha='center', fontsize=10)

    # Draw methods
    left_methods = []
    right_methods = []

    for method, rank in zip(methods, rank_values):
        if rank <= (n_methods + 1) / 2:
            left_methods.append((method, rank))
        else:
            right_methods.append((method, rank))

    # Left side (lower ranks = better)
    y_pos = 0.6
    for method, rank in left_methods:
        ax.plot([rank, rank], [0.85, y_pos + 0.05], 'k-', linewidth=1)
        ax.plot([rank, 0.3], [y_pos + 0.05, y_pos + 0.05], 'k-', linewidth=1)
        ax.text(0.25, y_pos + 0.05, method, ha='right', va='center', fontsize=10,
                fontweight='bold' if method in ['Static', 'DSS'] else 'normal')
        y_pos -= 0.12

    # Right side (higher ranks = worse)
    y_pos = 0.6
    for method, rank in reversed(right_methods):
        ax.plot([rank, rank], [0.85, y_pos + 0.05], 'k-', linewidth=1)
        ax.plot([rank, n_methods + 0.7], [y_pos + 0.05, y_pos + 0.05], 'k-', linewidth=1)
        ax.text(n_methods + 0.75, y_pos + 0.05, method, ha='left', va='center', fontsize=10)
        y_pos -= 0.12

    # Draw connections for methods not significantly different
    # Methods with rank difference < CD are connected
    connected_groups = []
    for i, (m1, r1) in enumerate(zip(methods, rank_values)):
        for j, (m2, r2) in enumerate(zip(methods, rank_values)):
            if i < j and abs(r1 - r2) < cd:
                connected_groups.append((r1, r2))

    # Draw thick lines connecting non-significant groups
    y_line = 0.82
    for r1, r2 in connected_groups:
        ax.plot([r1, r2], [y_line, y_line], 'k-', linewidth=3)
        y_line -= 0.02

    ax.axis('off')
    ax.set_title('Critical Difference Diagram (Nemenyi test, α=0.05)', fontsize=12, pad=20)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_file.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"Saved to {output_file}")

    return fig


if __name__ == "__main__":
    # Results from Phase 3 experiments
    ranks = {
        'RF+Thresh': 1.83,
        'Static': 2.67,
        'DSS': 3.00,
        'GB+Thresh': 3.50,
        'EasyEns': 4.00,
        'RUSBoost': 6.00
    }
    cd = 3.078

    print("Creating Critical Difference Diagram...")
    print(f"Ranks: {ranks}")
    print(f"CD = {cd}")

    fig = create_cd_diagram(ranks, cd, 'figures/cd_diagram.pdf')

    # Also create summary table
    print("\n" + "=" * 50)
    print("SIGNIFICANT DIFFERENCES")
    print("=" * 50)

    methods = list(ranks.keys())
    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            if i < j:
                diff = abs(ranks[m1] - ranks[m2])
                sig = "YES" if diff > cd else "no"
                print(f"{m1} vs {m2}: diff={diff:.2f}, significant={sig}")
