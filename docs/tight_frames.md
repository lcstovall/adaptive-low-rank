# Tight Frame Union Model

A flexible union-of-subspaces dataset where each subspace is populated by a tight frame.

## Quick Start

```python
from src.adaptive_low_rank.synthetic import generate_multiscale_dataset

# Generate dataset with 5 tight frames
ds = generate_multiscale_dataset(
    model='tight_frame_union',
    n_clusters=5,            # 5 orthogonal subspaces
    columns_per_cluster=10,  # 10 frame vectors per subspace
    tube_dim=4,              # dimension of each subspace
    seed=0
)

# Access results
print(ds.X.shape)        # (d, 50) - 5 clusters × 10 points
print(ds.labels)         # cluster membership for each column
print(ds.central_energies)  # frame levels λ_ℓ for each cluster
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_clusters` | `int` | - | Number of tight frames (subspaces) L |
| `columns_per_cluster` | `int \| Array` | - | Vectors per frame; int (uniform) or array of length n_clusters |
| `tube_dim` | `int \| Array` | - | Dimension of each subspace; int (uniform) or array of length n_clusters |
| `ambient_dim` | `int \| None` | None | Ambient space dimension d; auto-computed if None |
| `high_energy` | `float` | 1.0 | Highest frame level λ₁ |
| `condition_number` | `float` | 1e4 | Ratio κ = λ₁ / λ_L |
| `decay_rate` | `str` | `'polynomial'` | How energy levels decrease: `'linear'`, `'polynomial'`, or `'exponential'` |
| `seed` | `int` | 0 | Random seed for reproducibility |

## Energy Level Decay

The frame levels λ₁ > λ₂ > ... > λ_L are always strictly decreasing with ratio κ = λ₁ / λ_L.

### Decay Patterns

**Linear** (`'linear'`): Arithmetic sequence
```
λ = [1.0, 0.75, 0.5, 0.25, 0.01]  (uniform decrease)
```

**Polynomial** (`'polynomial'`, default): 1/t² decay
```
λ = [1.0, 0.0947, 0.0331, 0.0167, 0.01]  (slower decay)
```

**Exponential** (`'exponential'`): Geometric sequence
```
λ = [1.0, 0.316, 0.1, 0.0316, 0.01]  (faster decay)
```

## Flexible Per-Cluster Parameters

Each cluster can have its own `columns_per_cluster` and `tube_dim`.

### Uniform (Backward Compatible)

```python
# All clusters identical
ds = generate_multiscale_dataset(
    model='tight_frame_union',
    n_clusters=4,
    columns_per_cluster=10,  # int: same for all
    tube_dim=5,              # int: same for all
    seed=0
)
```

### Variable Per-Cluster

```python
# Each cluster different
ds = generate_multiscale_dataset(
    model='tight_frame_union',
    n_clusters=4,
    columns_per_cluster=[10, 15, 20, 25],  # array: one per cluster
    tube_dim=[3, 4, 5, 6],                 # array: one per cluster
    seed=0
)
```

### Constraints

- **Array length must equal n_clusters**
- **tube_dim ≥ columns_per_cluster for each cluster** (no overcomplete frames)
- **Same parameter type for both**: use int for uniform or array for per-cluster

## Mathematical Properties

### Tight Frame Definition

Each cluster ℓ generates a data matrix X_ℓ ∈ ℝ^(d × n_ℓ) satisfying:

```
X_ℓ X_ℓᵀ = λ_ℓ P_ℓ
```

where:
- λ_ℓ is the frame level (central energy)
- P_ℓ is the orthogonal projector onto subspace W_ℓ

### Covariance Structure

The combined dataset X = [X₁ | X₂ | ... | X_L] has covariance:

```
C = X Xᵀ = Σ_ℓ λ_ℓ P_ℓ
```

This is a **block-diagonal** structure with energy levels λ₁ > λ₂ > ... > λ_L.

## Common Patterns

### Pattern 1: Uniform Tight Frames (Standard)

All clusters identical:
```python
ds = generate_multiscale_dataset(
    model='tight_frame_union',
    n_clusters=100,
    columns_per_cluster=10,  # 10 frame vectors in each subspace
    tube_dim=5,               # each in 5D
    decay_rate='polynomial',  # polynomial energy hierarchy
    seed=0
)
# Result: 100 subspaces × 5 dimensions = 500 total rows
```

### Pattern 2: Hierarchical Tight Frames

Mixed dimensions and densities:
```python
ds = generate_multiscale_dataset(
    model='tight_frame_union',
    n_clusters=4,
    columns_per_cluster=[10, 10, 2, 2],  # Dense then sparse
    tube_dim=[3, 3, 10, 10],              # Low then high dimension
    decay_rate='exponential',
    seed=0
)
# Cluster 0-1: Dense low-dimensional (10 points in 3D)
# Cluster 2-3: Sparse high-dimensional (2 points in 10D)
```

### Pattern 3: Sparse Undercomplete Frames

Few points in high-dimensional subspaces:
```python
ds = generate_multiscale_dataset(
    model='tight_frame_union',
    n_clusters=10,
    columns_per_cluster=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    tube_dim=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    decay_rate='linear',
    seed=0
)
# Each cluster has fewer points than dimension (undercomplete)
```

## Configuration Files

Create YAML files for reproducible experiments:

```yaml
# configs/tight_frame_union.yml
model: tight_frame_union
n_clusters: 10
columns_per_cluster: 5
tube_dim: 6
ambient_dim: null  # auto-computed
high_energy: 1.0
condition_number: 1000
decay_rate: polynomial
seed: 0
```

Load and use:
```python
import yaml
from src.adaptive_low_rank.synthetic import generate_multiscale_dataset

with open('configs/tight_frame_union.yml') as f:
    config = yaml.safe_load(f)

ds = generate_multiscale_dataset(**config)
```

## Validation & Error Handling

| Error | Cause | Solution |
|-------|-------|----------|
| `tube_dim < columns_per_cluster` | Overcomplete frame | Use `tube_dim ≥ columns_per_cluster` |
| `length X but n_clusters=Y` | Array length mismatch | Ensure array has exactly `n_clusters` elements |
| `decay_rate must be one of` | Invalid decay rate | Use `'linear'`, `'polynomial'`, or `'exponential'` |
| `ambient_dim too small` | Insufficient space | Increase `ambient_dim` to at least `sum(tube_dim)` |

## Performance Notes

- **Uniform parameters** (scalar): Fast and memory-efficient
- **Variable parameters** (array): Same performance, slightly more flexible
- All decay patterns have equivalent computational cost
- Central energies always strictly ordered: λ₁ > λ₂ > ... > λ_L

## References

The tight frame model is based on the assumption described in:
- **Assumption**: Union-of-subspaces where each subspace contains a tight frame
- **Property**: Covariance decomposes as C = Σ_ℓ λ_ℓ P_ℓ with strict energy hierarchy
- **Applications**: Benchmarking column selection algorithms on structured data

---

**Last Updated**: 2024  
**Features**: Flexible decay rates, per-cluster parameters, full backward compatibility
