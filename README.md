# (1+ε)-Approximate Ball Tree Search

An implementation of the Arya & Mount (1998) ANN relaxation applied to ball trees. The idea is simple: exact ball-tree search prunes a subtree when its closest possible point is already farther than the current best candidate. The (1+ε) relaxation prunes more aggressively — skip a subtree if even its closest point can't beat the current best *by more than a factor of (1+ε)*. The returned neighbor is guaranteed within (1+ε)·d\* of the true nearest neighbor.

This is one line of code:

```python
# exact
if lb_sq > best_dist_sq:
    return

# (1+ε)-ANN
if lb_sq > best_dist_sq / (1 + eps)**2:   # note: squared distance space
    return
```

The same trick has been standard for kd-trees since Arya & Mount; applying it explicitly to ball trees doesn't appear to have been written up before, though the transfer is straightforward.

## What's in the notebook

- Ball tree construction (flat numpy arrays, Numba-friendly)
- Exact and approximate query with branch counting
- Correctness verification against sklearn
- Guarantee verification across ε values
- Benchmarks on GloVe-25, SIFTsmall, and synthetic clustered data
- Dimensionality sweep at fixed ε

## Scope

This is an exploration, not a proposal to use ball trees for production ANN. Ball trees don't compete with HNSW or DiskANN at scale, and the ε trick doesn't change that. The interest here is in whether the guarantee transfers cleanly and what the pruning behaviour looks like empirically — it does, and it's well-behaved.

## Reference

Arya, S. & Mount, D. M. (1998). Approximate nearest neighbor queries in fixed dimensions. *Journal of the ACM*, 45(6), 891–923.
