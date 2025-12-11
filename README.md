# ReAPtree: Global Explanations for Tree Ensembles through Relaxation Optimization

This repository provides **ReAPtree**, a global explanation method for tree ensemble models (e.g., Random Forest, GBDT) based on **relaxation optimization**.  

`example.py` demonstrates how to load an explanation task, set parameters, and run ReAPtree.

---

## Key Parameters

### `vector_distance`
Distance metric used in the optimization:  
`L1`, `L2`, `KL`, `CL`, `NL`

Rules:
- `CL` / `NL` → `value_space` must be `probability`
- If the model is **GBDT**, `NL` is **not supported**

### `value_space`
Specifies how model outputs are represented:  
- `probability`  
- `onehot` 

### Other parameters
- `max_step`: maximum search depth of ReAPtree  
- `subaptree_max_step`: maximum depth for SubspaceAPtree refinement  
- `n_jobs`: number of parallel threads (`-1` uses all available cores)

---

## Run Example

```bash
python example.py
