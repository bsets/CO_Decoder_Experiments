# Minimal HGS inference/decoder dependencies

This bundle removes `Sampler.py`.

## Why `Sampler.py` was removed

The only function needed from `Sampler.py` was:

```text
getclicnum(...)
```

That function is now inlined inside:

```text
hgs_aws/code/run_hgs_large_graph_decoder.py
```

Therefore `Sampler.py` is no longer needed for the large-graph inference/decoder pipeline.

## Files still required

These four files are still required:

```text
models.py
layers.py
diff_module.py
utils.py
```

Reason:

1. The trained `.pth` models are loaded with `torch.load(...)`.
2. If the model was saved as a full Python object, PyTorch needs the original model class definitions at load time.
3. `models.py` defines the HGS model classes and imports:
   - `diff_module.py`
   - `layers.py`
4. `layers.py` and `diff_module.py` both import `utils.py`.

So removing any of these four files may cause either:

```text
ModuleNotFoundError
```

during `torch.load(...)`, or a runtime error during the HGS forward pass.

## Minimal code folder

The minimal code folder is:

```text
hgs_aws/code/
├── build_hgs_large_graph_preposs.py
├── run_hgs_large_graph_decoder.py
├── utils.py
├── models.py
├── layers.py
└── diff_module.py
```

## Optional future reduction

A more aggressive cleanup is possible only if the trained models are re-saved as `state_dict` files and the repo defines a clean packaged HGS model class. For the current `.pth` files, keep the four model-support files.
