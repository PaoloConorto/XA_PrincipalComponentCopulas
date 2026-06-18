# Principal Component Copula

A Python implementation of the Principal Component Copula (PCC), which combines
copula-based dependence modelling with PCA to capture tail dependence along the
leading principal directions of high-dimensional data (Gubbels et al., 2025).
It accompanies the MSc thesis *Principal Component Copulas for Cross-Asset
Modelling* (Conorto, 2026, University of Groningen).

The core is the `PrincipalComponentCopula` class in
[principal_component_copula.py](principal_component_copula.py), which supports
the `"t"`, `"cross"`, and `"normal"` generator families and provides simulation
and confidence-interval methods for fitted instances.

## Installation

```
pip install -r requirments.txt
```

## Citation

If you use this code in academic or published work, please cite it. A
machine-readable description is provided in [CITATION.cff](CITATION.cff); GitHub
renders it as a "Cite this repository" button.

> Conorto, P. (2026). *Principal Component Copulas for Cross-Asset Modelling*
> [Master's thesis, University of Groningen].

## License

Released under the [MIT License](LICENSE).
