# UMPFEA

UMPFEA is a toolset for 2D linear magnetic finite element analysis. The
analysis is magnetostatic: the flux distribution in a model is fully
determined by the electrical currents flowing in it.

Its primary purpose is the analysis of electrical machines in which
transverse forces act between the rotor and stator. It also applies
directly to magnetic bearings, shaft displacement sensing, and
semi-active magnetic devices that introduce circulatory forces onto a
rotor.

UMPFEA is fully scriptable, with structured meshing, bay replication,
and matrix reduction for faster solves.

## Installation

Requires Python 3.11 or later.

```
pip install git+https://github.com/eldarrahimov/umpfea.git
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
