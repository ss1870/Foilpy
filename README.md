# Foilpy: A python toolkit for the design and analysis of hydrofoils
![Tests](https://github.com/ss1870/Foilpy/actions/workflows/tests.yml/badge.svg)

Foilpy enables a user to parametrically define and visualise hydrofoil wings. The wings can then be assembled into a conventional hydrofoil setup i.e. with main wing, stabiliser wing, and mast. Foilpy can also calculate the quasi-steasy forces/moments produced by the hydrofoil assembly, at a given speed and orientation to the flow. Hydrodynamic analysis is conducted using vortex lifting line theory with a free vortex wake, therefore, the effect of induced flow is captured. Lastly, it is also possible to output a high-resolution .stl file of any given wing for processing in CAD/3D printing.

Please try out the notebooks in the examples directory to see how to use the various functionality within Foilpy.


## Installing Foilpy
Please note that Foilpy makes use of the [JAX](https://github.com/google/jax) package which requires a Mac or Linux operating system. 

### pip installation

Foilpy can be installed with the following steps:

1. Clone the repository into a suitable location.
2. Activate your desired python environment.
3. Change the current directory to the local Foilpy directory.
4. Run 'python setup.py install'. Or 'python setup.py develop' if you wish to make modifications to the codebase.

