import numpy as np
from scipy.optimize import minimize

class Optimiser():
    """
    This class defines an optimisation within Foilpy.
    """
    def __init__(self, design_variables=None, objective=None, constraints=None):
        self.design_variables = design_variables
        self.objective = objective
        self.constraints = constraints
        self.wrapper = OptiWrapper()

    def initialise_dvs(self):
        """Initialises information about design variables."""
        # loop through DVs and add relevant info to dictionary
        for variable in self.design_variables:
            print(variable)

    def optimise(self):
        """
        Call optimiser function.
        """
        obj_fun = lambda x: self.wrapper.call(x, return_type='objective')
        x0 = []
        res = minimize(obj_fun,
                        x0,
                        args=(),
                        method=None,
                        jac=None,
                        hess=None,
                        hessp=None,
                        bounds=None,
                        constraints=(),
                        tol=None,
                        callback=None,
                        options=None)


class OptiWrapper():
    """
    This class defines the optiwrapper. It tracks the state of the objective/constraints,
    and also calls/defines the optiwrapper.
    """
    def __init__(self):
        self.dv_state = None
        self.current_objective = None
        self.current_constraints = None

    def call(self, x, return_type='objective'):
        """
        This method is essentially a decorator on the wrapper that checks if
        it has already been called for a given set of DVs. This avoids
        re-running the same code for both the objective and constraint calls.
        """
        # if current DV state is equivalent to the call being requested
        # then do not need to perform the wrapper.
        # Just return existing objective/constraint states.
        if np.any(self.dv_state != x):
            # if call being requested is new, then need to call the wrapper
            # to compute the new objective/constraint states
            self.current_objective, self.current_constraints = self.wrapper(x)

        if return_type == 'objective':
            return self.current_objective
        else:
            return self.current_constraints

    def wrapper(self, x):
        # Update design with new DV vector (x)


        # Perform requested analyses


        # Extract objective and constraints

        return [] # fx, gx
