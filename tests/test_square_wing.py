import numpy as np
from foilpy.foildef import LiftingSurface

def test_square_wing():
    """
    This function tests that the wing area, aspect ratio, and volume calculators are
    producing the expected outputs for a simple square wing.
    """

    afoil = [['tests/square', 0], 
            ['tests/square', 1]]
    square_wing = LiftingSurface(rt_chord=1,
                                 tip_chord=1,
                                 span=5,
                                 afoil=afoil,
                                 nsegs=40,
                                 units='m')

    assert np.isclose(square_wing.afoil_table[afoil[0][0]]['area'], 1.0, rtol=1e-03, atol=1e-03)
    assert np.isclose(square_wing.calc_AR(), 5.0, rtol=1e-03, atol=1e-03)
    assert np.isclose(square_wing.calc_proj_wing_area(), 5.0, rtol=1e-03, atol=1e-03)
    assert np.isclose(square_wing.calc_actual_wing_area(), 5.0, rtol=1e-03, atol=1e-03)
    assert np.isclose(square_wing.calc_wing_volume(), 5, rtol=1e-03, atol=1e-03)
