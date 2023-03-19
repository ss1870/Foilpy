from foilpy.foildef import FoilAssembly, LiftingSurface
from foilpy.utils import knts2ms

import machupX as MX
import airfoil_db as adb
import numpy as np
import math as m
import json
import os

class MachUpXWrapper():
    """
    This class provides an interface between foilpy and MachUpX.
    """
    def __init__(self, foil: FoilAssembly, baseDir: str) -> None:
        
        self._baseDir: str = baseDir
        self._afoilDir: str = os.path.join(baseDir, 'afoil_polars')
        if not os.path.exists(self._afoilDir):
            os.makedirs(self._afoilDir)
        self._afoil_fNames: list = []
        self._airfoils = dict()
        self._getAirfoils(foil.main_wing)
        self._getAirfoils(foil.stabiliser)
        self._getAirfoils(foil.mast, minRe=50e3)

        # Definition of foil assembly (modelled as a single aircraft)
        foilDef = dict(
            CG = foil.cog.reshape(-1).tolist(),
            weight = foil.total_mass,
            reference = dict(),
            controls = dict(),
            airfoils = self._airfoils,
            wings = dict(
                mainWing = self._surf2dict(foil.main_wing, 
                                    id=1, isMain=True,
                                    mountingAngle = foil.wing_angle),
                stabiliser = self._surf2dict(foil.stabiliser, 
                                  id=2, isMain=False,
                                  mountingAngle = foil.stabiliser_angle),
                mast = self._surf2dict(foil.mast, id=3, isMain=False, mast=True),
            )
        )

        # Generic MachUpX inputs
        self._atmosphere = dict(
            rho = 1025,
            viscosity = 1.307e-6  # https://www.engineersedge.com/physics/water__density_viscosity_specific_weight_13146.htm
        )
        self._run = dict(
            display_wireframe = dict(show_legend = True),
            solve_forces = dict(non_dimensional = False),
            derivatives = dict()
        )
        self._solver = dict(
            type = 'linear', # 'linear', 'nonlinear', or 'scipy_fsolve'
            convergence = 1e-10,
            relaxation = 1.0,
            max_iterations = 100,
        )
        self._inputDict = dict(
            run = self._run,
            solver = self._solver,
            units = 'SI',
            scene = dict(
                atmosphere = self._atmosphere,
                # dict of aircraft in the scene (typically only need 1 foil assembly)
                aircraft = dict(
                    foil1 = dict(
                        file = foilDef,  # dict containing aircraft description
                        state = dict(
                            position = [0,0,0],
                            velocity = knts2ms(15),
                            alpha = 2,
                            beta = 0,
                        )
                    )
                )
            )
        )

    def _getAirfoils(self, surface: LiftingSurface,
                     minRe: float = 1e4) -> None:
        for i, afoil in enumerate(surface.afoil_table):
            afoilPath = os.path.join(self._afoilDir, afoil + '.txt')
            if not os.path.exists(afoilPath):
                afoilInput = dict(
                    type = 'database',
                    geometry = dict(
                        # NACA = afoil
                    )
                )
                if "naca" in afoil:
                    afoilInput['geometry']['NACA'] = afoil.replace('naca', '')
                else:
                    raise Exception('Coords not yet implemented.')

                airfoil = adb.Airfoil("my_airfoil", afoilInput, verbose=True)

                dofs = dict(
                    alpha = dict(
                        range = [m.radians(-15.0), m.radians(15.0)],
                        steps = 31,
                        index = 0
                    ),
                    Mach = 0.01,
                    Rey = dict(
                        range = [minRe, 5e6],
                        steps = 10,
                        index = 1
                    )
                )
                airfoil.generate_database(degrees_of_freedom=dofs,
                                N = 200,
                                max_iter = 100,
                                x_trip = 0.4,
                                N_crit = 0.5,
                                show_xfoil_output=False)
                airfoil.export_database(filename=afoilPath)

            self._afoil_fNames.append(afoilPath)
            
            self._airfoils[afoil] = dict(
                type = 'database',
                input_file = self._afoil_fNames[-1]
            )
            
    def _surf2dict(self, surface: LiftingSurface, 
                    id: int, isMain: bool,
                    mountingAngle: float = 0.0,
                    mast: bool = False,
                    connectTo: int = 0,
                    dx: float = 0.0,
                    dy: float = 0.0,
                    dz: float = 0.0,
                    nSegs = 40) -> dict:
        out = dict(
            ID = id,
            is_main = isMain,
            side = 'both',
            connect_to = dict(),
            twist = [],
            chord = [],
            quarter_chord_locs = [],
            airfoil = [],
            grid = dict(
                N = nSegs,
                distribution = 'cosine_cluster'
            )
        )

        if mast:
            out['side'] = 'right'

        # Supply quarter chord locations
        quChord = surface.qu_chord_loc
        nptsSemi = 0
        if not mast:
            nptsSemi = int(np.floor(quChord.shape[0]/2))
            quChord = quChord[nptsSemi:,:]

        out['quarter_chord_locs'] = np.stack((
                                        quChord[:,1] - quChord[:,1], 
                                        quChord[:,0] - quChord[0,0],
                                        quChord[0,2] - quChord[:,2]
                                    ), axis=1).tolist()
        normSpan = quChord[:,0] / quChord[-1,0]

        if connectTo > 0:
            out['connect_to']['ID'] = connectTo
            out['connect_to']['location'] = 'root'
            out['connect_to']['dx'] = dx
            out['connect_to']['dy'] = dy
            out['connect_to']['dz'] = dz
        else:
            out['connect_to']['ID'] = 0
            out['connect_to']['dx'] = quChord[0,1]
            out['connect_to']['dy'] = quChord[0,0]
            out['connect_to']['dz'] = -quChord[0,2]

        # Chord
        chord = np.linalg.norm(surface.LE[nptsSemi:] - surface.TE[nptsSemi:], axis=1)
        if np.all(chord == chord[0]):
            out['chord'] = chord[0]
        else:
            assert len(chord) == quChord.shape[0]
            out['chord'] = np.stack((normSpan, chord), axis=1).tolist()

        # Twist
        twist = surface.washout_curve[nptsSemi:]
        if np.all(twist == twist[0]):
            out['twist'] = twist[0] + mountingAngle
        else:
            assert len(twist) == quChord.shape[0]
            out['twist'] = np.stack((normSpan,
                                twist + mountingAngle), axis=1).tolist()

        # Airfoils
        out['airfoil'] = [[span, name] for name, span in surface.afoil]

        return out







