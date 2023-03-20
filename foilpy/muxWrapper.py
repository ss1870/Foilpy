from foilpy.foildef import FoilAssembly, LiftingSurface
from foilpy.utils import knts2ms
import foilpy.splines.curve as spl

import machupX as MX
import airfoil_db as adb
import numpy as np
import math as m
import json
import os

class MachUpXWrapper():
    """
    This class provides an interface between foil designs and MachUpX.
    """
    def __init__(self, baseDir: str) -> None:
        
        self._atmosphere: dict or None = None
        self._run: dict or None = None
        self._solver: dict or None = None
        self._foilDef: dict or None = None
        self._inputDict: dict or None = None
        self.setupGenericDicts()

        self._baseDir: str = baseDir
        self._afoil_fNames: list = []
        self._airfoils = dict()
        self._afoilDir: str = os.path.join(baseDir, 'afoil_polars')
        if not os.path.exists(self._afoilDir):
            os.makedirs(self._afoilDir)

    def foilFromFoilAssembly(self, foil: FoilAssembly):

        self._getAirfoils(foil.main_wing, minRe=1e3)
        self._getAirfoils(foil.stabiliser, minRe=1e3)
        self._getAirfoils(foil.mast, minRe=50e3)

        # Definition of foil assembly (modelled as a single aircraft)
        self._foilDef = dict(
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
        self._inputDict['scene']['aircraft']['foil1']['file'] = self._foilDef

    def foilFromFoilDef(self, foil: dict, unsweep: bool =True):
        # Definition of foil assembly from .foil file
        elmts = foil['Elements']
        nElmts = len(elmts)
        wings = dict()
        wings[elmts[1]['Name']] = self._foilElmt2dict(elmts[1], 1, isMain=True, unsweep=unsweep)
        # wings[elmts[0]['Name']] = self._foilElmt2dict(elmts[0], 2, isMain=False)

        self._foilDef = dict(
            CG = [0,0,0],
            weight = 0,
            reference = dict(),
            controls = dict(),
            # airfoils = dict(),
            wings = wings
        )
        self._inputDict['scene']['aircraft']['foil1']['file'] = self._foilDef

    def setupGenericDicts(self,
                   rho: float = 1025,
                   viscosity: float = 1.307e-6, # https://www.engineersedge.com/physics/water__density_viscosity_specific_weight_13146.htm
                   solver: str = 'linear', # 'linear', 'nonlinear', or 'scipy_fsolve'
                   ) -> dict:
        # Atmosphere dict
        self._atmosphere = dict(
            rho = rho,
            viscosity = viscosity  
        )
        # Run dict
        self._run = dict(
            display_wireframe = dict(show_legend = True),
            solve_forces = dict(non_dimensional = False),
            derivatives = dict()
        )
        # Solver
        self._solver = dict(
            type = solver, 
            convergence = 1e-10,
            relaxation = 1.0,
            max_iterations = 100,
        )
        # Main input dict
        self._inputDict = dict(
            run = self._run,
            solver = self._solver,
            units = 'SI',
            scene = dict(
                atmosphere = self._atmosphere,
                # dict of aircraft in the scene (typically only need 1 foil assembly)
                aircraft = dict(
                    foil1 = dict(
                        file = dict(),  # dict containing aircraft description
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

    def setVelocitykts(self, velKts: float):
        self._inputDict['scene']['aircraft']['foil1']['state']['velocity'] = knts2ms(velKts)

    def setAngleDeg(self, alpha: float):
        self._inputDict['scene']['aircraft']['foil1']['state']['alpha'] = alpha

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
    
    def _foilElmt2dict(self, foilElmt: dict,
                       id: int, isMain: bool,
                       unsweep: bool = False,
                       nSegs = 40,
                       nPts = 500
                       ) -> dict:
        out = dict(
            ID = id,
            is_main = isMain,
            side = 'right',
            connect_to = dict(),
            twist = [],
            chord = [],
            quarter_chord_locs = [],
            # airfoil = [],
            grid = dict(
                N = nSegs,
                distribution = 'cosine_cluster'
            )
        )

        if foilElmt['DuplicatedAndReflected']:
            out['side'] = 'both'
        
        # Get rondure
        knots = foilElmt['Rondure']['tKU']
        tStations = np.array(foilElmt['Rondure']['tStations'])
        B = np.array(foilElmt['Rondure']['B'])
        U = np.concatenate(([knots[0]]*3, knots, [knots[-1]]*3))
        rond = spl.BSplineCurve(3, U=U, P=B)
        rond.def_mapping(plot_flag=False)

        # Get sweep
        knots = tStations[foilElmt['Sweep']['stationFlags']]
        B = np.array(foilElmt['Sweep']['B']).reshape(-1,1)
        U = np.concatenate(([knots[0]]*3, knots, [knots[-1]]*3))
        sweep = spl.BSplineCurve(3, U=U, P=B)
        sweep.def_mapping(plot_flag=False)

        # Get twist
        knots = tStations[foilElmt['Twist']['stationFlags']]
        B = np.array(foilElmt['Twist']['B']).reshape(-1,1)
        U = np.concatenate(([knots[0]]*3, knots, [knots[-1]]*3))
        twistSpl = spl.BSplineCurve(3, U=U, P=B)
        twistSpl.def_mapping(plot_flag=False)

        # Get chord
        knots = tStations[foilElmt['Chord']['stationFlags']]
        B = np.array(foilElmt['Chord']['B']).reshape(-1,1)
        U = np.concatenate(([knots[0]]*3, knots, [knots[-1]]*3))
        chordSpl = spl.BSplineCurve(3, U=U, P=B)
        chordSpl.def_mapping(plot_flag=False)

        # Get sectionID
        knots = tStations[foilElmt['SectionID']['stationFlags']]
        B = np.array(foilElmt['SectionID']['B']).reshape(-1,1)
        U = np.concatenate(([knots[0]]*3, knots, [knots[-1]]*3))
        sectionID = spl.BSplineCurve(3, U=U, P=B)
        sectionID.def_mapping(plot_flag=False)

        # Get sections
        sec = foilElmt['Sections']
        nSections = len(sec)
        origin = []
        for s in sec:
            if 'Origin' in s:
                if s['Origin'] == 'Hinge':
                    origin.append(s['HingePosition'])
                else:
                    raise Exception('')
                
            else:
                origin.append([0.25, 0])
        origin = np.array(origin)
        
        t = np.linspace(0, 1, nPts)
        quChord = np.zeros((nPts, 3))
        chord = np.zeros((nPts))
        twist = np.zeros((nPts))
        for i in range(nPts):
            quChord[i, 1:] = rond.eval_curve(t[i])
            quChord[i, 2] = quChord[i, 2]
            quChord[i, 0] = sweep.eval_curve(t[i])
            chord[i] = chordSpl.eval_curve(t[i])
            twist[i] = twistSpl.eval_curve(t[i])
            secID = sectionID.eval_curve(t[i])
            if nSections == 1:
                orig = origin[0,:]
            else:
                orig = np.interp(secID, np.arange(nSections), origin)
            quChord[i, 0] += chord[i] * (orig[0] - 0.25)
        
        if unsweep:
            out['quarter_chord_locs'] = np.stack((
                                            quChord[:,0] - quChord[:,0], 
                                            quChord[:,1] - quChord[0,1],
                                            quChord[0,2] - quChord[:,2]
                                        ), axis=1).tolist()
        else:
            out['quarter_chord_locs'] = np.stack((
                                            quChord[:,0] - quChord[0,0], 
                                            quChord[:,1] - quChord[0,1],
                                            quChord[0,2] - quChord[:,2]
                                        ), axis=1).tolist()
        normSpan = quChord[:,1] / quChord[-1,1]
        out['connect_to']['ID'] = 0
        out['connect_to']['dx'] = quChord[0,0]
        out['connect_to']['dy'] = quChord[0,1]
        out['connect_to']['dz'] = -quChord[0,2]
        out['chord'] = np.stack((normSpan, chord), axis=1).tolist()
        out['twist'] = np.stack((normSpan, twist), axis=1).tolist()
        
        return out

    def _surf2dict(self, surface: LiftingSurface, 
                    id: int, isMain: bool,
                    unsweep: bool = False,
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

        if unsweep:
            out['quarter_chord_locs'] = np.stack((
                                            quChord[:,1] - quChord[:,1], 
                                            quChord[:,0] - quChord[0,0],
                                            quChord[0,2] - quChord[:,2]
                                        ), axis=1).tolist()
        else:
            out['quarter_chord_locs'] = np.stack((
                                            quChord[:,1] - quChord[0,1], 
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







