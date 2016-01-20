import pdb
import numpy as np
import simtk.unit as unit
import simtk.openmm as mm
from simtk.openmm import app
from copy import deepcopy
from sys import stdout
import itertools
from examolhelpers import *

'''
This script houses the integrators for the basisSim examol objects. Since they get long, I put it here
'''

class HybridLDMCIntegratorEngine(object):
    '''
    Custom hybrid Lambda-Dynamics/MonteCarlo Integrator Housing and operations.

    Carries out a number of lambda-dynamics MD steps then does a hybrid MD accept/reject at the end to determine if it shoudl accept the results

    NOTE: Computing dU/dL at each timestep makes this a slow process. Untenable in its current form
    -----------------------
    TODO:
    - Implement actual MC accept/reject at end of timestep
    '''

    def _constructIntegrator(self):
        #Set default theta (continuous variable that maps back to lambda) mappings
        integrator = mm.CustomIntegrator(self.timestep.value_in_unit(unit.picoseconds))
        standardParameter = 'lam{i:s}x{j:s}{b:s}'
        crossParameter = 'lam{i:s}x{j:s}x{i2:s}x{j2:s}{b:s}'
        integrator.addGlobalVariable('unaffectedPotential', 0)
        integrator.addGlobalVariable('nStandard', self._basisSim.standardNumBasis)
        integrator.addGlobalVariable('nCross', self._basisSim.crossNumBasis)
        derivsStandard = {}
        switchStandard = {}
        derivsCross = {}
        switchCross = {}
        for i in xrange(self._basisSim.Ni):
            for j in xrange(self._basisSim.Nj):
                integrator.addGlobalVariable('theta{0:s}x{1:s}'.format(str(i),str(j)), 0)
                integrator.addGlobalVariable('thetaV{0:s}x{1:s}'.format(str(i),str(j)), 0)
                integrator.addGlobalVariable('curentB{0:s}x{1:s}'.format(str(i),str(j)), 0)
                integrator.addGlobalVariable('f{0:s}x{1:s}'.format(str(i),str(j)), 0)
        #Determine how each parameter is linked to its bonded term
        for uniqueSetCount in xrange(self._basisSim.standardNumBasis):
            basisSet = self._basisSim._flatStandardUniqueBasis[uniqueSetCount]
            for basis in basisSet:
                if basis is 'R':
                    derivsStandard[basis] = "4*1.61995584*({0:s}^3) + 3*-0.8889962*({0:s}^2) + 2*0.02552684*({0:s}) + (1-1.61995584--0.8889962-0.2552684)"
                    switchStandard[basis] = "1.61995584*({0:s}^4) + -0.8889962*({0:s}^3) + 0.02552684*({0:s}^2) + (1-1.61995584--0.8889962-0.2552684)*{0:s}"
                else:
                    derivsStandard[basis] = '1+0*{0:s}' #For the sake of generality
                    switchStandard[basis] = '{0:s}' #
        for uniqueSetCount in xrange(self._basisSim.crossNumBasis):
            basisSet = self._basisSim._flatCrossUniqueBasis[uniqueSetCount]
            for basis in basisSet:
                derivsCross[basis] = '1+0*{0:s}' #For the sake of generality
                switchCross[basis] = '{0:s}' #

        #Helper Functions
        def assignBasisParameters(integrator):
            #Redistribute the dependent parameters based on current lamB
            for i in xrange(self._basisSim.Ni):
                for j in xrange(self._basisSim.Nj):
                    offSetCount = 0
                    for stageCount in xrange(len(self._basisSim.standardBasisPerStage)):
                        #Through each unique basis in that stage
                        for uniqueSetCount in xrange(self._basisSim.standardBasisPerStage[stageCount]):
                            basisSet = self._basisSim.standardUniqueBasis[stageCount][uniqueSetCount]
                            if not 'C' in basisSet:
                                for basis in basisSet:
                                    function = "((nStandard-1)*lam{i:d}x{j:d}B-{off:d})*max(step((nStandard-1)*lam{i:d}x{j:d}B-{off:d})*step(1-((nStandard-1)*lam{i:d}x{j:d}B-{off:d})),1*step((nStandard-1)*lam{i:d}x{j:d}B-{off:d}-1))".format(i=i,j=j,off=offSetCount)
                                    integrator.addComputeGlobal(standardParameter.format(i=str(i),j=str(j),b=basis), function)
                                dropOffset = False
                            else:
                                dropOffset = True
                                function = "1*step((nStandard-1)*lam{i:d}x{j:d}B-1)".format(i=i,j=j)
                                integrator.addComputeGlobal(standardParameter.format(i=str(i),j=str(j),b='C'), function)
                        if not dropOffset:
                            offSetCount += 1
                    for i2 in xrange(i+1, self._basisSim.Ni):
                        for j2 in xrange(self._basisSim.Nj):
                            offSetCount =0
                            for stageCount in xrange(len(self._basisSim.crossBasisPerStage)):
                                #Through each unique basis in that stage
                                for uniqueSetCount in xrange(self._basisSim.crossBasisPerStage[stageCount]):
                                    basisSet = self._basisSim.crossUniqueBasis[stageCount][uniqueSetCount]
                                    if not 'C' in basisSet:
                                        for basis in basisSet:
                                            function = "((nCross)*lam{i:d}x{j:d}B*lam{i2:d}x{j2:d}B-{off:d})*max(step((nCross)*lam{i:d}x{j:d}B*lam{i2:d}x{j2:d}B-{off:d})*step(1-((nCross)*lam{i:d}x{j:d}B*lam{i2:d}x{j2:d}B-{off:d})),1*step((nCross)*lam{i:d}x{j:d}B*lam{i2:d}x{j2:d}B-{off:d}-1))".format(i=i,j=j,i2=i2,j2=j2,off=offSetCount)
                                            integrator.addComputeGlobal(crossParameter.format(i=str(i),j=str(j),i2=str(i2),j2=str(j2),b=basis), function)
                                        dropOffset = False
                                    else:
                                        dropOffset = True
                                        function = "1*step((nCross-1)*lam{i:d}x{j:d}B*lam{i2:d}x{j2:d}B-1)".format(i=i,j=j,i2=i2,j2=j2)
                                        integrator.addComputeGlobal(standardParameter.format(i=str(i),j=str(j),b='C'), function)
                                if not dropOffset:
                                    offSetCount += 1
            return
        def computeBasisForce(integrator):
            #Fetch current lambdaB then set all to 0
            for i in xrange(self._basisSim.Ni):
                for j in xrange(self._basisSim.Nj):
                    integrator.addComputeGlobal('f{0:s}x{1:s}'.format(str(i),str(j)), '0')
                    integrator.addComputeGlobal('curentB{0:s}x{1:s}'.format(str(i),str(j)), 'lam{0:s}x{1:s}B'.format(str(i),str(j)))
                    integrator.addComputeGlobal(standardParameter.format(i=str(i),j=str(j),b='B'), '0')
                    for uniqueSetCount in xrange(self._basisSim.standardNumBasis):
                        #Grab the unique basis functions
                        basisSet = self._basisSim._flatStandardUniqueBasis[uniqueSetCount]
                        for basis in basisSet:
                            integrator.addComputeGlobal(standardParameter.format(i=str(i),j=str(j),b=basis), '0')
                    #Loop through i2/j2 interactions
                    for i2 in xrange(i+1,self._basisSim.Ni): #no need to loop backwards, otherwise will double up on energy calculations
                        for j2 in xrange(self._basisSim.Nj): #All j affected, just not same i
                            for uniqueSetCount2 in xrange(self._basisSim.crossNumBasis):
                                basisSet2 = self._basisSim._flatCrossUniqueBasis[uniqueSetCount2]
                                for basis in basisSet2:
                                    integrator.addComputeGlobal(crossParameter.format(i=str(i),j=str(j),i2=str(i2),j2=str(j2), b=basis), '0')
            #Now that they are zero, compute the derivatives
            integrator.addComputeGlobal('unaffectedPotential', 'energy')
            for i in xrange(self._basisSim.Ni):
                for j in xrange(self._basisSim.Nj):
                    integrator.addComputeGlobal('curentB{0:s}x{1:s}'.format(str(i),str(j)), 'lam{0:s}x{1:s}B'.format(str(i),str(j)))
                    integrator.addComputeGlobal(standardParameter.format(i=str(i),j=str(j),b='B'), '1')
                    integrator.addComputeGlobal('f{0:s}x{1:s}'.format(str(i),str(j)), 'f{0:s}x{1:s} + energy - unaffectedPotential'.format(str(i),str(j)))
                    integrator.addComputeGlobal(standardParameter.format(i=str(i),j=str(j),b='B'), '0')
                    offSetCount = 0
                    #Through Each Stage
                    for stageCount in xrange(len(self._basisSim.standardBasisPerStage)):
                        #Through each unique basis in that stage
                        for uniqueSetCount in xrange(self._basisSim.standardBasisPerStage[stageCount]):
                            basisSet = self._basisSim.standardUniqueBasis[stageCount][uniqueSetCount]
                            if not 'C' in basisSet:
                                for basis in basisSet:
                                    integrator.addComputeGlobal(standardParameter.format(i=str(i),j=str(j),b=basis), '1*step(  (nStandard-1)*curentB{0:d}x{1:d}-{2:d})*step(1-((nStandard-1)*curentB{0:d}x{1:d}-{2:d}))'.format(i,j,offSetCount))
                                derivString = str(self._basisSim.standardNumBasis-1) + '*curentB{0:d}x{1:d}-{2:d}'.format(i,j,offSetCount)
                                passMultiplier = derivsStandard[basisSet[0]].format(derivString)
                                integrator.addComputeGlobal('f{0:s}x{1:s}'.format(str(i),str(j)), 'f{0:s}x{1:s} + (energy - unaffectedPotential)*{2:s}'.format(str(i),str(j),passMultiplier))
                                for basis in basisSet:
                                    integrator.addComputeGlobal(standardParameter.format(i=str(i),j=str(j),b=basis), '0')
                                offSetCount += 1
                    for i2 in xrange(i+1, self._basisSim.Ni):
                        for i2 in xrange(i+1, self._basisSim.Ni):
                            for stageCount in xrange(len(self._basisSim.crossBasisPerStage)):
                                #Through each unique basis in that stage
                                offSetCount = 0
                                for uniqueSetCount in xrange(self._basisSim.crossBasisPerStage[stageCount]):
                                    basisSet = self._basisSim.crossUniqueBasis[stageCount][uniqueSetCount]
                                    if not 'C' in basisSet:
                                        for basis in basisSet:
                                            integrator.addComputeGlobal(crossParameter.format(i=str(i),j=str(j),i2=str(i2),j2=str(j2),b=basis), '1*step(  (nCross)*curentB{0:d}x{1:d}*curentB{2:d}x{3:d}-{4:d})*step(1-((nCross)*curentB{0:d}x{1:d}*curentB{2:d}x{3:d}-{4:d}))'.format(i,j,i2,j2,offSetCount))
                                        derivString = str(self._basisSim.crossNumBasis) + '*curentB{0:d}x{1:d}*curentB{2:d}x{3:d}-{4:d}'.format(i,j,i2,j2,offSetCount)
                                        passMultiplier = derivsCross[basisSet[0]].format(derivString)
                                        integrator.addComputeGlobal('f{0:s}x{1:s}'.format(str(i),str(j)), 'f{0:s}x{1:s} + (energy - unaffectedPotential)*{2:s}'.format(str(i),str(j),passMultiplier))
                                        for basis in basisSet:
                                            integrator.addComputeGlobal(crossParameter.format(i=str(i),j=str(j),i2=str(i2),j2=str(j2),b=basis), '0')
                                        offSetCount += 1
                    #Cast dU/dL -> dU/dTheta.   dU/dTheta = dU/dL * dL/dTheta
                    integrator.addComputeGlobal('f{0:s}x{1:s}'.format(str(i),str(j)), 'f{i:s}x{j:s}*'.format(i=str(i),j=str(j)) + self.thetaFunctions['dftheta'].format(i=str(i),j=str(j)))
            return

        #Velocity Verlet integrator with constraints
        #Because theta is (-inf,inf) and no other constraints are on it, we do not have to constrain its "position"
        integrator.addPerDofVariable("x1", 0);
        integrator.addComputeGlobal('unaffectedPotential', '0')
        integrator.addComputePerDof("v", "v+0.5*dt*f/m")
        computeBasisForce(integrator)
        #Compute the velocity of each theta term
        for i in xrange(self._basisSim.Ni):
            for j in xrange(self._basisSim.Nj):
                integrator.addComputeGlobal('thetaV{0:s}x{1:s}'.format(str(i),str(j)), 'thetaV{i:d}x{j:d} + 0.5*dt*f{i:d}x{j:d}/{m:f}'.format(i=i,j=j,m=self.thetaMasses[i,j]))
        integrator.addComputePerDof("x", "x+dt*v")
        #Update alchemcial positions
        for i in xrange(self._basisSim.Ni):
            for j in xrange(self._basisSim.Nj):
                integrator.addComputeGlobal('theta{0:s}x{1:s}'.format(str(i),str(j)), 'theta{i:d}x{j:d} + dt*thetaV{i:d}x{j:d}'.format(i=i,j=j))
                integrator.addComputeGlobal('lam{i:s}x{j:s}B'.format(i=str(i),j=str(j)), self.thetaFunctions['ftheta'].format(i=str(i),j=str(j)))
        assignBasisParameters(integrator)
        integrator.addComputePerDof("x1", "x")
        integrator.addConstrainPositions()
        integrator.addComputePerDof("v", "v+0.5*dt*f/m+(x-x1)/dt")
        computeBasisForce(integrator)
        for i in xrange(self._basisSim.Ni):
            for j in xrange(self._basisSim.Nj):
                integrator.addComputeGlobal('thetaV{0:s}x{1:s}'.format(str(i),str(j)), 'thetaV{i:d}x{j:d} + 0.5*dt*f{i:d}x{j:d}/{m:f}'.format(i=i,j=j,m=self.thetaMasses[i,j]))
                integrator.addComputeGlobal('lam{i:s}x{j:s}B'.format(i=str(i),j=str(j)), self.thetaFunctions['ftheta'].format(i=str(i),j=str(j)))
        assignBasisParameters(integrator)
        integrator.addConstrainVelocities()
        return integrator

    def initilizeTheta(self):
        if not hasattr(self._basisSim, 'context'):
            raise Exception("Cannot initilize theta until context exists")
        else:
            currentLambda = self._basisSim.getLambda()
            Ni = self._basisSim.Ni
            Nj = self._basisSim.Nj
            thetas = self.thetaFunctions['inverselambdaftheta'](currentLambda)
            for i in xrange(Ni):
                for j in xrange(Nj):
                    self.integrator.setGlobalVariableByName('theta{j:d}x{i:d}'.format(i=i,j=j), thetas[i,j])
            
        
    def __init__(self, basisSim, timestep, thetaMasses=None, thetaFunctions = None):
        self._basisSim = basisSim
        if thetaFunctions is None:
            #Mapping of theta -> lambda
            thetaFunctions = {}
            thetaFunctions['ftheta'] = '0.5 + 0.5*sin(theta{i:s}x{j:s})'
            thetaFunctions['lambdaftheta'] = lambda theta: 0.5 + 0.5*np.sin(theta)
            thetaFunctions['inverselambdaftheta'] = lambda lam: np.arcsin((lam-0.5)/0.5)
            thetaFunctions['dftheta'] = '0.5*cos(theta{i:s}x{j:s})'
        self.thetaFunctions = thetaFunctions
        if thetaMasses is None:
            #Mass assuming the default 0.5 amu A^2 from Knight and Brooks, JCTC 7 2011, pp 2728-2739
            defaultMass = (5 * unit.amu * unit.angstrom**2)
            thetaMasses = np.empty([self._basisSim.Ni,self._basisSim.Nj],dtype=float)
            thetaMasses.fill(defaultMass.value_in_unit(unit.amu * unit.nanometer**2))
        self.thetaMasses = thetaMasses.reshape([self._basisSim.Ni,self._basisSim.Nj])
        self.timestep = timestep
        self.integrator = self._constructIntegrator()
        return

class BasisMCJumpIntegratorEngine(object):
    '''
    Standard integrator mixed with a MC jumps to new alchemical states
    '''
    def __init__(self, basisSim, timestep):
        self._basisSim = basisSim
    
