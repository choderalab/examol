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

    Uses the following operators to take a timestep, q = positions, p = momentum, L is in lambda
    pL(t/2) + p(t/2) + qL(t) + q(t) + p(t/2) + pL(t/2)
    pL is offset by 1/2 timestep initially to do the following 
    pL(t) + p(t/2) + qL(t) + q(t) + p(t/2)
    To reduce the number of integrator steps needed per timestep
    -----------------------
    '''

    def _constructIntegrator(self):
        #Set default theta (continuous variable that maps back to lambda) mappings
        integrator = mm.CustomIntegrator(self.timestep.value_in_unit(unit.picoseconds))
        standardParameter = 'lam{i:d}x{j:d}B'
        crossParameter = 'lam{i:d}x{j:d}x{i2:d}x{j2:d}{b:s}'
        integrator.addGlobalVariable('pi', np.pi)
        # HMC integrator, derived from the OpenMMTools repo for HMC integrator by J. Chodera
        temp = self._basisSim.temperature
        kT = kB * temp
        #Set integrator variables
        #Global Variables
        integrator.addPerDofVariable("sigma", 0)
        integrator.addGlobalVariable("ke", 0)  # kinetic energy
        integrator.addGlobalVariable("kePass", 0)  # ke that passes up to the outer MC chain as the velocities get reshuffled on the inner step, but on the last step of the outer chain, this would cause an issue
        integrator.addGlobalVariable('kT', kT)
        #Statistics
        integrator.addGlobalVariable("nacceptOuter", 0)
        integrator.addGlobalVariable("nacceptInner", 0)
        integrator.addGlobalVariable("ntrialsOuter", 0)  # number of Metropolization trials
        integrator.addGlobalVariable("ntrialsInner", 0)  # number of Metropolization trials
        integrator.addPerDofVariable("xoldOuter", 0)  # old positions
        integrator.addGlobalVariable("EoldOuter", 0)  # old energy
        integrator.addPerDofVariable("xoldInner", 0) 
        integrator.addGlobalVariable("EoldInner", 0)  
        integrator.addGlobalVariable("EoldInnerEval", 0) #Special storage of the approximate old potential for the outer evaluation 
        integrator.addGlobalVariable("EnewOuter", 0)  # new energy
        integrator.addGlobalVariable("EnewInner", 0)  # new energy
        integrator.addGlobalVariable("EInnerPass", 0)  # Inner energy passed up to the outer
        #Accept/Reject
        integrator.addGlobalVariable("acceptOuter", 0)
        integrator.addGlobalVariable("acceptInner", 0)
        integrator.addGlobalVariable("counterMCInner", 0) #While block counter for MC steps 
        integrator.addGlobalVariable("counterMCOuter", 0)  
        integrator.addGlobalVariable("stepsPerMCInner", self.stepsPerMCInner) #While block counter for number of MD steps before taking hybrid MC step
        integrator.addGlobalVariable("stepsPerMCOuter", self.stepsPerMCOuter) #While block counter for outter approximate potential MC chain, counts number of Hybrid MC steps inner
        #Velocity Verlet integrator with constraints
        integrator.addPerDofVariable("x1", 0); #For constraints in cartesian
        for i in xrange(self._basisSim.Ni):
            for j in xrange(self._basisSim.Nj):
                integrator.addGlobalVariable('old{0:d}x{1:d}BOuter'.format(i,j), 0)
                integrator.addGlobalVariable('old{0:d}x{1:d}BInner'.format(i,j), 0)
                #Due to odd unit module issues, I need to compute sigma by hand
                mass = self.lamMasses[i,j] # Unit in am
                sigma = np.sqrt(kT.value_in_unit(unit.kilojoules_per_mole)/mass)
                integrator.addGlobalVariable('sigma{0:d}x{1:d}'.format(i,j), sigma)
                integrator.addGlobalVariable('lamV{0:d}x{1:d}'.format(i,j), 0)
        #Storage variables for NVE MD steps and reducing force calls
        integrator.addGlobalVariable("forceStored", 0)
        integrator.addPerDofVariable("storedForce", 0)
        #DEBUG
        integrator.addGlobalVariable("masterReject", 0)
        integrator.addPerDofVariable("xFDebug", 0)
        for i in xrange(self.stepsPerMCInner):
            integrator.addPerDofVariable("x{0:d}Debug".format(i), 0)
            integrator.addGlobalVariable("U{0:d}Debug".format(i), 0)
            integrator.addGlobalVariable("K{0:d}Debug".format(i), 0)
        integrator.addGlobalVariable("keDebug", 0)
        integrator.addPerDofVariable("xPassDebug", 0)
        integrator.addGlobalVariable("kePassDebug", 0)
        integrator.addGlobalVariable("UPassDebug", 0)
        integrator.addPerDofVariable("xOutDebug", 0)
        integrator.addGlobalVariable("keOutDebug", 0)
        integrator.addGlobalVariable("UOutDebug", 0)
        integrator.addPerDofVariable("vOld", 0)

        #Integrator proper
        #Initilize velocities for MC V-randomization
        #Draw new velocities from a maxwell boltzman distribution
        integrator.addComputePerDof("sigma", "sqrt(kT/m)")
        integrator.addUpdateContextState()
        #DEBUG
        integrator.addComputePerDof("v", "sigma*gaussian")
        #integrator.addComputePerDof("vOld", "v")
        if not self._cartesianOnly:
            for i in xrange(self._basisSim.Ni):
                for j in xrange(self._basisSim.Nj):
                    integrator.addComputeGlobal('old{0:d}x{1:d}BOuter'.format(i,j), "lam{0:d}x{1:d}B".format(i,j))
                    #For initial inner loop step, lambda value is the same, we'll update at end of inner loop
                    integrator.addComputeGlobal('old{0:d}x{1:d}BInner'.format(i,j), "lam{0:d}x{1:d}B".format(i,j))
                    integrator.addComputeGlobal('lamV{0:d}x{1:d}'.format(i,j), "sigma{0:d}x{1:d}*gaussian".format(i,j))
        integrator.addConstrainVelocities()
        #Start Outer MC loop
        # Store old position and energy.
        integrator.addComputeSum("kePass", "0.5*m*v*v")
        if not self._cartesianOnly:
            #Compute alchemical kinetic energy, writing this as 1 string to keep number of steps small
            keStrComps = []
            for i in xrange(self._basisSim.Ni):
                for j in xrange(self._basisSim.Nj):
                    keStrComps.append('{m:f}*lamV{i:d}x{j:d}^2'.format(m=self.lamMasses[i,j], i=i, j=j))
            integrator.addComputeGlobal("kePass", "kePass + 0.5*(" + " + ".join(keStrComps) + ")")
        integrator.addComputeGlobal("EoldOuter", "kePass + energy")
        integrator.addComputePerDof("xoldOuter", "x")
        integrator.addComputePerDof("xoldInner", "x")
        #Deactivate cross-alchemical terms to compute primed energy
        integrator.addComputeGlobal("includeSecond", "0")
        #Store approximate old energy for outter eval later
        integrator.addComputeGlobal("EoldInnerEval", "kePass + energy")
        #Speed up inner calculations
        integrator.addComputeGlobal("EoldInner", "EoldInnerEval")
        #Switch to Inner MC Loop, adding a visual indintation to help code
        #Outer MC Loop counter
        integrator.addComputeGlobal("counterMCOuter", "0")
        integrator.beginWhileBlock("counterMCOuter < stepsPerMCOuter")
        if True:
            # Inner symplectic steps using velocity Verlet.
            #for step in xrange(self.stepsPerMC):
            integrator.addComputeGlobal("counterMCInner", "0")
            integrator.beginWhileBlock("counterMCInner < stepsPerMCInner")
            if True: #This is a visual indentation block to show what is falling under the integrator's "while" block
                #DEBUG:
                #for i in xrange(self.stepsPerMCInner):
                #    integrator.addComputePerDof("x{0:d}Debug".format(i), "x*delta(counterMCInner-{0:d}) + x{0:d}Debug*(1-delta(counterMCInner-{0:d}))".format(i))
                #    integrator.addComputeGlobal("U{0:d}Debug".format(i), "energy*delta(counterMCInner-{0:d}) + U{0:d}Debug*(1-delta(counterMCInner-{0:d}))".format(i))
                #    integrator.addComputeSum("keDebug", "0.5*m*v*v")
                #    integrator.addComputeGlobal("K{0:d}Debug".format(i), "keDebug*delta(counterMCInner-{0:d}) + K{0:d}Debug*(1-delta(counterMCInner-{0:d}))".format(i))
                if not self._cartesianOnly:
                    #pL(t)
                    integrator.addComputeGlobal("derivMode", "1")
                    #Compute the velocity of each lambda term
                    for i in xrange(self._basisSim.Ni):
                        for j in xrange(self._basisSim.Nj):
                            group = self._basisSim.calcGroup(i,j)
                            #This is MINUS the force term since force is -dU/dL. "f" variable = "-dU/dr", but since we manually coded 
                            integrator.addComputeGlobal('lamV{0:d}x{1:d}'.format(i,j), 'lamV{i:d}x{j:d} - dt*energy{group:d}/{m:f}'.format(i=i,j=j,group=group,m=self.lamMasses[i,j]))
                    integrator.addComputeGlobal("derivMode", "0");
                #Check if force is stored
                #integrator.beginIfBlock("forceStored = 0")
                #if True: #Visual
                #    integrator.addComputePerDof("storedForce", "f")
                #    #The force stored flag is set here
                #    integrator.addComputeGlobal("forceStored", "1")
                #    integrator.endBlock()
                #p(t/2)
                integrator.addComputePerDof("v", "v+0.5*dt*f/m")
                #integrator.addComputePerDof("v", "v+0.5*dt*storedForce/m")
                #q(t)
                integrator.addComputePerDof("x", "x+dt*v")
                integrator.addComputePerDof("x1", "x")
                if not self._cartesianOnly:
                    #qL(t)
                    #Update alchemcial positions
                    for i in xrange(self._basisSim.Ni):
                        for j in xrange(self._basisSim.Nj):
                            integrator.addComputeGlobal('lam{0:d}x{1:d}B'.format(i,j), 'lam{i:d}x{j:d}B + dt*lamV{i:d}x{j:d}'.format(i=i,j=j))
                            #Add reflective BC steps
                            #ISSUE: IfBlocks with all R-groups is painfuly slow to construct integrator
                            #Check if lam < 0 
                            integrator.addComputeGlobal('lam{0:d}x{1:d}B'.format(i,j), 'step(lam{i:d}x{j:d}B)*lam{i:d}x{j:d}B + (1-step(lam{i:d}x{j:d}B))*(-lam{i:d}x{j:d}B)'.format(i=i,j=j))
                            integrator.addComputeGlobal('lamV{0:d}x{1:d}'.format(i,j), 'step(lam{i:d}x{j:d}B)*lamV{i:d}x{j:d} + (1-step(lam{i:d}x{j:d}B))*(-lamV{i:d}x{j:d}B)'.format(i=i,j=j))
                            #integrator.beginIfBlock("lam{0:d}x{1:d}B < 0".format(i,j))
                            #if True: #Visual integrator IF block
                            #    integrator.addComputeGlobal("lam{0:d}x{1:d}B".format(i,j), "-lam{0:d}x{1:d}B".format(i,j))
                            #    integrator.addComputeGlobal("lamV{0:d}x{1:d}".format(i,j), "-lamV{0:d}x{1:d}".format(i,j))
                            #    integrator.endBlock()
                            #Check if lam > 1 
                            integrator.addComputeGlobal('lam{0:d}x{1:d}B'.format(i,j), 'step(1-lam{i:d}x{j:d}B)*lam{i:d}x{j:d}B + (1-step(1-lam{i:d}x{j:d}B))*(2-lam{i:d}x{j:d}B)'.format(i=i,j=j))
                            integrator.addComputeGlobal('lamV{0:d}x{1:d}'.format(i,j), 'step(1-lam{i:d}x{j:d}B)*lamV{i:d}x{j:d} + (1-step(1-lam{i:d}x{j:d}B))*(-lamV{i:d}x{j:d}B)'.format(i=i,j=j))
                            #integrator.beginIfBlock("lam{0:d}x{1:d}B > 1".format(i,j))
                            #if True: #Visual integrator IF block
                            #    integrator.addComputeGlobal("lam{0:d}x{1:d}B".format(i,j), "2-lam{0:d}x{1:d}B".format(i,j))
                            #    integrator.addComputeGlobal("lamV{0:d}x{1:d}".format(i,j), "-lamV{0:d}x{1:d}".format(i,j))
                            #    integrator.endBlock()
                integrator.addConstrainPositions()
                #p(t/2)
                #integrator.addComputePerDof("storedForce", "f")
                integrator.addComputePerDof("v", "v+0.5*dt*f/m+(x-x1)/dt")
                #integrator.addComputePerDof("v", "v+0.5*dt*storedForce/m+(x-x1)/dt")
                integrator.addConstrainVelocities()
                integrator.addComputeGlobal("counterMCInner", "counterMCInner + 1")
                integrator.endBlock()
            #Inner Accept/Reject Step
            #DEBUG:
            #integrator.addComputePerDof("xFDebug", "x")
            integrator.addComputeSum("ke", "0.5*m*v*v")
            if not self._cartesianOnly:
                integrator.addComputeGlobal("ke", "ke + 0.5*(" + " + ".join(keStrComps) + ")") #Just reusing the strings generated before
            integrator.addComputeGlobal("EnewInner", "ke + energy")
            integrator.addComputeGlobal("acceptInner", "step(exp(-(EnewInner-EoldInner)/kT) - uniform)")
            #DEBUG:
            if self._forceAcceptMC:
                integrator.addComputeGlobal("acceptInner", "1")
            #NaN Discard block. This is a full discard of the run (not just reject), so decrease counters
            integrator.beginIfBlock("EnewInner != EnewInner")
            if True:
                #Reset everything that could have NaN'd
                integrator.addComputePerDof("x", "xoldInner")
                integrator.addComputeGlobal("EnewInner", "EoldInner")
                integrator.addComputeGlobal("ke", "kePass")
                if not self._cartesianOnly:
                    for i in xrange(self._basisSim.Ni):
                        for j in xrange(self._basisSim.Nj):
                            integrator.addComputeGlobal(standardParameter.format(i=i,j=j), "old{i:d}x{j:d}BInner".format(i=i,j=j))
                integrator.addComputeGlobal("acceptInner", "0")
                #Decrease counters
                integrator.addComputeGlobal("counterMCOuter", "counterMCOuter - 1")
                integrator.addComputeGlobal("ntrialsInner", "ntrialsInner - 1") #Only trials since naccept wont incremenet
                integrator.addComputeGlobal("masterReject", "masterReject + 1")
                integrator.addUpdateContextState()
                integrator.endBlock()
            integrator.addComputePerDof("x", "x*acceptInner + xoldInner*(1-acceptInner)")
            integrator.addComputePerDof("xoldInner", "x")
            if not self._cartesianOnly:
                for i in xrange(self._basisSim.Ni):
                    for j in xrange(self._basisSim.Nj):
                        integrator.addComputeGlobal(standardParameter.format(i=i,j=j), "lam{i:d}x{j:d}B*acceptInner + old{i:d}x{j:d}BInner*(1-acceptInner)".format(i=i,j=j))
                        integrator.addComputeGlobal("old{i:d}x{j:d}BInner".format(i=i,j=j), "lam{i:d}x{j:d}B".format(i=i,j=j))
            #Set new energy to pass up
            integrator.addComputeGlobal("EInnerPass", "EnewInner*acceptInner + EoldInner*(1-acceptInner)")
            integrator.addComputeGlobal("kePass", "ke*acceptInner + kePass*(1-acceptInner)")
            #DEBUG
            #integrator.addComputeGlobal("UPassDebug", "energy")
            #integrator.addComputeGlobal("kePassDebug", "kePass")
            #integrator.addComputePerDof("xPassDebug", "x")
            #Generate new velocities, does not effect outter loop since KE is stored
            integrator.addUpdateContextState()
            #DEBUG
            #integrator.addComputeGlobal("UOutDebug", "energy")
            #integrator.addComputeSum("keOutDebug", "0.5*m*v*v")
            #integrator.addComputePerDof("xOutDebug", "x")
            #DEBUG
            integrator.addComputePerDof("v", "sigma*gaussian")
            if not self._cartesianOnly:
                for i in xrange(self._basisSim.Ni):
                    for j in xrange(self._basisSim.Nj):
                        integrator.addComputeGlobal('lamV{0:d}x{1:d}'.format(i,j), "sigma{0:d}x{1:d}*gaussian".format(i,j))
            integrator.addConstrainVelocities()
            #Reset the stored force to be safe. If accept, its fine, otherwise have to store initial old force and add more conditional checks, just resetting it.
            integrator.addComputeGlobal("forceStored", "0") #Reset stored force
            #Finally set the new old inner energy since KE has changed: E = Epass - kePass + newKE
            integrator.addComputeSum("ke", "0.5*m*v*v")
            if not self._cartesianOnly:
                integrator.addComputeGlobal("ke", "ke + 0.5*(" + " + ".join(keStrComps) + ")") #Just reusing the strings generated before
            integrator.addComputeGlobal("EoldInner", "EInnerPass - kePass + ke")
            # Accumulate Inner statistics.
            integrator.addComputeGlobal("nacceptInner", "nacceptInner + acceptInner")
            integrator.addComputeGlobal("ntrialsInner", "ntrialsInner + 1")
            integrator.addComputeGlobal("counterMCOuter", "counterMCOuter + 1")
            integrator.endBlock()
        #Outer loop Accept/Reject Step
        #Flip secondary interactions back on
        integrator.addComputeGlobal("includeSecond", "1")
        integrator.addComputeGlobal("EnewOuter", "kePass + energy")
        #The apprximate potential eval here is EoldInner-EoldInnerEval since I moved EnewInner -> EoldInner or preserved old EoldInner from above
        integrator.addComputeGlobal("acceptOuter", "step(exp(-((EnewOuter-EoldOuter)-(EInnerPass-EoldInnerEval))/kT) - uniform)")
        if self._forceAcceptMC:
            integrator.addComputeGlobal("acceptOuter", "1")
        #DEBUG:
        #integrator.addComputePerDof("v", "v*acceptOuter + vOld*(1-acceptOuter)")
        integrator.addComputePerDof("x", "x*acceptOuter + xoldOuter*(1-acceptOuter)")
        if not self._cartesianOnly:
            for i in xrange(self._basisSim.Ni):
                for j in xrange(self._basisSim.Nj):
                    integrator.addComputeGlobal(standardParameter.format(i=i,j=j), "lam{i:d}x{j:d}B*acceptOuter + old{i:d}x{j:d}BOuter*(1-acceptOuter)".format(i=i,j=j))
        # Accumulate Outer statistics.
        integrator.addComputeGlobal("nacceptOuter", "nacceptOuter + acceptOuter")
        integrator.addComputeGlobal("ntrialsOuter", "ntrialsOuter + 1")
        #DEBUG
        #integrator.setRandomNumberSeed(0)
        return integrator

    @property
    def n_accept(self):
        """The number of accepted (Outer Approximate MC, Inner HMC) moves."""
        return (self.integrator.getGlobalVariableByName("nacceptOuter"), self.integrator.getGlobalVariableByName("nacceptInner"))

    @property
    def n_trials(self):
        """The total number of attempted (Outer Apprixmate, Inner HMC) moves."""
        return (self.integrator.getGlobalVariableByName("ntrialsOuter"), self.integrator.getGlobalVariableByName("ntrialsInner") )

    @property
    def acceptance_rate(self):
        """The acceptance rate: n_accept  / n_trials."""
        return tuple(na / float(nt) for na,nt in zip(self.n_accept,self.n_trials))

    def initilizeIntegrator(self):
        if not hasattr(self._basisSim, 'context'):
            raise Exception("Cannot initilize theta until context exists")
        else:
            #Take initial half time step in lamV 
            pdb.set_trace() #Still a work in progress
            self.integrator.setGlobalVariable('derivMode'.format(i,j), 1)
            ts = self._basisSim.timestep.value_in_unit(unit.picosecond)
            for i in xrange(self._basisSim.Ni):
                for j in xrange(self._basisSim.Nj):
                    #Draw initial velocities from the Maxwell-Boltzman dist.
                    v0 = self.integrator.getGlobalVariableByName("sigma{0:d}x{1:d}".format(i,j)) * np.random.normal()
                    group = self._basisSim.calcGroup(i,j)
                    #Take the half timestep
                    energyGroup = self._basisSim.context.getState(getEnergy=True, groups=group).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
                    self.integrator.setGlobalVariable('lamV{0:d}x{1:d}'.format(i,j), v0 + 0.5*ts*energyGroup/self.lamMasses[i,j])
            self.integrator.setGlobalVariable('derivMode'.format(i,j), 0)
        return 
        
    def __init__(self, basisSim, timestep, lamMasses=None, stepsPerMCInner=10, stepsPerMCOuter=None, cartesianOnly=False, forceAcceptMC=False):
        '''
        The cartesianOnly flag disables alchemical updates and only updates the cartesian coordinates. Helpful for debugging and doing Check Ensemble
        '''
        self._basisSim = basisSim
        if lamMasses is None:
            #Mass assuming the default 0.5 amu A^2 from Knight and Brooks, JCTC 7 2011, pp 2728-2739
            defaultMass = (500 * unit.amu * unit.angstrom**2)
            lamMasses = np.empty([self._basisSim.Ni,self._basisSim.Nj],dtype=float)
            lamMasses.fill(defaultMass.value_in_unit(unit.amu * unit.nanometer**2))
        self.lamMasses = lamMasses.reshape([self._basisSim.Ni,self._basisSim.Nj])
        self.timestep = timestep
        self.stepsPerMCInner = stepsPerMCInner
        if stepsPerMCOuter is None:
            stepsPerMCOuter = self.stepsPerMCInner
        self.stepsPerMCOuter = stepsPerMCOuter
        self._cartesianOnly = cartesianOnly
        self._forceAcceptMC = forceAcceptMC
        self.integrator = self._constructIntegrator()
        return

class VelocityVerletNVT(mm.CustomIntegrator):
    def __init__(self, timestep=1.0 * unit.femtoseconds, temp=298*unit.kelvin, steps=1000):

        super(VelocityVerletNVT, self).__init__(timestep)

        kT = kB * temp
        self.addPerDofVariable("sigma", 0)
        self.addPerDofVariable("x1", 0)
        self.addGlobalVariable('kT', kT)
        self.addGlobalVariable("stepper", 0)
        #Storage variables for NVE MD steps and reducing force calls
        self.addGlobalVariable("forceStored", 0)
        self.addPerDofVariable("storedForce", 0)

        self.addComputePerDof("sigma", "sqrt(kT/m)")
        self.addComputePerDof("v", "sigma*gaussian")
        self.addComputeGlobal("stepper", "0")
        self.beginWhileBlock("stepper < {0:d}".format(steps))
        self.addUpdateContextState()
        self.beginIfBlock("forceStored = 0")
        if True: #Visual
            self.addComputePerDof("storedForce", "f")
            #The force stored flag is set here
            self.addComputeGlobal("forceStored", "1")
            self.endBlock()
        #self.addComputePerDof("v", "v+0.5*dt*f/m")
        self.addComputePerDof("v", "v+0.5*dt*storedForce/m")
        self.addComputePerDof("x", "x+dt*v")
        self.addComputePerDof("x1", "x")
        self.addConstrainPositions()
        self.addComputePerDof("storedForce", "f")
        self.addComputePerDof("v", "v+0.5*dt*storedForce/m+(x-x1)/dt")
        #self.addComputePerDof("v", "v+0.5*dt*f/m+(x-x1)/dt")
        self.addConstrainVelocities()    
        self.addComputeGlobal("stepper", "stepper + 1")
        self.endBlock()

class VelocityVerletIntegrator(mm.CustomIntegrator):

    """Verlocity Verlet integrator.
    Notes
    -----
    This integrator is taken verbatim from Peter Eastman's example appearing in the CustomIntegrator header file documentation.
    References

    ----------
    W. C. Swope, H. C. Andersen, P. H. Berens, and K. R. Wilson, J. Chem. Phys. 76, 637 (1982)
    Examples
    --------
    Create a velocity Verlet integrator.
    >>> timestep = 1.0 * simtk.unit.femtoseconds
    >>> integrator = VelocityVerletIntegrator(timestep)
    """

    def __init__(self, timestep=1.0 * unit.femtoseconds):
        """Construct a velocity Verlet integrator.
        Parameters
        ----------
        timestep : numpy.unit.Quantity compatible with femtoseconds, default: 1*simtk.unit.femtoseconds
           The integration timestep.
        """

        super(VelocityVerletIntegrator, self).__init__(timestep)

        self.addPerDofVariable("x1", 0)

        self.addUpdateContextState()
        self.addComputePerDof("v", "v+0.5*dt*f/m")
        self.addComputePerDof("x", "x+dt*v")
        self.addComputePerDof("x1", "x")
        self.addConstrainPositions()
        self.addComputePerDof("v", "v+0.5*dt*f/m+(x-x1)/dt")
        self.addConstrainVelocities()    
