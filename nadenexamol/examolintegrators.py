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
        integrator.addGlobalVariable('kT', kT)
        integrator.addGlobalVariable("naccept", 0)  # number accepted
        integrator.addGlobalVariable("ntrials", 0)  # number of Metropolization trials
        integrator.addPerDofVariable("sigma", 0)
        integrator.addGlobalVariable("ke", 0)  # kinetic energy
        integrator.addPerDofVariable("xold", 0)  # old positions
        integrator.addGlobalVariable("Eold", 0)  # old energy
        integrator.addGlobalVariable("Eprimeold", 0)  # old energy
        integrator.addGlobalVariable("Enew", 0)  # new energy
        integrator.addGlobalVariable("Eprimenew", 0)  # new energy
        integrator.addGlobalVariable("accept", 0)  # accept or reject
        integrator.addGlobalVariable("counterMC", 0) #While block counter for MC steps 
        integrator.addGlobalVariable("stepsPerMC", self.stepsPerMC) #While block counter for MC steps 
        #Velocity Verlet integrator with constraints
        #Because theta is (-inf,inf) and no other constraints are on it, we do not have to constrain its "position"
        integrator.addPerDofVariable("x1", 0); #For constraints in cartesian
        for i in xrange(self._basisSim.Ni):
            for j in xrange(self._basisSim.Nj):
                integrator.addGlobalVariable('old{0:d}x{1:d}B'.format(i,j), 0)
                #Due to odd unit module issues, I need to compute sigma by hand
                mass = self.lamMasses[i,j] # Unit in am
                sigma = np.sqrt(kT.value_in_unit(unit.kilojoules_per_mole)/mass)
                integrator.addGlobalVariable('sigma{0:d}x{1:d}'.format(i,j), sigma)
                integrator.addGlobalVariable('lamV{0:d}x{1:d}'.format(i,j), 0)

        #Integrator proper
        #Initilize velocities for MC V-randomization
        #Draw new velocities from a maxwell boltzman distribution
        integrator.addComputePerDof("sigma", "sqrt(kT/m)")
        integrator.addUpdateContextState()
        integrator.addComputePerDof("v", "sigma*gaussian")
        for i in xrange(self._basisSim.Ni):
            for j in xrange(self._basisSim.Nj):
                integrator.addComputeGlobal('old{0:d}x{1:d}B'.format(i,j), "lam{0:d}x{1:d}B".format(i,j))
                integrator.addComputeGlobal('lamV{0:d}x{1:d}'.format(i,j), "sigma{0:d}x{1:d}*gaussian".format(i,j))
        integrator.addConstrainVelocities()
        # Store old position and energy.
        integrator.addComputeSum("ke", "0.5*m*v*v")
        integrator.addComputeGlobal("Eold", "ke + energy")
        integrator.addComputePerDof("xold", "x")
        #Deactivate cross-alchemical terms to compute primed energy
        integrator.addComputeGlobal("includeSecond", "0")
        integrator.addComputeGlobal("Eprimeold", "ke + energy")

        # Inner symplectic steps using velocity Verlet.
        #for step in xrange(self.stepsPerMC):
        integrator.addComputeGlobal("counterMC", "0")
        integrator.beginWhileBlock("counterMC < stepsPerMC")
        if True: #This is a visual indentation block to show what is falling under the integrator's "while" block
            #pL(t)
            integrator.addComputeGlobal("derivMode", "1")
            #Compute the velocity of each lambda term
            for i in xrange(self._basisSim.Ni):
                for j in xrange(self._basisSim.Nj):
                    group = self._basisSim.calcGroup(i,j)
                    integrator.addComputeGlobal('lamV{0:d}x{1:d}'.format(i,j), 'lamV{i:d}x{j:d} + dt*energy{group:d}/{m:f}'.format(i=i,j=j,group=group,m=self.lamMasses[i,j]))
            integrator.addComputeGlobal("derivMode", "0");
            #p(t/2)
            integrator.addComputePerDof("v", "v+0.5*dt*f/m")
            #q(t)
            integrator.addComputePerDof("x", "x+dt*v")
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
            #q(t)
            integrator.addComputePerDof("x1", "x")
            integrator.addConstrainPositions()
            #p(t/2)
            integrator.addComputePerDof("v", "v+0.5*dt*f/m+(x-x1)/dt")
            integrator.addConstrainVelocities()
            integrator.addComputeGlobal("counterMC", "counterMC + 1")
            integrator.endBlock()
        #Accept/Reject Step
        integrator.addComputeSum("ke", "0.5*m*v*v")
        integrator.addComputeGlobal("Eprimenew", "ke + energy0")
        #Flip secondary interactions back on
        integrator.addComputeGlobal("includeSecond", "1")
        integrator.addComputeGlobal("Enew", "ke + energy")
        #integrator.addComputeGlobal("accept", "step(exp(-(Enew-Eold)/kT) - uniform)") #Condition for non-approximate potential
        integrator.addComputeGlobal("accept", "step(exp(-((Enew-Eold)-(Eprimenew-Eprimeold))/kT) - uniform)")
        integrator.addComputePerDof("x", "x*accept + xold*(1-accept)")
        for i in xrange(self._basisSim.Ni):
            for j in xrange(self._basisSim.Nj):
                integrator.addComputeGlobal(standardParameter.format(i=i,j=j), "lam{i:d}x{j:d}B*accept + old{i:d}x{j:d}B*(1-accept)".format(i=i,j=j))
        #
        # Accumulate statistics.
        #
        integrator.addComputeGlobal("naccept", "naccept + accept")
        integrator.addComputeGlobal("ntrials", "ntrials + 1")
        return integrator

    @property
    def n_accept(self):
        """The number of accepted HMC moves."""
        return self.integrator.getGlobalVariableByName("naccept")

    @property
    def n_trials(self):
        """The total number of attempted HMC moves."""
        return self.integrator.getGlobalVariableByName("ntrials")

    @property
    def acceptance_rate(self):
        """The acceptance rate: n_accept  / n_trials."""
        return self.n_accept / float(self.n_trials)

    def initilizeIntegrator(self):
        if not hasattr(self._basisSim, 'context'):
            raise Exception("Cannot initilize theta until context exists")
        else:
            #Take initial half time step in lamV 
            import pdb
            pdb.set_trace()
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
        
    def __init__(self, basisSim, timestep, lamMasses=None, stepsPerMC=10):
        self._basisSim = basisSim
        if lamMasses is None:
            #Mass assuming the default 0.5 amu A^2 from Knight and Brooks, JCTC 7 2011, pp 2728-2739
            defaultMass = (5 * unit.amu * unit.angstrom**2)
            lamMasses = np.empty([self._basisSim.Ni,self._basisSim.Nj],dtype=float)
            lamMasses.fill(defaultMass.value_in_unit(unit.amu * unit.nanometer**2))
        self.lamMasses = lamMasses.reshape([self._basisSim.Ni,self._basisSim.Nj])
        self.timestep = timestep
        self.stepsPerMC = stepsPerMC
        self.integrator = self._constructIntegrator()
        return

class BasisMCJumpIntegratorEngine(object):
    '''
    Standard integrator mixed with a MC jumps to new alchemical states
    '''
    def __init__(self, basisSim, timestep):
        self._basisSim = basisSim

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
