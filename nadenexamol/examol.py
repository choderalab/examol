import pdb
import numpy as np
import simtk.unit as unit
import simtk.openmm as mm
from simtk.openmm import app
from copy import deepcopy
from examolclasses import *
from examolhelpers import *
import matplotlib.pyplot as plt
import cProfile, pstats, StringIO
import sys

#=== DEFINE CONSTANTS  ===
DEBUG_MODE = False
#ff = app.ForceField('xmlfiles/gaff.xml', 'xmlfiles/examol.xml', 'xmlfiles/examolresidue.xml', 'tip3p.xml')
if DEBUG_MODE:
    ff = app.ForceField('xmlfiles/gaff.xml', 'xmlfiles/examolcharge.xml', 'xmlfiles/testresidue.xml', 'tip3p.xml')
    Ni = 3 #Number of ith groups
    Nj = 1 #Number of jth groups
else:
    ff = app.ForceField('xmlfiles/gaff.xml', 'xmlfiles/examolcharge.xml', 'xmlfiles/examolresiduecharge.xml', 'tip3p.xml')
    Ni = 3 #Number of ith groups
    Nj = 10 #Number of jth groups

#Nonbonded methods
NBM=app.PME
#NBM=app.NoCutoff
#Nonbonded cutoff
NBCO=9*unit.angstrom
#Constraint
constraints=app.HBonds
# rigid water
rigidWater=True
#Ewald Tolerance 
eET=0.0005
#=== END CONSTANTS ===

def loadamber(basefilename, NBM=NBM, NBCO=NBCO, constraints=constraints, rigidWater=rigidWater, eET=eET):
    prmtop = app.AmberPrmtopFile(basefilename + '.prmtop')
    inpcrd = app.AmberInpcrdFile(basefilename + '.inpcrd')
    system = prmtop.createSystem(
     nonbondedMethod=NBM,
     nonbondedCutoff=NBCO,
     constraints=constraints,
     rigidWater=rigidWater,
     ewaldErrorTolerance=eET)
    return system

def loadpdb(basefilename, NBM=NBM, NBCO=NBCO, constraints=constraints, rigidWater=rigidWater, eET=eET, boxbuffer=0.2):
    #Box Buffer is % extra away from peak-to-peak distances to draw pbc
    pdbfile = app.PDBFile(basefilename + '.pdb')
    #Check for not PBC
    if pdbfile.topology.getPeriodicBoxVectors() is None:
        pos = pdbfile.getPositions(asNumpy=True)
        #Peak to peak distances across all atoms
        ptpdistances = np.ptp(pos,axis=0)
        #Maximum absolute coordinate
        absmaxdists = np.abs(pos).max(axis=0)
        #3x3 box vector, sum of abs + ptp, then scaled up by box buffer, and reapply unit
        boxvecs = np.eye(3)*(absmaxdists + ptpdistances) * (1+boxbuffer) * pos.unit
        pdbfile.topology.setPeriodicBoxVectors(boxvecs)
    system = ff.createSystem(
     pdbfile.topology,
     nonbondedMethod=NBM,
     nonbondedCutoff=NBCO,
     constraints=constraints,
     rigidWater=rigidWater,
     ewaldErrorTolerance=eET)
    #FF sanity check. Make sure there are an equal number of bonds/constraints in system as there are in topology
    #Assuming not multiple constraints per atom
    if len(pdbfile.topology._bonds) != system.getForce(0).getNumBonds() + system.getNumConstraints():
        print("WARNING: {0:s} does not have matching topology and system bonds/constraints!".format(basefilename))
    return system, pdbfile

def writePDBCoords(filename, topology, positions):
    #Write out a PDB file positions based on the topology names and positions
    writestr = ''
    writestr += "REMARK CREATED WITH EXAMOL\n"
    #PBC
    box = topology.getPeriodicBoxVectors()
    writestr += "CRYSTL   {0:0.3f}   {1:0.3f}   {2:0.3f}  90.00  90.00  90.00 P 1           1\n".format(*(box/unit.angstrom).diagonal())
    writestr += "MODEL\n"
    #                   num           name         res        chainid    resid
    pointstr = "ATOM {atnum: >6d} {atname: >4s} {resname:>3s} {chain:1s}{resid: >4d}      {xcoord: >5.3f}  {ycoord: >5.3f}  {zcoord: >5.3f}  1.00  0.00 {element: >11s}\n"
    #                idnum
    termstr = "TER {atnum: >6d}      {resname:>3s} {chain:1s}{resid: >4d}\n"
    #Construct atoms
    ci = 0 
    cstr = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ai = 1
    for chain in topology.chains():
        cs = cstr[ci]
        ri = 1
        for res in chain.residues():
            resname = res.name
            for atom in res.atoms():
                x,y,z = positions[ai-1,:].value_in_unit(unit.angstrom)
                name = atom.name
                element = atom.element.symbol
                atdic = {"atnum":ai, "atname":name, "resname":resname, "chain":cs, "resid":ri, "xcoord":x, "ycoord":y, "zcoord":z, "element":element }
                writestr += pointstr.format(**atdic)
                ai += 1
            ri += 1
        #Remove one so last residue is used as term
        ri -= 1
        terdic = {"atnum":ai, "resname":resname, "chain":cs, "resid":ri}
        writestr += termstr.format(**terdic)
        ci +=1
    with open(filename, 'w') as pdbfile:
        pdbfile.write(writestr)
    return

def addRParticles(mainSystem, coreSystem, corecoords, Rsystem, Rcoords):
    #Detect differences between core and R group
    Ncore = coreSystem.getNumParticles()
    NR = Rsystem.getNumParticles()
    Nmain = mainSystem.getNumParticles()
    new_atoms = xrange(Ncore,NR)
    #Attach R group to main system
    for new_atom in new_atoms:
        mainSystem.addParticle(Rsystem.getParticleMass(new_atom))
        #Map this particle to the new atom number
    ##Atoms, bonds, angles, torsions, dihedrals
    return range(Nmain, mainSystem.getNumParticles())

def checkEnergies(sim, steps=100):
    context = sim.context
    integrator = sim.integrator
    if sim.barostat is not None or sim.thermostat is not None or isinstance(integrator,mm.LangevinIntegrator) or isinstance(integrator,mm.BrownianIntegrator):
        print("Cannot valitate energy if not in NVE ensemble")
        return
    currentState = context.getState(getPositions=True,getVelocities=True,getEnergy=True)
    pos0 = currentState.getPositions(asNumpy=True)
    vel0 = currentState.getVelocities(asNumpy=True)
    velLast = vel0
    #Take 1 step to offset velocity
    integrator.step(1)        
    ke = np.zeros(steps)*unit.kilojoules_per_mole
    pe = np.zeros(steps)*unit.kilojoules_per_mole
    for step in xrange(steps):
        state = context.getState(getEnergy=True, getVelocities=True)
        pe[step] = state.getPotentialEnergy()
        velNext = state.getVelocities(asNumpy=True)
        context.setVelocities((velLast+velNext)/2)
        ke[step] = context.getState(getEnergy=True).getKineticEnergy()
        context.setVelocities(velNext)
        velLast = velNext
        integrator.step(1)
    context.setPositions(pos0)
    context.setVelocities(vel0)
    return ke+pe

def timeSteps(sim, steps):
    print("Timing {0:d} timesteps".format(steps))
    if hasattr(sim, "integratorEngine"):
        from examolintegrators import HybridLDMCIntegratorEngine as HLDMC
        IE = sim.integratorEngine
        if isinstance(IE, HLDMC):
            #Convert to the true number of steps 
            steps = steps/IE.stepsPerMC
    steps=1
    print steps, IE.stepsPerMC
    pr = cProfile.Profile()
    pr.enable()
    sim.integrator.step(steps)
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print s.getvalue()
    return

def timeEnergy(sim):
    pr = cProfile.Profile()
    pr.enable()
    basisSim.computeBasisEnergy()
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print s.getvalue()

def sensitivityChecks(dicts=None):
    #Set default protocols
    defProto = {}
    if dicts is not None:
        for key in dicts.keys():
            defProto[key] = dicts[key]
    Ni=3
    Nj=10
    numTimeSteps = 1000
    stockLam = np.array([0.1]*(Ni*Nj))
    stockLam = stockLam.reshape([Ni,Nj])
    fs = unit.femtosecond
    lm = unit.amu * unit.angstrom**2
    comb = []
    natomsperR = np.array([[1, 2, 4, 7, 3, 1, 1, 11, 3, 9],
                           [1, 2, 4, 7, 3, 1, 1, 11, 3, 9],
                           [1, 2, 4, 7, 3, 1, 1, 11, 3, 9]], dtype=int)
    meanParticles = natomsperR.sum()/float(natomsperR.size)
    timesteps = (1.25*fs, 1.5*fs, 1.75*fs, 2.0*fs)
    baseMass = (50*lm, 100*lm, 500*lm)
    propMass = (False, True)
    stepsPerMCOuter = (1, 5, 10)
    stepsPerMCInner = (1, 5, 10, 50)

    #timesteps = (1.5*fs,)
    #baseMass = (500*lm,)
    #propMass = (False, True)
    #stepsPerMCInner = (10,)
    #stepsPerMCOuter = (1,)
    fileline = '{ts:.2f}    {baseMass:f}    {propMass}    {stepsPerMCOuter:d}    {stepsPerMCInner:d}    {wallClock:f}    {acceptOuter:f}    {acceptInner:f}    {deltaLamPerTimeStep:f}\n'
    nPerm = len(timesteps)*len(baseMass)*len(propMass)*len(stepsPerMCInner)*len(stepsPerMCOuter)
    accepts = {}
    deltaLam = {}
    times = {}
    counter = 1
    for timestep in timesteps:
        for mass in baseMass:
            for prop in propMass:
                for spmco in stepsPerMCOuter:
                    for spmci in stepsPerMCInner:
                        if counter > 153:
                            nSteps = numTimeSteps/(spmci*spmco)
                            print("Working on combo {0:d}/{1:d} ({2:f}%)".format(counter, nPerm, counter/float(nPerm)))
                            comb.append((timestep, mass, prop, spmci, spmco))
                            print comb[-1]
                            proto = {}
                            for key in defProto.keys(): proto[key] = defProto[key]
                            proto['stepsPerMCInner'] = spmci
                            proto['stepsPerMCOuter'] = spmco
                            proto['timestep'] = timestep
                            masses =  np.empty([Ni,Nj],dtype=float)
                            masses.fill(mass.value_in_unit(unit.amu * unit.nanometer**2))
                            if mass is True:
                                #Using % of the mean, figuring out finer details will come later
                                masses*(natomsperR/meanParticles)
                            proto['lamMasses'] = masses
                            #try:
                            if True:
                                sim = initilizeSimulation(filename='scratch.nc', systemname='examolsystem4th.xml', filedebug=True, coordsFromFile='examoleqNVT.nc', protocol=proto)
                                sim.assignLambda(stockLam)
                                #Time
                                pr = cProfile.Profile()
                                pr.enable()
                                sim.integrator.step(nSteps)
                                pr.disable()
                                s = StringIO.StringIO()
                                sortby = 'cumulative'
                                ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
                                ps.print_stats()
                                #Get the time of the run
                                with open('sens.txt', 'a') as sens:
                                    times[comb[-1]] = float(s.getvalue().split('\n')[0].split()[-2])
                                    accepts[comb[-1]] = sim.integratorEngine.acceptance_rate
                                    newLam = sim.getLambda()
                                    deltaLam[comb[-1]] = np.sum((newLam-stockLam)**2)/numTimeSteps
                                    sens.write(fileline.format(ts=timestep._value, baseMass=mass._value, propMass=prop, stepsPerMCOuter=spmco, stepsPerMCInner=spmci, wallClock=times[comb[-1]], acceptOuter=accepts[comb[-1]][0], acceptInner=accepts[comb[-1]][1], deltaLamPerTimeStep=deltaLam[comb[-1]]))
                                #Cleanup
                                del sim.context, sim.integrator
                                del sim
                            #except:
                            #    pass
                        counter += 1
    pdb.set_trace()
    return


def initilizeSimulation(**kwargs):
    '''
    Create the simulation object
    '''
    basisSim = basisExamol(Ni, Nj, ff, **kwargs)
    
    if not basisSim.coordsLoaded:
        #Set the positions so all particles are in the box and do no wrap oddly
        box=basisSim.mainSystem.getDefaultPeriodicBoxVectors()
        box = np.array([unit.norm(vector.value_in_unit(unit.nanometer)) for vector in box])*unit.nanometer
        mincoords = np.min(basisSim.mainPositions,axis=0)
        newPositions = basisSim.mainPositions - mincoords
        nudgeDistance = (box - newPositions.max(axis=0))/2
        newPositions += nudgeDistance
        basisSim.mainPositions = newPositions
    
    #Quick code to create the PDB file with all the correct CONNECT entries (visualization)
    #Atommaps from PDB file
    #amap = range(1,Noriginal+1) #Base 1
    #amap.extend(range(Noriginal+1+1, Nnew+Noriginal+1+1)) #Base 1, ter command occupies 1
    #nC = mainSystem.getNumConstraints()
    #nB = 0
    #bondforces = []
    #for forceidx in xrange(mainSystem.getNumForces()):
    #    force = mainSystem.getForce(forceidx)
    #    if isinstance(force, mm.HarmonicBondForce) or isinstance(force, mm.CustomBondForce):
    #        bondforces.append(force)
    #        nB += force.getNumBonds()
    #bondlist = np.zeros([nB+nC, 2], dtype=int)
    #count = 0
    #for constraint in xrange(nC):
    #    atomi, atomj, r0 = mainSystem.getConstraintParameters(constraint)
    #    bondlist[count,:] = (amap[atomi], amap[atomj])
    #    count +=1
    #for force in bondforces:
    #    for bond in xrange(force.getNumBonds()):
    #        bondparam = force.getBondParameters(bond)
    #        atomi, atomj = bondparam[0], bondparam[1]
    #        bondlist[count,:] = (amap[atomi], amap[atomj])
    #        count +=1
    #conline = "CONECT{a1: >5d}{a2: >5d}\n"
    #output = ''
    #for bond in xrange(nB+nC):
    #    output += conline.format(a1=bondlist[bond,0], a2=bondlist[bond,1])
    #file = open('connects.pdb', 'w')
    #file.write(output)
    #file.close()
    #pdb.set_trace()
    
    
    #DEBUG: Testing built in reporters to see if I need something else
    #simulation = app.Simulation(mainTopology, mainSystem, integrator, platform)
    #simulation.context.setPositions(mainPositions)
    #simulation.context.setVelocitiesToTemperature(equilibriumTemperature)
    #reporter = app.PDBReporter('trajectory.pdb',1)
    ##reporter = app.DCDReporter('trajectory.dcd',1)
    #reporter.report(simulation, simulation.context.getState(getPositions=True,getParameters=True, enforcePeriodicBox=True))
    #pdb.set_trace()
    #simulation.minimizeEnergy(1.0 * unit.kilojoules_per_mole / unit.nanometers, 0)
    #reporter.report(simulation, simulation.context.getState(getPositions=True,getParameters=True, enforcePeriodicBox=True))
    ##reporter.report(simulation, simulation.context.getState(getPositions=True,getParameters=True))
    #simulation.reporters.append(reporter)
    #print simulation.context.getState(getEnergy=True).getPotentialEnergy()
    #pdb.set_trace()
    #simulation.step(1)
    
    #Test taking a formal step to see if wrapping is handled correctly and if energies go to NaN
    context = basisSim.context
    
    #=== MINIMIZE ENERGIES ===
    context.setVelocitiesToTemperature(basisSim.temperature)
    return basisSim

def execute():
    if DEBUG_MODE:
        filename = 'examoldebug.nc'
        systemname = 'examoldebugsystem.xml'
    else:
        filename = 'examol.nc'
        systemname = 'examolsystem.xml'
    #basisSim = initilizeSimulation(filename=filename, equilibrate=True, systemname=systemname, coordsFromFile='examoleqNVT.nc', protocol={'nIterations':2000, 'stepsPerIteration':1000, 'pressure':1*unit.atmosphere,'timestep':1.0*unit.femtosecond})
    #basisSim = initilizeSimulation(filename=filename, equilibrate=True, systemname=systemname, coordsFromFile='examol.nc.initPos.npz', protocol={'nIterations':2000, 'stepsPerIteration':1000, 'timestep':1.0*unit.femtosecond})
    crossSwitches = {'R':'fourth', 'E':'fourth', 'A':'fourth', 'C':'fourth', 'B':'fourth'}
    standardSwitches = {'B':'fourth'}
    #Without alchemical change, ts = 1.5 for optimal HMC, with alchemical change is 1.25fs
    #sensitivityChecks(dicts={'crossSwitches':crossSwitches, 'standardSwitches':standardSwitches, 'nIterations':1, 'stepsPerIteration':1, 'devIndex':1})
    #Set optimal options from the sensitivity check
    protocol = {}
    protocol['timestep'] = 1.25 * unit.femtosecond
    baseMass = 50 * unit.amu * unit.angstrom**2
    masses =  np.empty([Ni,Nj],dtype=float)
    masses.fill(baseMass.value_in_unit(unit.amu * unit.nanometer**2))
    protocol['lamMasses'] = masses
    protocol['stepsPerMCInner'] = 10
    protocol['stepsPerMCOuter'] = 1
    #Set the switches
    protocol['crossSwitches'] = crossSwitches
    protocol['standardSwitches'] = standardSwitches
    #Set the write out
    protocol['nIterations'] = 200
    timestepsPerIteration = 500
    protocol['stepsPerIteration'] = int(timestepsPerIteration/float(protocol['stepsPerMCInner'])) 
    #Choose to disable alchemical updates
    protocol['cartesianOnly'] = False
    #Choose to force moves to accept (used for equilibration)
    protocol['forceAcceptMC'] = False
    #Device
    protocol['devIndex'] = 0
    #Temperature
    protocol['temperature'] = 298*unit.kelvin
    #platoform
    protocol['platform'] = 'OpenCL'
     
    #basisSim = initilizeSimulation(filename=filename[:-3]+'4th.nc', systemname=systemname[:-4]+'4th.xml', coordsFromFile='examoleqNVT.nc', protocol=protocol)
    #basisSim = initilizeSimulation(filename='FRand3-1.0.nc', systemname=systemname[:-4]+'4th.xml', coordsFromFile='examoleqNVT.nc', protocol=protocol)
    #basisSim = initilizeSimulation(filename='SRand3-1.0.nc', systemname=systemname[:-4]+'4th.xml', coordsFromFile='FRand3-1.0.nc', protocol=protocol)
    basisSim = initilizeSimulation(filename='SNIa.nc', systemname=systemname[:-4]+'4th.xml', coordsFromFile='examoleqNVT.nc', protocol=protocol)
    #pdb.set_trace()

    #context.applyConstraints(1E-6)
    #Pull the initial energy to allocte the Context__getStateAsLists call for "fair" testing, still looking into why the initial call is slow
    #writeGROCoords('allatomwtap.gro', basisSim.mainTopology, basisSim.context.getState(getPositions=True,enforcePeriodicBox=True).getPositions(asNumpy=True))
    #mod = app.Modeller(basisSim.mainTopology, basisSim.mainPositions)
    #mod.delete([res for res in mod.topology.residues() if res.name == 'COC'])
    #writeGROCoords('watergro.gro', mod.getTopology(), listCoordsToNumpy(mod.getPositions()))
    
    #Write out bond connectivity
    #hbf=[basisSim.mainSystem.getForce(i) for i in xrange(basisSim.mainSystem.getNumForces()) if isinstance(basisSim.mainSystem.getForce(i), mm.HarmonicBondForce)][0]
    #with open('grotop.txt', 'w') as grotop:
    #    #Handle atomtypes
    #    writestr = '[ atoms ]\n'
    #    writestr += '; id  attype res nr  res name  at name  cg nr  charge    mass\n'
    #    c0 = basisSim.mainTopology._chains[0]
    #    counter = 0
    #    for atom in c0.atoms():
    #        m = basisSim.mainSystem.getParticleMass(counter)/unit.amu
    #        writestr += ' {i: >3d} ?????? {resnum: >2d} {resname: >s} {atname: >6s} 1 0.00000 {mass: >6.4f}\n'.format(i=int(atom.id), resnum=int(atom.residue.id), resname=atom.residue.name, atname=atom.name, mass=m)
    #        counter +=1
    #    #Bonds
    #    writestr += '\n[ bonds ]\n'
    #    writestr += '; i   j   funct   length  force_constant\n'
    #    for iBond in xrange(hbf.getNumBonds()):
    #        i, j, l, k = hbf.getBondParameters(iBond)
    #        writestr += ' {i: >3d} {j: >3d} 1  {l: >5.3f}  {k: >6.3f}\n'.format(i=i+1, j=j+1, l=l/unit.nanometer, k=k.value_in_unit(unit.kilojoules_per_mole/unit.nanometer**2))
    #    #Constraints
    #    writestr += '\n[ constraints ]\n'
    #    writestr += '; i   j   funct   length\n'
    #    for iConst in xrange(basisSim.mainSystem.getNumConstraints()):
    #        i, j, l = basisSim.mainSystem.getConstraintParameters(iConst)
    #        if i not in basisSim.solventNumbers and j not in basisSim.solventNumbers:
    #            writestr += ' {i: >3d} {j: >3d} 1  {l: >5.3f} \n'.format(i=i+1, j=j+1, l=l/unit.nanometer)
    #    grotop.write(writestr)
    
    #pdb.set_trace()

    #One off code to bring in new water gro file
    #watergro = app.gromacsgrofile.GromacsGroFile('outframe.gro')
    #waterpos = watergro.getPositions(asNumpy=True)
    #waterbox = listCoordsToNumpy(watergro.getPeriodicBoxVectors())
    #basisSim.mainPositions[basisSim.solventNumbers,:] = waterpos
    #basisSim.boxVectors = waterbox
    #basisSim.context.setPositions(basisSim.mainPositions)
    #basisSim.context.setPeriodicBoxVectors(basisSim.boxVectors[0,:], basisSim.boxVectors[1,:], basisSim.boxVectors[2,:])
    #writeGROCoords('newwaterwrap.gro', basisSim.mainTopology, basisSim.context.getState(getPositions=True,enforcePeriodicBox=True).getPositions(asNumpy=True))

    if not basisSim.resume:
        #basisSim.assignLambda(np.array([0.1]*(basisSim.Ni*basisSim.Nj)))
        #basisSim.assignLambda(np.array([[0,0,0,0.2,0,0,0,0,0,0.2],
        #                                [0,0,0,0,0,0.2,0,0,0,0],
        #                                [0,0,0.2,0,0,0,0,0.2,0,0]]))
        #basisSim.assignLambda(np.array([[0,0,0,0,0,0,0,0,0,0.5],
        #                                [0,0,0,0,0,0,0.5,0,0,0],
        #                                [0,0,0.5,0,0,0,0,0,0,0]]))
        #basisSim.assignLambda(np.array([0]*(basisSim.Ni*basisSim.Nj)))
        lamin = np.array([0.1]*(basisSim.Ni*basisSim.Nj))
        lamin = lamin.reshape([basisSim.Ni,basisSim.Nj])
        lamin[:,4] = np.array([1,1,1])*0.6
        basisSim.assignLambda(lamin)
        if False:
           if basisSim.verbose: print("Minimizing Positions")
           mm.LocalEnergyMinimizer.minimize(basisSim.context)
           basisSim._updatePositions() 
    initialU = basisSim.computeBasisEnergy()
    if False:
        debugPosVel = np.load('debugPosVel.npz')
        debugPos = debugPosVel['pos']
        debugVel = debugPosVel['vel']
        basisSim.context.setPositions(debugPos)
        basisSim.context.setVelocities(debugVel)
        basisSim.integrator.step(5)
        #E0 = []
        #E1 = []
        #K0 = []
        #K1 = []
        #for i in xrange(basisSim.calcGroup(2,9,2,9)+3):
        #    E0.append(basisSim.getPotential(groups=i)/unit.kilojoules_per_mole)
        #    K0.append(basisSim.context.getState(getForces=True, groups=i).getForces(asNumpy=True))
        #basisSim.integrator.step(1)
        #for i in xrange(basisSim.calcGroup(2,9,2,9)+3):
        #    E1.append(basisSim.getPotential(groups=i)/unit.kilojoules_per_mole)
        #    K1.append(basisSim.context.getState(getForces=True, groups=i).getForces(asNumpy=True))
        #dbs = 50
        #utot = np.zeros(dbs)
        #ktot = np.zeros(dbs)
        #KS = np.zeros((dbs, 4884, 3))*unit.kilojoules_per_mole/unit.nanometer
        #xout = np.zeros((dbs, 4884, 3))*unit.nanometer
        #for s in xrange(dbs):
        #    state = basisSim.context.getState(getForces=True, getPositions=True, getEnergy=True)
        #    utot[s] = state.getPotentialEnergy()/unit.kilojoules_per_mole
        #    ktot[s] = state.getKineticEnergy()/unit.kilojoules_per_mole
        #    KS[s] = state.getForces(asNumpy=True)
        #    xout[s] = state.getPositions(asNumpy=True)
        #    basisSim.integrator.step(1)
        #etot = utot+ktot
        #forcesum = np.sqrt(np.sum(KS**2,axis=2))
        #f,(a,u,k,e) = plt.subplots(4,1, figsize=(8, 4*4))
        ##b = plt.twinx(ax=a)
        #xval = np.array(xrange(dbs))*protocol['timestep']/unit.femtosecond
        #a.set_ylabel('Acting force in kJ/mol/nm')
        #a.set_xlabel('timestep')
        #a.plot(xval, forcesum[:,1], '-k', label='Core C')
        #a.plot(xval, forcesum[:,6], '--k', label='Core H')
        #a.plot(xval, forcesum[:,52], '-r', label='Hydroxyl O')
        #a.plot(xval, forcesum[:,53], '--r', label='Hydroxyl H')
        #a.legend(loc="upper right")
        #xlim = a.get_xlim()
        #u.scatter(xval, utot/1000, c='g', label='Potential')
        #k.scatter(xval, ktot/1000, c='b', label='Kinetic')
        #e.scatter(xval, etot/1000, c='k', label='Total E')
        #for plot in (u,k,e):
        #    plot.legend(loc='center right')
        #    plot.set_xlim(xlim)
        #k.set_ylabel(r'Energy in kJ/mol $\cdot 10^3$')
        ##Figure out Maxes to draw lines at
        #coreCpeaks = np.argsort(forcesum[:,1])[-1:]
        #coreHpeaks = np.argsort(forcesum[:,6])[-1:]
        #ohOpeaks = np.argsort(forcesum[:,52])[-1:]
        #ohHpeaks = np.argsort(forcesum[:,53])[-1:]
        #for plot in (a,u,k,e):
        #    for peaks, fo in zip((coreCpeaks, coreHpeaks, ohOpeaks, ohHpeaks), (forcesum[:,1], forcesum[:,6], forcesum[:,52], forcesum[:,53])):
        #       for peak in peaks:
        #           if fo[peak] > 10000:
        #               plot.axvline(x=xval[peak])
        #f.savefig("actingForcesFix.png")#,bbox_inches='tight')
        #writeGROCoords('forceTwistFix.gro', basisSim.mainTopology, xout)
        #pdb.set_trace()
        
        def checkMD(sim):
            mci = protocol['stepsPerMCInner']
            xout=np.array(sim.integrator.getPerDofVariableByName('xFDebug'))
            x2 = np.zeros((mci,)  + xout.shape)
            uout = np.zeros(mci)
            u2 = np.zeros(mci)
            kout = np.zeros(mci)
            for i in xrange(mci):
                x2[i] = np.array(sim.integrator.getPerDofVariableByName('x{0:d}Debug'.format(i)))
                uout[i] = sim.integrator.getGlobalVariableByName('U{0:d}Debug'.format(i))
                try:
                    kout[i] = sim.integrator.getGlobalVariableByName('K{0:d}Debug'.format(i))
                except: pass
            xout = x2 * unit.nanometer
            uout *= unit.kilojoules_per_mole
            kout *= unit.kilojoules_per_mole
            u2   *= unit.kilojoules_per_mole
            state = sim.context.getState(getPositions=True, enforcePeriodicBox=True)
            pos0 = state.getPositions(asNumpy=True)
            sim.context.setParameter('includeSecond',0)
            for i in xrange(mci):
                sim.context.setPositions(xout[i])
                u2[i] =  basisSim.getPotential()
            sim.context.setParameter('includeSecond',1)
            sim.context.setPositions(pos0)
            #f,a = plt.subplots(1,1)
            #a.plot(range(mci), uout+kout)
            #a.set_ylabel('Total E in kJ/mol')
            #plt.show()
            #pdb.set_trace()
            #writeGROCoords('nevermore.gro', sim.mainTopology, xout)
            return xout, uout, kout

        steps = 250
        state = basisSim.context.getState(getPositions=True)
        pos = np.zeros((steps,) + state.getPositions(asNumpy=True).shape)*unit.nanometer
        box = np.zeros((steps,) + state.getPeriodicBoxVectors(asNumpy=True).shape)*unit.nanometer

        #basisSim.context.setParameter('includeSecond', 0)
        #pdb.set_trace()
        print("Debugging")
        eo = np.zeros(steps)
        en = np.zeros(steps)
        epn = np.zeros(steps)
        epo = np.zeros(steps)
        epnt = np.zeros(steps)
        epot = np.zeros(steps)
        eVal = np.zeros(steps)
        ke = np.zeros(steps)
        kep = np.zeros(steps)
        fullE = np.zeros(steps)
        try:
            kT=basisSim.integrator.getGlobalVariableByName('kT')
        except:
            kT = basisSim.kT/unit.kilojoules_per_mole
        mR = 0
        stopReject = True
 
        #s0 = basisSim.context.getState(getPositions=True,getVelocities=True,getForces=True,getEnergy=True,getParameters=True)
        #basisSim.integrator.step(1)
        #s1 = basisSim.context.getState(getPositions=True,getVelocities=True,getForces=True,getEnergy=True,getParameters=True)
        #fs = []
        #ke = []
        #parms = []
        #pbve = []
        #pbvo = []
        #pos = []
        #pe = []
        #tm = []
        #vels = []
        #for system in [s0,s1]:
        #    fs.append(system.getForces(asNumpy=True))
        #    ke.append(system.getKineticEnergy())
        #    parms.append(system.getParameters())
        #    pbve.append(system.getPeriodicBoxVectors(asNumpy=True))
        #    pbvo.append(system.getPeriodicBoxVolume())
        #    pos.append(system.getPositions(asNumpy=True))
        #    pe.append(system.getPotentialEnergy())
        #    tm.append(system.getTime())
        #    vels.append(system.getVelocities(asNumpy=True))
        #pdb.set_trace()
        xs = []
        us = []
        ks = []
        es = []
        for i in xrange(steps):
            try:
                en[i] = basisSim.integrator.getGlobalVariableByName('EnewOuter')
                eo[i] = basisSim.integrator.getGlobalVariableByName('EoldOuter')
                epo[i] = basisSim.integrator.getGlobalVariableByName('EoldInnerEval')
                epn[i] = basisSim.integrator.getGlobalVariableByName('EInnerPass') #Because I shuffle new->old, this is the correct value to grab for testing MC accept/reject\
                #2 debug values which track the inner loop, only at inner MC step 0 and -1 do the old and new match the value used in outer MC respectivley
                epnt[i] = basisSim.integrator.getGlobalVariableByName('EnewInner')
                epot[i] = basisSim.integrator.getGlobalVariableByName('EoldInner')
                ke[i] = basisSim.integrator.getGlobalVariableByName('ke')
                kep[i] = basisSim.integrator.getGlobalVariableByName('kePass')
                eVal[i] = -((en[i]-eo[i])-(epn[i]-epo[i]))/kT
                #if np.abs(en[i]-epn[i]) > 1:
                #    uPass = basisSim.integrator.getGlobalVariableByName('UPassDebug')
                #    kePass = basisSim.integrator.getGlobalVariableByName('kePassDebug')
                #    xPass = np.array(basisSim.integrator.getPerDofVariableByName('xPassDebug'))
                #    uOut = basisSim.integrator.getGlobalVariableByName('UOutDebug')
                #    keOut = basisSim.integrator.getGlobalVariableByName('keOutDebug')
                #    xOut = np.array(basisSim.integrator.getPerDofVariableByName('xOutDebug'))
                #    pdb.set_trace()
            except: pass
            fullE[i] = basisSim.getPotential()/(kT*unit.kilojoules_per_mole)
            state = basisSim.context.getState(getPositions=True, enforcePeriodicBox=True)
            pos[i] = state.getPositions(asNumpy=True)
            box[i] = state.getPeriodicBoxVectors(asNumpy=True)
            basisSim.integrator.step(1)
            #basisSim.integrator.step(10)
            #try:
            #    for ls, val in zip([xs,us,ks], checkMD(basisSim)):
            #        ls.append(val)
            #    es.append(us[-1]+ks[-1])
            #    writeGROCoords('twisting.gro', basisSim.mainTopology, xs[-1])
            #except:
            #    pass
            try:
                mRsim =basisSim.integrator.getGlobalVariableByName('masterReject')
                if mR < mRsim:
                    mR = mRsim
                    #if stopReject:
                    #    stopReject = False
                    #    pdb.set_trace()
            except: pass
        #E0 = np.array(E0)
        #E1 = np.array(E1)
        #dE = E1-E0
        pdb.set_trace()
    #timeSteps(basisSim, 1000)
    #checkMD(basisSim)
    #pdb.set_trace()
    basisSim.run()

if __name__ == "__main__":
    execute()
