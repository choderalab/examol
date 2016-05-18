import pdb
import numpy as np
import simtk.unit as unit
import simtk.openmm as mm
from simtk.openmm import app
from copy import deepcopy
from examolclasses import *
from examolhelpers import *
import cProfile, pstats, StringIO
import sys

#=== DEFINE CONSTANTS  ===
DEBUG_MODE = False
#ff = app.ForceField('xmlfiles/gaff.xml', 'xmlfiles/examol.xml', 'xmlfiles/examolresidue.xml', 'tip3p.xml')
if DEBUG_MODE:
    ff = app.ForceField('xmlfiles/gaff.xml', 'xmlfiles/examolcharge.xml', 'xmlfiles/testresidue.xml', 'tip3p.xml')
else:
    ff = app.ForceField('xmlfiles/gaff.xml', 'xmlfiles/examolcharge.xml', 'xmlfiles/examolresiduecharge.xml', 'tip3p.xml')
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

def writeGROCoords(filename, topology, positions):
    #Write out a GRO file positions based on the topology names and positions
    writestr = ''
    writestr += "CREATED WITH EXAMOL\n"
    writestr += "{0:d}\n".format(positions.shape[0])
    #            resnum        resname        atomname     atnum    
    pointstr = "{resid: >5d}{resname: >5s}{atname:>5s}{atnum:>5d}{xcoord:> 8.3f}{ycoord:> 8.3f}{zcoord:> 8.3f}\n"
    #Construct atoms
    ci = 0 
    ai = 1
    for chain in topology.chains():
        ri = 1
        for res in chain.residues():
            resname = res.name
            for atom in res.atoms():
                x,y,z = positions[ai-1,:].value_in_unit(unit.nanometer)
                name = atom.name
                atdic = {"atnum":ai, "atname":name, "resname":resname, "resid":ri, "xcoord":x, "ycoord":y, "zcoord":z}
                writestr += pointstr.format(**atdic)
                ai += 1
            ri += 1
        ci +=1
    #PBC
    box = topology.getPeriodicBoxVectors()
    writestr += "{0:.3f} {1:.3f} {2:.3f}\n".format(*(box/unit.nanometer).diagonal())
    with open(filename, 'w') as grofile:
        grofile.write(writestr)
    return

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
    fileline = '{ts:.2f}    {baseMass:f}    {propMass}    {stepsPerMCOuter:d}    {stepsPerMCInner:d}    {wallClock:f}    {acceptOuter:f}/{acceptInner:f}    {deltaLamPerTimeStep:f}\n'
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
    if DEBUG_MODE:
        #DEBUG: 3 sites, 1 R-group per site (hydrogens)
        Ni = 3 #Number of ith groups
        Nj = 1 #Number of jth groups
    else:
        Ni = 3 #Number of ith groups
        Nj = 10 #Number of jth groups
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
    #basisSim = initilizeSimulation(filename=filename, systemname=systemname, coordsFromFile='examoleqNVT.nc', protocol={'nIterations':1})
    #basisSim = initilizeSimulation(filename=filename, systemname=systemname, coordsFromFile='examoleqNVT.nc', protocol={'nIterations':1, 'stepsPerIteration':1, 'stepsPerMC':1})
    #basisSim = initilizeSimulation(filename=filename, systemname=systemname, coordsFromFile='examoleqNVT.nc', protocol={'nIterations':1, 'stepsPerIteration':1})
    #basisSim = initilizeSimulation(filename=filename, systemname=systemname, coordsFromFile='examoleqNVT.nc', protocol={'nIterations':1, 'stepsPerIteration':1, 'timestep':2.0*unit.femtosecond})
    crossSwitches = {'R':'fourth', 'E':'fourth', 'A':'fourth', 'C':'fourth', 'B':'fourth'}
    standardSwitches = {'B':'fourth'}
    #Without alchemical change, ts = 1.5 for optimal HMC, with alchemical change is 1.25fs
    #basisSim = initilizeSimulation(filename=filename[:-3]+'4th.nc', systemname=systemname[:-4]+'4th.xml', coordsFromFile='examoleqNVT.nc', protocol={'nIterations':1, 'stepsPerIteration':1, 'timestep':1.25*unit.femtosecond, 'crossSwitches':crossSwitches, 'standardSwitches':standardSwitches, 'stepsPerMCOuter':1})
    sensitivityChecks(dicts={'crossSwitches':crossSwitches, 'standardSwitches':standardSwitches, 'nIterations':1, 'stepsPerIteration':1, 'devIndex':1})
     
    basisSim = initilizeSimulation(filename=filename[:-3]+'4th.nc', systemname=systemname[:-4]+'4th.xml', coordsFromFile='examoleqNVT.nc', protocol={'nIterations':2000, 'stepsPerIteration':500, 'timestep':1.5*unit.femtosecond, 'crossSwitches':crossSwitches, 'standardSwitches':standardSwitches, 'stepsPerMCOuter':1, 'stepsPerMCInner':10})
    #basisSim = initilizeSimulation(filename=filename, systemname=systemname, protocol={'nIterations':4})
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

    basisSim.assignLambda(np.array([0.5]*(basisSim.Ni*basisSim.Nj)))
    basisSim.assignLambda(np.array([0.1]*(basisSim.Ni*basisSim.Nj)))
    #basisSim.assignLambda(np.array([[0,0,0,0,0,0,0,1,0,0],
    #                                [0,0,0,0,0,1,0,0,0,0],
    #                                [0,0,1,0,0,0,0,0,0,0]]))
    #basisSim.assignLambda(np.array([0]*(basisSim.Ni*basisSim.Nj)))
    initialU = basisSim.computeBasisEnergy()
    if True:
        E0 = []
        E1 = []
        for i in xrange(basisSim.calcGroup(2,9,2,9)+3):
            E0.append(basisSim.getPotential(groups=i)/unit.kilojoules_per_mole)
        basisSim.integrator.step(1)
        for i in xrange(basisSim.calcGroup(2,9,2,9)+3):
            E1.append(basisSim.getPotential(groups=i)/unit.kilojoules_per_mole)
        try:
            print("Debugging")
            steps = 600
            eo = np.zeros(steps)
            en = np.zeros(steps)
            epn = np.zeros(steps)
            epo = np.zeros(steps)
            eVal = np.zeros(steps)
            ai = np.zeros(steps)
            ao = np.zeros(steps)
            kT=basisSim.integrator.getGlobalVariableByName('kT')
            for i in xrange(steps):
                en[i] = basisSim.integrator.getGlobalVariableByName('EnewOuter')
                eo[i] = basisSim.integrator.getGlobalVariableByName('EoldOuter')
                epo[i] = basisSim.integrator.getGlobalVariableByName('EoldInnerEval')
                epn[i] = basisSim.integrator.getGlobalVariableByName('EoldInner') #Because I shuffle new->old, this is the correct value to grab for testing MC accept/reject
                #epn[i] = basisSim.integrator.getGlobalVariableByName('EnewInner') #This is debugging value to track what inner proposed values were, not what was accepted.
                eVal[i] = -((en[i]-eo[i])-(epn[i]-epo[i]))/kT
                basisSim.integrator.step(1)
            deo = eo-epo
        except: pass
        E0 = np.array(E0)
        E1 = np.array(E1)
        dE = E1-E0
    #timeSteps(basisSim, 1000)
    pdb.set_trace()
    basisSim.run()

if __name__ == "__main__":
    execute()
