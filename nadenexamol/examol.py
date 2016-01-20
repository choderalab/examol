import pdb
import numpy as np
import simtk.unit as unit
import simtk.openmm as mm
from simtk.openmm import app
from copy import deepcopy
from examolclasses import *
from examolhelpers import *
import cProfile, pstats, StringIO

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

def writePDBCoords(filename, topology, positions):
    #Cast positions into on-demand generator
    def posgen(positions):
        nPos = positions.shape[0]
        i = 0
        while i < nPos:
            yield positions[i,:]
            i += 1

    Pgen = posgen(positions)
    #Write out a PDB file positions based on the topology names and positions
    wrtiestr = ''
    writestr += "REMARK CREATED WITH EXAMOL\n"
    #PBC
    writestr += "CRYSTL   {0:0.3f}   {1:0.3f}   {2:0.3f}  90.00  90.00  90.00 P 1           1\n".format()
    writestr += "MODEL\n"
    #                   num           name         res        chainid    resid
    pointstr = "ATOM {atnum: >6d} {atname: >4s} {resname:>3s} {chain:1s}{resid: >4d}      {xcoord: >5.3f}  {ycoord: >5.3f}  {zcoord: >5.3f}  1.00  0.00 {element: >11s}\n"
    #                idnum
    termstr = "TER {atnum: >6d}      {resname:>3s} {chain:1s}{resid: >4d}\n"
    #Construct atoms
    getPos 
    ci = 0 
    cstr = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ai = 1
    for chain in topology.chains():
        cs = cstr[ci]
        ri = 1
        for res in chain.resudies():
            resname = res.name
            for atom in res.atoms():
                x,y,z = next(posgen).value_in_unit(unit.angstrom)
                name = atom.name
                element = atom.element
                atdic = {"atnum":ai, "atname":name, "resname":resname, "chain":cs, "resid":ri, "xcoord":x, "ycoord":y, "zcoord":z, "element":element }
                writestr += pointstr.format(atdic)
            ri += 1
        #Remove one so last residue is used as term
        ri -= 1
        terdic = {"atnum":ai, "resname":resname, "chain":cs, "resid":ri}
        writestr += termstr.format(terdic)
        ci +=1
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

if DEBUG_MODE:
    #DEBUG: 3 sites, 1 R-group per site (hydrogens)
    Ni = 3 #Number of ith groups
    Nj = 1 #Number of jth groups
else:
    Ni = 3 #Number of ith groups
    Nj = 10 #Number of jth groups
basisSim = basisExamol(Ni, Nj, ff)

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
#context = mm.Context(basisSim.mainSystem, integrator, platform)
context = basisSim.buildContext(provideContext=True)

#=== MINIMIZE ENERGIES ===
context.setPositions(basisSim.mainPositions)
context.setVelocitiesToTemperature(basisSim.temperature)
#context.applyConstraints(1E-6)
#Assign random lambda vector (testing)
profile = True
if profile:
    pr = cProfile.Profile()
    pr.enable()
    basisSim.computeBasisEnergy()
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print s.getvalue()
pdb.set_trace()
#cProfile.run(basisSim.computeBasisEnergy())
basisSim.computeBasisEnergy()
randLam = np.random.random(Ni*Nj)
pdb.set_trace()
basisSim.assignLambda(randLam)
basisSim.computeBasisEnergy()
#randLam = np.zeros(Ni*Nj)
#randLam[8:-1:10] = 1
pdb.set_trace()
randLam = np.ones(Ni*Nj) * 0.9
assignLambda(context, randLam, Ni, Nj, skipij2=False)
checkLam = getLambda(context, Ni, Nj)
platformMode = platform.getPropertyValue(context, 'OpenCLPrecision')
platform.setPropertyValue(context, 'OpenCLPrecision', 'double')
pdb.set_trace()
computeBasisEnergy(context, Ni, Nj)
mm.Platform.setPropertyValue(context, 'OpenCLPrecision', platformMode)
print context.getState(getEnergy=True, groups=1).getPotentialEnergy()
print context.getState(getEnergy=True).getPotentialEnergy()
#Minimize positions
mm.LocalEnergyMinimizer.minimize(context, 1.0 * unit.kilojoules_per_mole / unit.nanometers, 0)

#Test energy Evaluation
computeBasisEnergy(context, Ni, Nj)
pdb.set_trace()
    
#Test a step
integrator.step(10)

#Sanity Checks
state = context.getState(getPositions=True,enforcePeriodicBox=True,getEnergy=True)
#Check energies
energy = state.getPotentialEnergy()
print energy
#Check Positions
coords = state.getPositions(asNumpy=True)
print coords

pdb.set_trace()
