import pdb
import numpy as np
import simtk.unit as unit
import simtk.openmm as mm
from simtk.openmm import app
from copy import deepcopy
from customexamolforces import *

#=== DEFINE CONSTANTS  ===
ff = app.ForceField('xmlfiles/gaff.xml', 'xmlfiles/examol.xml', 'xmlfiles/examolresidue.xml', 'tip3p.xml')
#Nonbonded methods
NBM=app.PME
#Nonbonded cutoff
NBCO=1*unit.nanometer
#Constraint
constraints=app.HBonds
# rigid water
rigidWater=False
#Ewald Tolerance 
eET=0.0005
#=== END CONSTANTS ===

def listCoordsToNumpy(Coords):
    #Cast the coordinates in a list format to numpy format. Some getPositions() functions allow asNumpy=True keyword, others (like the modeler) do not. This function handles thoes that do not
    nCoords = len(Coords)
    numpyCoords = np.zeros([nCoords,3])
    baseunit = Coords.unit
    for n in xrange(nCoords):
        numpyCoords[n,:] = Coords[n].value_in_unit(baseunit)
    return numpyCoords * baseunit

def stripAndUnifyUnits(A, B):
    #Cast both A and B to the same base unit and return them stripped along with the base
    #Many NumPy functions loose do not preserve units, this function is mostly used for this
    baseunit = A.unit
    return A.value_in_unit(baseunit), B.value_in_unit(baseunit), baseunit

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
    #ff = app.ForceField('xmlfiles/examol.xml', 'xmlfiles/examolresidue.xml')
    #ff = app.ForceField('examol.xml')
    system = ff.createSystem(
     pdbfile.topology,
     nonbondedMethod=NBM,
     nonbondedCutoff=NBCO,
     constraints=constraints,
     rigidWater=rigidWater,
     ewaldErrorTolerance=eET)
    return system, pdbfile

def addTopologyToSystem(system, topology, NBM=NBM, NBCO=NBCO, constraints=constraints, rigidWater=rigidWater, eET=eET):
    #Fully combine the topology into the system, no bonds between system and topology

    pass

def addRParticles(mainSystem, coresystem, corecoords, Rsystem, Rcoords):
    #Detect differences between core and R group
    Ncore = coresystem.getNumParticles()
    NR = Rsystem.getNumParticles()
    Nmain = mainSystem.getNumParticles()
    new_atoms = xrange(Ncore,NR)
    #Attach R group to main system
    for new_atom in new_atoms:
        mainSystem.addParticle(Rsystem.getParticleMass(new_atom))
        #Map this particle to the new atom number
    ##Atoms, bonds, angles, torsions, dihedrals
    return range(Nmain, mainSystem.getNumParticles())

def maxPBC(targetSystem, additionalSystem, percentNudge=1):
    #set the periodic boundary vectors (PBV) of the target system to the max of target and additional
    addPBV = additionalSystem.getDefaultPeriodicBoxVectors()
    targetPBV = targetSystem.getDefaultPeriodicBoxVectors()
    newPBV = [[1,0,0] * addPBV[0].unit, [0,1,0] * addPBV[1].unit, [0,0,1] * addPBV[2].unit]
    for idim in range(3):
        vecmax = np.max((addPBV[idim][idim],targetPBV[idim][idim]))
        newPBV[idim] *= vecmax.value_in_unit(newPBV[idim].unit)*percentNudge
    targetSystem.setDefaultPeriodicBoxVectors(*newPBV)
    return

def appendPositions(mainSet, additionalSet):
    #Append additionalSet positions to the mainSet positions and return mainSet
    mainSet, additionalSet, baseunit = stripAndUnifyUnits(mainSet, additionalSet)
    mainSet = np.append(mainSet,additionalSet,axis=0)
    return mainSet * baseunit

def alignCoords(referenceCoords, toXformCoords, weights=None, pointOfOrigin=0):
    #Code from "Python Programming for Biology: Bioinformatics and Beyond" pg 301-302
    #Accepts a PDBFile.getPositions(asNumpy=True) argument for referenceCoords and toXformCoords
    #Point of Origin is the atom to translate to first
    refCoords, xformCoords, baseunit = stripAndUnifyUnits(referenceCoords, toXformCoords)
    nalign = len(refCoords)
    nxform = len(xformCoords)
    translate = refCoords[pointOfOrigin,:] - xformCoords[pointOfOrigin,:]
    xformCoords += translate
    if weights is None:
        weights = np.zeros(nxform)
        weights[:nalign] = 1 
        weights = np.ones(nalign)
    rMat = np.dot(xformCoords[:nalign,:].T*weights, refCoords)
    rMat1, scales, rMat2 = np.linalg.svd(rMat)
    sign = np.linalg.det(rMat1) * np.linalg.det(rMat2)
    if sign < 0:
        rMat1[:,2] *= -1
    rotation = np.dot(rMat1, rMat2)
    newCoords = np.dot(xformCoords, rotation)
    translate2 = refCoords[pointOfOrigin,:] - newCoords[pointOfOrigin,:]
    newCoords += translate2
    return newCoords * baseunit

def addToMainTopology(maintopology, addontopology, Ncore):
    #Add the atoms from the addontopology to the maintopology
    #Atom->Residue->Chain->Topology
    #Chain ownership has to be passed from the addontopology to the main topology before adding the atom to the main topology
    #Alternatley, choose the residue that the atom is attached to as the 
    #Grab the main residue
    for residue in maintopology.residues():
        mainres = residue
        break
    for atom in addontopology.atoms():
        if int(atom.id) > Ncore:
            maintopology.addAtom(atom.name, atom.element, mainres)
    #Bonds are not really needed since all we need is buffer for water
    return

#Load the core
#coresystem = loadamber('testcore')
coresystem, corecoords = loadpdb('pdbfiles/core/corem')
corePositions = corecoords.getPositions(asNumpy=True) #Positions of core atoms (used for alignment)
Ncore = coresystem.getNumParticles()


#Start mainSystem
mainSystem = deepcopy(coresystem)
'''
Note: The mainSystem is NOT built from the combined topologies because that would add torsions and angle forces to R-groups on the same core carbon, which we wond want.
'''
maintopology = deepcopy(corecoords.getTopology())
mainPositions = deepcopy(corePositions)
mainBondForce = getArbitraryForce(mainSystem, mm.HarmonicBondForce)
mainAngleForce = getArbitraryForce(mainSystem, mm.HarmonicAngleForce)
mainTorsionForce = getArbitraryForce(mainSystem, mm.PeriodicTorsionForce)
mainNonbondedForce = getArbitraryForce(mainSystem, mm.NonbondedForce)
mainCMRForce = getArbitraryForce(mainSystem, mm.CMMotionRemover)

#Start the Rgroups
Ni = 3 #Number of ith groups
Nj = 10 #Number of jth groups
#allocate the housing objects
Rsystems=np.empty([Ni,Nj],dtype=np.object)
Rcoords=np.empty([Ni,Nj],dtype=np.object)
RMainAtomNumbers = np.empty([Ni,Nj],dtype=np.object)
#Import the Rgroups
for i in xrange(Ni):
    for j in xrange(Nj):
        #Rgroup = loadamber('testR')
        #Rsystems[i,j], Rcoords[i,j] = loadpdb('j1mt')
        Rsystem, Rcoord = loadpdb('pdbfiles/i%d/j%dm'%(i+1,j+1))
        Rsystems[i,j], Rcoords[i,j] = Rsystem, Rcoord
        #Add the Rgroup atoms to the main system
        RMainAtomNumber = addRParticles(mainSystem, coresystem, corecoords, Rsystem, Rcoord)
        RMainAtomNumbers[i,j] = RMainAtomNumber
        RPos = Rcoord.getPositions(asNumpy=True)
        #align the new group to the core structure (to which the main was already alligned)
        alignedPositions = alignCoords(corePositions, RPos)
        #Append the newly aligned R-group structure to the main structure
        mainPositions = appendPositions(mainPositions,alignedPositions[Ncore:,:])
        #set PBC's, probably not needed here
        maxPBC(mainSystem, Rsystem)
        #Add topologies together, only needed to add solvent to the system
        addToMainTopology(maintopology, Rcoord.getTopology(), Ncore)
        maintopology.setPeriodicBoxVectors(mainSystem.getDefaultPeriodicBoxVectors())
        # === Add forces (exclusions later, for now, just get in all the defaults) ===
        for constraintIndex in range(Rsystem.getNumConstraints()):
            atomi, atomj, r0 = Rsystem.getConstraintParameters(constraintIndex)
            atomi, atomj = mapAtomsToMain([atomi, atomj], RMainAtomNumber, Ncore)
            if atomi >= Ncore or atomj >= Ncore:
                mainSystem.addConstraint(atomi, atomj, r0)
        for forceIndex in xrange(Rsystem.getNumForces()):
            referenceForce = Rsystem.getForce(forceIndex)
            if isinstance(referenceForce, mm.HarmonicBondForce):
                nRBonds = referenceForce.getNumBonds()
                for bondIndex in xrange(nRBonds):
                    atomi, atomj, eqdist, k = referenceForce.getBondParameters(bondIndex)
                    #if atomi >= Ncore or atomj >= Ncore: pdb.set_trace()
                    #Map atoms to core system
                    atomi, atomj = mapAtomsToMain([atomi, atomj], RMainAtomNumber, Ncore)
                    if atomi >= Ncore or atomj >= Ncore:
                        mainBondForce.addBond(atomi, atomj, eqdist, k)
            elif isinstance(referenceForce, mm.HarmonicAngleForce):
                customAngleForce = addAngleForceWithCustom(mainAngleForce, referenceForce, RMainAtomNumber, i, j, Ncore)
                mainSystem.addForce(customAngleForce)
            elif isinstance(referenceForce, mm.PeriodicTorsionForce):
                customTorsionForce = addTorsionForceWithCustom(mainTorsionForce, referenceForce, RMainAtomNumber, i, j, Ncore)
                mainSystem.addForce(customTorsionForce)
            elif isinstance(referenceForce, mm.NonbondedForce):
                #Add the particle to the main nonbonded force. Custom will come after
                nParticles = referenceForce.getNumParticles()
                for atomi in xrange(nParticles):
                    q, sig, epsi = referenceForce.getParticleParameters(atomi)
                    (atomi,) = mapAtomsToMain([atomi], RMainAtomNumber, Ncore) #If you dont trap the returned atomi, it returns a list of atomi, e.g. [0], which is > int for some reason?
                    if atomi >= Ncore:
                        mainNonbondedForce.addParticle(q, sig, epsi)
                nException = referenceForce.getNumExceptions()
                for exceptionIndex in xrange(nException):
                    atomi, atomj, chargeProd, sig, epsi = referenceForce.getExceptionParameters(exceptionIndex)
                    atomi, atomj = mapAtomsToMain([atomi, atomj], RMainAtomNumber, Ncore)
                    if atomi >= Ncore or atomj >= Ncore:
                        mainNonbondedForce.addException(atomi, atomj, chargeProd, sig, epsi)
                    

######## BRING IN SOLVENT ##########
#Adjust the residue in the main topology to match the combined name so the modeler does not throw an error
for res in maintopology.residues():
    res.name = 'COC'
#Add water with the modeler
modeller = app.Modeller(maintopology, mainPositions)
modeller.addSolvent(ff, padding=1.2*unit.nanometer)
#Deelete non solvent residues
modeller.delete([res for res in modeller.topology.residues() if res.name == 'COC'])
#Get Positions
modellerCoords = listCoordsToNumpy(modeller.getPositions())
#Combine positions
mainPositions = appendPositions(mainPositions, modellerCoords)
#Combine solvent with system, this can probably can be made into function form at some point
addSystem = ff.createSystem( 
 modeller.topology,
 nonbondedMethod=NBM,
 nonbondedCutoff=NBCO,
 constraints=constraints,
 rigidWater=rigidWater,
 ewaldErrorTolerance=eET)
Noriginal = mainSystem.getNumParticles()
Nnew = addSystem.getNumParticles()
maxPBC(mainSystem, addSystem, percentNudge=1.1)
solventNumbers = range(Noriginal,Nnew+Noriginal)
for atomIndex in xrange(Nnew):
    mainSystem.addParticle(addSystem.getParticleMass(atomIndex))
for constraintIndex in range(addSystem.getNumConstraints()):
    atomi, atomj, r0 = addSystem.getConstraintParameters(constraintIndex)
    mainSystem.addConstraint(solventNumbers[atomi], solventNumbers[atomj], r0)
for forceIndex in xrange(addSystem.getNumForces()):
    referenceForce = addSystem.getForce(forceIndex)
    if isinstance(referenceForce, mm.HarmonicBondForce):
        nRBonds = referenceForce.getNumBonds()
        for bondIndex in xrange(nRBonds):
            atomi, atomj, eqdist, k = referenceForce.getBondParameters(bondIndex)
            mainBondForce.addBond(solventNumbers[atomi], solventNumbers[atomj], eqdist, k)
    elif isinstance(referenceForce, mm.HarmonicAngleForce):
        nAngle = referenceForce.getNumAngles()
        for angleIndex in xrange(nAngle):
            atomi, atomj, atomk, angle, k = referenceForce.getAngleParameters(angleIndex)
            mainAngleForce.addAngle(solventNumbers[atomi], solventNumbers[atomj], solventNumbers[atomk], angle, k)
    elif isinstance(referenceForce, mm.PeriodicTorsionForce):
        nTorsion = referenceForce.getNumTorsions()
        for torsionIndex in xrange(nTorsion):
            atomi, atomj, atomk, atoml, period, phase, k = referenceForce.getTorsionParameters(torsionIndex)
            mainTorsionForce.addTorsion(solventNumbers[atomi], solventNumbers[atomj], solventNumbers[atomk], solventNumbers[atoml], period, phase, k)
    elif isinstance(referenceForce, mm.NonbondedForce):
        #Add the particle to the main nonbonded force. Custom will come after
        nParticles = referenceForce.getNumParticles()
        for atomi in xrange(nParticles):
            q, sig, epsi = referenceForce.getParticleParameters(atomi)
            mainNonbondedForce.addParticle(q, sig, epsi)
        nException = referenceForce.getNumExceptions()
        for exceptionIndex in xrange(nException):
            atomi, atomj, chargeProd, sig, epsi = referenceForce.getExceptionParameters(exceptionIndex)
            mainNonbondedForce.addException(solventNumbers[atomi], solventNumbers[atomj], chargeProd, sig, epsi)


#=== NONBONDED AND CUSTOM NONBONDED ===
#Now that all atoms are at least in the system, build the (custom) nonbonded forces
buildNonbonded(mainSystem, Rsystems, RMainAtomNumbers, solventNumbers, Ni, Nj)


#=== ATTACH INTEGRATOR, TEMPERATURE/PRESSURE COUPLING, AND MAKE CONTEXT ===
integrator = mm.LangevinIntegrator(298*unit.kelvin, 1.0/unit.picosecond, 2*unit.femtosecond)
barostat = mm.MonteCarloBarostat(1*unit.bar, 298*unit.kelvin, 1)
mainSystem.addForce(barostat)
platform = mm.Platform.getPlatformByName('OpenCL')
context = mm.Context(mainSystem, integrator, platform)

#=== MINIMIZE ENERGIES ===
context.setPositions(mainPositions)
context.applyConstraints(1E-6)
#Assign random lambda vector (testing)
pdb.set_trace()
randLam = np.random.random(Ni*Nj)
assignLambda(context, randLam, Ni, Nj)
checkLam = getLambda(context, Ni, Nj)
print context.getState(getEnergy=True, groups=1).getPotentialEnergy()
print context.getState(getEnergy=True).getPotentialEnergy()
#Minimize positions
mm.LocalEnergyMinimizer.minimize(context, 1.0 * unit.kilojoules_per_mole / unit.nanometers, 0)
    
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
