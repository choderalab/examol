import pdb
import numpy as np
import simtk.unit as unit
import simtk.openmm as mm
from simtk.openmm import app
from sys import stdout

'''
This module has the helper functions for examol, mostly system combinations
I split this off to reduce the clutter in the main examol script.
'''
# Cosntants #
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

def getArbitraryForce(system, force):
    #Get the instance of specified force from the openmm system
    for forceIndex in xrange(system.getNumForces()):
        referenceForce = system.getForce(forceIndex)
        if isinstance(referenceForce,force):
            return referenceForce
    return None

def mapAtomsToMain(atomIndices, mainMap, Ncore):
    #Map the atom to the correct index in the main system
    if type(atomIndices) == int:
        singleEntry = True
        atomIndices = [atomIndices]
    else:
        singleEntry = False
    nAtom = len(atomIndices)
    for i in xrange(nAtom):
        atomIndex = atomIndices[i]
        if atomIndex > Ncore-1:
            atomIndices[i] = mainMap[atomIndex - Ncore]
    if singleEntry:
        return atomIndices[0]
    else:
        return atomIndices

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
    #Find COGeometry of the known common structureand translate
    refCenter = np.mean(refCoords, axis=0)
    xformCenter = np.mean(xformCoords[:nalign], axis=0)
    refCoords -= refCenter
    xformCoords -= xformCenter
    #Compute rotation
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
    #Rotate new coordinates
    newCoords = np.dot(xformCoords, rotation)
    #Remove COG translation
    newCoords += refCenter
    return newCoords * baseunit

def copyTopologyBtoA(topA, topB):
    #Function to add on topology B to topology A by a copy.
    #This is a FULL copy
    #Map linking atoms in B to A so I can make bonds
    atommap = []
    for chainB in topB.chains():
        chainA = topA.addChain()
        for resB in chainB.residues():
            resA = topA.addResidue(resB.name, chainA)
            for atomB in resB.atoms():
                atomA = topA.addAtom(atomB.name, atomB.element, resA)
                atommap.append((atomB,atomA))
    natoms = len(atommap)
    for bond in topB.bonds():
        bondBA1, bondBA2 = bond #Break up atom bonds
        bondAA1 = None
        bondAA2 = None
        for i in xrange(natoms):
            if bondBA1 is atommap[i][0]:
                bondAA1 = atommap[i][1]
            if bondBA2 is atommap[i][0]:
                bondAA2 = atommap[i][1]
            if bondAA1 is not None and bondAA2 is not None:
                #Stop loop if both atoms found
                break
        topA.addBond(bondAA1, bondAA2)
    return

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

def addToMainTopology(mainTopology, addontopology, Ncore, addBonds=False):
    #Add the atoms from the addontopology to the mainTopology
    #Atom->Residue->Chain->Topology
    #Chain ownership has to be passed from the addontopology to the main topology before adding the atom to the main topology
    #Alternatley, choose the residue that the atom is attached to as the 
    #Grab the main residue
    atommap = []
    for residue in mainTopology.residues():
        mainres = residue
        break
    coreAtoms = [atom for atom in mainTopology.atoms()][:Ncore]
    for atom in addontopology.atoms():
        if int(atom.id) > Ncore:
            atomMain = mainTopology.addAtom(atom.name, atom.element, mainres)
            atommap.append((atom,atomMain))
        else: #Map the core atoms
            #Atom id's start with numbers, -1 to align with list index
            atommap.append((atom, coreAtoms[int(atom.id)-1]))
    natoms = len(atommap)
    if addBonds:
        for bond in addontopology.bonds():
            bondAddA1, bondAddA2 = bond #Break up atom bonds
            bondMainA1 = None
            bondMainA2 = None
            for i in xrange(natoms):
                if bondAddA1 is atommap[i][0]:
                    bondMainA1 = atommap[i][1]
                if bondAddA2 is atommap[i][0]:
                    bondMainA2 = atommap[i][1]
                if bondMainA1 is not None and bondMainA2 is not None:
                    #Stop loop if both atoms found
                    break
            #Only add if not in core atoms
            if bondMainA1 not in coreAtoms or bondMainA2 not in coreAtoms:
                mainTopology.addBond(bondMainA1, bondMainA2)
    return

