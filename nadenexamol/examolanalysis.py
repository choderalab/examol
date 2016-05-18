import numpy as np
from pymbar import MBAR
from examolhelpers import *
from examolclasses import *
import simtk.unit as unit
import simtk.openmm as mm
from simtk.openmm import app
import pdb

def ncvarToPython(ncvar, Ni=None, Nj=None):
    '''
    Turn an NetCDF4 variables for protocols and constants back into Python variables
    '''
    # Get option value.
    varValue = ncvar[:]
    # Get python types
    varType = getattr(ncvar, 'protoType')
    #Cast to correct type    
    try:
        if varType == 'bool':
            varValue = bool(varValue)
        elif varType == 'int':
            varValue = int(varValue)
        elif varType == 'float':
            varValue = float(varValue)
        elif varType == 'list':
            varValue = varValue.split('---')
        elif varType == 'dict':
            varValue = dict([ (varValue[i,0],varValue[i,1]) for i in xrange(varValue.shape[0]) ])
        elif varType == 'str':
            varValue = str(varValue)
        elif varType == 'ndarray':
            if Ni is None or Nj is None:
                Exception("Need Ni and Nj to recast an ndarray from the varcols")
                raise
            varValue = varValue.reshape([Ni,Nj])
        elif varType == 'NoneType':
            varValue = None
        else:
            print "wut m8?"
            raise
    except: pdb.set_trace()
    # If Quantity, assign units.
    if hasattr(ncvar, 'units'):
        unitName = getattr(ncvar, 'units')
        if unitName[0] == '/': unitName = '1' + unitName
        varUnit = eval(unitName, vars(unit))
        varValue = unit.Quantity(varValue, varUnit)
    return varValue

def readSimulation(filename):
    #Made it its own function in case I want to do more with it
    return loadnc(filename, full=True)

def runAnalysis(filenames):
    ncfiles = []
    energies = []
    iterations = []
    for filename in filenames:
        ncfile, energy = readSimulation(filename)
        iterations.append(iterations = ncfile.variables['positions'].shape[0])
        ncfiles.append(ncfile)
        energies.append(energy)
    #Ger some basic shape information
    totalIterations = sum(iterations)
    #Figure out what size and shape the basis functions will have
    standardShape = (totalIterations,) + energies[0]['energies']['standardBasis'].shape
    crossShape = (totalIterations,) + energies[0]['energies']['crossBasis'].shape
    #Get the protocols and constants out of the NetCDF files
    Ni = ncvarToPython(ncfiles[0].variables['Ni'][0])
    Nj = ncvarToPython(ncfiles[0].variables['Nj'][0])
    standardSwitches = ncvarToPython(ncfiles[0].groups['protocols'].variables['standardSwitches'])
    crossSwitches = ncvarToPython(ncfiles[0].groups['protocols'].variables['crossSwitches'])
    standardBasisCoupling = ncvarToPython(ncfiles[0].groups['protocols'].variables['standardBasisCoupling'])
    crossBasisCoupling = ncvarToPython(ncfiles[0].groups['protocols'].variables['crossBasisCoupling'])
    basisManipulator = basisManipulation(Ni, Nj, standardSwitches, standardBasisCoupling, crossSwitches, crossBasisCoupling)
    #List of lambda -> state index map 
    stateMap = {}
    #Pre Allocate the energy matricies (we'll trim them later)
    u_nk = np.zeros([totalIterations,totalIterations])
    N_k = np.zeros(totalIterations)
    standardBasis = np.zeros(standardShape)
    crossBasis = np.zeros(crossShape)
    n=0
    for ncfile, energy, indvIteration in zip(ncfiles, energies, iterations):
        for iteration in xrange(indvIteration):
            state = energy['state']
            n += 1
    return

if __name__== "__main__":
    #Import files
    #Stich the frames together
    #Split the states
    #Figure out the switches for each state
    ##Store them in dictionaries using the lambda vector tuple as a key
    #Generate MBAR energies
    #Generate MBAR object
    #Go through the endstates
    #Go through the bias states
    #Write out the bias file
    #See if FE is self-consistant
    runAnalysis()
