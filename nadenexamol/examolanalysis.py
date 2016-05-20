import numpy as np
from scipy import linspace
import itertools
from pymbar import MBAR
from examolhelpers import *
from examolclasses import *
import simtk.unit as unit
import simtk.openmm as mm
from simtk.openmm import app
import pdb

try:
    savez = np.savez_compressed
except:
    savez = np.savez

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

def computeFEBias(mbar, basis, basisManipulator, nPoints = 10):
    '''
    Compute the Free Energy Bias along each of the lambda vectors
    '''
    Ni = basisManipulator.Ni
    Nj = basisManipulator.Nj
    FE = np.zeros([Ni,Nj,nPoints])
    K = mbar.K
    N = mbar.N
    standardBasis = basis['standardBasis']
    crossBasis = basis['crossBasis']
    unaffected = basis['unaffected']
    L = Ni*Nj*nPoints
    u_ln = np.zeros([L,N])
    pointValue = linspace(0,1,nPoints)
    for i in xrange(Ni):
        for j in xrange(Nj):
            for nP in xrange(nPoints):
                #There are Ni*Nj repeated states at 0, so the first l is the decoupled state. A bit redundant but it ensures we have a reference state
                l = i*Nj*nPoints + j*nPoints + nP
                state = np.zeros([Ni,Nj])
                state[i,j] = pointValue[nP]
                hStandard, hCross = basisManipulator.computeSwitches(state, flat=True)
                u_ln[l,:] = unaffected + np.sum(standardBasis*hStandard, axis=-1) + np.sum(crossBasis*hCross, axis=-1)
    Deltaf_ij, dDeltaf_ij = mbar.computePerturbedFreeEnergies(u_ln)
    for i in xrange(Ni):
        for j in xrange(Nj):
            for nP in xrange(nPoints):
                l = i*Nj*nPoints + j*nPoints + nP
                FE[i,j,nP] = Deltaf_ij[0,l]
    np.save('FEBias.npy', FE)
    return

def computeFEEndstates(mbar, basis, basisManipulator, filename=None):
    '''
    Compute the Free energy of the endstates
    '''
    Ni = basisManipulator.Ni
    Nj = basisManipulator.Nj
    FE = np.zeros([Ni,Nj,nPoints])
    K = mbar.K
    N = mbar.N
    standardBasis = basis['standardBasis']
    crossBasis = basis['crossBasis']
    unaffected = basis['unaffected']
    L = Nj**Ni
    u_ln = np.zeros([L+1,N]) #Add 1 for the decoupled state to take FE with reference to
    stateMap = {}
    #Loop over all endstates
    for l, combo in enumerate(itertools.product(xrange(Nj), repeat=Ni)):
        #Each 'combo' is len = Ni and each value is the j index to set for position i
        state = np.zeros([Ni,Nj])
        for i in xrange(Ni):
            state[i,combo[i]] = 1
        hStandard, hCross = basisManipulator.computeSwitches(state, flat=True)
        u_ln[l,:] = unaffected + np.sum(standardBasis*hStandard, axis=-1) + np.sum(crossBasis*hCross, axis=-1)
        stateMap[combo] = l
    #Handle the reference state
    state = np.zeros([Ni,Nj])
    hStandard, hCross = basisManipulator.computeSwitches(state, flat=True)
    u_ln[-1,:] = unaffected + np.sum(standardBasis*hStandard, axis=-1) + np.sum(crossBasis*hCross, axis=-1)
    #Compute MBAR
    Deltaf_ij, dDeltaf_ij = mbar.computePerturbedFreeEnergies(u_ln)
    #Sort free energies, indicies now match the R-group "j" you want to specify is coupled at site "i"
    DF = np.zeros([Nj]*Ni) 
    dDF = np.zeros(DF.shape)
    for stateCombo in stateMap.keys():
        l = stateMap[stateCombo]
        DF[stateCombo] = Deltaf_ij[-1,l]
        dDF[stateCombo] = dDeltaf_ij[-1,l]
    if filename is not None:
        savez(filename, DF=DF, dDF=dDF)
    return DF, dDF

def constructConsts(filenames, verbose=False):
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
    u_kn = np.zeros([totalIterations,totalIterations])
    N_k = np.zeros(totalIterations)
    standardBasis = np.zeros(standardShape)
    crossBasis = np.zeros(crossShape)
    unaffected = np.zeros(totalIterations)
    kAssign = 0
    n = 0 
    #Stich together frames, fiugre out states
    #This would be the place to detect correlated samples, this loop would need to change a bit though
    for filei, (ncfile, energy, indvIteration) in enumerate(zip(ncfiles, energies, iterations)):
        if verbose: print("Workign on ncfile {0:d}/{1:d}".format(filei+1,len(filenames)))
        for iteration in xrange(indvIteration):
            state = tuple(energy['state'][iteration])
            if state not in stateMap.keys():
                stateMap[state] = kAssign
                k = kAssign
                kAssign += 1
            else:
                k = stateMap[state]
            N_k[k] += 1
            standardBasis[n] = energy['standardBasis'][iteration]
            crossBasis[n] = energy['crossBasis'][iteration]
            unaffected[n] = energy['unaffected'][iteration]
            n += 1
    totalN = n #should also be N_k.sum()
    totalK = kAssign
    #Trim up
    standardBasis = standardBasis[:totalN]
    crossBasis    = crossBasis[:totalN]
    unaffected    = unaffected[:totalN]
    N_k           = N_k[:totalK]
    u_kn          = u_kn[:totalK,:totalN]
    #Create Switches
    hStandard_k   = np.zeros( (totalK,)+standardShape.shape[1:])
    hCross_k   = np.zeros( (totalK,)+crossShape.shape[1:])
    #Create the switches
    for state in stateMap.keys():
        k = stateMap(state)
        hStandard, hCross = basisManipulator.computeSwitches(state, flat=True)
        hStandard_k[k] = hStandard
        hCross_k[k] = hCross
    #Assign u_kn
    for k in xrange(totalK):
        u_kn[k,:] = unaffected + np.sum(hStandard_k[k] * standardBasis) + np.sum(hCross_k[k] * crossBasis)
    if verbose: print("Building MBAR")
    mbar = MBAR(u_kn, N_k, verbose=verbose)
    basisOut = {'unaffected':unaffected, 'standardBasis':standardBasis, 'crossBasis':crossBasis}
    return mbar, basisOut, basisManipulator

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
    mbar, basis, basisManipulator = constructConsts(filenames)
    computeFEBias(mbar, basis, basisManipulator)
    computeFEEndstates(mbar, basis, basisManipulator)
