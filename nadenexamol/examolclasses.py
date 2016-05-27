import pdb
import numpy as np
import simtk.unit as unit
import simtk.openmm as mm
from simtk.openmm import app
from copy import deepcopy
from sys import stdout
import os
import os.path
import itertools
from scipy.interpolate import UnivariateSpline
from examolhelpers import *
from examolintegrators import *
import netCDF4 as netcdf
import time
import datetime
import cProfile
import pstats
import pickle
'''
This module houses all the custom force functions used by the main exmaol script.
I split this off to reduce the clutter in the main examol script.
'''

kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA

try:
    savez = np.savez_compressed
except:
    savez = np.savez

def basisEnergy(i, j, i2=None, j2=None, LJ=True, Electro=True):
    '''
    Houses the basic energy string for the basis function energy. Based on Naden and Shirts, JCTC 11 (6), 2015, pp. 2536-2549

    Can pass in i2 and j2 for alchemical <-> alchemical groups, It may be possible to simplify this interation to linear (one basis function), but I think having it as full basis function may be better since I have flexible groups

    Energy expression kept here to reduce clutter elsewhere.
    All of the "If" statements are so the order of the global parameters for setting defaults is preserved
    
    CURRENTLY HARD CODED: R C EA
    '''
    ONE_4PI_EPS0 = 138.935456 #From OpenMM's OpenCL kernel
    if i2 is None and j2 is None:
        lamE = 'lam{0:s}x{1:s}E'.format(str(i),str(j))
        lamP = 'lam{0:s}x{1:s}P'.format(str(i),str(j))
        lamC = 'lam{0:s}x{1:s}C'.format(str(i),str(j))
        lamA = 'lam{0:s}x{1:s}A'.format(str(i),str(j))
        lamR = 'lam{0:s}x{1:s}R'.format(str(i),str(j))
        lamB = 'lam{0:s}x{1:s}B'.format(str(i),str(j))
    else:
        lamE = 'lam{0:s}x{1:s}x{2:s}x{3:s}E'.format(str(i),str(j),str(i2),str(j2))
        lamP = 'lam{0:s}x{1:s}x{2:s}x{3:s}P'.format(str(i),str(j),str(i2),str(j2))
        lamC = 'lam{0:s}x{1:s}x{2:s}x{3:s}C'.format(str(i),str(j),str(i2),str(j2))
        lamA = 'lam{0:s}x{1:s}x{2:s}x{3:s}A'.format(str(i),str(j),str(i2),str(j2))
        lamR = 'lam{0:s}x{1:s}x{2:s}x{3:s}R'.format(str(i),str(j),str(i2),str(j2))
        lamB = 'lam{0:s}x{1:s}B*lam{2:s}x{3:s}B'.format(str(i),str(j),str(i2),str(j2))
   
    #Start energy Expression
    if i2 is None and j2 is None: #Conditional for approximate potential
        energy_expression = "theEnergy;"
    else:
        energy_expression = "theEnergy*includeSecond;"
    #DEBUG:
    energy_expression = "0;"
    switchRules = ''
    if LJ and Electro:
        energy_expression +=  "theEnergy = epsilon*(RepSwitchCappedBasis + AttSwitchBasis + CappingSwitchBasis) + electrostatics;"
    elif LJ:
        energy_expression +=  "theEnergy = epsilon*(RepSwitchCappedBasis + AttSwitchBasis + CappingSwitchBasis);"
    elif Electro:
        energy_expression +=  "theEnergy = electrostatics;"
    else:
        print("I need some type of nonbonded force!")
        raise

    if LJ:
        ###NOTE: the epsilon has been extracted but the 4 is still embeded!###########
        #energy_expression += "CappingSwitchBasis = {0:s}*CappingBasis;".format(lamC)
        energy_expression += "CappingSwitchBasis = capSwitch*CappingBasis;"
        energy_expression += "CappingBasis = repUncap - RepCappedBasis;"
        energy_expression += "RepSwitchCappedBasis = repSwitch*RepCappedBasis;"
        energy_expression += "RepCappedBasis = Rnear + Rtrans + Rfar;"
        energy_expression += "AttSwitchBasis = attSwitch*attBasis;"
        #energy_expression += "repSwitch = pa*({0:s}^4) + pb*({0:s}^3) + pc*({0:s}^2) + (1-pa-pb-pc)*{0:s};".format(lamR) #Repulsive Term
        #energy_expression += "pa = %f; pb = %f; pc = %f;" % (1.61995584, -0.8889962, 0.02552684) #Values found from optimization routine
        #energy_expression += "attSwitch = {0:s};".format(lamA)
        energy_expression += "attBasis = Anear+Afar;"
        energy_expression += "Anear = -1 * step(1-r/((2^(1.0/6.0))*sigma));" #WCA attrcative plateau near r=0
        energy_expression += "Afar = LJ*(1-step(1-r/((2^(1.0/6.0))*sigma)));" #WCA attractive comp far away.
        energy_expression += "Rnear = (Ucap1+1)*step(1-r/(cap1*sigma));" #Define platau near r=0 for repulsion
        energy_expression += "Rtrans = (T + 1 + Ucap2)*step(1-r/(cap2*sigma))*(1-step(1-r/(cap1*sigma)));"
        energy_expression += "Rfar = (LJ + 1)*step(1-r/((2^(1.0/6.0))*sigma))*(1-step(1-r/(cap2*sigma)));"
        #Curveing function, its plenty easier to write it like this instead of not having it range from zero to one
        energy_expression += "T=a*tvar^4+b*tvar^3+c*tvar^2+d*tvar+e;"
        energy_expression += "a=d2Cap2*idt^2/2 - 3*deltaC - 2*dCap2*idt;"
        energy_expression += "b=-d2Cap2*idt^2 + 8*deltaC + 5*dCap2*idt;"
        energy_expression += "c=d2Cap2*idt^2/2 - 6*deltaC - 3*dCap2*idt;"
        energy_expression += "d=0;"
        energy_expression += "e=deltaC;"
        energy_expression += "tvar=(r-cap1*sigma)/(cap2*sigma-cap1*sigma);"
        energy_expression += "idt=1.0/dt;"
        energy_expression += "dt = 1.0/(cap2*sigma-cap1*sigma);"
        energy_expression += "deltaC = Ucap1-Ucap2;"
        energy_expression += "repUncap = (LJ + 1)*step(1-r/((2^(1.0/6.0))*sigma));" #Uncapped Repulsive Basis
        energy_expression += "LJ = 4*((sigma/r)^12 - (sigma/r)^6);" #Lennard-Jones statment
        energy_expression += "Ucap1 = 4*((1.0/cap1)^12 - (1.0/cap1)^6);" #Potential at cap 1 = 0.8sigma
        energy_expression += "Ucap2 = 4*((1.0/cap2)^12 - (1.0/cap2)^6);" #Potential at cap 2 = 0.9sigma
        energy_expression += "dCap2 = 4*(-12.0/(cap2^13*sigma) + 6.0/(cap2^7*sigma));"#Derivative at cap 2 = .9 sigma
        energy_expression += "d2Cap2 = 4*(13.0*12.0/(cap2^14*sigma^2) - 7.0*6.0/(cap2^8*sigma^2));"# Second Derivative at cap 2 = .9 sigma
        energy_expression += "cap1 = %f; cap2 = %f;" % (0.8, 0.9)
        energy_expression += "epsilon = sqrt(epsilon1*epsilon2);" # mixing rule for epsilon
        energy_expression += "sigma = 0.5*(sigma1 + sigma2);" # mixing rule for sigma
        #Write the rules for the individual switches
        energy_expression += switchRules
        switchRules += "repSwitch = (derivMode*repDeriv) + ((1-derivMode)*repBase);"
        switchRules += "repBase = pa*(repVar^4) + pb*(repVar^3) + pc*(repVar^2) + (1-pa-pb-pc)*repVar;"
        #Derivative dU/dLB = pU/pLR * pLR/pLB
        switchRules += "repDeriv = (4*pa*(repVar^3) + 3*pb*(repVar^2) + 2*pc*(repVar) + (1-pa-pb-pc))*RDerivCheck;"
        #                              pLR/pLB     Should it be 0 or not
        switchRules += "RDerivCheck = (nStage-1)*Rcheck1*Rcheck2;"
        switchRules += "repVar = (singleSwitchMode * {0:s}) + ((1-singleSwitchMode)*lamMapR);".format(lamR)
        switchRules += "pa = %f; pb = %f; pc = %f;" % (1.61995584, -0.8889962, 0.02552684) #Values found from optimization routine
        switchRules += "lamMapR = min(1,Rscale*max(Rcheck1*Rcheck2,Rcheck3));"
        switchRules += "Rcheck1 = step(Rscale); Rcheck2 = step(1-Rscale); Rcheck3 = step(Rscale-1);"
        switchRules += "Rscale = ((nStage-1)*{lam:s}-{off:d});".format(lam=lamB,off=0)
        #Attractive
        switchRules += "attSwitch = (derivMode*attDeriv) + ((1-derivMode)*attBase);"
        switchRules += "attBase = attVar;"
        switchRules += "attDeriv = 1*ADerivCheck;"
        switchRules += "ADerivCheck = (nStage-1)*Acheck1*Acheck2;"
        switchRules += "attVar = (singleSwitchMode * {0:s}) + ((1-singleSwitchMode)*lamMapA);".format(lamA)
        switchRules += "lamMapA = min(1,Ascale*max(Acheck1*Acheck2,Acheck3));"
        switchRules += "Acheck1 = step(Ascale); Acheck2 = step(1-Ascale); Acheck3 = step(Ascale-1);"
        switchRules += "Ascale = ((nStage-1)*{lam:s}-{off:d});".format(lam=lamB,off=1)
        #Cap rules
        switchRules += "capSwitch = (derivMode*capDeriv) + ((1-derivMode)*capBase);"
        switchRules += "capBase = capVar;"
        switchRules += "capDeriv = 1*CDerivCheck;"
        switchRules += "CDerivCheck = (nStage-1)*delta({lam:s} - 1.0/(nStage-1));".format(lam=lamB)
        switchRules += "capVar = (singleSwitchMode * {0:s}) + ((1-singleSwitchMode)*step({1:s} - 1.0/(nStage-1) - 0.000001));".format(lamC,lamB)
    if Electro:
        #=== Electrostatics ===
        # This block commented out until I figure out how to do long range PME without calling updateParametersInContext(), Switched to reaction field below
        #err_tol = nonbonded_force.getEwaldErrorTolerance()
        #rcutoff = nonbonded_force.getCutoffDistance() / unit.nanometer #getCutoffDistance Returns a unit object, convert to OpenMM default (nm)
        #alpha = numpy.sqrt(-numpy.log(2*err_tol))/rcutoff #NOTE: OpenMM manual is wrong here
        #energy_expression = "alchEdirect-(Edirect * eitheralch);" #Correction for the alchemical1=0 alchemical2=0 case)
        #energy_expression += "Edirect = ({0:f} * (switchPME*alchemical1 + 1 -alchemical1) * charge1 * (switchPME*alchemical2 + 1 -alchemical2) * charge2 * erfc({1:f} * r)/r);".format(ONE_4PI_EPS0, alpha) #The extra bits correct for alchemical1 and alhemical being on
        #energy_expression += "alchEdirect = switchE * {0:f} * charge1 * charge2 * erfc({1:f} * r)/r;".format(ONE_4PI_EPS0, alpha)
        #energy_expression += "switchPME = {0:s};".format(lamP)
        #energy_expression += "switchE = {0:s};".format(lamE)

        #energy_expression += "electrostatics = {0:s}*charge1*charge2*{1:f}*reaction_field;".format(lamE, ONE_4PI_EPS0)
        energy_expression += "electrostatics = elecSwitch*charge1*charge2*{0:f}*reaction_field;".format(ONE_4PI_EPS0)
        energy_expression += "reaction_field = (1/r) + krf*r^2 - crf;"
        energy_expression += "krf = (1/(rcut^3)) * ((dielectric-1)/(2*dielectric+1));"
        energy_expression += "crf = (1/rcut) * ((3*dielectric)/(2*dielectric+1));"
        switchRules += "elecSwitch = (derivMode*elecDeriv) + ((1-derivMode)*elecBase);"
        switchRules += "elecBase = elecVar;"
        switchRules += "elecDeriv = 1*EDerivCheck;"
        switchRules += "EDerivCheck = (nStage-1)*Echeck1*Echeck2;"
        switchRules += "elecVar = (singleSwitchMode * {0:s}) + ((1-singleSwitchMode)*lamMapE);".format(lamE)
        switchRules += "lamMapE = min(1,Escale*max(Echeck1*Echeck2,Echeck3));"
        switchRules += "Echeck1 = step(Escale); Echeck2 = step(1-Escale); Echeck3 = step(Escale-1);"
        switchRules += "Escale = ((nStage-1)*{lam:s}-{off:d});".format(lam=lamB,off=1)
    
    switchRules += "nStage = {0:d};".format(3)
    energy_expression += switchRules
    custom_nonbonded_force = mm.CustomNonbondedForce(energy_expression)
    #All the global/PerParticle parameters dont matter, they just will ocupy a bit extra memory
    if Electro:
        custom_nonbonded_force.addPerParticleParameter("charge")
        custom_nonbonded_force.addGlobalParameter("{0:s}".format(lamE), 1)
        #custom_nonbonded_force.addGlobalParameter("{0:s}".format(lamP), 1)
        custom_nonbonded_force.addGlobalParameter("dielectric", 70)
        custom_nonbonded_force.addGlobalParameter("rcut", 1)
    if LJ:
        custom_nonbonded_force.addPerParticleParameter("sigma") # Lennard-Jones sigma
        custom_nonbonded_force.addPerParticleParameter("epsilon") # Lennard-Jones epsilon
        custom_nonbonded_force.addGlobalParameter("{0:s}".format(lamC), 1)
        custom_nonbonded_force.addGlobalParameter("{0:s}".format(lamA), 1)
        custom_nonbonded_force.addGlobalParameter("{0:s}".format(lamR), 1)
    custom_nonbonded_force.addGlobalParameter("lam{0:d}x{1:d}B".format(i,j), 1)
    if i2 is not None and j2 is not None:
        custom_nonbonded_force.addGlobalParameter("lam{0:d}x{1:d}B".format(i2,j2), 1)
        custom_nonbonded_force.addGlobalParameter("includeSecond", 1)
    custom_nonbonded_force.addGlobalParameter("derivMode", 0)
    custom_nonbonded_force.addGlobalParameter("singleSwitchMode", 0)
    
    return custom_nonbonded_force

def basisUncapLinearEnergy(i, j, i2=None, j2=None, LJ=True, Electro=True):
    '''
    This is an uncapped potential energy function, where all energies are wrapped in a standard linear switch
    '''
    ONE_4PI_EPS0 = 138.935456 #From OpenMM's OpenCL kernel
    if i2 is None and j2 is None:
        lamE = 'lam{0:s}x{1:s}E'.format(str(i),str(j))
        lamP = 'lam{0:s}x{1:s}P'.format(str(i),str(j))
        lamA = 'lam{0:s}x{1:s}A'.format(str(i),str(j))
        lamR = 'lam{0:s}x{1:s}R'.format(str(i),str(j))
        lamB = 'lam{0:s}x{1:s}B'.format(str(i),str(j))
    else:
        lamE = 'lam{0:s}x{1:s}x{2:s}x{3:s}E'.format(str(i),str(j),str(i2),str(j2))
        lamP = 'lam{0:s}x{1:s}x{2:s}x{3:s}P'.format(str(i),str(j),str(i2),str(j2))
        lamA = 'lam{0:s}x{1:s}x{2:s}x{3:s}A'.format(str(i),str(j),str(i2),str(j2))
        lamR = 'lam{0:s}x{1:s}x{2:s}x{3:s}R'.format(str(i),str(j),str(i2),str(j2))
        lamB = 'lam{0:s}x{1:s}B*lam{2:s}x{3:s}B'.format(str(i),str(j),str(i2),str(j2))
   
    #Start energy Expression
    if i2 is None and j2 is None: #Conditional for approximate potential
        energy_expression = "theEnergy;"
    else:
        energy_expression = "theEnergy*includeSecond;"
    switchRules = ""
    if LJ and Electro:
        energy_expression +=  "theEnergy = epsilon*(RepSwitchBasis + AttSwitchBasis) + electrostatics;"
    elif LJ:
        energy_expression +=  "theEnergy = epsilon*(RepSwitchBasis + AttSwitchBasis);"
    elif Electro:
        energy_expression +=  "theEnergy = electrostatics;"
    else:
        print("I need some type of nonbonded force!")
        raise
    if LJ:
        ###NOTE: the epsilon has been extracted but the 4 is still embeded!###########
        energy_expression += "RepSwitchBasis = repSwitch*RepBasis;"
        energy_expression += "RepBasis = Rnear + Rfar;"
        energy_expression += "AttSwitchBasis = attSwitch*attBasis;"
        #energy_expression += "repSwitch = {0:s};".format(lamR) #Repulsive Term
        #energy_expression += "attSwitch = {0:s};".format(lamA)
        energy_expression += "attBasis = Anear+Afar;"
        energy_expression += "Anear = -1 * step(1-r/((2^(1.0/6.0))*sigma));" #WCA attrcative plateau near r=0
        energy_expression += "Afar = LJ*(1-step(1-r/((2^(1.0/6.0))*sigma)));" #WCA attractive comp far away.
        energy_expression += "Rnear = (LJ + 1)*step(1-r/((2^(1.0/6.0))*sigma));" #Uncapped Repulsive Basis
        energy_expression += "Rfar = 0;" #*step(1-step(1-r/((2^(1.0/6.0))*sigma)));"
        energy_expression += "LJ = 4*((sigma/r)^12 - (sigma/r)^6);" #Lennard-Jones statment
        energy_expression += "epsilon = sqrt(epsilon1*epsilon2);" # mixing rule for epsilon
        energy_expression += "sigma = 0.5*(sigma1 + sigma2);" # mixing rule for sigma
        #set up the Switch rules, since these all change together
        switchRules += "repSwitch = (derivMode*repDeriv) + ((1-derivMode)*repBase);"
        #switchRules += "repBase = repVar;"
        #switchRules += "repDeriv = 1;"
        switchRules += "repBase = repVar^4;"
        switchRules += "repDeriv = 4*repVar^3;"
        switchRules += "repVar = (singleSwitchMode * {0:s}) + ((1-singleSwitchMode)*{1:s});".format(lamR,lamB)
        switchRules += "attSwitch = (derivMode*attDeriv) + ((1-derivMode)*attBase);"
        #switchRules += "attBase = attVar;"
        #switchRules += "attDeriv = 1;"
        switchRules += "attBase = attVar^4;"
        switchRules += "attDeriv = 4*attVar^3;"
        switchRules += "attVar = (singleSwitchMode * {0:s}) + ((1-singleSwitchMode)*{1:s});".format(lamA,lamB)

    if Electro:
        #=== Electrostatics ===
        # This block commented out until I figure out how to do long range PME without calling updateParametersInContext(), Switched to reaction field below
        #err_tol = nonbonded_force.getEwaldErrorTolerance()
        #rcutoff = nonbonded_force.getCutoffDistance() / unit.nanometer #getCutoffDistance Returns a unit object, convert to OpenMM default (nm)
        #alpha = numpy.sqrt(-numpy.log(2*err_tol))/rcutoff #NOTE: OpenMM manual is wrong here
        #energy_expression = "alchEdirect-(Edirect * eitheralch);" #Correction for the alchemical1=0 alchemical2=0 case)
        #energy_expression += "Edirect = ({0:f} * (switchPME*alchemical1 + 1 -alchemical1) * charge1 * (switchPME*alchemical2 + 1 -alchemical2) * charge2 * erfc({1:f} * r)/r);".format(ONE_4PI_EPS0, alpha) #The extra bits correct for alchemical1 and alhemical being on
        #energy_expression += "alchEdirect = switchE * {0:f} * charge1 * charge2 * erfc({1:f} * r)/r;".format(ONE_4PI_EPS0, alpha)
        #energy_expression += "switchPME = {0:s};".format(lamP)
        #energy_expression += "switchE = {0:s};".format(lamE)

        energy_expression += "electrostatics = elecSwitch*charge1*charge2*{0:f}*reaction_field;".format(ONE_4PI_EPS0)
        energy_expression += "reaction_field = (1/r) + krf*r^2 - crf;"
        energy_expression += "krf = (1/(rcut^3)) * ((dielectric-1)/(2*dielectric+1));"
        energy_expression += "crf = (1/rcut) * ((3*dielectric)/(2*dielectric+1));"
        switchRules += "elecSwitch = (derivMode*elecDeriv) + ((1-derivMode)*elecBase);"
        #switchRules += "elecBase = elecVar;"
        #switchRules += "elecDeriv = 1;"
        switchRules += "elecBase = elecVar^4;"
        switchRules += "elecDeriv = 4*elecVar^3;"
        switchRules += "elecVar = (singleSwitchMode * {0:s}) + ((1-singleSwitchMode)*{1:s});".format(lamE,lamB)

    energy_expression += switchRules
    custom_nonbonded_force = mm.CustomNonbondedForce(energy_expression)
    #All the global/PerParticle parameters dont matter, they just will ocupy a bit extra memory
    if Electro:
        custom_nonbonded_force.addPerParticleParameter("charge")
        custom_nonbonded_force.addGlobalParameter("{0:s}".format(lamE), 1)
        #custom_nonbonded_force.addGlobalParameter("{0:s}".format(lamP), 1)
        custom_nonbonded_force.addGlobalParameter("dielectric", 70)
        custom_nonbonded_force.addGlobalParameter("rcut", 1)
    if LJ:
        custom_nonbonded_force.addPerParticleParameter("sigma") # Lennard-Jones sigma
        custom_nonbonded_force.addPerParticleParameter("epsilon") # Lennard-Jones epsilon
        custom_nonbonded_force.addGlobalParameter("{0:s}".format(lamA), 1)
        custom_nonbonded_force.addGlobalParameter("{0:s}".format(lamR), 1)
    custom_nonbonded_force.addGlobalParameter("lam{0:d}x{1:d}B".format(i,j), 1)
    if i2 is not None and j2 is not None:
        custom_nonbonded_force.addGlobalParameter("lam{0:d}x{1:d}B".format(i2,j2), 1)
        custom_nonbonded_force.addGlobalParameter("includeSecond", 1)
    custom_nonbonded_force.addGlobalParameter("derivMode", 0)
    custom_nonbonded_force.addGlobalParameter("singleSwitchMode", 0)
    return custom_nonbonded_force

def biasDerivative(i, j, Ni, Nj, lamMin = 0.3,  K = 50.0):
    '''
    Bias derivative in the lambda dims. Flat bottom conditional harmonic restraint
    
    lamMin : Float
        The minimum threshold for Heaviside step saying "this group is approaching fully coupled"
    K       : float
        The "spring constant" in the harmonic restraint, units of kJ/(mol lam^2)
    '''
    energy_expression = ''
    lamVar = 'lam{i:d}x{j:d}B'
    basisij = lamVar.format(i=i,j=j)
    #Set up Heaviside step and energy expression, we also want bias OFF in single switch mode
    energy_expression += "(1-singleSwitchMode)*derivMode*biasSum;"
    biasObjs = []
    biasVars = []
    for loopi in xrange(Ni):
        for loopj in xrange(Nj):
            if loopi == i and loopj == j: #check for ij == ik
                #Generate step function bias
                ikObjs = ["step({basisik:s} - {lamMin:f})*{K:f}*({basisik:s} - {lamMin:f})^2".format(basisik=lamVar.format(i=i,j=loopk), lamMin=lamMin, K=K) for loopk in xrange(Nj) if loopk != j]
                if len(ikObjs) == 0:
                    biasObjs.append('delta({basisij:s} - {lamMin:f})'.format(basisij=basisij,lamMin=lamMin) + '*(0)')
                else:
                    biasObjs.append('delta({basisij:s} - {lamMin:f})'.format(basisij=basisij,lamMin=lamMin) + '*(' + ' + '.join(ikObjs) + ')')
            else:
                basisij2 = lamVar.format(i=loopi,j=loopj)
                biasObjs.append("step({basisij2:s} - {lamMin:f})*step({basisij:s} - {lamMin:f})*2*{K:f}*({basisij:s} - {lamMin:f})".format(basisij=basisij, lamMin=lamMin,  basisij2=basisij2, K=K))
            biasVars.append(lamVar.format(i=loopi,j=loopj))
    biasSum = ' + '.join(biasObjs)
    energy_expression += "biasSum = {0};".format(biasSum)
    custom_external_force = mm.CustomExternalForce(energy_expression)
    custom_external_force.addGlobalParameter('singleSwitchMode', 0)
    custom_external_force.addGlobalParameter('derivMode', 0)
    for var in biasVars:
        custom_external_force.addGlobalParameter(var, 1)
    return custom_external_force

def biasPotential(Ni, Nj, lamMin = 0.3 , K = 50.0):
    '''
    Bias derivative in the lambda dims. Flat bottom conditional harmonic restraint
    
    lamMin : Float
        The minimum threshold for Heaviside step saying "this group is approaching fully coupled"
    K       : float
        The "spring constant" in the harmonic restraint, units of kJ/(mol lam^2)
    '''
    energy_expression = ''
    lamVar = 'lam{i:d}x{j:d}B'
    biasObjs = []
    biasVars = []
    #energy_expression += "(1-singleSwitchMode)*(1-derivMode)*biasSum;"
    energy_expression += "(1-derivMode)*biasSum;"
    for i in xrange(Ni):
        for j in xrange(Nj):
            basisij = lamVar.format(i=i,j=j)
            ikObjs = ["step({basisik2:s} - {lamMin:f})*{K:f}*({basisik2:s} - {lamMin:f})^2".format(basisik2=lamVar.format(i=i,j=k), lamMin=lamMin, K=K) for k in xrange(Nj) if k != j]
            if len(ikObjs) == 0:
                biasObjs.append('step({basisij:s} - {lamMin:f})'.format(basisij=basisij,lamMin=lamMin) + '*(0)')
            else:
                biasObjs.append('step({basisij:s} - {lamMin:f})'.format(basisij=basisij,lamMin=lamMin) + '*(' + ' + '.join(ikObjs) + ')')
            biasVars.append(basisij)
    energy_expression += 'biasSum = ' + ' + '.join(biasObjs) + ';'
    custom_external_force = mm.CustomExternalForce(energy_expression)
    #custom_external_force.addGlobalParameter('singleSwitchMode', 0)
    custom_external_force.addGlobalParameter('derivMode', 0)
    for var in biasVars:
        custom_external_force.addGlobalParameter(var, 1)
    return custom_external_force

def initFreeEnergyUBias(Ni, Nj):
    '''
    Bias with respect to the free energy.

    This is implemented in a CustomCompoundBondForce as tabulated functions cannot be implemented in a CustomExternalForce
    This force is not affected by du/dr calculations so will never update the cartesian positions. By making this force act on 2 bonded atoms 
    of the core, we ensure the 2 particles stay nearby and the "bond" is not overstreched

    Imperfect soultion, but it works!

    Tabulated functions are intilized to 0
   
    Ni and Nj : integers
        Dimentions of the free energy biases. These are NOT the du/dL force values, just the natrual bias.
        This is a single force object, but multiple tabulated functions
    '''
    energy_expression = ''
    energy_expression += "(1-derivMode)*biasFESum;"
    biasObjs = []
    tabFunctions = np.empty([Ni, Nj], dtype=object)
    for i in xrange(Ni):
        for j in xrange(Nj):
            tabFunctions[i,j] = mm.Continuous1DFunction([0,0], 0, 1)
            biasObjs.append('FEU{i:d}x{j:d}Bias(lam{i:d}x{j:d}B)'.format(i=i,j=j))
    energy_expression += 'biasFESum = ' + ' + '.join(biasObjs) + ';'
    custom_compound_bond_force = mm.CustomCompoundBondForce(2, energy_expression)
    custom_compound_bond_force.addGlobalParameter('derivMode', 0)
    for i in xrange(Ni):
        for j in xrange(Nj):
            custom_compound_bond_force.addTabulatedFunction('FEU{i:d}x{j:d}Bias'.format(i=i,j=j), tabFunctions[i,j])
            custom_compound_bond_force.addGlobalParameter('lam{i:d}x{j:d}B'.format(i=i,j=j), 0)
    custom_compound_bond_force.addBond([0,1],[])
    return custom_compound_bond_force, tabFunctions

def initFreeEnergyForceBias(i, j):
    '''
    Derivative of the bias with respect to the free energy.
 
    This creates the force object relative to a single lambda to evaluate dU/dL.
    See comments in initFreeEnergyUBias for details as to implementation

    i and j : integers
        Indicies of which lambda this function belongs to.
    '''
    energy_expression = ''
    energy_expression += "derivMode*biasFEForce;"
    energy_expression += "biasFEForce = FEF{i:d}x{j:d}Bias(lam{i:d}x{j:d}B);".format(i=i,j=j)
    custom_compound_bond_force = mm.CustomCompoundBondForce(2, energy_expression)
    custom_compound_bond_force.addGlobalParameter('derivMode', 0)
    tabFunction = mm.Continuous1DFunction([0,0], 0, 1)
    custom_compound_bond_force.addTabulatedFunction('FEF{i:d}x{j:d}Bias'.format(i=i,j=j), tabFunction)
    custom_compound_bond_force.addGlobalParameter('lam{i:d}x{j:d}B'.format(i=i,j=j), 0)
    custom_compound_bond_force.addBond([0,1],[])
    return custom_compound_bond_force, tabFunction

class basisSwitches(object):
    def _setProtocol(self,protocol):
        defaultProtocol = {}
        defaultProtocol['R'] = "nadenOptimal"
        defaultProtocol['E'] = "linear"
        defaultProtocol['C'] = "linear"
        defaultProtocol['A'] = "linear"
        defaultProtocol['B'] = "linear"
        if protocol is None:
            self.protocol = defaultProtocol
        else:
            self.protocol = {}
            try:
                for key in protocol.keys():
                    self.protocol[key] = protocol[key]
                for key in defaultProtocol.keys():
                    if key not in self.protocol.keys():
                        self.protocol[key] = defaultProtocol[key]
            except:
                errorMsg = "Protocol needs to be a dictionary. Valid keys are: "
                for key in defaultProtocols.keys():
                    errorMsg += "%s "
                print errorMsg % tuple(defaultProtocols.keys())
                print "Assuming default protocol"
                self.protocol = defaultProtocols
        for key in self.protocol.keys():
            self.protocol[key] = '_' + self.protocol[key]
        return

    def _linear(self, lam):
        return lam
    
    def _nadenOptimal(self, lam):
        repA = 1.61995584
        repB = -0.8889962
        repC = 0.02552684
        return repA*lam**4 + repB*lam**3 + repC*lam**2 + (1-repA-repB-repC)*lam
    def _square(self, lam):
        return lam**2
    def _fourth(self, lam):
        return lam**4
        
    def __init__(self, protocol=None):
        '''
        This class hosues the function deffinitions used by the basisExamol. Users can define their own functions here.
        Users should write their own functions here and cast them as private "_functionName". When setting the protocol, use the function name as the values without the _
        
        protocol : Dict., checked keys are "R" "E" "C" "A" "B". The values for the keys should be the names of the functions the user has defined in this class without the leading _
        '''
        self._setProtocol(protocol)
        self.R = getattr(self, self.protocol['R'])
        self.A = getattr(self, self.protocol['A'])
        self.C = getattr(self, self.protocol['C'])
        self.E = getattr(self, self.protocol['E'])
        self.B = getattr(self, self.protocol['B'])
        return

class basisManipulation(object):

    def _countTotalBasisFunctions(self):
        #Utility function to count up the total number of basis given the current protocols
        basisCount = 0
        standardBasisCount = 0 
        crossBasisCount = 0
        for i in xrange(self.Ni):
            for j in xrange(self.Nj):
                #Count the bonded basis
                basisCount += 1
                standardBasisCount += 1
                for uniqueSetCount in xrange(self.standardNumBasis):
                    #Add in the unique stages
                    basisCount += 1
                    standardBasisCount += 1
                #Loop through i2/j2 interactions
                for i2 in xrange(i+1,self.Ni): 
                    for j2 in xrange(self.Nj): 
                        for uniqueSetCount2 in xrange(self.crossNumBasis):
                            #Add in the cross interactions unique counts
                            basisCount += 1
                            crossBasisCount += 1
        self.standardBasisCount = standardBasisCount
        self.crossBasisCount = crossBasisCount
        return

    def castLamVector(self, lamVector):
        #Formatting function
        if isinstance(lamVector, list):
            lamVector = np.array(lamVector)
        lamVector = lamVector.reshape((self.Ni,self.Nj))
        return lamVector
   
    def computeSwitches(self, lamVector, flat=False):
        '''
        Computes the H(lambda) and H(lambda_1 * lambda_2) values for a given lambda vector. Helpful for determining what the multipliers will be for a given set of basis functions.
       
        The "flat" boolean determines if a flat set of switches should be returned instead of a fully expanded one.
        '''
        lamVector = self.castLamVector(lamVector)
        standardHValues = np.zeros([self.Ni,self.Nj,self.standardNumBasis + 1]) #Add 1 since the bonded terms are on here
        crossHValues = np.zeros([self.Ni,self.Nj,self.Ni,self.Nj,self.crossNumBasis])
        #Compute alchemical switches
        for i in xrange(self.Ni):
            for j in xrange(self.Nj):
                lams = basisMap(lamVector[i,j], self._standardBasisCoupling)
                standardHValues[i,j,-1] = self.standardH.B(lamVector[i,j])
                for uniqueSetCount in xrange(self.standardNumBasis):
                    basisSet = self._flatStandardUniqueBasis[uniqueSetCount]
                    arbitraryBasis = basisSet[0]
                    arbitraryLam = lams[arbitraryBasis]
                    standardHValues[i,j,uniqueSetCount] = getattr(self.standardH, arbitraryBasis)(arbitraryLam)
                for i2 in xrange(i+1,self.Ni): #no need to loop backwards, otherwise will double up on energy calculations
                    for j2 in xrange(self.Nj): #All j affected, just not same i
                        lamsij = basisMap(lamVector[i,j], self._crossBasisCoupling)
                        lamsij2 = basisMap(lamVector[i2,j2], self._crossBasisCoupling)
                        for uniqueSetCount2 in xrange(self.crossNumBasis):
                            basisSet2 = self._flatCrossUniqueBasis[uniqueSetCount2]
                            arbitraryBasis2 = basisSet2[0]
                            arbitraryLamij = lamsij[arbitraryBasis2]
                            arbitraryLamij2 = lamsij2[arbitraryBasis2]
                            arbitraryCrossLam = arbitraryLamij*arbitraryLamij2
                            crossHValues[i,j,i2,j2,uniqueSetCount2] = getattr(self.crossH, arbitraryBasis2)(arbitraryCrossLam)
        if flat:
            standardHValues = self.flattenBasis(standardHValues)
            crossHValues = self.flattenBasis(crossHValues)
        return standardHValues, crossHValues 

    def computeArbitraryAlchemicalEnergy(self, basis, lamVector, derivative=False, provideHValues=False):
        '''
        Compute the potential energy at an arbitary alchemical state defined by 'lamVector'. Assumes current lambda if not passed in.
        If the basis functions are already known, then they are fed in with 'basis', otherwise, they are computed from the computeBasisEnergy block

        lamVector : list of floats len(lamVector)= Ni*Nj OR ndarray of floats of (lamVector.shape = Ni*Nj or lamVector.shape = [Ni,Nj])
        basis : dict with entries returned from computeBasisEnergy
        derivative : bool, if True, returnes dU/dL instead of just U
        provideHValues: bool, if true, the 'standardHValues' and the 'crossHValues' will be returned as keys in the potential energy
        '''
        
        lamVector = self.castLamVector(lamVector)
        standardBasis = basis['standardBasis']
        crossBasis = basis['crossBasis']
        unaffectedPotential = basis['unaffectedPotential']
        if derivative:
            raise Exception("Still working on it")
        standardHValues, crossHValues = self.computeSwitches(lamVector)
        returns = {}
        if provideHValues:
            returns['standardHValues'] = standardHValues
            returns['crossHValues'] = crossHValues
        basisPotential = np.sum(standardBasis * standardHValues) + np.sum(crossBasis * crossHValues)
        basisPotential += unaffectedPotential
        basisPotential += basis['harmonicBias']
        basisPotential += basis['freeEnergyBias']
        returns['potential'] = basisPotential
        return returns

    def expandBasis(self, flatBasis):
        #Helper function to unravel a flat, non-zero array into the the basis array indexed by i, j, i2, and j2
        #Accepts either flatStandard or flatCross and determines based on size
        #Consider moving this to helpers
        counter = 0
        #Strip units
        hasUnit = False
        if isinstance(flatBasis, unit.Quantity):
            hasUnit = True
            basisUnit = flatBasis.unit
            flatBasis /= basisUnit
        if flatBasis.size == self.standardBasisCount:
            output = flatBasis.reshape(self.Ni, self.Nj, self.standardNumBasis + 1)
        elif flatBasis.size == self.crossBasisCount:
            output = np.zeros([self.Ni,self.Nj,self.Ni,self.Nj,self.crossNumBasis])
            for i in xrange(self.Ni):
                for j in xrange(self.Nj):
                    for i2 in xrange(i+1,self.Ni): 
                        for j2 in xrange(self.Nj): 
                            for uniqueSetCount2 in xrange(self.crossNumBasis):
                                output[i,j,i2,j2,uniqueSetCount2] = flatBasis[counter]
                                counter += 1
        if hasUnit:
            output *= basisUnit
        return output

    def flattenBasis(self, basis):
        '''
        Remove all the excess zeros from the cross basis and return a flat array
        Consider moving this to helpers
        '''
        hasUnit = False
        if isinstance(basis, unit.Quantity):
            hasUnit = True
            basisUnit = basis.unit
            basis /= basisUnit
        if basis.ndim == 3:
            output = basis.flatten()
        elif basis.ndim == 5:
            output = np.zeros([self.crossBasisCount])
            counter = 0
            for i in xrange(self.Ni):
                for j in xrange(self.Nj):
                    for i2 in xrange(i+1,self.Ni): 
                        for j2 in xrange(self.Nj): 
                            for uniqueSetCount2 in xrange(self.crossNumBasis):
                                output[counter] = basis[i,j,i2,j2,uniqueSetCount2]
                                counter += 1
        if hasUnit:
            output *= basisUnit
        return output

    def __init__(self, Ni, Nj, standardSwitches, standardBasisCoupling, crossSwitches, crossBasisCoupling):
        '''
        This class manipulates the basis outputs (flatten/expanding) and allows computation at arbitrary alchemical energy
        It was split off from the main basisExamol class as the analysis code would need it also.
        This class also handles the various counts of basis functions needed by the basisExamol and analysis
        '''
        self.Ni = Ni
        self.Nj = Nj
        self.standardH = basisSwitches(protocol=standardSwitches)
        self.crossH = basisSwitches(protocol=crossSwitches)
        self._standardBasisCoupling = standardBasisCoupling
        self._crossBasisCoupling = crossBasisCoupling
        #Set some constants, handling counting now for book keeping later.
        self.standardUniqueBasis = [findUniqueBasis(stage, self.standardH) for stage in self._standardBasisCoupling]
        self.crossUniqueBasis = [findUniqueBasis(stage, self.crossH) for stage in self._crossBasisCoupling]
        self.standardBasisPerStage = [len(stage) for stage in self.standardUniqueBasis]
        self.standardNumBasis = np.sum(np.array(self.standardBasisPerStage))
        self.crossBasisPerStage = [len(stage) for stage in self.crossUniqueBasis]
        self.crossNumBasis = np.sum(np.array(self.crossBasisPerStage))
        #Flatten the unique lists for computing energies later
        self._flatStandardUniqueBasis = list(itertools.chain.from_iterable(self.standardUniqueBasis))
        self._flatCrossUniqueBasis = list(itertools.chain.from_iterable(self.crossUniqueBasis))
        #Count total unique basis
        self._countTotalBasisFunctions()

        return

class basisExamol(object):

    def _addAngleForceWithCustom(self, RForce, i, j):
        #Copy angle forces from the RForce to the mainAngleForce. Uses info from RAtomNubmers to map the unique angles in the RSystems to the mainSystem. Creates a custom angle force for bonds to the core where R attaches
        lamExpression = 'lam{0:s}x{1:s}B'.format(str(i), str(j))
        energyExpression = '((1-derivMode)*{0:s} + derivMode*{1:s})*0.5*k*(theta-theta0)^2;'.format(lamExpression+'^4', '4*'+lamExpression+'^3')
        customAngleForce = mm.CustomAngleForce(energyExpression)
        customAngleForce.addGlobalParameter(lamExpression, 1)
        customAngleForce.addGlobalParameter('derivMode', 0)
        customAngleForce.addPerAngleParameter('theta0')
        customAngleForce.addPerAngleParameter('k')
        nRAngle = RForce.getNumAngles()
        for angleIndex in xrange(nRAngle):
            atomi, atomj, atomk, angle, k = RForce.getAngleParameters(angleIndex)
            atomi, atomj, atomk = mapAtomsToMain([atomi, atomj, atomk], self.RMainAtomNumbers[i,j], self.Ncore)
            #Is the angle part of the new R group?
            if (atomi >= self.Ncore or atomj >= self.Ncore or atomk >= self.Ncore):
                #is the angle both core AND R group? (and when combined with previous)
                if atomi < self.Ncore or atomj < self.Ncore or atomk < self.Ncore:
                    customAngleForce.addAngle(atomi, atomj, atomk, (angle, k)) 
                else: #Standard angle in R group
                    self.mainAngleForce.addAngle(atomi, atomj, atomk, angle, k)
        return customAngleForce
    
    def _addTorsionForceWithCustom(self, RForce, i, j):
        #Copy torsion forces from the RForce to the mainTorsionForce. Uses info from RAtomNubmers to map the unique torsions in the RSystems to the mainSystem. Creates a custom torsion force for bonds to the core where R attaches
        lamExpression = 'lam{0:s}x{1:s}B'.format(str(i), str(j))
        energyExpression = '((1-derivMode)*{0:s} + derivMode*{1:s})*k*(1+cos(n*theta-theta0));'.format(lamExpression+'^4', '4*'+lamExpression+'^3')
        customTorsionForce = mm.CustomTorsionForce(energyExpression)
        customTorsionForce.addGlobalParameter(lamExpression, 1)
        customTorsionForce.addGlobalParameter('derivMode', 0)
        customTorsionForce.addPerTorsionParameter('n')
        customTorsionForce.addPerTorsionParameter('theta0')
        customTorsionForce.addPerTorsionParameter('k')
        #customAngleForce.setEnergyFunction(energyExpression)
        nRTorsion = RForce.getNumTorsions()
        for torsionIndex in xrange(nRTorsion):
            atomi, atomj, atomk, atoml, period, phase, k = RForce.getTorsionParameters(torsionIndex)
            atomi, atomj, atomk, atoml = mapAtomsToMain([atomi, atomj, atomk, atoml], self.RMainAtomNumbers[i,j], self.Ncore)
            #Is the torsion part of the new R group?
            if (atomi >= self.Ncore or atomj >= self.Ncore or atomk >= self.Ncore or atoml >= self.Ncore):
                #is the torsion both core AND R group? (and when combined with previous)
                if atomi < self.Ncore or atomj < self.Ncore or atomk < self.Ncore or atoml < self.Ncore:
                    customTorsionForce.addTorsion(atomi, atomj, atomk, atoml, (period, phase, k)) 
                else: #Standard torsion in R group
                    self.mainTorsionForce.addTorsion(atomi, atomj, atomk, atoml, period, phase, k)
        return customTorsionForce

    def loadpdb(self, basefilename, NBM=NBM, NBCO=NBCO, constraints=constraints, rigidWater=rigidWater, eET=eET, boxbuffer=0.2):
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
        system = self.ff.createSystem(
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

    def _addRParticles(self,  Rsystem):
        #Detect differences between core and R group
        NR = Rsystem.getNumParticles()
        Nmain = self.mainSystem.getNumParticles()
        new_atoms = xrange(self.Ncore,NR)
        #Attach R group to main system
        for new_atom in new_atoms:
            self.mainSystem.addParticle(Rsystem.getParticleMass(new_atom))
            #Map this particle to the new atom number
        ##Atoms, bonds, angles, torsions, dihedrals
        return range(Nmain, self.mainSystem.getNumParticles())
    
    def _addRSystemToMain(self, i, j):
        Rsystem = self.Rsystems[i,j]
        RMainAtomNumber = self.RMainAtomNumbers[i,j]
        for constraintIndex in range(Rsystem.getNumConstraints()):
            atomi, atomj, r0 = Rsystem.getConstraintParameters(constraintIndex)
            atomi, atomj = mapAtomsToMain([atomi, atomj], RMainAtomNumber, self.Ncore)
            if atomi >= self.Ncore or atomj >= self.Ncore:
                self.mainSystem.addConstraint(atomi, atomj, r0)
        for forceIndex in xrange(Rsystem.getNumForces()):
            referenceForce = Rsystem.getForce(forceIndex)
            if isinstance(referenceForce, mm.HarmonicBondForce):
                nRBonds = referenceForce.getNumBonds()
                for bondIndex in xrange(nRBonds):
                    atomi, atomj, eqdist, k = referenceForce.getBondParameters(bondIndex)
                    #Map atoms to core system
                    atomi, atomj = mapAtomsToMain([atomi, atomj], RMainAtomNumber, self.Ncore)
                    if atomi >= self.Ncore or atomj >= self.Ncore:
                        self.mainBondForce.addBond(atomi, atomj, eqdist, k)
            elif isinstance(referenceForce, mm.HarmonicAngleForce):
                customAngleForce = self._addAngleForceWithCustom(referenceForce, i, j)
                #customAngleForce.setForceGroup(i+1)
                customAngleForce.setForceGroup(self.calcGroup(i,j))
                self.mainSystem.addForce(customAngleForce)
            elif isinstance(referenceForce, mm.PeriodicTorsionForce):
                customTorsionForce = self._addTorsionForceWithCustom(referenceForce, i, j)
                #customTorsionForce.setForceGroup(i+1)
                customTorsionForce.setForceGroup(self.calcGroup(i,j))
                self.mainSystem.addForce(customTorsionForce)
            elif isinstance(referenceForce, mm.NonbondedForce):
                #Add the particle to the main nonbonded force. Custom will come after
                nParticles = referenceForce.getNumParticles()
                for atomi in xrange(nParticles):
                    q, sig, epsi = referenceForce.getParticleParameters(atomi)
                    (atomi,) = mapAtomsToMain([atomi], RMainAtomNumber, self.Ncore) #If you dont trap the returned atomi, it returns a list of atomi, e.g. [0], which is > int for some reason?
                    if atomi >= self.Ncore:
                        self.mainNonbondedForce.addParticle(q, sig, epsi)
                nException = referenceForce.getNumExceptions()
                for exceptionIndex in xrange(nException):
                    atomi, atomj, chargeProd, sig, epsi = referenceForce.getExceptionParameters(exceptionIndex)
                    atomi, atomj = mapAtomsToMain([atomi, atomj], RMainAtomNumber, self.Ncore)
                    if atomi >= self.Ncore or atomj >= self.Ncore:
                        self.mainNonbondedForce.addException(atomi, atomj, chargeProd, sig, epsi)
        return

    def _buildRGroups(self, defaultPath='pdbfiles/i%d/j%dc'):
        #allocate the housing objects
        self.Rsystems=np.empty([self.Ni,self.Nj],dtype=np.object)
        self.Rcoords=np.empty([self.Ni,self.Nj],dtype=np.object)
        self.RMainAtomNumbers = np.empty([self.Ni,self.Nj],dtype=np.object)
        #Import the Rgroups
        for i in xrange(self.Ni):
            for j in xrange(self.Nj):
                Rsystem, Rcoord = self.loadpdb(defaultPath%(i+1,j+1))
                self.Rsystems[i,j], self.Rcoords[i,j] = Rsystem, Rcoord
                #Add the Rgroup atoms to the main system
                self.RMainAtomNumbers[i,j] = self._addRParticles(Rsystem)
                RPos = Rcoord.getPositions(asNumpy=True)
                #align the new group to the core structure (to which the main was already alligned)
                alignedPositions = alignCoords(self.corePositions, RPos)
                #Append the newly aligned R-group structure to the main structure
                self.mainPositions = appendPositions(self.mainPositions,alignedPositions[self.Ncore:,:])
                #set PBC's, probably not needed here
                maxPBC(self.mainSystem, Rsystem)
                #Add topologies together, only needed to add solvent to the system
                addToMainTopology(self.mainTopology, Rcoord.getTopology(), self.Ncore, i, j)
                self.mainTopology.setPeriodicBoxVectors(self.mainSystem.getDefaultPeriodicBoxVectors())
                # === Add forces (exclusions later, for now, just get in all the defaults) ===
                self._addRSystemToMain(i,j)
        return

    def _assignBasisForceDefaults(self, basisForce, interactionGroups=None, longRange=False):
        #Assign the default values to the basisForce object based on mainSystem
        #If interactionGroups is a size 2 iterable object of sets, then interaction groups are added
        mainNonbondedCutoff = self.mainNonbondedForce.getCutoffDistance()
        mainNonbondedDielectric = self.mainNonbondedForce.getReactionFieldDielectric()
        #Set parameters NB method, cutoff, other long range terms
        basisForce.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
        basisForce.setCutoffDistance(mainNonbondedCutoff.value_in_unit(unit.nanometer))
        basisForce.setUseLongRangeCorrection(longRange)
        #Set global parameters
        for parameterIndex in xrange(basisForce.getNumGlobalParameters()):
            if "dielectric" == basisForce.getGlobalParameterName(parameterIndex):
                basisForce.setGlobalParameterDefaultValue(parameterIndex, mainNonbondedDielectric)
            if "rcut" == basisForce.getGlobalParameterName(parameterIndex):
                basisForce.setGlobalParameterDefaultValue(parameterIndex, mainNonbondedCutoff.value_in_unit(unit.nanometer))
        if interactionGroups is not None:
            basisForce.addInteractionGroup(interactionGroups[0], interactionGroups[1])
        self.mainSystem.addForce(basisForce)
        return basisForce
 
    def _buildNonbonded(self):
        '''
        Constuct all of the nonbonded objects
        List of interactions to account for:
        Solvent <-> Solvent   : mainNonbondedForce
        Solvent <-> R(i,j)
        Solvent <-> Core
        R(i,*)  <-> R(i,*)    : 0, does not interact
        R(i,*)  <-> R(k!=i,*)
        R(i,j)  <-> Core
        Core    <-> Solvent
        Core    <-> Core
    
        The Core's sigma/epsilon will be handled with main NB force and the R-group interactions
        A separate set of charge interactions will handle the core electrostatics
        '''
        coreAtomNumbers = range(self.Ncore)
        #Build the i to common interactions
        nbForceList = [] #will loop through this afterwords to add particles
        coreForceList = [] #Special group of forces handling the core
        #forceGroupI = 1 #starting force group for ith->solvent
        #forceGroupII = self.Ni+1 #Starting force group for i->i interaction
        for i in xrange(self.Ni):
            #i-> solvent force groups are all in the initial force groups
            for j in xrange(self.Nj):
                forceGroupI = self.calcGroup(i,j)
                #forceGroupII = (self.Ni+1)+(i*self.Ni - (i**2+i)/2) #starting Force group for i->i interactions
                #ij -> solvent interactions
                #Fetch energy
                basisForce = basisEnergy(i,j)
                #Build Defaults and attach basisForce, append force to list to loop through later
                basisForce = self._assignBasisForceDefaults(basisForce, interactionGroups=(set(self.RMainAtomNumbers[i,j]), set(self.solventNumbers)))
                #Add (i,j) R-group to (i,j) R-group interactions
                basisForce.addInteractionGroup(set(self.RMainAtomNumbers[i,j]), set(self.RMainAtomNumbers[i,j]))
                #Add sig/epsilon interaction of (i,j) R-group to (i,j) corei (Core charge will be 0)
                basisForce.addInteractionGroup(set(self.RMainAtomNumbers[i,j]), set(coreAtomNumbers))
                #Add (i,j) Core -> Solvent Electrostatic interactions
                coreForce = basisEnergy(i,j, LJ=False, Electro=True) 
                coreForce = self._assignBasisForceDefaults(coreForce, interactionGroups=(set(coreAtomNumbers), set(self.solventNumbers)))
                #(i,j )Core -> (i,j) R-group Electrostatics
                coreForce.addInteractionGroup(set(coreAtomNumbers),set(self.RMainAtomNumbers[i,j]))
                #(i,j) Core -> (i,j) Core Electrostatics
                coreForce.addInteractionGroup(set(coreAtomNumbers),set(coreAtomNumbers))
                basisForce.setForceGroup(forceGroupI)
                coreForce.setForceGroup(forceGroupI)
                nbForceList.append(basisForce)
                coreForceList.append({'i':i, 'j':j, 'i2':None, 'j2':None, 'force':coreForce})
                #Loop through all j on NOT the same i
                for i2 in xrange(i+1,self.Ni): #no need to loop backwards, otherwise will double up on force
                    for j2 in xrange(self.Nj): #All j affected, just not same i
                        #Check for how the cross terms are handled.
                        forceGroupII = self.calcGroup(i,j,i2,j2)
                        useCrossCap = False
                        for step in self.protocol['crossBasisCoupling']:
                            if 'C' in step:
                                useCrossCap = True
                        if useCrossCap:
                            basisForce2 = basisEnergy(i, j, i2=i2, j2=j2)
                        else:
                            basisForce2 = basisUncapLinearEnergy(i,j, i2=i2, j2=j2)
                        basisForce2 = self._assignBasisForceDefaults(basisForce2, interactionGroups=(set(self.RMainAtomNumbers[i,j]), set(self.RMainAtomNumbers[i2,j2])), longRange=False)
                        #(i,j) Core -> R(i2,j2) sigma/epsilon REDUNDANT, NOT NEEDED since same particles as (i,j) R-core
                        #(i,j) Core to (i2,j2) R-group Electrostatics
                        if useCrossCap:
                            coreForce2 = basisEnergy(i, j, i2=i2, j2=j2, LJ=False, Electro=True)
                        else:
                            coreForce2 = basisUncapLinearEnergy(i,j, i2=i2, j2=j2, LJ=False, Electro=True)
                        coreForce2 = self._assignBasisForceDefaults(coreForce2, interactionGroups=(set(coreAtomNumbers), set(self.RMainAtomNumbers[i2,j2])))
                        #Core -> Core(i2,j2) Electrostatics: Disabled since I assume that the individual pull from the R-groups and the self (ij) (ij) sums linearly to make a net result
                        basisForce2.setForceGroup(forceGroupII)
                        coreForce2.setForceGroup(forceGroupII)
                        nbForceList.append(basisForce2)
                        coreForceList.append({'i':i, 'j':j, 'i2':i2, 'j2':j2, 'force':coreForce2})
                    #forceGroupII += 1
            #forceGroupI += 1
        #Bring over Nonbonded parameters to the forces
        nParticles = self.mainSystem.getNumParticles()
        #Get all alchemical atoms
        alchemicalParticles = []
        for atomList in self.RMainAtomNumbers.flatten():
            alchemicalParticles.extend(atomList)
        particleNB = [] #Empty list to store nonbonded parameters data is not accidentally deleted before it is added to the customNB forces
        #Fetch default NB parameters
        for particleIndex in xrange(nParticles):
            #Order: q, sigma, epsilon
            q, sigma, epsilon = self.mainNonbondedForce.getParticleParameters(particleIndex)
            #Special case for the sigma = 0 on hydroxyl and other hydrogens poorly defined in the FF. causes issues for H<->H interaction where .5*sigma1*sigma2 = 0 and throws an error
            if sigma.value_in_unit(unit.nanometer) == 0:
                if epsilon.value_in_unit(unit.kilojoules_per_mole) != 0:
                    print("Warrning! Particle {0:d} has sigma = 0 but epsilon != 0. This will cause a force field error!")
                    raise
                sigma = 1*unit.nanometer
            particleNB.append((q, sigma, epsilon))
            #Remove from the main NB force the parts that are handled with custom forces
            if particleIndex in alchemicalParticles:
                #Remove R-group aprticles
                self.mainNonbondedForce.setParticleParameters(particleIndex, 0*unit.elementary_charge, sigma, 0*unit.kilojoules_per_mole)
            elif particleIndex in coreAtomNumbers:
                #Remove Core charge, leave sigma/epsilon for those interactions with solvent
                self.mainNonbondedForce.setParticleParameters(particleIndex, 0*unit.elementary_charge, sigma, epsilon)
        alchemicalExceptions = [] #Exception indicies involving alchemical atoms, will speed up loop later
        for exceptionIndex in xrange(self.mainNonbondedForce.getNumExceptions()):
            atomi, atomj, chargeProd, sigma, epsilon = self.mainNonbondedForce.getExceptionParameters(exceptionIndex)
            if atomi in alchemicalParticles or atomj in alchemicalParticles or atomi in coreAtomNumbers or atomj in coreAtomNumbers:
                alchemicalExceptions.append(exceptionIndex)
        #Assign NB parameters to the custom NB forces
        for (i,(Rforce,coreForce)) in enumerate(zip(nbForceList,coreForceList)):
            stdout.flush()
            stdout.write('\rWorking on force {0:d}/{1:d}'.format(i+1,len(nbForceList)))
            for particleIndex in xrange(nParticles):
               q, sig, epsi = particleNB[particleIndex]
               if particleIndex in coreAtomNumbers:
                   #Set the q that will pass to the Rforce
                   q = 0 *unit.elementary_charge
                   #Get the charge for the i,j core
                   corei = coreForce['i']
                   corej = coreForce['j']
                   qcore, sigmacore, epsiloncore  = getArbitraryForce(self.Rsystems[corei,corej], mm.NonbondedForce).getParticleParameters(particleIndex)
               else:
                   qcore = q
               #Assign parameters. Each particle must be in every customNonbondedForce, but the InteractionGroups defined eariler will handle which particles interact
               coreForce['force'].addParticle((qcore.value_in_unit(unit.elementary_charge),))
               Rforce.addParticle((q.value_in_unit(unit.elementary_charge), sig.value_in_unit(unit.nanometer), epsi.value_in_unit(unit.kilojoules_per_mole)))
            for exceptionIndex in alchemicalExceptions:
                atomi, atomj, chargeProd, sigma, epsilon = self.mainNonbondedForce.getExceptionParameters(exceptionIndex)
                Rforce.addExclusion(atomi, atomj)
                coreForce['force'].addExclusion(atomi, atomj)
        stdout.write('\n')
        return
    
    def _buildBiasForces(self):
        #Build the conditional multi-dim flat-bottom harmonic bias in the lambda dimentions
        #Get the normal bias force
        biasU = biasPotential(self.Ni,self.Nj)
        #Tack the bias potential as the last force group
        biasUGroup = self.calcGroup(self.Ni-1, self.Nj-1, self.Ni-1, self.Nj-1) + 1
        biasU.setForceGroup(biasUGroup)
        #Add the forces ONLY to the first particle so that we dont get the energy added to EVERY particle
        biasU.addParticle(0)
        self.mainSystem.addForce(biasU)
        #Create the holders for the free energy tabulated potentials
        freeEnergyBiasUTabs = np.empty([self.Ni, self.Nj],dtype=object)
        freeEnergyBiasForceTabs = np.empty([self.Ni, self.Nj],dtype=object)
        biasFEForces = []
        biasFEU, freeEnergyBiasUTabs[:,:] = initFreeEnergyUBias(self.Ni, self.Nj)
        #Append to the back of the force groups.
        biasFEUGroup = self.calcGroup(self.Ni-1, self.Nj-1, self.Ni-1, self.Nj-1) + 2
        biasFEU.setForceGroup(biasFEUGroup)
        self.mainSystem.addForce(biasFEU)
        biasFEForces.append(biasFEU)
        for i in xrange(self.Ni):
            for j in xrange(self.Nj):
                biasForce = biasDerivative(i, j, self.Ni, self.Nj)
                biasForceGroup = self.calcGroup(i,j)
                biasForce.setForceGroup(biasForceGroup)
                biasForce.addParticle(0)
                self.mainSystem.addForce(biasForce)
                #Handle the FE Force
                biasFEForce, freeEnergyBiasForceTabs[i,j] = initFreeEnergyForceBias(i,j)
                biasFEForce.setForceGroup(biasForceGroup)
                self.mainSystem.addForce(biasFEForce)
        #Store the tabulated FE biases and derivatives for later use (activley updating later)
        self.freeEnergyTabs = {}
        self.freeEnergyTabs['potential'] = freeEnergyBiasUTabs
        self.freeEnergyTabs['force'] = freeEnergyBiasForceTabs
        return

    def _addSolvent(self):
        #Adjust the residue in the main topology to match the combined name so the modeler does not throw an error
        for res in self.mainTopology.residues():
            res.name = 'COC'
        #Add water with the modeler
        modeller = app.Modeller(self.mainTopology, self.mainPositions)
        modeller.addSolvent(self.ff, padding=1.2*unit.nanometer)
        #Deelete non solvent residues. This includes the neutralizing ions which will be added since we have not handled electrostatics yet
        modeller.delete([res for res in modeller.topology.residues() if res.name == 'COC' or res.name == 'CL' or res.name=='NA'])
        copyTopologyBtoA(self.mainTopology, modeller.topology)
        #Get Positions
        modellerCoords = listCoordsToNumpy(modeller.getPositions())
        #Combine positions
        self.mainPositions = appendPositions(self.mainPositions, modellerCoords)
        #Combine solvent with system, this can probably can be made into function form at some point
        addSystem = self.ff.createSystem(
         modeller.topology,
         nonbondedMethod=NBM,
         nonbondedCutoff=NBCO,
         constraints=constraints,
         rigidWater=rigidWater,
         ewaldErrorTolerance=eET)
        Noriginal = self.mainSystem.getNumParticles()
        Nnew = addSystem.getNumParticles()
        maxPBC(self.mainSystem, addSystem, percentNudge=1.0)
        self.mainTopology.setPeriodicBoxVectors(self.mainSystem.getDefaultPeriodicBoxVectors())
        self.solventNumbers = range(Noriginal,Nnew+Noriginal)
        for atomIndex in xrange(Nnew):
            self.mainSystem.addParticle(addSystem.getParticleMass(atomIndex))
        for constraintIndex in range(addSystem.getNumConstraints()):
            atomi, atomj, r0 = addSystem.getConstraintParameters(constraintIndex)
            self.mainSystem.addConstraint(self.solventNumbers[atomi], self.solventNumbers[atomj], r0)
        for forceIndex in xrange(addSystem.getNumForces()):
            referenceForce = addSystem.getForce(forceIndex)
            if isinstance(referenceForce, mm.HarmonicBondForce):
                nRBonds = referenceForce.getNumBonds()
                for bondIndex in xrange(nRBonds):
                    atomi, atomj, eqdist, k = referenceForce.getBondParameters(bondIndex)
                    self.mainBondForce.addBond(self.solventNumbers[atomi], self.solventNumbers[atomj], eqdist, k)
            elif isinstance(referenceForce, mm.HarmonicAngleForce):
                nAngle = referenceForce.getNumAngles()
                for angleIndex in xrange(nAngle):
                    atomi, atomj, atomk, angle, k = referenceForce.getAngleParameters(angleIndex)
                    self.mainAngleForce.addAngle(self.solventNumbers[atomi], self.solventNumbers[atomj], self.solventNumbers[atomk], angle, k)
            elif isinstance(referenceForce, mm.PeriodicTorsionForce):
                nTorsion = referenceForce.getNumTorsions()
                for torsionIndex in xrange(nTorsion):
                    atomi, atomj, atomk, atoml, period, phase, k = referenceForce.getTorsionParameters(torsionIndex)
                    self.mainTorsionForce.addTorsion(self.solventNumbers[atomi], self.solventNumbers[atomj], self.solventNumbers[atomk], self.solventNumbers[atoml], period, phase, k)
            elif isinstance(referenceForce, mm.NonbondedForce):
                #Add the particle to the main nonbonded force. Custom will come after
                nParticles = referenceForce.getNumParticles()
                for atomi in xrange(nParticles):
                    q, sig, epsi = referenceForce.getParticleParameters(atomi)
                    self.mainNonbondedForce.addParticle(q, sig, epsi)
                nException = referenceForce.getNumExceptions()
                for exceptionIndex in xrange(nException):
                    atomi, atomj, chargeProd, sig, epsi = referenceForce.getExceptionParameters(exceptionIndex)
                    self.mainNonbondedForce.addException(self.solventNumbers[atomi], self.solventNumbers[atomj], chargeProd, sig, epsi)
        return


    def updateFreeEnergyBias(self, freeEnergies, knotMultiplier = 1):
        '''
        NOTE: UNTESTED
 
        Update the free energy biases based on the freeEnergies fed in to the function
        This is a slow process that requires collapsing the Context and rebuilding it from scratch.

        Splines are first constructed from scipy.interpolate.UnivariateSpline then a series of spline values are generated from 
        the knots, followed by derivative of the spline. Both these outputs are fed into the OpenMM splines that make up the 
        Free Energy Bias
        
        freeEnergies : ndarray of floats of shape [Ni,Nj,S] where S is # of spline points
            The free energies from MBAR in units of kJ/mol for each {i,j} pair assuming lam_{k,l} = 0 for all (i != k and j != l)
            Points in the array should be uniformly measured along [0,1] domain for each i,j and is assumed as much in this function.

        knotMultiplier : float >= 1
            Multiplier on the number of spline points, S, to use internally. 
            May improve internal representation of derivative mostly for instability (not known to happen)
            Final count is closet integer.

        OpenMM interpolates between all points, does not use any smoothing, so UnivariateSpline(s=0), may want to use that.
        '''
        freeEnergyUTabs = self.freeEnergyTabs['potential']
        freeEnergyFTabs = self.freeEnergyTabs['force']
        #Determine spline sizes
        nKnots = freeEnergies.shape[-1]
        x = np.linspace(0,1,nKnots)
        xMM = np.linspace(0,1, int(nKnots * knotMultiplier))
        for i in xrange(self.Ni):
            for j in xrange(self.Nj):
                #Build Spline
                spline = UnivariateSpline(x, freeEnergy[i,j,:], s=0)
                y = spline(xMM)*self.Kt
                dy = spline(xMM,1)*self.kT
                #Update free energy tabulated functions
                freeEnergyUTabs.setFunctionParameters(y, 0, 1)
                freeEnergyFTabs.setFunctionParameters(dy, 0, 1)
        if self.context is not None:
            #Extract all the parameters/values needed to rebuild the context.
            state = self.context.getState(getPositions = True, getVelocities = True, getForces = True, getEnergy = True, getParameters = True, enforcePeriodicBox=False)
            #Rebuild context
            self.context.reinitialize()
            #Restore state now that the context has been rebuilt
            self.context.setState(state)
        return

    def getLambda(self, flat=False):
        lamVector = np.zeros([self.Ni,self.Nj])
        for i in xrange(self.Ni):
            for j in xrange(self.Nj):
                lamVector[i,j] = self.context.getParameter('lam{0:s}x{1:s}B'.format(str(i),str(j)))
        if flat:
            lamVector = lamVector.flatten()
        return lamVector
    
    def groupFlag(self, listin):
        #Take a list of force group IDs (ints from [0,31]) and cast it to the bitwise flag for unmodified openmm
        bits = '0'*32
        if type(listin) is int:
            listin = [listin]
        bits = list(bits)
        for flag in listin:
            bits[-flag-1] = '1'
        return int(''.join(bits), 2)
 
    def calcGroup(self, *args):
        #Given an i and j, then optionally an i2 and j2, calculate the force group with the expanded groups needed.
        if len(args) == 2:
            i,j = args
            return 1 + i*self.Nj + j
        elif len(args) == 4:
            i,j,i2,j2 = args
            maxMainGroup = self.Ni*self.Nj + 1
            #Because the i2 variable is a decreasing function of i, flattening the groups requires knowledge of the previous sizes
            #\sum{Ni-(x+1)}_{x=0}^{i-1}
            sizeOfPreviousISets = (i*(2*self.Ni-i-1)/2 * self.Nj**2)
            sizeOfCurrentISet = self.Ni-(i+1)
            return maxMainGroup + sizeOfPreviousISets + (j * sizeOfCurrentISet * self.Nj) + ((i2-(i+1)) * self.Nj) + j2
        else:
            print("Only i,j or  i,j,i2,j2 arguments accepted!")
            raise(Exception)

    def buildThermostat(self):
        self.thermostat = mm.AndersenThermostat(self.temperature, 1.0/unit.picosecond)
        self.mainSystem.addForce(self.thermostat)

    def buildIntegrator(self):
        if self.context is not None:
            print "Cannot make new integrator with existing context!"
        else:
            thermostat = True
            if self.equilibrate:
                if self.pressure is not None and False:
                    self.integrator = VelocityVerletIntegrator(self.timestep)
                else:
                    self.integrator = mm.LangevinIntegrator(self.temperature, 1.0/unit.picosecond, self.timestep)
                    thermostat=False
            else:
                if self.pressure is not None:
                    print("The Hybrid LDMC Integrator is coded to to NVT simulations, not NPT!")
                    raise(Exception)
                self.integratorEngine = HybridLDMCIntegratorEngine(self, self.timestep, stepsPerMCInner = self.stepsPerMCInner, stepsPerMCOuter = self.stepsPerMCOuter, lamMasses = self.protocol['lamMasses'], cartesianOnly = self.protocol['cartesianOnly'])
                self.integrator = self.integratorEngine.integrator
                #hybrid LDMC runs NVE MD with NVT sampling at MC step
                thermostat = False
            #--
            #self.integrator = mm.VerletIntegrator(self.timestep)
            #--
            #self.integrator = VelocityVerletIntegrator(self.timestep)
            #--
            #self.integrator  = VelocityVerletNVT(self.timestep)
            if thermostat:
                self.buildThermostat()
            else:
                self.thermostat = None
        return

    def buildBarostat(self):
        if self.context is not None:
            print "Cannot make new barostat with existing context!"
        else:
            self.barostat = mm.MonteCarloBarostat(self.pressure, self.temperature, 100)
            self.mainSystem.addForce(self.barostat)
        return

    def buildPlatform(self):
        if self.context is not None:
            print "Cannot make new Platform with existing context!"
        else:
            platformName = self.protocol['platform']
            deviceIndex = self.protocol['devIndex']
            self.platform = mm.Platform.getPlatformByName(platformName)
            try:
                self.platform.setPropertyDefaultValue(platformName+'DeviceIndex', str(deviceIndex))
            except: pass
        return

    def _buildContext(self,provideContext=False, force=False):
        if self.context is not None:
            print "Context already made! Use the \"force=True\" keyword to rebuild it!"
            return
        if self.integrator is None:
            if self.verbose: print("Building Integrator") 
            self.buildIntegrator()
        if self.barostat is None and self.pressure is not None:
            if self.verbose: print("Building Barostat") 
            self.buildBarostat()
        if self.platform is None:
            self.buildPlatform()
        if self.verbose: print("Building Context")
        if self.protocol['skipContextUnits']:
            #Overload the self.context.setParameter function to skip the try stripUnits arg at the start
            from simtk.openmm import _openmm as _mm
            class skipUnitsInParameterContext(mm.Context):
                def setParameter(self, *args):
                    return _mm.Context_setParameter(self, *args)
            self.context = skipUnitsInParameterContext(self.mainSystem, self.integrator, self.platform)
        else:
            self.context = mm.Context(self.mainSystem, self.integrator, self.platform)
        #Ensure initial values are set right
        self.assignLambda(self.getLambda())
        try: 
            self.integratorEnginge.initilizeIntegrator(self.currentLambda)
        except:
            pass
        #Assign positions
        #Note: BOX MUST BE SET FIRST!!!
        self.context.setPeriodicBoxVectors(self.boxVectors[0,:], self.boxVectors[1,:], self.boxVectors[2,:])
        self.context.setPositions(self.mainPositions)
        #DEBUG
        #self.context.setPositions(self.context.getState(getPositions=True,enforcePeriodicBox=True).getPositions())
        if not self.coordsLoaded:
            if self.verbose: print("Minimizing Context")
            #mm.LocalEnergyMinimizer.minimize(self.context, 1.0 * unit.kilojoules_per_mole / unit.nanometers, 0)
            self.mainPositions = self.context.getState(getPositions=True).getPositions(asNumpy=True)
        else:
            #Reinitilize to lambda
            try:
                self.assignLambda(self.state)
            except:
                print("WARNING! No alchemical state in loaded file! Assuming fully coupled!")
                self.state = np.ones([self.Ni,self.Nj])
                self.assignLambda(self.state)
        return

    def _castLambda(self, lam, method, protocol):
        '''
        Helper function which takes an several methods for determining the basis function lambdas and condenses it into one area, reduces code replicaiton
        '''
        lams = {}
        if method is 'directDict':
            lams.update(lam)
        elif method is 'structuredNumpyArray':
            lams.update({key:float(value) for (key,value) in zip(lam.dtype.names,lam)})
        elif method is 'bondOnly':
            lams['B'] = lam
            lams.update(basisMap(lams['B'],protocol))
        return lams
    
    def assignLambda(self, lamVector):
        '''
        Set the switch values inside the context. Done by either passing in a vector of lambda values, or dictionaries housing each of the switch values
        '''
        if self.context is None:
            print "Cannot assign Lambda when no context exists!"
            raise
        #Take the lamVector and assign the global parameters
        #lamP is not implemented for lamVectors which include the basis functions
        if isinstance(lamVector, list):
            lamVector = np.array(lamVector)
        #Check if explicit values for the basis have been passed in as dictionaries or structured numpy arrays
        if isinstance(lamVector.flat[0],dict):
            method = 'directDict'
        elif (lamVector.dtype.fields is not None):
            method = 'structuredNumpyArray'
        else:
            method = 'bondOnly'
        lamVector = lamVector.reshape((self.Ni,self.Nj))
        #Determine which basis need computed for interactions
        standardBasis = []
        for stage in self.protocol['standardBasisCoupling']:
            standardBasis.extend([basis for basis in stage])
        crossBasis = []
        for stage in self.protocol['crossBasisCoupling']:
            crossBasis.extend([basis for basis in stage])
        for i in xrange(self.Ni):
            for j in xrange(self.Nj):
                #lam values are cast to float since openmm can't convert numpy.float32 dtypes
                lams = self._castLambda(lamVector[i,j], method, self.protocol['standardBasisCoupling'])
                self.context.setParameter('lam{0:s}x{1:s}B'.format(str(i),str(j)), float(lams['B']))
                for basis in standardBasis:
                    self.context.setParameter('lam{0:s}x{1:s}{2:s}'.format(str(i),str(j),basis), float(lams[basis]))
                for i2 in xrange(i+1,self.Ni): #no need to loop backwards, otherwise will double up on force
                    for j2 in xrange(self.Nj): #All j affected, just not same i
                        lams1 = self._castLambda(lamVector[i,j], method, self.protocol['crossBasisCoupling'])
                        lams2 = self._castLambda(lamVector[i2,j2], method, self.protocol['crossBasisCoupling'])
                        for basis in crossBasis:
                            self.context.setParameter('lam{0:s}x{1:s}x{2:s}x{3:s}{4:s}'.format(str(i),str(j),str(i2),str(j2),basis), float(lams1[basis]*lams2[basis]))
        try: 
            self.integratorEnginge.initilizeTheta(self.currentLambda)
        except:
            pass
        return

    def getPotential(self, groups=None):
        #Helper function to shorthand the context.getState(...).getPotentialEnergy() command
        if groups is None:
            groups = -1
        return self.context.getState(enforcePeriodicBox=False,getEnergy=True,groups=groups).getPotentialEnergy()

    def computeBasisEnergy(self):
        '''
        Compute all the basis function energies given a context, sites, and molecules per site.
    
        Energy list:
        Solvent <-> R(i,j) including core: Electro, cap, R, A, Bonded
        R(i,j) <-> R(i2,j2) including core: Electro, cap, R, A
        ''' 
        if self.context is None:
            print "Cannot comptue energy when no context exists!"
            raise
        #Initilize energies
        # E, C, R, A, B
        standardParameter = 'lam{i:s}x{j:s}{b:s}'
        crossParameter = 'lam{i:s}x{j:s}x{i2:s}x{j2:s}{b:s}'
        #Determine the number of basis functions per stage
        rijSolvBasis = np.zeros([self.Ni,self.Nj,self.basisManipulator.standardNumBasis + 1]) * unit.kilojoules_per_mole #Add 1 since the bonded terms are on here
        rijRij2Basis = np.zeros([self.Ni,self.Nj,self.Ni,self.Nj,self.basisManipulator.crossNumBasis]) * unit.kilojoules_per_mole
        #Create blank structured numpy array (behaves similar to an array of dictionary objects)
        blankLamVector = np.zeros([self.Ni, self.Nj], dtype={'names':['E','C','R','A','B',], 'formats':['f','f','f','f','f']})
        #Get current total potential and state
        currentLambda = self.getLambda()
        currentPotential = self.getPotential()
        currentDerivMode = self.context.getParameter('derivMode')
        currentSingleSwitchMode = self.context.getParameter('singleSwitchMode') 
        #Set the mode to compute basis
        self.context.setParameter('derivMode', 0)
        self.context.setParameter('singleSwitchMode', 1) 
        #Get the Harmonic Bias Potential before setting states to 0
        groups = self.calcGroup(self.Ni-1, self.Nj-1, self.Ni-1, self.Nj-1) + 1
        harmonicBias = self.getPotential(groups=groups)
        #Get the Free Energy Bias
        groups = self.calcGroup(self.Ni-1, self.Nj-1, self.Ni-1, self.Nj-1) + 2
        freeEnergyBias = self.getPotential(groups=groups)
        #Start at fully decoupled potential
        self.assignLambda(blankLamVector)
        #import pdb
        #pdb.set_trace()
        #Cycle through the lambda
        forceGroupI = 1 #starting force group for ith->solvent
        forceGroupII = self.Ni+1 #Starting force group for i->i interaction
        for i in xrange(self.Ni):
            for j in xrange(self.Nj):
                forceGroupII = (self.Ni+1)+(i*self.Ni - (i**2+i)/2) #starting Force group for i->i interactions
                #rijIndex = i*Ni + j
                #groups = self.groupFlag(forceGroupI)
                #groups = forceGroupI
                groups = self.calcGroup(i,j)
                #Compute the bonded terms
                self.context.setParameter(standardParameter.format(i=str(i), j=str(j), b='B'), 1)
                rijSolvBasis[i,j,-1] = self.getPotential(groups=groups)
                self.context.setParameter(standardParameter.format(i=str(i), j=str(j), b='B'), 0)
                for uniqueSetCount in xrange(self.basisManipulator.standardNumBasis):
                    #Grab the unique basis functions
                    basisSet = self.basisManipulator._flatStandardUniqueBasis[uniqueSetCount]
                    #Set the switch, we only need to grab one of the lambdas since they are the same function
                    for basis in basisSet:
                        self.context.setParameter(standardParameter.format(i=str(i), j=str(j), b=basis), 1)
                    rijSolvBasis[i,j,uniqueSetCount] = self.getPotential(groups=groups)
                    for basis in basisSet:
                        self.context.setParameter(standardParameter.format(i=str(i), j=str(j), b=basis), 0)
                #Loop through i2/j2 interactions
                for i2 in xrange(i+1,self.Ni): #no need to loop backwards, otherwise will double up on energy calculations
                    for j2 in xrange(self.Nj): #All j affected, just not same i
                        #groups = self.groupFlag(forceGroupII)
                        #groups = forceGroupII
                        groups = self.calcGroup(i,j,i2,j2)
                        for uniqueSetCount2 in xrange(self.basisManipulator.crossNumBasis):
                            basisSet2 = self.basisManipulator._flatCrossUniqueBasis[uniqueSetCount2]
                            for basis in basisSet2:
                                self.context.setParameter(crossParameter.format(i=str(i), j=str(j), i2=str(i2), j2=str(j2), b=basis), 1)
                            rijRij2Basis[i,j,i2,j2,uniqueSetCount2] = self.getPotential(groups=groups)
                            for basis in basisSet2:
                                self.context.setParameter(crossParameter.format(i=str(i), j=str(j), i2=str(i2), j2=str(j2), b=basis), 0)
                    #forceGroupII += 1
            #forceGroupI += 1
        #Lastly, unaffected energies
        #groups = self.groupFlag(0)
        groups = 0
        unaffectedPotential = self.getPotential(groups=groups)
        returns = {}
        returns['standardBasis'] = rijSolvBasis
        returns['crossBasis'] = rijRij2Basis
        returns['harmonicBias'] = harmonicBias
        returns['freeEnergyBias'] = freeEnergyBias
        returns['unaffectedPotential'] = unaffectedPotential
        returns['totalPotential'] = currentPotential

        #Ensure total energy = bais function energy. This is a debug sanity check
        basisPotential = self.basisManipulator.computeArbitraryAlchemicalEnergy(returns, currentLambda)['potential']
        tolerance = 10**-2 #kJ/mol
        err = np.abs((currentPotential - basisPotential)/unit.kilojoules_per_mole)
        if err >= tolerance:
            print "=== WARNING: POTENTIAL ENERGY FROM BASIS FUNCTIONS != OPENMM ENERGY WITHIN {0:f} ===".format(tolerance)
            print "Net Total Energy: {0:f}".format(currentPotential / unit.kilojoules_per_mole)
            print "Basis Total Energy: {0:f}".format(basisPotential / unit.kilojoules_per_mole)
            print "Delta Energy: {0:f}".format(err)
        #Reset the state
        self.assignLambda(currentLambda)
        self.context.setParameter('derivMode', currentDerivMode)
        self.context.setParameter('singleSwitchMode', currentSingleSwitchMode) 
        #Bundle energies
        return returns

    def _loadCoordinates(self, data, iteration=-1):
        '''
        Load and store the particle positions, box vectors, and chemical state (alchemical coordinates) from ncType. This is its own function as the resuming entierly from file and resuming from equilibration are different, but this process is the same

        data : str of filename OR netCDF Dataset, filename can be either a netCDF file or the saved npz file.
            Object where data is located. Given a string, loads the ncfile in readonly then processes, otherwise just processes nc file
        iteration : int of index position
            Index of the 'iteration' dimention in the ncType data to grab positions/vectors from, assumes last position (-1)
        '''
        if type(data) is str:
            try:
                data = netcdf.Dataset(data, 'r')
            except:
                data = np.load(data)
        if isinstance(data, netcdf.Dataset):
            self.mainPositions = data.variables['positions'][iteration,:,:] * unit.nanometer
            self.boxVectors = data.variables['box_vectors'][iteration,:,:] * unit.nanometer
            self.state = data.variables['state'][iteration,:] 
            #DEBUG
            #self.boxVectors = 1.5*ncType.variables['box_vectors'][iteration,:,:] * unit.nanometer
        else:
            self.mainPositions = data['pos'] * unit.nanometer
            self.boxVectors = data['box'] * unit.nanometer
        return

    def _resumeFromFile(self):
        '''
        Resume the simulation from the netcdf file
        '''
        ncfile = netcdf.Dataset(self.filename, 'r')
        self.iteration = ncfile.variables['positions'].shape[0] - 1
        self.iteration = ncfile.variables['positions'].shape[0] - 2
        #Load coordinate, box vectors, and alchemical coordinates
        self._loadCoordinates(ncfile, iteration=self.iteration)
        try:
            self.naccept = ncfile.groups['MCStats'].variables['naccept'][self.iteration]
            self.ntrials = ncfile.groups['MCStats'].variables['ntrials'][self.iteration]
        except:
            pass
        self.protocol = {}
        protocols = ncfile.groups['protocols']
        for protoName in protocols.variables.keys():
           # Get NetCDF variable.
            ncvar = protocols.variables[protoName]
            # Get option value.
            protoValue = ncvar[:]
            # Get python types
            protoType = getattr(ncvar, 'protoType')
            #Cast to correct type    
            try:
                if protoType == 'bool':
                    protoValue = bool(protoValue)
                elif protoType == 'int':
                    protoValue = int(protoValue)
                elif protoType == 'float':
                    protoValue = float(protoValue)
                elif protoType == 'list':
                    protoValue = protoValue.split('---')
                elif protoType == 'dict':
                    protoValue = dict([ (protoValue[i,0],protoValue[i,1]) for i in xrange(protoValue.shape[0]) ])
                elif protoType == 'str':
                    protoValue = str(protoValue)
                elif protoType == 'ndarray':
                    protoValue = protoValue.reshape([self.Ni,self.Nj])
                elif protoType == 'NoneType':
                    protoValue = None
                else:
                    print "wut m8?"
                    raise
            except: pdb.set_trace()
            # If Quantity, assign units.
            if hasattr(ncvar, 'units'):
                unitName = getattr(ncvar, 'units')
                if unitName[0] == '/': unitName = '1' + unitName
                protoUnit = eval(unitName, vars(unit))
                protoValue = unit.Quantity(protoValue, protoUnit)
            self.protocol[protoName] = protoValue
        ncfile.close()
        self.iteration += 1
        self.ncfile = netcdf.Dataset(self.filename, 'a')
        return
    
    def _initilizeNetCDF(self):
        '''
        Create the NetCDF file to store and operate on
 
        '''
        ncfile = netcdf.Dataset(self.filename, 'w', version='NETCDF4')

        # Create dimensions.
        ncfile.createDimension('iteration', 0) # unlimited number of iterations
        ncfile.createDimension('scalar', 1) # scalar dim
        ncfile.createDimension('particle', self.nParticles) # number of atoms in system
        ncfile.createDimension('spatial', 3) # number of spatial dimensions
        ncfile.createDimension('lambda', self.Ni*self.Nj) # number of alchemical/lambda Dimentions
        ncfile.createDimension('standard', self.basisManipulator.standardBasisCount) # number of standard basis functions
        ncfile.createDimension('cross', self.basisManipulator.crossBasisCount) # number of cross functions
        ncfile.createDimension('biases', 2) # Number of Biases, Harmonic and Free Energy
        ncfile.createDimension('iterableLen', 0) # arbitrary iterable length
        ncfile.createDimension('dict', 2) # Dictionary variable, stores key:value

        # Create variables.
        ncvar_positions = ncfile.createVariable('positions', 'f', ('iteration','particle','spatial'))
        ncvar_state     = ncfile.createVariable('state', 'f', ('iteration','lambda'))
        ncgrp_energies  = ncfile.createGroup('energies') #Energies group
        ncvar_energy    = ncgrp_energies.createVariable('energy', 'f', ('iteration'))
        ncvar_unaffected= ncgrp_energies.createVariable('unaffected', 'f', ('iteration'))
        ncvar_bias      = ncgrp_energies.createVariable('bias', 'f', ('iteration', 'biases'))
        ncvar_standard  = ncgrp_energies.createVariable('standardBasis', 'f', ('iteration', 'standard'))
        ncvar_cross     = ncgrp_energies.createVariable('crossBasis', 'f', ('iteration', 'cross'))
        ncvar_box_vectors = ncfile.createVariable('box_vectors', 'f', ('iteration','spatial','spatial'))
        ncvar_volumes  = ncfile.createVariable('volumes', 'f', ('iteration'))

        # Define units for variables.
        setattr(ncvar_positions, 'units', 'nm')
        setattr(ncvar_state,     'units', 'none')
        setattr(ncvar_energy,    'units', 'kT')
        setattr(ncvar_unaffected,'units', 'kT')
        setattr(ncvar_bias,      'units', 'kT')
        setattr(ncvar_standard,  'units', 'kT')
        setattr(ncvar_cross,     'units', 'kT')
        setattr(ncvar_box_vectors, 'units', 'nm')
        setattr(ncvar_volumes, 'units', 'nm**3')

        # Define long (human-readable) names for variables.
        setattr(ncvar_positions, "long_name", "positions[iteration][particle][spatial] is position of coordinate 'spatial' of atom 'particle' for iteration 'iteration'.")
        setattr(ncvar_state,     "long_name", "states[iteration][lambda] is the value of alchemical variable 'lambda' of iteration 'iteration'.")
        setattr(ncvar_energy,    "long_name", "energies[iteration] is the reduced (unitless) energy of the configuration from iteration 'iteration'.")
        setattr(ncvar_unaffected,"long_name", "unaffected[iteration] is the reduced (unitless) energy of the non-alchemical energy of configuration from iteration 'iteration'.")
        setattr(ncvar_standard,  "long_name", "standardBasis[iteration][standard] is the reduced (unitless) energy of number of alchemical/non-alchemical energy of 'standardBasis' from configuration of iteration 'iteration'.")
        setattr(ncvar_cross,     "long_name", "crossBasis[iteration][cross] is the reduced (unitless) energy of number of alchemical/alchemical energy of 'crossBasis' from configuration of iteration 'iteration'.")
        setattr(ncvar_bias,      "long_name", "bias[iteration][biases] is the reduced (unitless) energy of the [HarmonicBias, FreeEnergyBias] from the current state of iteration 'iteration'.")
        setattr(ncvar_box_vectors, "long_name", "box_vectors[iteration][i][j] is dimension j of box vector i from iteration 'iteration-1'.")
        setattr(ncvar_volumes, "long_name", "volume[iteration] is the box volume for replica 'replica' from iteration 'iteration-1'.")

        # Create timestamp variable.
        ncvar_timestamp = ncfile.createVariable('timestamp', str, ('iteration',))

        # Create group for performance statistics.
        ncgrp_timings = ncfile.createGroup('timings')
        ncvar_iteration_time = ncgrp_timings.createVariable('iteration', 'f', ('iteration',)) # total iteration time (seconds)
 
        #MC statistics, accept/reject rates of the 
        ncgrp_MC = ncfile.createGroup('MCStats')
        ncgrp_MC.createVariable('naccept', int, ('iteration',))
        ncgrp_MC.createVariable('ntrials', int, ('iteration',))

        #Create some helpful constants to pass along
        ncvar_Ni = ncfile.createVariable('Ni', int)
        ncvar_Ni[0] = self.Ni
        ncvar_Nj = ncfile.createVariable('Nj', int)
        ncvar_Nj[0] = self.Nj
 
        #Create group of constants
        ncgrp_proto = ncfile.createGroup('protocols')
        for protoName in self.protocol.keys():
            #Get protocol
            protocol = self.protocol[protoName]
            protoUnit = None
            if isinstance(protocol,  unit.Quantity):
                protoUnit = protocol.unit
                protocol  /= protoUnit
            protoType = type(protocol)
            #Handle Types
            if protoType is type(None): #Have to type(None) since <type 'NoneType'> != None
                ncvar = ncgrp_proto.createVariable(protoName, str)
                ncvar[0] = "None"
            elif protoType is bool:
                protocol = int(protocol)
                ncvar = ncgrp_proto.createVariable(protoName, int)
                ncvar.assignValue(protocol)
            elif protoType is dict:
                ncvar = ncgrp_proto.createVariable(protoName, str, ('iterableLen','dict'))
                for i, key in enumerate(protocol.keys()):
                    ncvar[i,0] = key
                    ncvar[i,1] = protocol[key]
            elif protoType is list:
                ncvar = ncgrp_proto.createVariable(protoName, str)
                protocol = '---'.join(protocol)
                ncvar[0] = protocol
            elif protoType is np.ndarray:
                ncvar = ncgrp_proto.createVariable(protoName, protocol.dtype, ('lambda'))
                ncvar[:] = protocol.flatten()
            else:
                ncvar = ncgrp_proto.createVariable(protoName, protoType)
                ncvar[0] = protocol
            setattr(ncvar, 'protoType', protoType.__name__)
            if protoUnit:
                setattr(ncvar, "units", str(protoUnit))

        self.ncfile = ncfile
        return

    def writeIteration(self):
        #Get State
        initial_time = time.time()
        state = self.context.getState(getPositions=True, getEnergy=True) 
  
        #Store Positions
        self.mainPositions = state.getPositions(asNumpy=True)
        self.ncfile.variables['positions'][self.iteration,:,:] = self.mainPositions/unit.nanometer
  
        #Store Alchemical State
        self.state = self.getLambda(flat=True)
        self.ncfile.variables['state'][self.iteration,:] = self.state
 
        #Store Energies
        energies = self.computeBasisEnergy()
        self.ncfile.groups['energies'].variables['energy'][self.iteration] = energies['totalPotential']/self.kT
        self.ncfile.groups['energies'].variables['unaffected'][self.iteration] = energies['unaffectedPotential']/self.kT
        self.ncfile.groups['energies'].variables['bias'][self.iteration, 0] = energies['harmonicBias']/self.kT
        self.ncfile.groups['energies'].variables['bias'][self.iteration, 1] = energies['freeEnergyBias']/self.kT
        self.ncfile.groups['energies'].variables['standardBasis'][self.iteration, :] = self.basisManipulator.flattenBasis(energies['standardBasis']/self.kT)
        self.ncfile.groups['energies'].variables['crossBasis'][self.iteration, :] = self.basisManipulator.flattenBasis(energies['crossBasis']/self.kT)

        #Store box volumes:
        self.boxVectors = state.getPeriodicBoxVectors(asNumpy=True)
        self.ncfile.variables['box_vectors'][self.iteration,:,:] = self.boxVectors/unit.nanometer
        self.ncfile.variables['volumes'][self.iteration] = state.getPeriodicBoxVolume()/unit.nanometer**3

        #Store integrator values
        try:
            self.naccept = self.integrator.getGlobalVariableByName("naccept")
            self.ntrials = self.integrator.getGlobalVariableByName("ntrials")
            self.ncfile.groups['MCStats'].variables['naccept'][self.iteration] = self.naccept
            self.ncfile.groups['MCStats'].variables['ntrials'][self.iteration] = self.ntrials
        except:
            pass

        # Force sync to disk to avoid data loss.
        presync_time = time.time()
        self.ncfile.sync()

        # Print statistics.
        final_time = time.time()
        sync_time = final_time - presync_time
        elapsed_time = final_time - initial_time
        if self.verbose: print "Writing data to NetCDF file took {0:.3f} s ({1:.3f} s for sync)".format(elapsed_time, sync_time)
        return

    def run(self):
        if self.context is None:
            print('Cant run a simulation with out a context!')
            raise(Exception)

        run_start_time = time.time()
        run_start_iteration = self.iteration
        while self.iteration < self.nIterations:
            if self.verbose: 
                print "\nIteration {0:d} / {1:d}".format(self.iteration+1, self.nIterations)
            #Timing drawn from YANK (github.com/choderalab/yank)
            initial_time = time.time()

            #Timestep
            self.integrator.step(self.protocol['stepsPerIteration'])
            
            #Store information
            self.writeIteration()

            #Update Bias
            try:
                lasttime = os.stat('FEBias.npy')
                if lasttime > self._lastKnownTime:
                    FEBias = np.load('FEBias.npy')
                    self.updateFreeEnergyBias(FEBias)
            except:
                pass
           
            #Increment iteration
            self.iteration += 1

            #Final time
            final_time = time.time()
            elapsed_time = final_time - initial_time
            estimated_time_remaining = (final_time - run_start_time) / (self.iteration - run_start_iteration) * (self.nIterations - self.iteration)
            estimated_total_time = (final_time - run_start_time) / (self.iteration - run_start_iteration) * (self.nIterations)
            estimated_finish_time = final_time + estimated_time_remaining
            if self.verbose:
                print "Iteration took {0:.3f} s.".format(elapsed_time)
                print "Estimated completion in {0:s}, at {1:s} (consuming total wall clock time {2:s}).".format(str(datetime.timedelta(seconds=estimated_time_remaining)), time.ctime(estimated_finish_time), str(datetime.timedelta(seconds=estimated_total_time)))

        self.ncfile.close()
        return

    def _setProtocol(self, protocol):
        #Set defaults:
        defaultProtocols = {}
        defaultProtocols['standardBasisCoupling'] = ['R', 'C', 'EA']
        defaultProtocols['crossBasisCoupling'] = ['EAR']
        defaultProtocols['temperature'] = 298*unit.kelvin
        #defaultProtocols['pressure'] = 1*unit.atmosphere
        defaultProtocols['pressure'] = None
        defaultProtocols['platform'] = 'OpenCL'
        defaultProtocols['timestep'] = 1*unit.femtosecond
        defaultProtocols['standardSwitches'] = None
        defaultProtocols['crossSwitches'] = {'R':'linear'}
        defaultProtocols['skipContextUnits'] = True
        defaultProtocols['verbose'] = True
        defaultProtocols['stepsPerMCInner'] = 10
        defaultProtocols['stepsPerMCOuter'] = 10
        defaultProtocols['stepsPerIteration'] = 100 #makes 1ps per write out
        defaultProtocols['nIterations'] = 10000 # Makes 10ns at default values
        defaultProtocols['lamMasses'] = None
        defaultProtocols['devIndex'] = 0 #Device index for multiple walkers
        defaultProtocols['cartesianOnly'] = False #Set to use only carteisan updates (no alchemical change)
        if protocol is None:
            self.protocol = defaultProtocols
        else:
            self.protocol = {}
            try:
                for key in protocol.keys():
                    self.protocol[key] = protocol[key]
                for key in defaultProtocols.keys():
                    if key not in self.protocol.keys():
                        self.protocol[key] = defaultProtocols[key]
            except:
                errorMsg = "Protocol needs to be a dictionary. Valid keys are: "
                for key in defaultProtocols.keys():
                    errorMsg += "%s "
                print errorMsg % tuple(defaultProtocols.keys())
                print "Assuming default protocol"
                self.protocol = defaultProtocols
        return

    def __init__(self, Ni, Nj, ff, protocol=None, equilibrate=False, filename='examol.nc', systemname='examolsystem.xml', eqfile='examoleq.nc', coordsFromFile=None, filedebug=False, sysdebug=False):
        self.Ni = Ni
        self.Nj = Nj
        self.ff = ff
        self.equilibrate = equilibrate
        if self.equilibrate:
            ensemble = 'NVT'
            try:
                if protocol['pressure'] is not None:
                    ensemble = 'NPT'
            except:
                pass
            print("{0:s} EQULIBRATION RUN AT FIXED LAMBDA!\nChemical state will NOT change!".format(ensemble))
            if eqfile[-3:] == '.nc':
                eqfile = eqfile[:-3] + ensemble + '.nc'
            else:
                eqfile += ensemble
            self.filename = eqfile
        else:
            self.filename = filename
        self.systemname = systemname
        #Resume Functionality
        self.coordsLoaded = False 
        if os.path.isfile(self.filename) and not filedebug:
            self.resume = True
            self.coordsLoaded = True
            self._resumeFromFile()
        else:
            self._setProtocol(protocol)
            self.resume = False
        #Special check for verbosity
        try:
            self.protocol['verbose'] = protocol['verbose']
        except:
            pass 
        #Set some more easily accessed common variables
        self.temperature = self.protocol['temperature']
        self.kT = kB * self.temperature
        self.pressure = self.protocol['pressure']
        self.timestep = self.protocol['timestep']
        self.verbose = self.protocol['verbose']
        self.stepsPerMCInner = self.protocol['stepsPerMCInner']
        self.stepsPerMCOuter = self.protocol['stepsPerMCOuter']
        if coordsFromFile is not None and not self.coordsLoaded:
            if os.path.isfile(coordsFromFile):
                if self.verbose:
                    print("Loading coordinates and box vectors from final frame of {0:s}".format(coordsFromFile))
                self.coordsLoaded = True
                self._loadCoordinates(coordsFromFile)
        #Build the basisManipulation object. Handles counting, flattening, and expansion of basis function objects used to organize basis energies and handle switch multiplication
        self.basisManipulator = basisManipulation(self.Ni, self.Nj, self.protocol['standardSwitches'], self.protocol['standardBasisCoupling'], self.protocol['crossSwitches'], self.protocol['crossBasisCoupling'])
        #Load the core, we wont be using it in base form for long
        coreSystem, self.corecoords = self.loadpdb('pdbfiles/core/corec')
        self.corePositions = self.corecoords.getPositions(asNumpy=True) #Positions of core atoms (used for alignment)
        self.Ncore = coreSystem.getNumParticles()
        #Start mainSystem
        #Note: The mainSystem is NOT built from the combined topologies because that would add torsions and angle forces to R-groups on the same core carbon, which we wond want.
        if os.path.isfile(self.systemname) and not sysdebug:
            if self.verbose: print("Rebuilding System from file!")
            with open(self.systemname, 'r') as systemfile:
                self.mainSystem = mm.XmlSerializer.deserialize(systemfile.read())
            if self.verbose: print("Rebuilding Topology from file!")
            with open(self.systemname[:-4] + '.top', 'r') as topfile:
                self.mainTopology = pickle.load(topfile)
                #Recount the atom numbers to get the solvent numbers back out
                totalN = self.mainSystem.getNumParticles()
                soluteNAtoms = len([res for res in self.mainTopology.residues() if res.name == 'COC'][0]._atoms)
                self.soluteNumbers = np.arange(soluteNAtoms)
                self.solventNumbers = np.arange(soluteNAtoms,totalN)
            self.mainBondForce = getArbitraryForce(self.mainSystem, mm.HarmonicBondForce)
            self.mainAngleForce = getArbitraryForce(self.mainSystem, mm.HarmonicAngleForce)
            self.mainTorsionForce = getArbitraryForce(self.mainSystem, mm.PeriodicTorsionForce)
            self.mainNonbondedForce = getArbitraryForce(self.mainSystem, mm.NonbondedForce)
            self.mainCMRForce = getArbitraryForce(self.mainSystem, mm.CMMotionRemover)
            if not self.coordsLoaded:
                posdata = np.load(self.filename + '.initPos.npz')
                self.mainPositions = posdata['pos'] * unit.nanometer
                self.boxVectors    = posdata['box'] * unit.nanometer
                boxVectors = self.mainSystem.setDefaultPeriodicBoxVectors(self.boxVectors[0,:], self.boxVectors[1,:], self.boxVectors[2,:])
            self.mainTopology.setPeriodicBoxVectors(self.boxVectors)
            #self.mainTopology.setPeriodicBoxVectors(self.boxVectors[0,:], self.boxVectors[1,:], self.boxVectors[2,:])
        else:
            if self.coordsLoaded:
                tempPos = deepcopy(self.mainPositions)
                tempBox = deepcopy(self.boxVectors)
            self.mainSystem = deepcopy(coreSystem)
            self.mainTopology = deepcopy(self.corecoords.getTopology())
            self.mainPositions = deepcopy(self.corePositions)
            self.mainBondForce = getArbitraryForce(self.mainSystem, mm.HarmonicBondForce)
            self.mainAngleForce = getArbitraryForce(self.mainSystem, mm.HarmonicAngleForce)
            self.mainTorsionForce = getArbitraryForce(self.mainSystem, mm.PeriodicTorsionForce)
            self.mainNonbondedForce = getArbitraryForce(self.mainSystem, mm.NonbondedForce)
            self.mainCMRForce = getArbitraryForce(self.mainSystem, mm.CMMotionRemover)
            #Import R groups
            self._buildRGroups()
            #Bring in Solvent
            self._addSolvent()
            #Initilize box vectors, since these are serialized with the system, we dont need to save them explcitliy
            boxVectors = self.mainSystem.getDefaultPeriodicBoxVectors()
            #Cast box vectors to a more useful form
            self.boxVectors = np.array([x/unit.nanometer for x in boxVectors]) * unit.nanometer
            #Set up the Nonbonded Forces
            self._buildNonbonded()
            #Set up bias forces
            self._buildBiasForces()
            with open(self.systemname, 'w') as systemfile:
                systemfile.write(mm.XmlSerializer.serialize(self.mainSystem))
            #Write topology
            with open(self.systemname[:-4] + '.top', 'w') as topfile:
                pickle.dump(self.mainTopology, topfile)
            if self.coordsLoaded:
                self.mainPositions = tempPos
                self.boxVectors = tempBox
                del tempPos, tempBox
            else:    
                #Save positions with some clever name
                posfile = self.filename + '.initPos.npz'
                np.savez(posfile, pos=self.mainPositions/unit.nanometer, box=self.boxVectors/unit.nanometer)
        self.nParticles = self.mainSystem.getNumParticles()
        self.nIterations = self.protocol['nIterations']
        if not self.resume:
            #Set the iterations:
            self.iteration = 0
            if not filedebug:
                self._initilizeNetCDF()
        #Initilize objects
        self.barostat = None
        self.integrator = None
        self.context = None
        self.platform = None
        #DEBUG
        #1
        #self.mainNonbondedForce.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
        #2
        #for forceidx in xrange(self.mainSystem.getNumForces()):
        #    if isinstance(self.mainSystem.getForce(forceidx), mm.NonbondedForce):
        #        self.mainSystem.removeForce(forceidx)
        #        break
        #3
        #self.mainNonbondedForce.setUseDispersionCorrection(False)
        #4
        #forcelist = []
        #for forceidx in xrange(self.mainSystem.getNumForces()):
        #    if isinstance(self.mainSystem.getForce(forceidx), mm.CustomExternalForce):
        #        forcelist.append(forceidx)
        #for forceidx in forcelist[::-1]:
        #    self.mainSystem.removeForce(forceidx)
        #pdb.set_trace()
        #Load the FE biases
        try:
            self._lastKnownTime = os.stat('FEBias.npy').st_mtime
            FEBias = np.load('FEBias.npy')
            self.updateFreeEnergyBias(FEBias)
        except:
            self._lastKnownTime = 0
        self._buildContext()
        
        return

