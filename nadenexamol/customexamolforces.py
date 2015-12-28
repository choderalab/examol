import pdb
import numpy as np
import simtk.unit as unit
import simtk.openmm as mm
from simtk.openmm import app
from sys import stdout

'''
This module houses all the custom force functions used by the main exmaol script.
I split this off to reduce the clutter in the main examol script.
'''

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


def addAngleForceWithCustom(mainAngleForce, RForce, RAtomNumbers, lami, lamj, Ncore):
    #Copy angle forces from the RForce to the mainAngleForce. Uses info from RAtomNubmers to map the unique angles in the RSystems to the mainSystem. Creates a custom angle force for bonds to the core where R attaches
    lamExpression = 'lam{0:s}x{1:s}B'.format(str(lami), str(lamj))
    energyExpression = '0.5*%s*k*(theta-theta0)^2;' % lamExpression
    customAngleForce = mm.CustomAngleForce(energyExpression)
    customAngleForce.addGlobalParameter(lamExpression, 1)
    customAngleForce.addPerAngleParameter('theta0')
    customAngleForce.addPerAngleParameter('k')
    nRAngle = RForce.getNumAngles()
    for angleIndex in xrange(nRAngle):
        atomi, atomj, atomk, angle, k = RForce.getAngleParameters(angleIndex)
        atomi, atomj, atomk = mapAtomsToMain([atomi, atomj, atomk], RAtomNumbers, Ncore)
        #Is the angle part of the new R group?
        if (atomi >= Ncore or atomj >= Ncore or atomk >= Ncore):
            #is the angle both core AND R group? (and when combined with previous)
            if atomi < Ncore or atomj < Ncore or atomk < Ncore:
                customAngleForce.addAngle(atomi, atomj, atomk, (angle, k)) 
            else: #Standard angle in R group
                mainAngleForce.addAngle(atomi, atomj, atomk, angle, k)
    return customAngleForce

def addTorsionForceWithCustom(mainTorsionForce, RForce, RAtomNumbers, lami, lamj, Ncore):
    #Copy torsion forces from the RForce to the mainTorsionForce. Uses info from RAtomNubmers to map the unique torsions in the RSystems to the mainSystem. Creates a custom torsion force for bonds to the core where R attaches
    lamExpression = 'lam{0:s}x{1:s}B'.format(str(lami), str(lamj))
    energyExpression = '%s*k*(1+cos(n*theta-theta0));' % lamExpression
    customTorsionForce = mm.CustomTorsionForce(energyExpression)
    customTorsionForce.addGlobalParameter(lamExpression, 1)
    customTorsionForce.addPerTorsionParameter('n')
    customTorsionForce.addPerTorsionParameter('theta0')
    customTorsionForce.addPerTorsionParameter('k')
    #customAngleForce.setEnergyFunction(energyExpression)
    nRTorsion = RForce.getNumTorsions()
    for torsionIndex in xrange(nRTorsion):
        atomi, atomj, atomk, atoml, period, phase, k = RForce.getTorsionParameters(torsionIndex)
        atomi, atomj, atomk, atoml = mapAtomsToMain([atomi, atomj, atomk, atoml], RAtomNumbers, Ncore)
        #Is the torsion part of the new R group?
        if (atomi >= Ncore or atomj >= Ncore or atomk >= Ncore or atoml >= Ncore):
            #is the torsion both core AND R group? (and when combined with previous)
            if atomi < Ncore or atomj < Ncore or atomk < Ncore or atoml < Ncore:
                customTorsionForce.addTorsion(atomi, atomj, atomk, atoml, (period, phase, k)) 
            else: #Standard torsion in R group
                mainTorsionForce.addTorsion(atomi, atomj, atomk, atoml, period, phase, k)
    return customTorsionForce

def basisMap(lam, atHalf='down', lamP=None):
    #Basis function map. Converts a single value of lambda into the distribured value for repulsive and attractive basis functions.
    #atHalf controlls how lamCap (lamC) behaves at lam=0.5. 'down' sets lamC=0, 'up' sets lamC=1
    if lam > 0.5:
        lamE = 2.0*lam-1
        if lamP is None:
            lamP = lamE
        lamC = 1.0
        lamR = 1.0
        lamA = 1.0
    elif lam < 0.5:
        lamE = 0
        lamP = 0
        lamC = 0
        lamR = 2.0*lam
        lamA = 2.0*lam
    else: #lam =0.5
        lamE = 0.0
        if lamP is None:
            lamP = lamE
        lamR = 1.0
        lamA = 1.0
        if atHalf == 'down':
            lamC = 0.0
        elif atHalf == 'up':
            lamC = 1.0
    return (lamE, lamP, lamC, lamR, lamA)

def stockNBPotential(LJ=True, Electro=True):
    '''
    Returns a stock LJ nobbonded force with RF electrostatics. Used for decoupling instead of annihilation

    TODO: Figure out a better way to handle decoupling/annihilating
    '''
    ONE_4PI_EPS0 = 138.935456 #From OpenMM's OpenCL kernel
    energy_expression = ''
    if LJ and Electro:
        energy_expression += "epsilon*LJ + electrostatics" 
    elif LJ:
        energy_expression +=  "epsilon*LJ;"
    elif Electro:
        energy_expression +=  "electrostatics;"
    else:
        print("I need some type of nonbonded force!")
        raise
    if LJ:
        energy_expression += "LJ = 4*((sigma/r)^12 - (sigma/r)^6);" #Lennard-Jones statment
        energy_expression += "epsilon = sqrt(epsilon1*epsilon2);" # mixing rule for epsilon
        energy_expression += "sigma = 0.5*(sigma1 + sigma2);" # mixing rule for sigma
    if Electro:
        energy_expression += "electrostatics = charge1*charge2*{0:f}*reaction_field;".format(ONE_4PI_EPS0)
        energy_expression += "reaction_field = (1/r) + krf*r^2 - crf;"
        energy_expression += "krf = (1/(rcut^3)) * ((dielectric-1)/(2*dielectric+1));"
        energy_expression += "crf = (1/rcut) * ((3*dielectric)/(2*dielectric+1));"

    custom_nonbonded_force = mm.CustomNonbondedForce(energy_expression)
    if Electro:
        custom_nonbonded_force.addPerParticleParameter("charge")
        custom_nonbonded_force.addGlobalParameter("dielectric", 70)
        custom_nonbonded_force.addGlobalParameter("rcut", 1)
    if LJ:
        custom_nonbonded_force.addPerParticleParameter("sigma") # Lennard-Jones sigma
        custom_nonbonded_force.addPerParticleParameter("epsilon") # Lennard-Jones epsilon
    
    return custom_nonbonded_force

    
def basisEnergy(i, j, i2=None, j2=None, LJ=True, Electro=True):
    '''
    Houses the basic energy string for the basis function energy. Based on Naden and Shirts, JCTC 11 (6), 2015, pp. 2536-2549

    Can pass in i2 and j2 for alchemical <-> alchemical groups, It may be possible to simplify this interation to linear (one basis function), but I think having it as full basis function may be better since I have flexible groups

    Energy expression kept here to reduce clutter elsewhere.
    All of the "If" statements are so the order of the global parameters for setting defaults is preserved
    '''
    ONE_4PI_EPS0 = 138.935456 #From OpenMM's OpenCL kernel
    if i2 is None and j2 is None:
        lamE = 'lam{0:s}x{1:s}E'.format(str(i),str(j))
        lamP = 'lam{0:s}x{1:s}P'.format(str(i),str(j))
        lamC = 'lam{0:s}x{1:s}C'.format(str(i),str(j))
        lamA = 'lam{0:s}x{1:s}A'.format(str(i),str(j))
        lamR = 'lam{0:s}x{1:s}R'.format(str(i),str(j))
    else:
        lamE = 'lam{0:s}x{1:s}x{2:s}x{3:s}E'.format(str(i),str(j),str(i2),str(j2))
        lamP = 'lam{0:s}x{1:s}x{2:s}x{3:s}P'.format(str(i),str(j),str(i2),str(j2))
        lamC = 'lam{0:s}x{1:s}x{2:s}x{3:s}C'.format(str(i),str(j),str(i2),str(j2))
        lamA = 'lam{0:s}x{1:s}x{2:s}x{3:s}A'.format(str(i),str(j),str(i2),str(j2))
        lamR = 'lam{0:s}x{1:s}x{2:s}x{3:s}R'.format(str(i),str(j),str(i2),str(j2))
   
    #Start energy Expression
    energy_expression = ""
    if LJ and Electro:
        energy_expression +=  "epsilon*(RepSwitchCappedBasis + AttSwitchBasis + CappingSwitchBasis) + electrostatics;"
    elif LJ:
        energy_expression +=  "epsilon*(RepSwitchCappedBasis + AttSwitchBasis + CappingSwitchBasis);"
    elif Electro:
        energy_expression +=  "electrostatics;"
    else:
        print("I need some type of nonbonded force!")
        raise

    if LJ:
        ###NOTE: the epsilon has been extracted but the 4 is still embeded!###########
        energy_expression += "CappingSwitchBasis = {0:s}*CappingBasis;".format(lamC)
        energy_expression += "CappingBasis = repUncap - RepCappedBasis;"
        energy_expression += "RepSwitchCappedBasis = repSwitch*RepCappedBasis;"
        energy_expression += "RepCappedBasis = Rnear + Rtrans + Rfar;"
        energy_expression += "AttSwitchBasis = attSwitch*attBasis;"
        energy_expression += "repSwitch = pa*({0:s}^4) + pb*({0:s}^3) + pc*({0:s}^2) + (1-pa-pb-pc)*{0:s};".format(lamR) #Repulsive Term
        energy_expression += "pa = %f; pb = %f; pc = %f;" % (1.61995584, -0.8889962, 0.02552684) #Values found from optimization routine
        energy_expression += "attSwitch = {0:s};".format(lamA)
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
    if Electro:
        #=== Electrostatics ===
        # This block commented out until I figure out how to do long range PME without calling updateParametersInContext(), Switched to reaction field below
        #err_tol = nonbonded_force.getEwaldErrorTolerance()
        #rcutoff = nonbonded_force.getCutoffDistance() / units.nanometer #getCutoffDistance Returns a unit object, convert to OpenMM default (nm)
        #alpha = numpy.sqrt(-numpy.log(2*err_tol))/rcutoff #NOTE: OpenMM manual is wrong here
        #energy_expression = "alchEdirect-(Edirect * eitheralch);" #Correction for the alchemical1=0 alchemical2=0 case)
        #energy_expression += "Edirect = ({0:f} * (switchPME*alchemical1 + 1 -alchemical1) * charge1 * (switchPME*alchemical2 + 1 -alchemical2) * charge2 * erfc({1:f} * r)/r);".format(ONE_4PI_EPS0, alpha) #The extra bits correct for alchemical1 and alhemical being on
        #energy_expression += "alchEdirect = switchE * {0:f} * charge1 * charge2 * erfc({1:f} * r)/r;".format(ONE_4PI_EPS0, alpha)
        #energy_expression += "switchPME = {0:s};".format(lamP)
        #energy_expression += "switchE = {0:s};".format(lamE)

        energy_expression += "electrostatics = {0:s}*charge1*charge2*{1:f}*reaction_field;".format(lamE, ONE_4PI_EPS0)
        energy_expression += "reaction_field = (1/r) + krf*r^2 - crf;"
        energy_expression += "krf = (1/(rcut^3)) * ((dielectric-1)/(2*dielectric+1));"
        energy_expression += "crf = (1/rcut) * ((3*dielectric)/(2*dielectric+1));"

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
    
    return custom_nonbonded_force

def assignBasisForceDefaults(mainSystem, basisForce, interactionGroups=None, longRange=False):
    #Assign the default values to the basisForce object based on mainSystem
    #If interactionGroups is a size 2 iterable object of sets, then interaction groups are added
    mainNonbondedForce = getArbitraryForce(mainSystem, mm.NonbondedForce)
    mainNonbondedCutoff = mainNonbondedForce.getCutoffDistance()
    mainNonbondedDielectric = mainNonbondedForce.getReactionFieldDielectric()
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
    mainSystem.addForce(basisForce)
    return basisForce

def buildNonbonded(mainSystem, coreSystem, RSystems, RAtomNumbers, solventNumbers, Ni, Nj):
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
    mainNonbondedForce = getArbitraryForce(mainSystem, mm.NonbondedForce)
    Ncore = coreSystem.getNumParticles()
    coreAtomNumbers = range(Ncore)
    #Build the i to common interactions
    nbForceList = [] #will loop through this afterwords to add particles
    coreForceList = [] #Special group of forces handling the core
    forceGroupI = 1 #starting force group for ith->solvent
    forceGroupII = Ni+1 #Starting force group for i->i interaction
    for i in xrange(Ni):
        #i-> solvent force groups are all in the initial force groups
        for j in xrange(Nj):
            forceGroupII = (Ni+1)+(i*Ni - (i**2+i)/2) #starting Force group for i->i interactions
            #ij -> solvent interactions
            #Fetch energy
            basisForce = basisEnergy(i,j)
            #Build Defaults and attach basisForce, append force to list to loop through later
            basisForce = assignBasisForceDefaults(mainSystem, basisForce, interactionGroups=(set(RAtomNumbers[i,j]), set(solventNumbers)))
            #Add (i,j) R-group to (i,j) R-group interactions
            basisForce.addInteractionGroup(set(RAtomNumbers[i,j]), set(RAtomNumbers[i,j]))
            #Add sig/epsilon interaction of (i,j) R-group to (i,j) corei (Core charge will be 0)
            basisForce.addInteractionGroup(set(RAtomNumbers[i,j]), set(coreAtomNumbers))
            #Add (i,j) Core -> Solvent Electrostatic interactions
            coreForce = basisEnergy(i,j, LJ=False, Electro=True) 
            coreForce = assignBasisForceDefaults(mainSystem, coreForce, interactionGroups=(set(coreAtomNumbers), set(solventNumbers)))
            #(i,j )Core -> (i,j) R-group Electrostatics
            coreForce.addInteractionGroup(set(coreAtomNumbers),set(RAtomNumbers[i,j]))
            #(i,j) Core -> (i,j) Core Electrostatics
            coreForce.addInteractionGroup(set(coreAtomNumbers),set(coreAtomNumbers))
            basisForce.setForceGroup(forceGroupI)
            coreForce.setForceGroup(forceGroupI)
            nbForceList.append(basisForce)
            coreForceList.append({'i':i, 'j':j, 'i2':None, 'j2':None, 'force':coreForce})
            #Loop through all j on NOT the same i
            for i2 in xrange(i+1,Ni): #no need to loop backwards, otherwise will double up on force
                for j2 in xrange(Nj): #All j affected, just not same i
                    basisForce2 = basisEnergy(i, j, i2=i2, j2=j2)
                    basisForce2 = assignBasisForceDefaults(mainSystem, basisForce2, interactionGroups=(set(RAtomNumbers[i,j]), set(RAtomNumbers[i2,j2])), longRange=False)
                    #(i,j) Core -> R(i2,j2) sigma/epsilon REDUNDANT, NOT NEEDED since same particles as (i,j) R-core
                    #(i,j) Core to (i2,j2) R-group Electrostatics
                    coreForce2 = basisEnergy(i, j, i2=i2, j2=j2, LJ=False, Electro=True)
                    coreForce2 = assignBasisForceDefaults(mainSystem, coreForce2, interactionGroups=(set(coreAtomNumbers), set(RAtomNumbers[i2,j2])))
                    #Core -> Core(i2,j2) Electrostatics: Disabled since I assume that the individual pull from the R-groups and the self (ij) (ij) sums linearly to make a net result
                    basisForce2.setForceGroup(forceGroupII)
                    coreForce2.setForceGroup(forceGroupII)
                    nbForceList.append(basisForce2)
                    coreForceList.append({'i':i, 'j':j, 'i2':i2, 'j2':j2, 'force':coreForce2})
                forceGroupII += 1
        forceGroupI += 1

    #Bring over Nonbonded parameters to the forces
    nParticles = mainSystem.getNumParticles()
    #Get all alchemical atoms
    alchemicalParticles = []
    for atomList in RAtomNumbers.flatten():
        alchemicalParticles.extend(atomList)
    particleNB = [] #Empty list to store nonbonded parameters data is not accidentally deleted before it is added to the customNB forces
    #Fetch default NB parameters
    for particleIndex in xrange(nParticles):
        #Order: q, sigma, epsilon
        q, sigma, epsilon = mainNonbondedForce.getParticleParameters(particleIndex)
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
            mainNonbondedForce.setParticleParameters(particleIndex, 0*unit.elementary_charge, sigma, 0*unit.kilojoules_per_mole)
        elif particleIndex in coreAtomNumbers:
            #Remove Core charge, leave sigma/epsilon for those interactions with solvent
            mainNonbondedForce.setParticleParameters(particleIndex, 0*unit.elementary_charge, sigma, epsilon)
    alchemicalExceptions = [] #Exception indicies involving alchemical atoms, will speed up loop later
    for exceptionIndex in xrange(mainNonbondedForce.getNumExceptions()):
        atomi, atomj, chargeProd, sigma, epsilon = mainNonbondedForce.getExceptionParameters(exceptionIndex)
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
               qcore, sigmacore, epsiloncore  = getArbitraryForce(RSystems[corei,corej], mm.NonbondedForce).getParticleParameters(particleIndex)
           else:
               qcore = q
           #Assign parameters. Each particle must be in every customNonbondedForce, but the InteractionGroups defined eariler will handle which particles interact
           coreForce['force'].addParticle((qcore.value_in_unit(unit.elementary_charge),))
           Rforce.addParticle((q.value_in_unit(unit.elementary_charge), sig.value_in_unit(unit.nanometer), epsi.value_in_unit(unit.kilojoules_per_mole)))
        for exceptionIndex in alchemicalExceptions:
            atomi, atomj, chargeProd, sigma, epsilon = mainNonbondedForce.getExceptionParameters(exceptionIndex)
            Rforce.addExclusion(atomi, atomj)
            coreForce['force'].addExclusion(atomi, atomj)
    stdout.write('\n')
    return

def assignLambda(context, lamVector, Ni, Nj, skipij2=False):
    #Take the lamVector and assign the global parameters
    #lamP is not implemented for lamVectors which include the basis functions
    #skipij2 is a debug flag
    if isinstance(lamVector, list):
        lamVector = np.array(lamVector)
    #Check if the vector is a 2D array that is really just a 1D column.
    # True if there is a length of vectors but only 1 column entry
    if lamVector.shape[-1] == 1 and lamVector.shape[0] > 1:
        lamVector = lamVector.flatten()
    if len(lamVector.shape) >= 3: #The basis function values were passed in as a separate dimension
        hasBasis = True
        #Assuming the vector has been passed in correctly
    else:
        hasBasis = False
        lamVector = lamVector.reshape(Ni,Nj)
    for i in xrange(Ni):
        for j in xrange(Nj):
            if hasBasis:
                lamE, lamC, lamR, lamA, lamij = lamVector[i,j,:]
            else:        
                lamij = lamVector[i,j]
                lamE, lamP, lamC, lamR, lamA = basisMap(lamij)
            context.setParameter('lam{0:s}x{1:s}B'.format(str(i),str(j)), lamij)
            context.setParameter('lam{0:s}x{1:s}E'.format(str(i),str(j)), lamE)
            #context.setParameter('lam{0:s}x{1:s}P'.format(str(i),str(j)), lamP)
            context.setParameter('lam{0:s}x{1:s}C'.format(str(i),str(j)), lamC)
            context.setParameter('lam{0:s}x{1:s}A'.format(str(i),str(j)), lamA)
            context.setParameter('lam{0:s}x{1:s}R'.format(str(i),str(j)), lamR)
            if not skipij2:
                for i2 in xrange(i+1,Ni): #no need to loop backwards, otherwise will double up on force
                    for j2 in xrange(Nj): #All j affected, just not same i
                        if hasBasis:
                            lamE2, lamC2, lamR2, lamA2, lamij = lamVector[i2,j2,:]
                        else:
                            lamij2 = lamVector[i2,j2]
                            lamE2, lamP2, lamC2, lamR2, lamA2 = basisMap(lamij2)
                        context.setParameter('lam{0:s}x{1:s}x{2:s}x{3:s}E'.format(str(i),str(j),str(i2),str(j2)), lamE*lamE2)
                        #context.setParameter('lam{0:s}x{1:s}x{2:s}x{3:s}P'.format(str(i),str(j),str(i2),str(j2)), lamP*lamP2)
                        context.setParameter('lam{0:s}x{1:s}x{2:s}x{3:s}C'.format(str(i),str(j),str(i2),str(j2)), lamC*lamC2)
                        context.setParameter('lam{0:s}x{1:s}x{2:s}x{3:s}A'.format(str(i),str(j),str(i2),str(j2)), lamA*lamA2)
                        context.setParameter('lam{0:s}x{1:s}x{2:s}x{3:s}R'.format(str(i),str(j),str(i2),str(j2)), lamR*lamR2)
    return

def getLambda(context, Ni, Nj):
    lamVector = np.zeros([Ni,Nj])
    for i in xrange(Ni):
        for j in xrange(Nj):
            lamVector[i,j] = context.getParameter('lam{0:s}x{1:s}B'.format(str(i),str(j)))
    return lamVector

def groupFlag(listin):
    #Take a list of force group IDs (ints from [0,31]) and cast it to the bitwise flag for openmm
    bits = '0'*32
    if type(listin) is int:
        listin = [listin]
    bits = list(bits)
    for flag in listin:
        bits[-flag-1] = '1'
    return int(''.join(bits), 2)



def computeBasisEnergy(context, Ni, Nj):
    '''
    Compute all the basis function energies given a context, sites, and molecules per site.

    Energy list:
    Solvent <-> R(i,j) including core: Electro, cap, R, A, Bonded
    R(i,j) <-> R(i2,j2) including core: Electro, cap, R, A
    ''' 

    #Initilize energies
    # E, C, R, A, B
    #assign indices for more less hard code repition later
    basisIndex = {'E':0, 'C':1, 'R':2, 'A':3, 'B':4}
    numBasis = 5
    rijSolvBasis = np.zeros([Ni,Nj,numBasis]) * unit.kilojoules_per_mole
    rijRij2Basis = np.zeros([Ni,Nj,Ni,Nj,numBasis-1]) * unit.kilojoules_per_mole
    basisLambdaValues = np.zeros(rijSolvBasis.shape)
    #Get current total potential and statei
    currentLambda = getLambda(context, Ni, Nj)
    currentPotential = context.getState(enforcePeriodicBox=True,getEnergy=True).getPotentialEnergy()
    #Cycle through the lambda
    forceGroupI = 1 #starting force group for ith->solvent
    forceGroupII = Ni+1 #Starting force group for i->i interaction
    for i in xrange(Ni):
        for j in xrange(Nj):
            forceGroupII = (Ni+1)+(i*Ni - (i**2+i)/2) #starting Force group for i->i interactions
            #rijIndex = i*Ni + j
            groups = groupFlag(forceGroupI)
            #Set current basis values
            lamE, lamP, lamC, lamR, lamA = basisMap(currentLambda[i,j])
            basisLambdaValues[i,j,:] = np.array([lamE, lamC, lamR, lamA, currentLambda[i,j]])
            #Find the basis functions by setting the lambdaVector
            for nBasis in xrange(numBasis):
                lamVector = np.zeros([Ni,Nj,numBasis])
                lamVector[i,j,nBasis] = 1
                assignLambda(context, lamVector, Ni, Nj)
                rijSolvBasis[i,j,nBasis] = context.getState(enforcePeriodicBox=True,getEnergy=True,groups=groups).getPotentialEnergy()
            #Loop through i2/j2 interactions
            for i2 in xrange(i+1,Ni): #no need to loop backwards, otherwise will double up on energy calculations
                for j2 in xrange(Nj): #All j affected, just not same i
                    groups = groupFlag(forceGroupII)
                    for nBasis in xrange(numBasis-1):
                        lamVector = np.zeros([Ni,Nj,numBasis])
                        lamVector[i,j,nBasis] = 1
                        lamVector[i2,j2,nBasis] = 1
                        assignLambda(context, lamVector, Ni, Nj)
                        rijRij2Basis[i,j,i2,j2,nBasis] = context.getState(enforcePeriodicBox=True,getEnergy=True,groups=groups).getPotentialEnergy()
                forceGroupII += 1
        forceGroupI += 1
    #Lastly, unaffected energies
    groups = groupFlag(0)
    unaffectedPotential = context.getState(enforcePeriodicBox=True,getEnergy=True,groups=groups).getPotentialEnergy()
    #Recast repulsive lambda to h(lambda)
    repA = 1.61995584
    repB = -0.8889962
    repC = 0.02552684
    hRep = lambda lam: repA*lam**4 + repB*lam**3 + repC*lam**2 + (1-repA-repB-repC)*lam
    basisHValues = basisLambdaValues.copy()
    basisHValues[:,:,basisIndex['R']] = hRep(basisHValues[:,:,basisIndex['R']])
    #Ensure total energy = bais function energy
    basisPotential = unaffectedPotential
    basisPotential += np.sum(rijSolvBasis * basisHValues)
    for i in xrange(Ni):
        for j in xrange(Nj):
            for i2 in xrange(i+1,Ni):
                for j2 in xrange(Nj):
                    basisH2Values = basisLambdaValues[i,j,:-1] * basisLambdaValues[i2,j2,:-1]
                    basisH2Values[basisIndex['R']] = hRep(basisH2Values[basisIndex['R']])
                    basisPotential += np.sum(rijRij2Basis[i,j,i2,j2,:] * basisH2Values)
    print "Net Total Energy: {0:f}".format(currentPotential / unit.kilojoules_per_mole)
    print "Basis Total Energy: {0:f}".format(basisPotential / unit.kilojoules_per_mole)
    #Reset the state
    pdb.set_trace()
    assignLambda(context, currentLambda, Ni, Nj, skipij2=True)
    return

