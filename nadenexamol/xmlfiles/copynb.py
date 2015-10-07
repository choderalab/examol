import re
import os
import pdb

mainxml = open('examol.xml')
sourcefile = open('gaff.xml').readlines()

gaffclass = {}
nb = {}

typebreak = False
hitnb = False
for line in sourcefile:
    line = line.strip().strip('<>').split()
    if line[0] == '/AtomTypes':
        typebreak = True
    if not typebreak and line[0] == 'Type':
        name = re.search('"(.*)"',line[1]).group(1)
        molclass = re.search('"(.*)"',line[2]).group(1)
        gaffclass[molclass] = name
    if line[0] == "NonbondedForce":
        hitnb = True
    if hitnb and line[0] == 'Atom':
        name = re.search('"(.*)"',line[1]).group(1)
        sigma = re.search('"(.*)"',line[3]).group(1)
        epsilon = re.search('"(.*)"',line[4]).group(1)
        nb[name] = (sigma, epsilon)

typebreak = False
hitnb = False
donewithnb = False
names = []
classes = []
basenb = '  <Atom type="%s" charge="0" sigma="%s" epsilon="%s"/>\n'
for line in mainxml:
    print line,
    splitline = line.strip().strip('<>').split()
    if splitline[0] == '/AtomTypes':
        typebreak = True
    if not typebreak and splitline[0] == 'Type':
        names.append(re.search('"(.*)"',splitline[1]).group(1))
        classes.append(re.search('"(.*)"',splitline[2]).group(1))
    if splitline[0] == "NonbondedForce":
        hitnb = True
    if hitnb and not donewithnb:
        for i in xrange(len(names)):
            name = names[i]
            molclass = classes[i]
            gafftype = gaffclass[molclass]
            sigma, epsilon = nb[gafftype]
            print basenb % (name,sigma,epsilon),
        donewithnb = True
