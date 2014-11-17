import sys
import os
import subprocess
import string

printable = set(string.printable)
def sanitize(txt):
    txt = ''.join(filter(lambda c: c in printable, txt)) 
    return txt

def traverse(t, outfile):
    print>>outfile, sanitize(t.code+'\t'+t.description)
    for c in t.children:
        traverse(c, outfile)

def getEdges(t, outfile):
    for c in t.children:
        print >>outfile, sanitize(t.code+'\t'+c.code)
        getEdges(c, outfile)


print 'cloning github repository sirrice/icd9.git'
subprocess.call('git clone https://github.com/sirrice/icd9.git', shell=1)

sys.path.append('icd9')
from icd9 import ICD9

tree = ICD9('icd9/codes.json')
toplevelnodes = tree.children

print 'creating name file'
outfile = file('code.names', 'w')
traverse(tree, outfile)
outfile.close()

print 'creating edges file'
outfile = file('code.edges', 'w')
getEdges(tree, outfile)
outfile.close()

print 'cleaning up'
#os.chdir('..')
#subprocess.call('rm -rf icd9', shell=1)
