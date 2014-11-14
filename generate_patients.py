import sys
from xml.dom.minidom import parseString
from dicttoxml import dicttoxml
import string
import random
import shelve
import numpy as np 
import scipy.sparse as sparse
import cPickle as pickle
from collections import defaultdict, namedtuple

def randomString(length=16):
	return "".join([random.choice(string.letters) for _ in xrange(length)])

def token((disp, repr)):
	return {'disp':disp, 'repr':repr}

def randomText(length=30):
	return " ".join([random.choice(words) for _ in xrange(length)])

def remove_prefix(w):
	if '_' in w:
		return w.split('_', 1)[1]
	else:
		return w

words = file('/usr/share/dict/words').read().splitlines()
random.shuffle(words)
words = words[:500]
vocab = set(words)

def randomPatient():
	global vocab
	pat = {}
	pat['index'] = randomString()
	ChiefComplaint = pat['ChiefComplaint'] = randomText(2)
	TriageAssessment = pat['TriageAssessment'] = randomText(10)
	MDcomments = pat['MDcomments'] = randomText(50)
	Age = pat['Age'] = str(np.random.choice(range(20,80)))
	Sex = pat['Sex'] = np.random.choice(['M', 'F'])[0]
        return {'visit':pat}

if __name__ == "__main__":

    try:
        n = int(sys.argv[1])
    except:
        print "usage: python generate_patients.py numPatients"
        sys.exit()

    xml_str = ""
    for _ in xrange(n):
        pat = randomPatient()
        xml = dicttoxml(pat, attr_type=False, root=False)
        dom = parseString(xml)
        pat = '\n'.join(dom.toprettyxml().split('\n')[1:])
        xml_str += pat
    print xml_str
