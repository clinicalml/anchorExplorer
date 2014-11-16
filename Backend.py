from Tkinter import *
import os
from Anchors import Anchor
import random
from copy import deepcopy
import tkFileDialog
import itertools
from multiprocessing import Pool
import ttk
import shelve
from collections import defaultdict
import time
import scipy.sparse as sparse
import xml.etree.ElementTree as ET
from Logging import LogElement
from collections import namedtuple
from copy import *
import cPickle as pickle
import string
import numpy as np
from helpers import *
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.grid_search import GridSearchCV
import sklearn.metrics as metrics


def getPatient(v):
    visitShelf = shelve.open('visitShelf')
    pat = visitShelf[v]
    pat['anchors'] = set()
    return v, pat

def randomString(length=16):
    return "".join([random.choice(string.letters) for _ in xrange(length)])

def noPunctuation(w):
    return len(set("{}()") & set(w[0]))==0

def readLabels(filename):
    labels = {}
    f = file(filename)
    for l in f:
        id, label = l.split()
        labels[id] = int(label)
    return filename

def readAnchors(filename, parent, max_anchors=-1):
    Tree = ET.parse(filename)
    root = Tree.getroot()
    anchors = {}

    for concept in root.findall('.//concept'):
        name = concept.attrib['name']
        print 'initializing concept', name

        saveloc = concept.attrib['saveloc']+'/'+name+'.pk'
        label_saveloc = concept.attrib['saveloc']+'/'+name+'.labels.pk'
        flags_saveloc = concept.attrib['saveloc']+'/'+name+'.flags.pk'
        labels = None

        try:
            anch = pickle.load(file(saveloc))
        except:
            nodes = concept.text.strip().split('|')
            edges = []
            anch = [Anchor(n, [n], []) for n in nodes if len(n)]

        try:
            labels = pickle.load(file(label_saveloc))
        except:
            labels = {}

        try:
            flags = pickle.load(file(flags_saveloc))
        except:
            print 'could not load flags from ', flags_saveloc
            flags = {}

        anchors[name] = Concept(name, anch, parent=parent, saveloc=saveloc, labels=labels, flags=flags)

    for loc in os.listdir(parent.saveloc):
        if '.labels' in loc:
            continue
        if '.svn' in loc:
            continue
        if '.flags' in loc:
            continue
        if 'elkan' in loc:
            continue
        if '.eval' in loc:
            continue
        if '.weights' in loc:
            continue
        name = loc.split('/')[-1].replace('.pk', '')
        if not name in anchors:
            anch = pickle.load(file(parent.saveloc+'/'+loc))
            try:
                labels = pickle.load(file(parent.saveloc+'/'+loc.replace('.pk', '.labels.pk')))
            except:
                labels = {}
            anchors[name] = Concept(name, anch, parent=parent, saveloc=parent.saveloc+'/'+name+'.pk', labels=labels)

    print 'anchors initialized', anchors
    return anchors

def update_sparse_X(X):
    return csr_matrix(X)


class Concept:
    def __init__(self, name, anchors, parent=None, description="", saveloc='', labels=None, flags=None):
        self.anchors = set(anchors)
        self.evaluators = set()
        self.name = name
        self.id = randomString(16)
        self.anchoredPatients = {}
        self.evaluatorPatients = {}
        self.recall = 0.8
        
        self.pos_patients = []

        self.description=description
        self.human_labels = {}
        self.evaluations = []
        self.online=True

        if labels == None:
            self.labels = {}
        else:
            self.labels = labels

        for pid,label in self.labels.items():
            self.human_labels[pid] = label

        if flags == None:
            self.flagged_patients = {}
        else:
            self.flagged_patients = flags

        self.sparse_X = {}
        self.sparse_X_csr = None
        self.masked_elements = defaultdict(set)
        self.Y = []
        self.Y_counts = []
        self.Y_negCounts = []
        self.log = []
        self.vocab = None
        self.inv_vocab = None
        self.display_vocab = None
        self.patient_index = None
        self.patient_list = None
        self.estimator=None
        self.ranking=None
        self.recentPatients = set()
        self.initialized = False

        #state that is not preserved
        self.pool=Pool(2)
        self.wordShelf=None
        self.backend = parent
        self.saveloc=saveloc
        self.label_saveloc = saveloc.replace('.pk', '.labels.pk')
        self.flags_saveloc = saveloc.replace('.pk', '.flags.pk')
        self.eval_saveloc = saveloc.replace('.pk', '.eval.pk')
        
        try:
            self.evaluators = pickle.load(file(self.eval_saveloc))
        except:
            pass
        self.dumpAnchors()
        self.dumpLabels()

    def set_name(self, new_name):
        self.saveloc = self.saveloc.replace(self.name, new_name)
        self.label_saveloc = self.label_saveloc.replace(self.name, new_name)
        
        for pid in union(self.anchoredPatients.values()):
            if pid in self.backend.patients:
                self.backend.patients[pid]['anchors'].remove(self.name)
                self.backend.patients[pid]['anchors'].add(new_name)

        self.name = new_name

    def dumpAnchors(self):
        try:
            pickle.dump(self.anchors, file(self.saveloc, 'w'))
        except:
            print 'warning could not save to ', self.saveloc

    def dumpLabels(self):
        try:
            pickle.dump(self.human_labels, file(self.label_saveloc, 'w'))
        except:
            print 'warning could not save to ', self.saveloc

    def dumpFlags(self):
        try:
            print 'dumping flags', self.flagged_patients.items()
            pickle.dump(self.flagged_patients, file(self.flags_saveloc, 'w'))
        except:
            print 'warning could not save to ', self.flags_saveloc

    def dumpEvaluators(self):
        try:
            print 'dumping evaluators', self.evaluators
            pickle.dump(self.evaluators, file(self.eval_saveloc, 'w'))
        except:
            print 'warning could not save to ', self.eval_saveloc

    def dumpDecisionRule(self):
        loc = self.saveloc.replace('.pk', '.weights.pk')
        try:
            pickle.dump(zip(self.vocab, self.estimator.coef_), file(loc, 'w'))
        except Exception, e:
            print 'could not dump rule to ', loc, "%s", e

            pass
    def saveState(self):
        print 'saving state'
        f = file(self.saveloc, 'w')
        state = [self.anchors, 
                      self.name, 
                      self.id, 
                      self.anchoredPatients, 
                      self.description,
                      self.human_labels,
                      self.sparse_X,
                      self.sparse_X_csr,
                      self.masked_elements,
                      self.Y,
                      self.Y_counts,
                      self.log,
                      self.patient_index,
                      self.patient_list,
                      self.estimator,
                      self.ranking,
                      self.recentPatients,
                     ]
        pickle.dump(state, f)
        f.close()
        
    def loadState(self, parent, wordshelf):
        try:
            assert 0
            f = file(self.saveloc)
        except:
            print 'could not load from pickle'
            return False
        
        [self.anchors, 
          self.name, 
          self.id, 
          self.anchoredPatients, 
          self.description,
          self.human_labels,
          self.sparse_X,
          self.sparse_X_csr,
          self.masked_elements,
          self.Y,
          self.Y_counts,
          self.log,
          self.vocab,
          self.inv_vocab,
          self.display_vocab,
          self.patient_index,
          self.patient_list,
          self.estimator,
          self.ranking,
          self.recentPatients,
         ] = pickle.load(f)
        f.close()
        self.parent = parent
        self.wordshelf = wordshelf
        return True
        
    def initPatients(self, patients, wordShelf, vocab, inv_vocab, display_vocab):
        print 'concept initialize patients'
        self.vocab, self.inv_vocab, self.display_vocab = vocab, inv_vocab, display_vocab
        self.wordShelf = wordShelf
        for anchor in self.anchors:
            for a in anchor.getMembers():
                a = a.lstrip('!')
                if a in wordShelf:
                    self.anchoredPatients[a] = wordShelf[a]
                else:
                    print "warning: word", a, "not indexed!"
        
        for pid in union(self.anchoredPatients.values()):
            if pid in patients:
                patients[pid]['anchors'].add(self.name)

    def initLog(self):
        self.log.append(LogElement('init')) 

    
    def done_updating(self, result):
        self.sparse_X_csr = result
        print "done updating"

    def configureLearnButton(self, state):
        self.backend.parent.anchorDisplay.learnButton.configure({'state':state})

    def initRepresentation(self, patients, sparse_X):
        print 'init representation'
        print >> self.backend.parent.logfile, str(time.time())+' init representation', self.name
        self.patient_index = dict(zip([pat['index'] for pat in patients], xrange(len(patients))))
        self.patient_list = patients
        print len(patients)
        self.sparse_X = sparse_X.copy()
        self.sparse_X_csr = None
        self.sparse_X_csr_eval = None

        if self.online:
            self.pool.apply_async(update_sparse_X, args=[self.sparse_X], callback=self.done_updating)
            self.sparse_X_csr_eval = csr_matrix(self.backend.sparse_X_validate)
        else:
            self.done_updating(update_sparse_X(self.sparse_X))

        self.Y = [0]*len(patients)
        self.Y_counts = [0]*len(patients)
        self.Y_negCounts = [0]*len(patients)
        for anchor in self.anchors:
            self.addAnchor(anchor)
        for evaluator in self.evaluators:
            self.addEvaluator(evaluator)
    
    def addAnchor(self, new_anchor):

        if new_anchor.id == '':
            return
        print >> self.backend.parent.logfile, str(time.time())+' added anchor', new_anchor.id, self.name
        print 'new anchor', new_anchor.id
        self.backend.parent.logfile.flush()

        self.backend.doIndexing(new_anchor)

        assert type(new_anchor) == Anchor, type(new_anchor)
        self.anchors.add(new_anchor)

        newly_anchored_patients = set()
        for a in new_anchor.getMembers():
            a = a.lstrip('!')
            if a in self.wordShelf:
                newly_anchored_patients |= set(self.wordShelf[a])
                print 'anchor component', a, len(set(self.wordShelf[a])), 'total', len(newly_anchored_patients)
            else:
                print "anchor", a, "not indexed!"
                sys.exit()


        self.anchoredPatients[new_anchor.id] = newly_anchored_patients
        self.recentPatients = newly_anchored_patients
        print new_anchor in self.vocab
        print new_anchor in self.inv_vocab

        for pid in self.recentPatients:
            try:
                i = self.patient_index[pid]
            except:
                continue
            self.Y[i]=1
            self.Y_counts[i] += 1

            if new_anchor.id[0] == '!':
                self.Y_negCounts[i] += 1
                self.Y[i] = 0
                continue

            for a in new_anchor.getExclusions():
                if not a in self.inv_vocab:
                    continue

                j = self.inv_vocab[a]
                if self.sparse_X[i,j] > 0:
                    self.masked_elements[j].add(i)
                    self.sparse_X[i,j] = 0

        #self.configureLearnButton('disabled')
        if self.online:
            self.pool.apply_async(update_sparse_X, args=[self.sparse_X], callback=self.done_updating)
        else:
            self.done_updating(update_sparse_X(self.sparse_X))

        self.dumpAnchors()
        
    def addEvaluator(self, new_anchor):

        print >> self.backend.parent.logfile, str(time.time())+' added evaluator', new_anchor.id, self.name
        print 'new evaluator', new_anchor.id
        self.backend.parent.logfile.flush()

        self.backend.doIndexing(new_anchor)

        assert type(new_anchor) == Anchor, type(new_anchor)
        self.evaluators.add(new_anchor)

        newly_anchored_patients = set()
        for a in new_anchor.getMembers():
            if a in self.wordShelf:
                newly_anchored_patients |= (set(self.wordShelf[a.lstrip('!')]) & self.backend.validate_patient_set)
                print len(set(self.wordShelf[a])), 'intersect', len(self.backend.validate_patient_set), '=', len(newly_anchored_patients)
                #print set(self.wordShelf[a])
                #print self.backend.validate_patient_set
            else:
                print "anchor", a, "not indexed!"
                sys.exit()

        self.evaluatorPatients[new_anchor.id] = newly_anchored_patients
        self.dumpEvaluators()
        try:
            self.do_evaluation()
        except:
            pass

    def removeEvaluator(self, anchorid):
        print >> self.backend.parent.logfile, str(time.time())+' removed evaluator', anchorid, self.name
        self.backend.parent.logfile.flush()
        #find an anchor with the same name
        print 'removing id', anchorid
        for anchor in  [a for a in self.evaluators if a.id == anchorid]:
            print 'removing anchor', anchorid
            self.evaluators.remove(anchor)
            self.evaluatorPatients[anchorid] = set()

        self.dumpEvaluators()

    def do_evaluation(self):
        patients = []
        print 'there are', len(union(self.evaluatorPatients.values())), "evaluator patients"
        for pid in union(self.evaluatorPatients.values()):
            print pid, 'is a positive case'
            patients.append((self.ranking[pid], pid))

        patients.sort()
        self.threshold = patients[int((1-self.recall)*len(patients))][0]

        self.pos_patients = patients[int((1-self.recall)*len(patients)):]
        random.shuffle(self.pos_patients)
        self.targets = [p[1] for p in self.pos_patients[:10]]
        
        print "evaluated!"
        #print 'precision:', self.prec
        print 'recall:', self.recall
        print 'threshold', self.threshold


    def get_precision(self):
        total = 0
        pos = 0
        print 'getting precision'
        for r,pid in self.pos_patients:
            if pid in self.human_labels:
                if self.human_labels[pid] > 0:
                    pos += 1
                    total += 1
                elif self.human_labels[pid] < 0:
                    total += 1
        
        if total == 0:
            return '?'

        if all([pid in self.human_labels for pid in self.targets]):
            print "complete evaluation!"
            self.evaluations.append(pos/float(total))
        return str(pos) + '/' + str(total)



    def get_recall(self):
        return self.recall

    def removeAnchor(self, anchorid):
        print >> self.backend.parent.logfile, str(time.time())+' removed anchor', anchorid, self.name
        self.backend.parent.logfile.flush()
        #find an anchor with the same name
        print 'removing id', anchorid
        print 'here are ids', [a.id for a in self.anchors]
        for anchor in  [a for a in self.anchors if a.id == anchorid]:
            print 'removing anchor', anchorid
            self.anchors.remove(anchor)
            anchored_patients = self.anchoredPatients[anchor.id]
            for pid in anchored_patients:
                try:
                    i = self.patient_index[pid]
                except:
                    continue
                self.Y_counts[i] -= 1

                if anchor.id[0] == '!':
                    self.Y_negCounts[i] -= 1

                assert self.Y_counts >= 0, "Y_counts negative?"
                assert self.Y_negCounts >= 0, "Y_negCounts negative?"

                self.Y[i]= int(self.Y_counts[i] > 0 and self.Y_negCounts[i] == 0)

                for a in anchor.getExclusions():
                    if not a in self.inv_vocab:
                        continue

                    j = self.inv_vocab[a]
                    if i in self.masked_elements[j]:
                        self.sparse_X[i,j] = 1
                        self.masked_elements[j].remove(i)

        #self.configureLearnButton('disabled')
        self.pool.apply_async(update_sparse_X, args=[self.sparse_X], callback=self.done_updating)
        self.dumpAnchors()


    def doLearning(self):      
        C = [10**(k) for k in xrange(-4,4)]
        params = [{'C':C, 'penalty':['l1'],}]
        print "learning!"
        print >> self.backend.parent.logfile, str(time.time())+' learning' , self.name
        self.backend.parent.logfile.flush()
        s = time.time()

        if self.online:
            #Learner=SGDClassifier(loss='log', alpha=0.0001)
            Learner=GridSearchCV(LogisticRegression(), params, cv=3, scoring='log_loss')
        else:
            Learner=GridSearchCV(LogisticRegression(), params, cv=3, scoring='log_loss')

        X = self.sparse_X_csr
        while X == None:
            time.sleep(1)
            print 'waiting for sparse csr'
            X = self.sparse_X_csr

        print 'transform', time.time() -s
        print 'pos examples', sum(self.Y)
        print 'pos features', X.sum()
        try:
            Learner.fit(X, self.Y)
            print 'best params', Learner.best_params_
            print 'grid scores', Learner.grid_scores_
            Learner = Learner.best_estimator_
            
        except:
            print "could not learn!"
        
        self.estimator = Learner
        self.dumpDecisionRule()
        print 'fit', time.time() -s
        self.predictions = self.sparse_X_csr * Learner.coef_.T + Learner.intercept_
        print 'predict', time.time() -s
        self.predictions = np.exp(self.predictions) / (1+np.exp(self.predictions))
        print 'scale', time.time() -s
        

        self.eval_predictions = self.sparse_X_csr_eval * Learner.coef_.T + Learner.intercept_
        self.eval_predictions = np.exp(self.eval_predictions) / (1+np.exp(self.eval_predictions))


        self.ranking = zip([pat['index'] for pat in self.patient_list], np.ravel(self.predictions).tolist())
        self.ranking += zip(self.backend.validate_patient_ids, np.ravel(self.eval_predictions).tolist())
        self.ranking = dict(self.ranking)


        print 'rank', time.time() -s
        print "done"

        try:
            self.do_evaluation()
            print 'evaluating new model'
        except:
            pass

    def getSuggestions(self):
        suggestions = []
        try:
            return filter(noPunctuation, sorted(zip(self.vocab, self.estimator.coef_[0]), key=lambda e: e[1], reverse=True))

        except:
            return []

    
    def tag_patient(self, patid, tagval):
        if tagval == 0:
            if patid in self.human_labels:
                del self.human_labels[patid]

        else:
            self.human_labels[patid] = tagval
        
        print >> self.backend.parent.logfile, str(time.time())+' tagged patient', self.name, patid, tagval
        self.backend.parent.logfile.flush()
        self.dumpLabels()


class Backend:
    def __init__(self, parent, loadfile):
        self.settings = parent.settings
        self.parent = parent
        self.concepts = {}
        self.patients = {}
        self.validate_patients = set()
        self.visitShelf = None
        self.wordShelf = None
        self.patientList = []
        self.patientIndex = {}
        self.visitIDs = None
        self.sparse_X = None
        self.saveloc = self.settings.find('anchors').attrib['loc']

        print "loading vocab"
        self.vocab,  self.inv_vocab, self.display_vocab = pickle.load(file(self.settings.find('./vocab').attrib['src']))
        print "done"
        if not loadfile:
            print 'init patients'
            self.initPatients()
            print 'init anchors'
            self.initAnchors()
            print 'done'
        else:
            print "loading file", loadfile
            self.doLoad(loadfile)

    def doIndexing(self, anchor):
        for a in anchor.getMembers():
            a = a.lstrip('!')
            if not a in self.wordShelf:
                print 'indexing', a

                split_a = a.split()
                split_a_set = set(split_a)
                indexed = set()

                if len(split_a) > 1: #only index compound words
                    print a, 'is a compound word'
                    for p in self.patientList + self.validate_patient_list:
                        if split_a_set.issubset(set(p['Text'].split())):
                            print "it is!"
                            for f in p.keys():

                                if not 'parsed' in f:
                                    continue

                                for i in xrange(len(p[f])-len(split_a)):
                                    match = True
                                    for j in xrange(len(split_a)):
                                        print 'compare', split_a[j], p[f][i+j]['repr']
                                        if not split_a[j] in p[f][i+j]['repr']:
                                            match = False
                                            break
                                    if match:
                                        print 'match!', p['index']
                                        indexed.add(p['index']) 

                                        for j in xrange(len(split_a)):
                                            p[f][i+j]['repr'].append(a)

                                        self.visitShelf[p['index']] = p
                                        print 'indexed!', p['index']

                self.wordShelf[a] = indexed

        self.visitShelf.sync()
        self.wordShelf.sync()
        print 'done'

    def getActiveConcept(self):
            return self.concepts[self.parent.currentConcept]


    def initPatients(self, patientSet="train"):
        
        visitIDs = file(self.settings.find('./patients').attrib['src'])
        self.visitShelf = shelve.open(self.settings.find('./patients').attrib['shelf'])
        self.wordShelf = shelve.open(self.settings.find('./vocab').attrib['shelf'])
        
        start = int(filter(lambda s: s.attrib['name'] == "train", self.settings.findall('./patientSets/set'))[0].attrib['start'])
        end = int(filter(lambda s: s.attrib['name'] == "train", self.settings.findall('./patientSets/set'))[0].attrib['end'])

        visit_ids = [z.strip() for z in visitIDs.readlines()[start:end]]

        self.visitIDs = visit_ids
        print "reading in patients", len(visit_ids)

        print 'from shelve'
        sparse_X = []
        s = time.time()
        for i,v in enumerate(self.visitIDs):
            if i%1000 == 0:
                print i, time.time() - s
                if i > end:
                    break
            pat = self.visitShelf[v]
            pat['anchors'] = set()
            self.patients[v] = pat
            sparse_X.append(pat['sparse_X'])
    
        #print self.patients.keys()
        self.sparse_X = sparse.vstack(sparse_X, 'lil')

        self.train_patient_ids = visit_ids


        self.patientList = [self.patients[v] for v in self.visitIDs]
        self.patientIndex = dict(zip([pat['index'] for pat in self.patientList], xrange(len(self.patientList))))

        visitIDs.seek(0)
        start = int(filter(lambda s: s.attrib['name'] == "validate", self.settings.findall('./patientSets/set'))[0].attrib['start'])
        end = int(filter(lambda s: s.attrib['name'] == "validate", self.settings.findall('./patientSets/set'))[0].attrib['end'])
        visit_ids = [z.strip() for z in visitIDs.readlines()[start:end]]
        self.validate_patient_set = set(visit_ids)
        self.validate_patient_ids = visit_ids
        self.validate_patient_list = []
        print "reading in validate patients", len(visit_ids)

        print 'from shelve'
        sparse_X_validate = []
        s = time.time()
        for i,v in enumerate(visit_ids):
            if i%1000 == 0:
                print i, time.time() - s
                if i > end:
                    break
            pat = self.visitShelf[v]
            pat['anchors'] = set()
            self.patients[v] = pat
            self.validate_patient_list.append(pat)
            sparse_X_validate.append(pat['sparse_X'])
    
        self.sparse_X_validate = sparse.vstack(sparse_X_validate, 'lil')

    def initAnchors(self):
        conceptList = self.parent.conceptListbox
        anchorfilename = self.settings.find('./anchors').attrib['src']
        self.concepts = readAnchors(anchorfilename, self, 0)
        for concept in sorted(self.concepts.values(), key=lambda c: c.name):
            conceptList.insertConcept(concept.name, concept.id)


    def initConcept(self, concept, online=True):
        concept = self.concepts[concept]
        concept.online = online
        if concept.initialized == True:
            return True

        if not concept.loadState(self, self.wordShelf):
            for a in concept.anchors:
                self.doIndexing(a)

            concept.initPatients(self.patients, self.wordShelf, self.vocab, self.inv_vocab, self.display_vocab)
            concept.initLog()
            concept.initRepresentation(self.patientList, self.sparse_X)
            try:
                concept.doLearning()
            except:
                print "could not learn concept"
            #concept.saveState()
        else:
            print 'loaded from pickle'
        concept.initialized=True
        self.parent.conceptListbox.activateConcept(concept.name)
        return True

    def newConcept(self, name):
        concept =  Concept(name, [], parent=self, saveloc=self.saveloc+'/'+name+'.pk')
        self.concepts[name] = concept
        self.parent.conceptListbox.insertConcept(concept.name, concept.id)
    
    def delete_concept(self, name):
        try:
            del self.concepts[name]
            print >> self.parent.logfile, str(time.time())+' deleted concept', name
        except:
            print 'could not delete concept', name

        try:
            os.remove(self.saveloc +'/'+name+'.pk')
            os.remove(self.saveloc +'/'+name+'.labels.pk')
        except:
            print 'could not delete files for concept', name

    def rename_concept(self, oldname, newname):
        print 'renaming', oldname, 'as', newname
        self.concepts[newname] = self.concepts[oldname]
        self.concepts[oldname] = None
        self.concepts[newname].set_name(newname)
        print >> self.parent.logfile, str(time.time())+' renamed concept', oldname, newname
        
        try:
            os.rename(self.saveloc +'/'+oldname+'.pk', self.saveloc+'/'+newname+'.pk')
            os.rename(self.saveloc +'/'+oldname+'.labels.pk', self.saveloc+'/'+newname+'.labels.pk')
        except:
            print 'could not move files for concept', oldname, 'to', newname


        
