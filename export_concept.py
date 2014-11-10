import sys
import numpy as np
from helpers import union
import cPickle as pickle
from Tkinter import *
from dev_gui import Display
import shelve

def export(c, exportdir):
    weights = c.estimator.coef_[0]
    offset = c.estimator.intercept_
    vocab = c.vocab

    weightShelf = shelve.open(exportdir+'/'+c.name+'.weights', 'n')

    weightShelf['__offset__'] = offset
    for v,weight in zip(vocab, weights):
        weightShelf[v] = weight

    weightShelf.close()

    weightfile = file(exportdir+'/'+c.name+'.weights.txt', 'w')
    for weight, v in sorted(zip(weights, vocab), reverse=True):
        print >>weightfile, v, weight
    weightfile.close()

    return dict(zip(vocab + ['__offset__'], np.hstack([weights, offset])))

def calibrate(concept, backend, weights, sens=0.8):
    predictions = []

    validate_patients = backend.validate_patient_set
    anchored = union(concept.anchoredPatients.values()) & validate_patients
    for pat in backend.validate_patient_list:
        if pat['index'] in anchored:
            pred =weights['__offset__']
            for w in set(pat['Text'].split()):
                if w in weights:
                    pred += weights[w]
            predictions.append(pred)

    print 'calibrating with', len(predictions), 'elements'
    predictions.sort()
    k = int(len(predictions)*(1-sens))
    thresh = predictions[k]
    thresholdfile = file(exportdir+'/'+concept.name+'.threshold', 'w')
    print >>thresholdfile, thresh
    thresholdfile.close()

if __name__ == "__main__":
    try:
        settings = sys.argv[1]
        myconcept = ' '.join(sys.argv[2:-1]) #allows for multiword concepts
        exportdir = sys.argv[-1]
    except:
        print 'usage: export_concept.py settings conceptname exportdir'
        sys.exit()

    root = Tk()
    app = Display(root, settings)
    print 'initilized display'
    backend = app.backend
    backend.initConcept(myconcept, online=False)
    print 'initilized concept'
    weights = export(backend.concepts[myconcept], exportdir)
    print 'exported weights'
    calibrate(backend.concepts[myconcept], backend, weights)
    print 'calibrated'
