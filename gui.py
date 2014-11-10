#from Tkinter import *
import random
from copy import deepcopy
#import tkFileDialog
import itertools
from multiprocessing import Pool
import string
import ttk
import shelve
import time
import sys
import cPickle as pickle
from collections import defaultdict
import numpy as np
import re
from sklearn import metrics
from Displays import *
from Backend import *
import xml.etree.ElementTree as ET





def show_hand_cursor(event): 
    event.widget.configure(cursor="hand1") 

def show_arrow_cursor(event):
    event.widget.configure(cursor="")

def readSettings(filename):
    Tree = ET.parse(filename)
    return Tree
    
class Display:
    def __init__(self,parent, settings, loadfile=None):
        self.parent = parent
        self.currentConcept='all'
        self.recentPatients = {}
        self.displayMode=StringVar()
        self.displayMode.set('filter')
        self.nProcs = 2
        self.settings = readSettings(settings)
        self.logfile = file(self.settings.find('logfile').attrib['path'], 'a')
        self.dictionaries = []
        for dat in self.settings.findall('dataTypes/datum'):
            if not 'dictionary' in dat.attrib:
                continue
            else:
                dct = dat.attrib['dictionary']

            self.dictionaries.append((dat.attrib['type'], pickle.load(file(dct))))

        m1 = PanedWindow()
        m1.pack(fill=BOTH, expand=1)
        
        self.leftDisplay = PanedWindow(m1, orient=VERTICAL)
        m1.add(self.leftDisplay)
        
        m2 = PanedWindow(m1, orient=VERTICAL)
        m1.add(m2)

        #left pane -- anchor showing
        self.conceptListbox = ConceptListbox(self.leftDisplay, self)

        self.buttons = []
        b = Button(self.leftDisplay, text='new variable', command=self.addConceptWindow)
        self.leftDisplay.add(b)
        b.pack(side=TOP)

        self.displayString = Label(self.leftDisplay, text='')
        self.displayString.pack(side=TOP)


        b = Radiobutton(self.leftDisplay, text="view recently anchored", variable=self.displayMode, value="recent", command=self.refresh)
        self.leftDisplay.add(b)
        b.pack(side=BOTTOM)
        
        b = Radiobutton(self.leftDisplay, text="view selected anchored", variable=self.displayMode, value="select", command=self.refresh)
        self.leftDisplay.add(b)
        b.pack(side=BOTTOM)

        b = Radiobutton(self.leftDisplay, text="view all anchored", variable=self.displayMode, value="filter", command=self.refresh)

        self.leftDisplay.add(b)
        b.pack(side=BOTTOM)
        
        b = Radiobutton(self.leftDisplay, text="view not anchored", variable=self.displayMode, value="sort", command=self.refresh)
        self.leftDisplay.add(b)
        b.pack(side=BOTTOM)

        #b = Radiobutton(self.leftDisplay, text="do labeling", variable=self.displayMode, value="label", command=self.refresh)
        #self.leftDisplay.add(b)
        #b.pack(side=BOTTOM)

        self.anchorDisplay = AnchorDisplay(m2, self)
        self.patientDetailDisplay = PatientDetailDisplay(m2, self)
        self.patientListDisplay = PatientListDisplay(m2, self)
        self.backend = Backend(self, loadfile)



    def displayConcept(self, conceptID=None):
        if conceptID == None:
            conceptID = self.currentConcept
        else:
            self.currentConcept = conceptID

        self.backend.initConcept(conceptID)
        self.anchorDisplay.showAnchors(conceptID)
        self.patientListDisplay.displayPatients()
        self.patientDetailDisplay.clear()
        self.showStats()


    def showStats(self):
        displayString = ""
        displayString += "current var is "+ self.currentConcept+'\n'
        displayString += 'anchored patients: ' +str(len(union(self.backend.concepts[self.currentConcept].anchoredPatients.values()))) +'\n'
        displayString += 'hand labeled patients: ' + str(len(self.backend.concepts[self.currentConcept].human_labels.keys())) +'\n'
        #displayString += 'evaluator patients: ' + str(len(union(self.backend.concepts[self.currentConcept].evaluatorPatients.values()))) +'\n'
        #displayString += 'precision@'+str(self.backend.concepts[self.currentConcept].recall)+': ' + str(self.backend.concepts[self.currentConcept].get_precision()) + '\n'
        
        self.displayString.config(text=displayString)

        
    def debug(self):
        #IPython.getipython.get_ipython().launch_new_instance({'self':self})
        print "done with debugging session"

    def calculateStats(self):
        nAnchored = 0
        for pat in self.patients.values():
            if self.currentConcept in pat['anchors']:
                self.anchored_patients[self.currentConcept].add(pat['index'])
                nAnchored += 1
            else:
                self.anchored_patients[self.currentConcept].discard(pat['index'])

        display_str = ""

        if self.currentConcept in self.weights and self.weights[self.currentConcept]:
            status = 'up to date.'
        else:
            status = 'out of date!'

        display_str += "model is "+status+'\n'
        #display_str += "validate set size "+str(self.validate_size)+'\n'
        display_str += "anchored patients="+str(nAnchored)+'\n'
        display_str += "human labels (pos/neg)= ("+str(len([i for i in self.human_labels[self.currentConcept].values() if i == 1])) + '/'
        display_str += str(len([i for i in self.human_labels[self.currentConcept].values() if i == 0])) + ')\n'
        display_str += "display size is="+str(self.nDisplay)+'\n'
        display_str += "train size is="+str(self.nTrain)+'\n'
        self.stat_str.set(display_str)

    def addConceptWindow(self):
        display = Tk()
        display.title('Add a new variable')
        new_window = PanedWindow(display, orient=VERTICAL)
        new_window.pack(fill=BOTH, expand=1)
        label = Label(new_window, text = "Enter a new variable")
        new_window.add(label)
        l = Entry(new_window)
        l.bind("<Return>", self.addConcept)
        new_window.add(l)
        
    def addConcept(self, event):
        new_concept = event.widget.get().lower()
        self.backend.newConcept(new_concept)
        self.displayConcept(new_concept)
        event.widget.master.master.destroy()

    
            
    def suggestConcept(self):
        pass


    #select a patient and display
    def patientSelect(self, event):
        for p in event.widget.curselection():
            self.displayPatient(self.toplistIDs[int(p)])

    
    def onStructuredAnchorSuggest(self, event):
        for p in event.widget.selection():
            item = event.widget.item(p)
            self.enterAnchor.delete(0,END)
            self.enterAnchor.insert(0, item['values'][0])
        event.widget.master.master.destroy()


    def refresh(self):
        self.displayConcept()


    def resetModel(self, conceptID):
        self.weights[conceptID] = None
        self.orderedPatients[conceptID] = None
        #self.weight_vectors[conceptID] = None

if __name__ == "__main__":
    root = Tk()
    try:
        settings = sys.argv[1]
    except:
        settings = 'settings.xml'
    myapp = Display(root, settings)
    root.mainloop()
