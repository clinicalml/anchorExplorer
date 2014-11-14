from Tkinter import *
import cPickle as pickle
from ttk import *
from helpers import *
from Anchors import Anchor
from Structures import Structure 
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt



def tv_sort(display, col, root):
    l = [(display.set(k, 'repr'), k) for k in display.get_children(root)]
    l.sort()

# rearrange items in sorted positions
    for index, (val, k) in enumerate(l):
        display.move(k, root, index)
        tv_sort(display, col, k)

###################################
# Concept list box
#     handles selecting concepts
#     adding new concepts
#     deleting concepts
###################################
class ConceptListbox:
    def __init__(self, parent, root, side=TOP, fill=BOTH):
        self.parent=parent
        self.treeview=Treeview(parent, columns=['conceptID'], displaycolumns=[])
        self.root = root
        self.treeview.pack(side=side, fill=fill)

        self.concepts = {}

        #BINDINGS
        self.treeview.bind('<<TreeviewSelect>>', self.onConceptSelect)
        self.treeview.bind('-', self.deleteConcept)
        self.treeview.bind('r', self.initiateRenameConcept)
        self.treeview.tag_configure('grey', foreground='gray')
        self.treeview.tag_configure('black', foreground='black')
        
    def onConceptSelect(self, event):
        for p in event.widget.selection():
            conceptID = event.widget.item(p)['text']
            print "selecting", conceptID
            self.root.displayConcept(conceptID)

    def insertConcept(self, name,iden):
        self.concepts[name] = self.treeview.insert("", END, text=name, values=[id], tags=['grey'])
        

    def activateConcept(self,name):
        self.treeview.item(self.concepts[name], tags=['black'])

    def deleteConcept(self, event):
        for p in event.widget.selection():
            conceptID = event.widget.item(p)['text']
            print "deleting", conceptID
            self.root.backend.delete_concept(conceptID)
            self.treeview.delete(p)
    
    def initiateRenameConcept(self, event):
        for p in event.widget.selection():
            conceptID = event.widget.item(p)['text']
            print "renaming", conceptID
            self.renameConceptWindow(conceptID)


    def renameConceptWindow(self, oldConcept):
        display = Tk()
        display.title('RENAME')
        new_window = PanedWindow(display, orient=VERTICAL)
        new_window.pack(fill=BOTH, expand=1)
        label = Label(new_window, text = "Enter the new name")
        new_window.add(label)
        l = Entry(new_window)

        def renameConcept(event):
            newname = event.widget.get().lower()
            self.root.backend.rename_concept(oldConcept, newname)
            event.widget.master.master.destroy()
            self.treeview.item(self.concepts[oldConcept], text=newname)
            self.concepts[newname] = self.concepts[oldConcept]
            self.concepts[oldConcept] = None

            
        l.bind("<Return>", renameConcept)
        new_window.add(l)

class StructuredAnchorDisplay:
    def __init__(self, parent, root, source):
        self.parent = parent
        self.source = source
        self.root = root
        self.structure = pickle.load(file(source))
        self.display=Treeview(parent, columns=['repr'], displaycolumns=[])
        self.display.pack(fill=BOTH, expand=1)

        self.ref = {"":""}
        for repr,disp,p in self.structure.getStructure():
            try:
                self.ref[repr] = self.display.insert(self.ref[p], END, text=disp, val=[repr])
            except:
                pass

        root = self.display.get_children('')[0]
        tv_sort(self.display, 'repr', root)
        self.display.bind('+', self.selectAnchor) 
        #self.display.bind('E', self.selectEvaluator) 

    def selectAnchor(self, event):
        for p in event.widget.selection():
            anchor_name = event.widget.item(p)['values'][0]
            print 'new structured anchor', anchor_name
            nodes,names,edges = self.structure.getDescendents(anchor_name)
            new_anchor = Anchor(str(anchor_name), nodes, edges,names)

        self.root.backend.getActiveConcept().addAnchor(new_anchor)
        self.root.displayConcept()

    def selectEvaluator(self, event):
        for p in event.widget.selection():
            anchor_name = event.widget.item(p)['values'][0]
            print 'new structured evaluator', anchor_name
            nodes,names,edges = self.structure.getDescendents(anchor_name)
            new_anchor = Anchor(str(anchor_name), nodes, edges,names)

        self.root.backend.getActiveConcept().addEvaluator(new_anchor)
        self.root.displayConcept()

    def open(self, iden):
        item = self.ref[iden]
        self.display.see(item)
        self.display.selection_set(item)
        self.display.see(item)
        self.display.selection_set(item)

class AnchorDisplay:
    def __init__(self, parent, root, side=TOP, fill=BOTH):
        self.parent = parent
        self.root = root
        self.Windows = PanedWindow(parent, orient=HORIZONTAL)
        self.anchorDisplay = Treeview(self.Windows, columns=['repr'], displaycolumns=[])
        self.anchorDisplay.pack(side=LEFT, fill=BOTH, expand=1)
        self.anchorDisplay.heading('#0', text='anchors')
        #self.evalAnchorDisplay = Treeview(self.Windows, columns=['repr'], displaycolumns=[])
        #self.evalAnchorDisplay.pack(side=LEFT, fill=BOTH, expand=1)
        #self.evalAnchorDisplay.heading('#0', text='evaluators')

        self.selection = 0
        self.anchorSuggestions = Notebook(self.Windows)
        self.anchorSuggestions.pack(side=RIGHT, fill=BOTH, expand=1)
        self.Windows.add(self.anchorDisplay)
        self.Windows.add(self.anchorSuggestions)
        #self.Windows.add(self.evalAnchorDisplay)
        self.frames = {}
        self.trees = {}
        settings = root.settings

        self.dataTypes = [field.attrib['type'] for field in root.settings.findall('./dataTypes/datum')]
        self.prefixes =  [field.attrib['prefix'] for field in root.settings.findall('./dataTypes/datum')]

        f = self.frames['suggest'] = Frame(self.anchorSuggestions)
        self.anchorSuggestions.add(f, text='Suggestions')
        self.trees['suggest'] = t = Treeview(f, columns=['repr', 'weight'], displaycolumns=[])
        t.pack(fill=BOTH, expand=1)

        frame_list = [d for d in settings.findall('dataTypes/datum') if not d.attrib['heirarchy']=='']
        for frame in frame_list:
            f = self.frames[frame.attrib['type']] = Frame(self.anchorSuggestions)
            self.anchorSuggestions.add(f, text=frame.attrib['type'])
            self.trees[frame.attrib['type']] = StructuredAnchorDisplay(f, root, frame.attrib['heirarchy'])
            

        self.anchorEntry = Entry(parent)
        parent.add(self.Windows)
        parent.add(self.anchorEntry)
        buttonWindow = PanedWindow(parent, orient=HORIZONTAL)
        self.learnButton = Button(buttonWindow, text="Learn!", command=self.doLearning)
        buttonWindow.add(self.learnButton)
        self.learnButton.pack(side=LEFT)

        #self.evaluateButton = Button(buttonWindow, text="showEval!", command=self.showEval)
        #buttonWindow.add(self.evaluateButton)
        #self.evaluateButton.pack(side=LEFT)

        #self.learnButton = Button(buttonWindow, text="Evaluate", command=self.doEvaluation)
        #buttonWindow.add(self.learnButton)
        #self.learnButton.pack(side=LEFT)
        self.parent.add(buttonWindow)
        
        #BINDINGS
        self.anchorEntry.bind("<Return>", self.anchorAdded)
        self.anchorDisplay.bind("-", self.anchorRemoved)
        self.anchorDisplay.tag_configure("red", foreground="red")
        #self.evalAnchorDisplay.bind("-", self.evalRemoved)
        self.anchorDisplay.bind("<<TreeviewSelect>>", self.updateSelection)
        self.trees['suggest'].bind('+', self.suggestionAccepted)
        self.trees['suggest'].bind('E', self.evaluatorAccepted)
        for datatype in self.dataTypes:
            self.trees['suggest'].tag_bind(datatype, "g", self.gotoCode)

    def updateSelection(self, event):
        for p in event.widget.selection():
            self.selection = event.widget.item(p)['values'][0]
            if self.root.displayMode.get() == 'select':
                self.root.patientListDisplay.displayPatients()
                self.root.patientDetailDisplay.clear()

    def gotoCode(self, event):
        for p in event.widget.selection():
            tags = event.widget.item(p)['tags']
            datatype = ""
            val = ""
            for t in tags:
                t = str(t)
                for d in self.dataTypes:
                    if t == d:
                        datatype = d

                for p in self.prefixes:
                    if len(p) and p in t:
                        val = t

            self.showSuggestion(datatype, val)

    def showAnchors(self, conceptID):
        concept = self.root.backend.concepts[conceptID]
        
        for c in self.anchorDisplay.get_children():
            self.anchorDisplay.delete(c)
        
        #for c in self.evalAnchorDisplay.get_children():
         #   self.evalAnchorDisplay.delete(c)

        ref = {"":""}
        
        for anch in concept.anchors:
            for a,name,p in anch.getStructure():
                print 'adding anchor', a,name,p
                tags = []
                if name[0] == '!':
                    tags.append('red')
                ref[a] = self.anchorDisplay.insert(ref[p], END, text=name, values=[a], tags=tags)
                if a == self.selection:
                    self.anchorDisplay.selection_set(ref[a])

        #for anch in concept.evaluators:
            #for a,name,p in anch.getStructure():
                #ref[a] = self.evalAnchorDisplay.insert(ref[p], END, text=name, values=[a])
                #if a == self.selection:
                    #self.evalAnchorDisplay.selection_set(ref[a])

        for c in self.trees['suggest'].get_children():
            self.trees['suggest'].delete(c)

        for word,weight in concept.getSuggestions():
            if '\'' in word or '\"' in word:
                continue
            txt,tags = self.getDisplayVersion(word)
            prefix = ""
            if '_' in word:
                prefix = word.split('_')[0]+'_'

            self.trees['suggest'].insert('', END, text=prefix+txt, values=(word, weight), tags=tags)
            
    def doLearning(self):
        self.root.backend.getActiveConcept().doLearning()
        self.root.displayConcept()

    def showEval(self):
        evaluations = self.root.backend.getActiveConcept().evaluations
        recall = self.root.backend.getActiveConcept().recall
        plt.plot(evaluations, '*-')
        plt.xlabel('steps')
        plt.ylabel('prec@'+str(recall))
        plt.show()


    def getDisplayVersion(self, word):
        for datatype,dct in self.root.dictionaries:
            if word in dct:
                return dct[word], [word, datatype]
        else:
            return word, []

    def anchorAdded(self, event):
        new_anchor = Anchor(event.widget.get().lower())
        self.anchorEntry.delete(0,END)
        self.root.backend.getActiveConcept().addAnchor(new_anchor)
        self.root.displayConcept(self.root.currentConcept)


    def suggestionAccepted(self, event):
        for p in event.widget.selection():
            repr = (str(event.widget.item(p)['values'][0]))
            disp = (str(event.widget.item(p)['text']))
            new_anchor = Anchor(repr, members=[repr], display_names=[disp]) 
            self.root.backend.getActiveConcept().addAnchor(new_anchor)
        self.root.displayConcept(self.root.currentConcept)

    def evaluatorAccepted(self, event):
        for p in event.widget.selection():
            repr = (str(event.widget.item(p)['values'][0]))
            disp = (str(event.widget.item(p)['text']))
            new_anchor = Anchor(repr, members=[repr], display_names=[disp]) 
            self.root.backend.getActiveConcept().addEvaluator(new_anchor)
        self.root.displayConcept(self.root.currentConcept)

    def anchorRemoved(self, event):
        for p in event.widget.selection():
            anchor = event.widget.item(p)['values'][0]
            self.root.backend.getActiveConcept().removeAnchor(anchor)
            self.root.displayConcept(self.root.currentConcept)

    def evalRemoved(self, event):
        for p in event.widget.selection():
            anchor = event.widget.item(p)['values'][0]
            self.root.backend.getActiveConcept().removeEvaluator(anchor)
            self.root.displayConcept(self.root.currentConcept)        

    def showSuggestion(self, datatype, val):
        tab = self.frames[datatype]
        self.anchorSuggestions.select(tab)
        self.trees[datatype].open(val)
        













class PatientDetailDisplay:
    def __init__(self, parent, root, side=TOP, fill=BOTH):
        #middle listbox -- patient representation
        self.parent = parent
        self.root = root
        self.patientDetails = Text(parent, wrap=WORD)
        self.patientDetails.pack(side=TOP, fill=X, expand=0)
        self.settings = root.settings
        self.displayFields = [field.attrib['name'] for field in root.settings.findall('./displaySettings/detailedDisplay/displayFields/field')]
        self.dataTypes = [field.attrib['type'] for field in root.settings.findall('./dataTypes/datum')]
        self.prefixes =  [field.attrib['prefix'] for field in root.settings.findall('./dataTypes/datum')]
        parent.add(self.patientDetails)

    def clear(self):
        self.patientDetails.delete(1.0,END)


    def gotoCode(self, event):
        x,y = event.x, event.y
        tags = event.widget.tag_names("@%d,%d" % (x, y))
        datatype = ""
        val = ""
        for t in tags:
            t = str(t)
            for d in self.dataTypes:
                if t == d:
                    datatype = d

            for p in self.prefixes:
                if len(p) and p in t:
                    val = t

            self.root.anchorDisplay.anchorSuggestions.focus_set()
            self.root.anchorDisplay.showSuggestion(datatype, val)



    def displayPatient(self, id):

        currentConcept = self.root.currentConcept
        self.clear()
        
        self.patientDetails.tag_config("red", foreground="red")
        self.patientDetails.tag_config("blue", foreground="blue")
        self.patientDetails.tag_config("purple", foreground="purple")
        
        for datatype in self.dataTypes:
            self.patientDetails.tag_config(datatype, underline=1)
            self.patientDetails.tag_bind(datatype, "<Enter>", show_hand_cursor)
            self.patientDetails.tag_bind(datatype, "<Leave>", show_arrow_cursor)
            self.patientDetails.tag_bind(datatype, "<Button-1>", self.gotoCode)
        
        pat = self.root.backend.patients[id]
        
        for field in self.displayFields:
            self.patientDetails.insert(END, field+': ')
            try:
                txt = pat[field+'_parsed']
            except Exception, e:
                print 'error?', e
                continue 
            for w in txt:
                tags = []
                if len(set(w['repr']) & union(a.getMembers() for a in self.root.backend.concepts[currentConcept].anchors)):
                    tags.append('red')


                spacer = ' '
                for prefix, datatype in zip(self.prefixes, self.dataTypes):
                    if prefix == "":
                        continue
                    if any([prefix in r for r in w['repr']]):
                        tags.append(datatype)
                        for r in w['repr']:
                            tags.append(r)
                        spacer = '\n'


                for r in w['repr']:
                    if 'negxxx' in r:
                        tags.append('purple')

                self.patientDetails.insert(END, w['disp'], tuple(tags))
                self.patientDetails.insert(END, spacer)


                #for r in w['repr']:
                #    self.patientDetails.insert(END, r+' ', ('blue', ))

            self.patientDetails.insert(END, '\n'+'-'*50+'\n')

        if id in self.root.backend.validate_patient_set:
            self.patientDetails.insert(END, 'VALIDATE PATIENT\n'+'-'*50+'\n')
        else:
            self.patientDetails.insert(END, 'TRAIN PATIENT\n'+'-'*50+'\n')       

        if id in self.root.backend.getActiveConcept().flagged_patients:
            note = self.root.backend.getActiveConcept().flagged_patients[id]
            self.patientDetails.insert(END, '\nNote:'+note+'\n-'*50+'\n')       


class PatientListDisplay:
    def __init__(self, parent, root, side=TOP, fill=BOTH):
        #bottom listbox -- patient representation
        self.parent = parent
        self.root = root
        self.patientList = Treeview(parent, columns=['pid'], displaycolumns=[])
        self.patientList.pack(side=TOP, fill=BOTH, expand=1)

        scrollbar = Scrollbar(self.patientList)
        scrollbar.pack(side=RIGHT,fill=Y)
        self.patientList.configure(yscroll=scrollbar.set)
        scrollbar.config(command=self.patientList.yview)
        parent.add(self.patientList)
        self.summaryFields = [field.attrib['name'] for field in root.settings.findall('./displaySettings/patientSummary/displayFields/field')]

        #tags
        self.patientList.tag_configure("red", foreground="red")
        self.patientList.tag_configure("blue", foreground="blue")
        self.patientList.tag_configure("green", foreground="green")
        
        #bindings
        self.patientList.bind('<<TreeviewSelect>>', self.onPatientSelect)
        self.patientList.bind('+', self.posTagPatient)
        self.patientList.bind('-', self.negTagPatient)
        self.patientList.bind('0', self.unTagPatient)
        self.patientList.bind('f', self.flagPatient)


    def onPatientSelect(self, event):
        for p in event.widget.selection():
            pid= event.widget.item(p)['values'][0]
            self.root.patientDetailDisplay.displayPatient(pid)
            
    def addPatientToDisplay(self, pat, showPrediction=True):
        if pat==None:
            print 'adding empty patients'
            self.patientList.insert("", END, text="", values=(""))
            return

        pat_description = " ".join([ET.fromstring(pat[field]).text for field in self.summaryFields]) + ' : '+ ",".join(pat['anchors'] - set([self.root.currentConcept]))
        try:
            if showPrediction:
                pat_description = "{:.3f}".format(self.root.backend.getActiveConcept().ranking[pat['index']]) +': '+ pat_description
        except  Exception as inst:
            #print inst
            pass

        tags = []

        try:
            if self.root.backend.getActiveConcept().human_labels[pat['index']] > 0:
                tags = ['green']
            if self.root.backend.getActiveConcept().human_labels[pat['index']] < 0:
                tags = ['red']
        except:
            pass

        self.patientList.insert("", END, text=pat_description, values=(pat['index']), tags=tags)
            

    def displayPatients(self):
        listbox = self.patientList

        for c in listbox.get_children():
            listbox.delete(c)

        
        print "displaying patients"
        ranking = self.root.backend.getActiveConcept().ranking
        train_patients = set(self.root.backend.train_patient_ids)
        validate_patients = self.root.backend.validate_patient_set
        
        #if ranking:
        #    assert set(ranking.keys()) == all_patients
        
        displayMode = self.root.displayMode.get()
        if displayMode == 'recent':
            target_patients = self.root.backend.getActiveConcept().recentPatients & validate_patients
        elif displayMode == 'select':
            s = self.root.anchorDisplay.selection
            target_patients = set(self.root.backend.getActiveConcept().anchoredPatients[s]) & validate_patients
            
        else:
            anchors = self.root.backend.getActiveConcept().anchors
            print 'anchors'
            for a in anchors:
                print a, a.id
            anchored_patients = union(self.root.backend.getActiveConcept().anchoredPatients[a.id] for a in anchors) & validate_patients - union(self.root.backend.getActiveConcept().anchoredPatients[a.id] for a in anchors if a.id[0] == '!')
            if displayMode == 'filter':
                target_patients = anchored_patients
            elif displayMode == 'sort':
                target_patients = validate_patients - anchored_patients - union(self.root.backend.getActiveConcept().anchoredPatients[a.id] for a in anchors if a.id[0] == '!')
            elif displayMode == 'label':
                target_patients = self.root.backend.getActiveConcept().targets
            else:
                print "unknown display mode"

        patients = self.root.backend.patients
        try:
            patient_order =  sorted(target_patients, key=lambda pid: ranking[pid], reverse=True)
        except:
            patient_order = target_patients

        for pid in patient_order:
            try:
                pat = patients[pid]
            except:
                continue
            self.addPatientToDisplay(pat)
        
        if len(patient_order) == 0:
            for _ in xrange(10):
                self.addPatientToDisplay(None)
        

    def showRecentPatients(self, conceptID):
        listbox = self.toplistbox
        listbox.delete(0, END)
        self.toplistIDs =[]
        self.middletextbox.delete(1.0,END)
        if conceptID == 'all':
            anchors = set()
        else:
            anchors = set(self.anchors[conceptID])

        for pat in self.recentPatients[self.currentConcept]:
            pat = self.patients[pat]
            if self.currentConcept in pat['anchors']:
                self.addPatientToTopList(pat)

    def showEvalPatients(self, conceptID):
        self.evaluationIndex += 1
        listbox = self.toplistbox
        listbox.delete(0, END)
        self.toplistIDs =[]
        self.middletextbox.delete(1.0,END)
        if conceptID == 'all':
            anchors = set()
        else:
            anchors = set(self.anchors[conceptID])

        ids = self.test_patients.keys()
        denom = 0.0
        inv_denom = 0.0
        if (not self.currentConcept in self.weights) or self.weights[self.currentConcept] == None:
            print "cannot evaluate. The predictor is out of date"
            return 

        for patid in ids:
            pat = self.test_patients[patid]
            pred =  predict(self, self.currentConcept, pat['sparse_X'])
            if not 'predictions' in pat:
                pat['predictions'] = {}

            pat['predictions'][self.currentConcept] = pred
            pat['anchors'] = set()
            denom += pred
            inv_denom += 1-pred
        

        odds = []
        for patid in ids:
            pat = self.test_patients[patid]
            pred = pat['predictions'][self.currentConcept] 
            odds.append(0.5*pred/denom + 0.5*(1-pred)/inv_denom)

            if not 'weighting' in pat:
                pat['weighting'] = {}

            pat['weighting'][self.currentConcept] = odds[-1]

        for i in xrange(50):
            r = np.random.choice(xrange(len(odds)), p=odds)
            patid = ids[r]
            pat = self.test_patients[patid]

            if not pat['index'] in self.human_labels[self.currentConcept]:
                self.addPatientToTopList(pat, showPrediction=False)

    def posTagPatient(self, event):
        for p in event.widget.selection():
            self.patientList.item(p, tags=['green'])
            id = self.patientList.item(p)['values'][0]
            self.root.backend.getActiveConcept().tag_patient(id, 1)
        self.root.showStats()

    def negTagPatient(self, event):
        for p in event.widget.selection():
            self.patientList.item(p, tags=['red'])
            id = self.patientList.item(p)['values'][0]
            self.root.backend.getActiveConcept().tag_patient(id, -1)
        self.root.showStats()
    
    def unTagPatient(self, event):
        for p in event.widget.selection():
            self.patientList.item(p, tags = [])
            id = self.patientList.item(p)['values'][0]
            if id in  self.root.backend.getActiveConcept().human_labels: 
                self.root.backend.getActiveConcept().tag_patient(id, 0)
        self.root.showStats()

    def flagPatient(self, event):
        for p in event.widget.selection():
            patid = self.patientList.item(p)['values'][0]
            self.flagPatientWindow(patid)
        self.root.backend.getActiveConcept().dumpFlags()   
        
    def flagPatientWindow(self, patid):
        display = Tk()
        display.title('flag patient')
        new_window = PanedWindow(display, orient=VERTICAL)
        new_window.pack(fill=BOTH, expand=1)
        label = Label(new_window, text = "Insert comment here:")
        new_window.add(label)
        l = Entry(new_window)

        def reportFlag(event):
            message = event.widget.get().lower()
            self.root.backend.getActiveConcept().flagged_patients[patid] = message
            event.widget.master.master.destroy()

            
        l.bind("<Return>", reportFlag)
        new_window.add(l)


 

