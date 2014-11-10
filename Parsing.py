######## What the script does:
######## provides preprocessing functions for medical text
#########################

import re
import string
import pickle

try:
    stopwords=pickle.load(open('Stopwords.pk','rb'))
except:
    stopwords = []

abrev=[
('\.\.\.','.'),
('\+',' + '),
('\-',' - '),
('phi_phi_phi\[\*\*.{0,50}\*\*\]phi_phi_phi',''),
('/','/'),
('\. ',' . '),
(', ',' , '),
('; ',' ; '),
('/',' / '),
(' +',' ')
]


def process(line):
    res = line
    for (a,b) in abrev: 
        res=re.sub(a,b,res)
    return res


table = string.maketrans("","")
##Remove puntuation, transforms numbers (if not removed in process)
def processbis(line):
    res=line
    res = res.translate(table, string.punctuation)
    res=re.sub('0',' zero ',res)
    res=re.sub('1',' one ',res)
    res=re.sub('2',' two ',res)
    res=re.sub('3',' three ',res)
    res=re.sub('4',' four ',res)
    res=re.sub('5',' five ',res)
    res=re.sub('6',' six ',res)
    res=re.sub('7',' seven ',res)
    res=re.sub('8',' eight ',res)
    res=re.sub('9',' nine ',res)
    res=re.sub(' +',' ',res).strip()    
    return res


fullstops=['.',';', '[', '-']
midstops=['+','but','and','pt','except','reports','alert','complains','has','states','secondary','per','did','aox3']
negwords=['no','not','denies','without','non','unable']
shortnegwords = ['-']
keywords = fullstops + midstops + negwords + shortnegwords

## returns list of scopes and annotated sentence.
#Exple: Patient presents no sign of fever but complains of headaches
#Returns: [(2,5)], Patient presents no negxxxsignxxxneg negxxxofxxxneg negxxxfeverxxxneg but complains of headaches
def annotate(x):
    y=x.split()
    z=''
    flag=0
    res=[]
    for i in range(len(y)):
        wrd=y[i]
        if (wrd in fullstops or wrd in midstops) and flag==1:
            flag=0
            res+=[(a,i-1)]
        elif flag==1 and not wrd in negwords:
            y[i]='negxxx'+wrd+'xxxneg'
            if flag == 2:
                flag = 0
        if wrd in negwords:
            flag=1
            a=i
        if wrd in shortnegwords: #short negwords only last for one word
            flag = 2
            a = i
    return res,string.join(y)


try:
    stopwords=pickle.load(open('Stopwords.pk','rb'))
except:
    stopwords = []

try:
    bigramlist=pickle.load(open('Bigrams.pk','rb'))
    bigramlist = filter(lambda b: not any([k+' ' in b or ' '+k in b for k in keywords]), bigramlist)
except:
    bigramlist = []

def bigrammed(sen):
    sent=' '+sen.lower()+' '
    senlist=sen.split()
    stop=set(filter(lambda x:x in sent,stopwords))
    for w in stop:
        sent=re.sub(' '+w+' ',' ',sent)
    sent=re.sub(' +',' ',sent).strip()
    i=0
    res=''
    big=set(filter(lambda x:x in sent,bigramlist))
    while i<len(senlist):
        if i<len(senlist)-1 and senlist[i]+' '+senlist[i+1] in big:
            res+= 'bigram_'+senlist[i]+'_'+senlist[i+1]+' '
            i+=2
        elif i<len(senlist)-2 and senlist[i+1] in stop and senlist[i]+' '+senlist[i+2] in big:
            res+='bigram_'+senlist[i]+'_'+senlist[i+1]+'_'+senlist[i+2]+' '
            i+=3
        else:
            res+=senlist[i]+' '
            i+=1
    return res.strip()

def parse_text(orig_txt, prefix):
    try:
        orig_txt = re.sub('PHI_PHI_PHI.*?PHI_PHI_PHI', '',orig_txt)
    except:
        print 'cannot parse orig text', orig_txt
        sys.exit()

    orig_txt = re.sub('(['+re.escape(string.punctuation)+'])', ' \g<1> ', orig_txt)
    txt = orig_txt.lower()
    txt = process(txt)
    txt = bigrammed(txt)
    _, txt = annotate(txt)
    txt = txt.split()
    orig_txt = orig_txt.split()
    tokens = []
    for w in txt:
        if 'bigram_' in w:
           w = w.replace('bigram_', '')
        if not '_' in w:
           tokens.append((w.replace('negxxx', '').replace('xxxneg','') , [prefix+w]))
        else:
            for part in w.split('_'):
               tokens.append((part.replace('negxxx', '').replace('xxxneg','') , [prefix+w.replace('_', ' '), prefix+part]))

    return [{'disp':t[0], 'repr':t[1]} for t in tokens]



def readVisit(f):
    visit_str = ""
    index = None
    l = f.readline()
    if l == "": 
        return None

    assert "<visit>" in l

    visit_str += l
    while not "</visit>" in l:
        l = f.readline()
        visit_str += l
        if l=="":
            assert 0, "misformed file!"

    return visit_str


tagname_pat = re.compile('<([a-zA-Z0-9]+)')

def extract_tagval(l):
    return l.split('>')[1].split('<')[0]

def extract_tagname(l):
    empty = False
    key = None

    if '/>' in l:
        empty = True

    match_obj = tagname_pat.search(l)
    if match_obj:
        key = match_obj.groups(1)[0]

    endtag_pat = "</"+str(key) 
    return key, endtag_pat, empty
 
def shallow_parse_XML(incoming_text):
    data = {}
    val = ""
    key = None
    empty = False
    for l in incoming_text.split('\n')[1:-1]:
        if (key==None):
            key,endtag_pat,empty = extract_tagname(l)

        val += l+'\n'

        if empty or re.search(endtag_pat, l):
            data[key] = val.strip('\n')
            key = None
            val = ""
#    print data
    return data
