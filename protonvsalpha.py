import ROOT as rt
import numpy as np
#import matplotlib.pyplot as plt
#from tqdm import tqdm
#from itertools import chain

from IPython.display import clear_output

class PyListOfLeaves(dict):
    pass

class ProtonVsAlpha():
    def __init__(self, FILENAME, MIN_ENERGY, N_EVENTS, CHANNELS, CALIBS, CUTP, CUTA):
        self.filename = FILENAME
        self.min_energy = MIN_ENERGY
        self.n_events = N_EVENTS
        self.channels = CHANNELS
        self.calibrations = CALIBS # a dictionary, channel:[b,a], for y = a*x +b
        self.cut_alpha = CUTA # root TCutg for classification
        self.cut_proton = CUTP 
        self.training_data_class = []

    PROTON = "p" # risetime ranges for discrimination
    ALPHA = "a"
    BACK = "b"
    LABELS = {PROTON:0, ALPHA:1, BACK:2}
    #training_data_class = []
    alphacount = 0
    protoncount = 0
    
    def make_training_data(self, name):
        rf = rt.TFile(self.filename, "read")
        evt_tree = rf.Get("ggpData")
        leaves = evt_tree.GetListOfLeaves()
        x=0
        print("Reading the events on channels:",self.channels )
        #print(self.training_data_class)
        ent = evt_tree.GetEntries()
        print(ent)
        pyl = PyListOfLeaves()
        for i in range(0,leaves.GetEntries()) :
            leaf = leaves.At( i )
            names = leaf.GetName( )
            #print(names)
            pyl.__setattr__( names, leaf )
            
        for ievt in range(0,self.n_events):
            
            evt_tree.GetEntry(ievt)
            
            channel = pyl.channel.GetValue( )
            energy = pyl.energy.GetValue( )
            iMax = pyl.derivMax.GetValue( )
            baseline = pyl.baseline.GetValue()
            
            if (channel in self.channels) & (x<self.n_events) & (energy>self.min_energy):
                if ( self.cut_proton.IsInside(iMax, np.dot(self.calibrations[channel],[1,energy])) ):
                    self.protoncount +=1
                    self.training_data_class.append([(np.frombuffer(evt_tree.samples, dtype = "f" )) - baseline, 
                                                   np.eye(3)[self.LABELS["p"]] ])
                elif ( self.cut_alpha.IsInside(iMax, np.dot(self.calibrations[channel],[1,energy])) ):
                    self.alphacount +=1
                    self.training_data_class.append([(np.frombuffer(evt_tree.samples, dtype = "f" )) - baseline, 
                                                   np.eye(3)[self.LABELS["a"]] ])
                elif ( (self.cut_alpha.IsInside(iMax, np.dot(self.calibrations[channel],[1,energy])) ==0) 
                      & (self.cut_proton.IsInside(iMax, np.dot(self.calibrations[channel],[1,energy])) ==0 )):
                    self.training_data_class.append([(np.frombuffer(evt_tree.samples, dtype = "f" )) - baseline, 
                                                   np.eye(3)[self.LABELS["b"]] ])
                #print(len(self.training_data_class), self.protoncount, self.alphacount)
                if (x%1000==0):
                    clear_output()
                    print(ent)
                    print(round(ievt/self.n_events*100,3),"%")
                    print(x)
                    print("numero protoni: ",self.protoncount)
                    print("numero apha: ", self.alphacount)
                x+=1
        print("uscito!")
        np.random.shuffle(self.training_data_class)
        np.save(name+".npy", self.training_data_class)
        print("Protons:", self.protoncount)
        print("Alphas:", self.alphacount)
