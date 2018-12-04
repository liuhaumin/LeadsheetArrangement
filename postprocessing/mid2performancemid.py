import os
import pretty_midi
import numpy as np

'''
For chord-roll
'''

####################################
#    From mid to preformance mid   #
####################################

directory = '../musegan_lpd/exps/nowbar_hybrid/gen_4dbar/all_dyn/'
if not os.path.exists(directory):
    os.makedirs(directory)
i = 0

for root, dirs, files in os.walk('../musegan_lpd/exps/nowbar_hybrid/gen_4dbar/all/', topdown=False):    
    for name in files:
        print(i)
        file = os.path.join(root, name)
        pm = pretty_midi.PrettyMIDI(file)
        print(np.shape(pm.instruments))
        offset = 30
        
        try:
            notes0 = pm.instruments[0].notes
            for note in notes0:
                a = np.random.randint(15, size=1)
                note.velocity = note.velocity -offset +30 - a[0]
        except:
            pass
        
        try:
            notes1 = pm.instruments[1].notes
            for note in notes1:
                a = np.random.randint(15, size=1)
                note.velocity = note.velocity -offset - a[0]
        except:
            pass
        
        try:
            notes2 = pm.instruments[2].notes
            for note in notes2:
                a = np.random.randint(15, size=1)
                note.velocity = note.velocity -offset - a[0]
        except:
            pass
        
        try:
            notes3 = pm.instruments[3].notes
            for note in notes3:
                a = np.random.randint(15, size=1)
                note.velocity = note.velocity -offset - a[0]
        except:
            pass
        
        try:
            notes4 = pm.instruments[4].notes
            for note in notes4:
                a = np.random.randint(15, size=1)
                note.velocity = note.velocity -offset -10 - a[0]
        except:
            pass
        
        try:
            notes5 = pm.instruments[5].notes
            for note in notes5:
                a = np.random.randint(15, size=1)
                note.velocity = note.velocity -offset - a[0]
        except:
            pass
        try:
            notes6 = pm.instruments[6].notes
            for note in notes6:
                a = np.random.randint(15, size=1)
                note.velocity = note.velocity -offset - a[0]
        except:
            pass
        try:
            notes7 = pm.instruments[7].notes
            for note in notes7:
                a = np.random.randint(15, size=1)
                note.velocity = note.velocity -offset - a[0]
        except:
            pass
        pm.write('../musegan_lpd/exps/nowbar_hybrid/gen_4dbar/all_dyn/'+name)
        i += 1

