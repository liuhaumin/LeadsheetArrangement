import numpy as np
from pypianoroll import Multitrack, Track
from matplotlib import pyplot as plt
import os
import shutil
import pretty_midi
import librosa.display
import matplotlib.pyplot as plt

########################
#  From npy to midi    #
########################
print('------------Start data postprocessing !!-------------')
for i in range(1):
    print(i)
    gen = np.load('./exps/nowbar_hybrid/gen/gen.npy')
    tt = np.load('./data/chord_roll/val/x_bar_chroma_test.npy')
    madmom_cho = np.load('./data/chord_roll/val/y_bar_chroma_test.npy')
    # print(np.shape(madmom_cho))
    
    temp_tt_mel = tt[:64,:,:,:1]
    temp_tt_cho = tt[:64,:,:,1:2]
    temp_madmom_cho = madmom_cho[:64,:,:,:]
    
    # print(np.shape(gen))
    # print(np.shape(temp_tt_cho))

    gen_pr = np.zeros((64, 48, 128, 8), dtype=bool)
    #print(np.shape(gen_pr[:,:,24:108,:]))
    gen_pr[:,:,24:108,:5] = gen
    gen_pr[:,:,24:108,5:6] = temp_tt_mel
    gen_pr[:,:,24:108,6:7] = temp_tt_cho
    gen_pr[:,:,24:108,7:8] = temp_madmom_cho
    #print(np.shape(gen_pr[:,:,24:108,:]))
    gen_pr_float = gen_pr.astype(float)*1.0
    #-------------------------------#
    # deal with overfragmented note #
    #-------------------------------#
    gen_32th = np.zeros((64,16,128,8), dtype=float) # (interp_idx, time_idx, pitch_idx, track_idx)
    
    for idx_song in range(64):
        for idx in range(8):
            for pitch in range(128):
                for time in range(16):
                    if ((time%16) == 15) & (idx == 6 or 7):
                        gen_32th[idx_song,time,pitch,idx]=0.0
                        continue
                    else:
                        pass
                    cal = (gen_pr_float[idx_song,3*time,pitch,idx]+ gen_pr_float[idx_song,3*time+1,pitch,idx]+ gen_pr_float[idx_song,3*time+2,pitch,idx])/3
                    #print(cal)
                    if idx == 1:
                        if cal > 0:
                            gen_32th[idx_song,time,pitch,idx]=1.0
                        else:
                            gen_32th[idx_song,time,pitch,idx]=0.0
                    elif idx == 2 or idx == 3:
                        if cal > 0.8:
                            gen_32th[idx_song,time,pitch,idx]=1.0
                        else:
                            gen_32th[idx_song,time,pitch,idx]=0.0
                    else:
                        if cal > 0.6:
                            gen_32th[idx_song,time,pitch,idx]=1.0
                        else:
                            gen_32th[idx_song,time,pitch,idx]=0.0
    
    
    #------------------#
    # write midi files #
    #------------------#
        
    gen_32th = np.reshape(gen_32th,[-1,128,8])
    
    # 64 samples
    track_bass = Track(pianoroll=gen_32th[:,:,0], program=33, is_drum=False,
                      name= 'bass')
    track_drum = Track(pianoroll=gen_32th[:,:,1], program=0, is_drum=True,
                      name= 'drum')
    track_guitar = Track(pianoroll=gen_32th[:,:,2], program=25, is_drum=False,
                      name= 'guitar')
    track_piano = Track(pianoroll=gen_32th[:,:,3], program=0, is_drum=False,
                      name= 'piano')
    track_string = Track(pianoroll=gen_32th[:,:,4], program=41, is_drum=False,
                      name= 'string')
    track_tt_mel = Track(pianoroll=gen_32th[:,:,5], program=61, is_drum=False,
                      name= 'tt_mel')
    track_tt_cho = Track(pianoroll=gen_32th[:,:,6], program=0, is_drum=False,
                      name= 'tt_cho')
    track_madmom_cho = Track(pianoroll=gen_32th[:,:,7], program=0, is_drum=False,
                      name= 'madmom_cho')
    
    
    # Create a `pypianoroll.Multitrack` instance
    multitrack = Multitrack(tracks=[track_bass, track_drum, track_guitar, track_piano, track_string, track_tt_mel, track_tt_cho, track_madmom_cho], tempo=100.0, beat_resolution=4)
        
    # Write the `pypianoroll.Multitrack` instance to a MIDI file
    directory = './exps/nowbar_hybrid/gen_4dbar/all/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    path_o = directory + 'all_%s'%i + '.mid'
    multitrack.write(path_o)    
print("----------npy to midi completed !!------------")
#######################################
#  From midi to velocity dynamic midi #
#######################################

directory = './output/'
if not os.path.exists(directory):
    os.makedirs(directory)
i = 0
for root, dirs, files in os.walk('./exps/nowbar_hybrid/gen_4dbar/all/', topdown=False):    
    for name in files:
        # print(i)
        file = os.path.join(root, name)
        pm = pretty_midi.PrettyMIDI(file)
        # print(np.shape(pm.instruments))
        offset = 30
        
        try:
            notes0 = pm.instruments[0].notes
            for note in notes0:
                a = np.random.randint(15, size=1)
                note.velocity = note.velocity -offset +50 - a[0]
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
        pm.write('./output/'+name)
        i += 1

print("-----------midi to performance midi completed !!-------------")
print("Please find the output midi file in ./output/all_0.mid")
