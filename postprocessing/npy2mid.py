import numpy as np
from pypianoroll import Multitrack, Track
from matplotlib import pyplot as plt
import os
import shutil
import pretty_midi
import librosa.display
import matplotlib.pyplot as plt

'''
For chord-roll
'''
############################################
# From npy to mid (one bar) chord sequence #
############################################

for i in range(1):
    print(i)
    gen = np.load('../musegan_lpd/exps/nowbar_hybrid/gen/gen.npy')
    #tt = np.load('../musegan_lpd/data/chord_sequence/val/x_bar_chroma_amazing_grace.npy')
    tt = np.load('../musegan_lpd/data/chord_sequence/val/x_bar_chroma_4bar_vae.npy')
    #madmom_cho = np.load('../musegan_lpd/data/chord_sequence/val/y_bar_chroma_amazing_grace.npy')
    madmom_cho = np.load('../musegan_lpd/data/chord_sequence/val/y_bar_chroma_4bar_vae.npy')
    print(np.shape(madmom_cho))
    
    temp_tt_mel = tt[:64,:,:,:1]
    temp_tt_cho = tt[:64,:,:,1:2]
    temp_madmom_cho = madmom_cho[:64,:,:,:]
    ##temp_tt_mel = tt[64:128,:,:,:1]
    ##temp_tt_cho = tt[64:128,:,:,1:2]
    ##temp_madmom_cho = madmom_cho[64:128,:,:,:]
    
    print(np.shape(gen))
    print(np.shape(temp_tt_cho))
    
    # move a direction
    ##source = '../musegan/interpolation/gen_4dbar/interp_%s'%i +'/sample/'
    ##destination = '../musegan/interpolation/gen_4dbar/interp/pic_%s'%i+'/'
    ##if not os.path.exists(destination):
    ##    os.makedirs(destination)
    ##os.rename(source, destination)

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
                    elif idx == 4:
                        if cal > 0.8:
                            gen_32th[idx_song,time,pitch,idx]=1.0
                        else:
                            gen_32th[idx_song,time,pitch,idx]=0.0
                    else:
                        if cal > 0.5:
                            gen_32th[idx_song,time,pitch,idx]=1.0
                        else:
                            gen_32th[idx_song,time,pitch,idx]=0.0
    
    #------------------#
    # save numpy files #
    #------------------#
    ##directory = '../musegan/interpolation/gen_4dbar/interp/all_npy/'
    ##if not os.path.exists(directory):
    ##    os.makedirs(directory)
    ##file_o = directory + 'all_%s'%i + '.npy'
    ##np.save(file_o, gen_32th)
    
    
    #------------------#
    # write midi files #
    #------------------#
        
    # All
    for idx in range(len(gen_32th[:,0,0,0])):
        # Create a `pypianoroll.Track` instance
        # program can be choosed to apply various instruments
        track_bass = Track(pianoroll=gen_32th[idx,:,:,0], program=33, is_drum=False,
                      name= 'bass')
        track_drum = Track(pianoroll=gen_32th[idx,:,:,1], program=0, is_drum=True,
                      name= 'drum')
        track_guitar = Track(pianoroll=gen_32th[idx,:,:,2], program=25, is_drum=False,
                      name= 'guitar')
        track_piano = Track(pianoroll=gen_32th[idx,:,:,3], program=0, is_drum=False,
                      name= 'piano')
        track_string = Track(pianoroll=gen_32th[idx,:,:,4], program=41, is_drum=False,
                      name= 'string')
        track_tt_mel = Track(pianoroll=gen_32th[idx,:,:,5], program=0, is_drum=False,
                      name= 'tt_mel')
        track_tt_cho = Track(pianoroll=gen_32th[idx,:,:,6], program=0, is_drum=False,
                      name= 'tt_cho')
        track_madmom_cho = Track(pianoroll=gen_32th[idx,:,:,7], program=0, is_drum=False,
                      name= 'madmom_cho')
    
        # Create a `pypianoroll.Multitrack` instance
        multitrack = Multitrack(tracks=[track_bass, track_drum, track_guitar, track_piano, track_string, track_tt_mel, track_tt_cho, track_madmom_cho], tempo=100.0, beat_resolution=4)
        
        # Plot the piano-roll
        ##fig, ax = multitrack.plot()
        ##plt.show()
        
        # Write the `pypianoroll.Multitrack` instance to a MIDI file
        directory = '../musegan_lpd/exps/nowbar_hybrid/gen_4dbar/all/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        path_o = directory + 'all_%s'%i+'_%s'%idx + '.mid'
        multitrack.write(path_o)
        
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
    track_tt_mel = Track(pianoroll=gen_32th[:,:,5], program=0, is_drum=False,
                      name= 'tt_mel')
    track_tt_cho = Track(pianoroll=gen_32th[:,:,6], program=0, is_drum=False,
                      name= 'tt_cho')
    track_madmom_cho = Track(pianoroll=gen_32th[:,:,7], program=0, is_drum=False,
                      name= 'madmom_cho')
    
    
    # Create a `pypianoroll.Multitrack` instance
    multitrack = Multitrack(tracks=[track_bass, track_drum, track_guitar, track_piano, track_string, track_tt_mel, track_tt_cho, track_madmom_cho], tempo=100.0, beat_resolution=4)
        
    # Plot the piano-roll
    ##fig, ax = multitrack.plot()
    ##plt.show()
        
    # Write the `pypianoroll.Multitrack` instance to a MIDI file
    directory = '../musegan_lpd/exps/nowbar_hybrid/gen_4dbar/all/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    path_o = directory + 'all_%s'%i + '.mid'
    multitrack.write(path_o)   

'''
For all kinds
'''
# import numpy as np
# from pypianoroll import Multitrack, Track
# from matplotlib import pyplot as plt
# import os
# import shutil
# import pretty_midi
# import librosa.display
# import matplotlib.pyplot as plt

# for i in range(1):
#     print(i)
#     gen = np.load('../musegan_lpd/exps/nowbar_hybrid/gen/gen.npy')
#     mel = np.load('../musegan_lpd/data/chroma_vector/val/melody_tt.npy')
#     cho = np.load('../musegan_lpd/data/chroma_vector/val/chord_tt.npy')
    
#     temp_mel = np.reshape(mel[:64,:,:],(64,48,84,1))
#     temp_cho = np.reshape(cho[:64,:,:],(64,48,84,1))
#     print(np.shape(gen))
#     print(np.shape(temp_mel))
    
#     # move a direction
#     ##source = '../musegan/interpolation/gen_4dbar/interp_%s'%i +'/sample/'
#     ##destination = '../musegan/interpolation/gen_4dbar/interp/pic_%s'%i+'/'
#     ##if not os.path.exists(destination):
#     ##    os.makedirs(destination)
#     ##os.rename(source, destination)

#     gen_pr = np.zeros((64, 48, 128, 7), dtype=bool)
#     #print(np.shape(gen_pr[:,:,24:108,:]))
#     gen_pr[:,:,24:108,:5] = gen
#     gen_pr[:,:,24:108,5:6] = temp_mel
#     gen_pr[:,:,24:108,6:] = temp_cho
#     #print(np.shape(gen_pr[:,:,24:108,:]))
#     gen_pr_float = gen_pr.astype(float)*1.0
#     #-------------------------------#
#     # deal with overfragmented note #
#     #-------------------------------#
#     gen_32th = np.zeros((64,16,128,7), dtype=float) # (interp_idx, time_idx, pitch_idx, track_idx)
    
#     for idx_song in range(64):
#         for idx in range(7):
#             for pitch in range(128):
#                 for time in range(16):
#                     #if ((time%16) == 15) & (idx == 5):
#                     #    gen_32th[idx_song,time,pitch,idx]=0.0
#                     #    continue
#                     #else:
#                     #    pass
#                     cal = (gen_pr_float[idx_song,3*time,pitch,idx]+ gen_pr_float[idx_song,3*time+1,pitch,idx]+ gen_pr_float[idx_song,3*time+2,pitch,idx])/3
#                     #print(cal)
#                     if idx == 1:
#                         if cal > 0:
#                             gen_32th[idx_song,time,pitch,idx]=1.0
#                         else:
#                             gen_32th[idx_song,time,pitch,idx]=0.0
#                     elif idx == 4:
#                         if cal > 0.8:
#                             gen_32th[idx_song,time,pitch,idx]=1.0
#                         else:
#                             gen_32th[idx_song,time,pitch,idx]=0.0
#                     else:
#                         if cal > 0.5:
#                             gen_32th[idx_song,time,pitch,idx]=1.0
#                         else:
#                             gen_32th[idx_song,time,pitch,idx]=0.0
    
#     #------------------#
#     # save numpy files #
#     #------------------#
#     ##directory = '../musegan/interpolation/gen_4dbar/interp/all_npy/'
#     ##if not os.path.exists(directory):
#     ##    os.makedirs(directory)
#     ##file_o = directory + 'all_%s'%i + '.npy'
#     ##np.save(file_o, gen_32th)
    
    
#     #------------------#
#     # write midi files #
#     #------------------#
        
#     # All
#     for idx in range(len(gen_32th[:,0,0,0])):
#         # Create a `pypianoroll.Track` instance
#         # program can be choosed to apply various instruments
#         track_bass = Track(pianoroll=gen_32th[idx,:,:,0], program=33, is_drum=False,
#                       name= 'bass')
#         track_drum = Track(pianoroll=gen_32th[idx,:,:,1], program=0, is_drum=True,
#                       name= 'drum')
#         track_guitar = Track(pianoroll=gen_32th[idx,:,:,2], program=25, is_drum=False,
#                       name= 'guitar')
#         track_piano = Track(pianoroll=gen_32th[idx,:,:,3], program=0, is_drum=False,
#                       name= 'piano')
#         track_string = Track(pianoroll=gen_32th[idx,:,:,4], program=41, is_drum=False,
#                       name= 'string')
#         track_melody = Track(pianoroll=gen_32th[idx,:,:,5], program=0, is_drum=False,
#                       name= 'melody')
#         track_chord = Track(pianoroll=gen_32th[idx,:,:,6], program=0, is_drum=False,
#                       name= 'chord')
    
#         # Create a `pypianoroll.Multitrack` instance
#         multitrack = Multitrack(tracks=[track_bass, track_drum, track_guitar, track_piano, track_string, track_melody, track_chord], tempo=100.0, beat_resolution=4)
        
#         # Plot the piano-roll
#         ##fig, ax = multitrack.plot()
#         ##plt.show()
        
#         # Write the `pypianoroll.Multitrack` instance to a MIDI file
#         directory = '../musegan_lpd/exps/nowbar_hybrid/gen_4dbar/all/'
#         if not os.path.exists(directory):
#             os.makedirs(directory)
#         path_o = directory + 'all_%s'%i+'_%s'%idx + '.mid'
#         multitrack.write(path_o)
        
#     gen_32th = np.reshape(gen_32th,[-1,128,7])
    
#     # 64 samples
#     track_bass = Track(pianoroll=gen_32th[:,:,0], program=33, is_drum=False,
#                       name= 'bass')
#     track_drum = Track(pianoroll=gen_32th[:,:,1], program=0, is_drum=True,
#                       name= 'drum')
#     track_guitar = Track(pianoroll=gen_32th[:,:,2], program=25, is_drum=False,
#                       name= 'guitar')
#     track_piano = Track(pianoroll=gen_32th[:,:,3], program=0, is_drum=False,
#                       name= 'piano')
#     track_string = Track(pianoroll=gen_32th[:,:,4], program=41, is_drum=False,
#                       name= 'string')
#     track_melody = Track(pianoroll=gen_32th[:,:,5], program=0, is_drum=False,
#                       name= 'melody')
#     track_chord = Track(pianoroll=gen_32th[:,:,6], program=0, is_drum=False,
#                       name= 'chord')
    
#     # Create a `pypianoroll.Multitrack` instance
#     multitrack = Multitrack(tracks=[track_bass, track_drum, track_guitar, track_piano, track_string, track_melody, track_chord], tempo=100.0, beat_resolution=4)
        
#     # Plot the piano-roll
#     ##fig, ax = multitrack.plot()
#     ##plt.show()
        
#     # Write the `pypianoroll.Multitrack` instance to a MIDI file
#     directory = '../musegan_lpd/exps/nowbar_hybrid/gen_4dbar/all/'
#     if not os.path.exists(directory):
#         os.makedirs(directory)
#     path_o = directory + 'all_%s'%i + '.mid'
#     multitrack.write(path_o)    
    
    
    