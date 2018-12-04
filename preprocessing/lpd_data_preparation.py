import os
import numpy as np
from lookup_tables import *
import pretty_midi

################################
#  chord-roll (48, 84)         #
################################

#### chord-roll for lpd #######
X1 = np.load('../../music/lpd_4dbar_12_C/tra/phr_chord_clean.npy')
tra_song = X1[:20000,:,:,:]
val_song = X1[20000:,:,:,:]
tra_song_re = np.reshape(tra_song,(-1,48,84,6))
val_song_re = np.reshape(val_song,(-1,48,84,6))[:16000,:,:,:]
# print(np.shape(tra_song_re)) #(160000, 48, 84, 6)
# print(np.shape(val_song_re)) #(16000, 48, 84, 6)
np.save('../../musegan_lpd/data/chord_sequence/tra/x_bar_chroma',tra_song_re[:,:,:,:5])
np.save('../../musegan_lpd/data/chord_sequence/val/x_bar_chroma',val_song_re[:,:,:,:5])
np.save('../../musegan_lpd/data/chord_sequence/tra/y_bar_chroma',tra_song_re[:,:,:,5:])
np.save('../../musegan_lpd/data/chord_sequence/val/y_bar_chroma',val_song_re[:,:,:,5:])

#### chord-roll for mysong #######
d = np.load('../../music/mysong_npy_C/all_15_32.npy')
d_bar = np.reshape(d,(-1,48,84,3))
d_bar_64 = np.concatenate((d_bar,d_bar,d_bar,d_bar,d_bar,d_bar,d_bar,d_bar),axis=0)
# np.shape(d_bar_64) #(64, 48, 84, 3)

y_bar_chroma_mysong = d_bar_64[:,:,:,2:]
x_bar_chroma_mysong = d_bar_64[:,:,:,:2]
# print(np.shape(x_bar_chroma_mysong)) #(64, 48, 84, 2)
# print(np.shape(y_bar_chroma_mysong)) #(64, 48, 84, 1)
np.save('../../musegan_lpd/data/chord_sequence/val/y_bar_chroma_mysong',y_bar_chroma_mysong)
np.save('../../musegan_lpd/data/chord_sequence/val/x_bar_chroma_mysong',x_bar_chroma_mysong)


################################
#  chroma-roll (48, 12)        #
################################
#### chroma-roll for lpd #######
X1 = np.load('../../music/lpd_4dbar_12_C/tra/phr_chord_clean.npy')
tra_song = X1[:20000,:,:,:]
val_song = X1[20000:,:,:,:]
tra_song_re = np.reshape(tra_song,(-1,48,84,6))
val_song_re = np.reshape(val_song,(-1,48,84,6))
# print(np.shape(tra_song_re)) # (160000, 48, 84, 6)

tra_song_union = np.logical_or.reduce((tra_song_re[:,:,:,0], # drum track should not count in
                                      tra_song_re[:,:,:,2],
                                      tra_song_re[:,:,:,3],
                                      tra_song_re[:,:,:,4]))
val_song_union = np.logical_or.reduce((val_song_re[:,:,:,0],
                                      val_song_re[:,:,:,2],
                                      val_song_re[:,:,:,3],
                                      val_song_re[:,:,:,4]))
# print('tra_song_union',np.shape(tra_song_union)) # tra_song_union (160000, 48, 84)

tra_song_chroma = np.logical_or.reduce((tra_song_union[:,:,:12],
                                        tra_song_union[:,:,12:24],
                                        tra_song_union[:,:,24:36],
                                        tra_song_union[:,:,36:48],
                                        tra_song_union[:,:,48:60],
                                        tra_song_union[:,:,60:72],
                                        tra_song_union[:,:,72:84]))

val_song_chroma = np.logical_or.reduce((val_song_union[:,:,:12],
                                        val_song_union[:,:,12:24],
                                        val_song_union[:,:,24:36],
                                        val_song_union[:,:,36:48],
                                        val_song_union[:,:,48:60],
                                        val_song_union[:,:,60:72],
                                        val_song_union[:,:,72:84]))
# print('tra_song_chroma',np.shape(tra_song_chroma)) # tra_song_chroma (160000, 48, 12)
np.save('data_lpd/val_song_chroma',val_song_chroma)
np.save('data_lpd/tra_song_chroma',tra_song_chroma)

tra_bar_chroma = np.reshape(tra_song_chroma,(-1,48,12)) # tra_bar_chroma (160000, 48, 12)
val_bar_chroma = np.reshape(val_song_chroma,(-1,48,12)) # val_bar_chroma (16640, 48, 12)
x_tra_bar_chroma = tra_song_re[:,:,:,:5]
x_val_bar_chroma = val_song_re[:16000,:,:,:5]
y_tra_bar_chroma = np.reshape(tra_bar_chroma,(160000,48,12,1))
y_val_bar_chroma = np.reshape(val_bar_chroma[:16000,:,:],(16000,48,12,1))

# print('x_tra_bar_chroma:',np.shape(x_tra_bar_chroma)) # x_tra_bar_chroma: (160000, 48, 84, 5)
# print('x_val_bar_chroma:',np.shape(x_val_bar_chroma)) # x_val_bar_chroma: (16000, 48, 84, 5)
# print('y_tra_bar_chroma:',np.shape(y_tra_bar_chroma)) # y_tra_bar_chroma: (160000, 48, 12, 1)
# print('y_val_bar_chroma:',np.shape(y_val_bar_chroma)) # y_val_bar_chroma: (16000, 48, 12, 1)

np.save('../../musegan_lpd/data/chroma_sequence/tra/x_bar_chroma',x_tra_bar_chroma)
np.save('../../musegan_lpd/data/chroma_sequence/val/x_bar_chroma',x_val_bar_chroma)
np.save('../../musegan_lpd/data/chroma_sequence/tra/y_bar_chroma',y_tra_bar_chroma)
np.save('../../musegan_lpd/data/chroma_sequence/val/y_bar_chroma',y_val_bar_chroma)

################################
#  chroma-beats (4, 12)        #
################################
#### chroma-beats for lpd #######
x_tra_bar_chroma = np.load('../../musegan_lpd/data/chroma_sequence/tra/x_bar_chroma.npy')
x_val_bar_chroma = np.load('../../musegan_lpd/data/chroma_sequence/val/x_bar_chroma.npy')
y_tra_bar_chroma = np.load('../../musegan_lpd/data/chroma_sequence/tra/y_bar_chroma.npy')
y_val_bar_chroma = np.load('../../musegan_lpd/data/chroma_sequence/val/y_bar_chroma.npy')
# np.shape(y_tra_bar_chroma) # (160000, 48, 12, 1)
y_tra_bar_chroma_4 = np.zeros((160000,4,12,1))
y_tra_bar_chroma_4[:,0,:,:] = np.average(y_tra_bar_chroma[:,:12,:,:], axis=1)
y_tra_bar_chroma_4[:,1,:,:] = np.average(y_tra_bar_chroma[:,12:24,:,:], axis=1)
y_tra_bar_chroma_4[:,2,:,:] = np.average(y_tra_bar_chroma[:,24:36,:,:], axis=1)
y_tra_bar_chroma_4[:,3,:,:] = np.average(y_tra_bar_chroma[:,36:48,:,:], axis=1)
y_val_bar_chroma_4 = np.zeros((16000,4,12,1))
y_val_bar_chroma_4[:,0,:,:] = np.average(y_val_bar_chroma[:,:12,:,:], axis=1)
y_val_bar_chroma_4[:,1,:,:] = np.average(y_val_bar_chroma[:,12:24,:,:], axis=1)
y_val_bar_chroma_4[:,2,:,:] = np.average(y_val_bar_chroma[:,24:36,:,:], axis=1)
y_val_bar_chroma_4[:,3,:,:] = np.average(y_val_bar_chroma[:,36:48,:,:], axis=1)
# print(np.shape(y_val_bar_chroma_4)) # (16000, 4, 12, 1)
np.save('../../musegan_lpd/data/chroma_vector/tra/x_bar_chroma',x_tra_bar_chroma)
np.save('../../musegan_lpd/data/chroma_vector/val/x_bar_chroma',x_val_bar_chroma)
np.save('../../musegan_lpd/data/chroma_vector/tra/y_bar_chroma_4',y_tra_bar_chroma_4)
np.save('../../musegan_lpd/data/chroma_vector/val/y_bar_chroma_4',y_val_bar_chroma_4)