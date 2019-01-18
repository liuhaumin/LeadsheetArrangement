# LeadsheetArrangement 自動簡譜編曲 :musical_note:
[Lead Sheet Arrangement](https://liuhaumin.github.io/LeadsheetArrangement/) is a task to automatically accompany the existed or generated lead sheets with multiple instruments. The rhythm, pitch notes, onsets and even playing style of different instruments are all learned by the model. 

We train the model with Lakh Pianoroll Dataset (LPD) to generate pop song style arrangement consisting of bass, guitar, piano, strings and drums.

Sample results are available
[here](https://liuhaumin.github.io/LeadsheetArrangement/results).

## Papers

__Lead sheet generation and arrangement by conditional generative adversarial network__<br>
Hao-Min Liu and Yi-Hsuan Yang,
to appear in *International Conference on Machine Learning and Applications* (ICMLA), 2018.
[[arxiv](https://arxiv.org/abs/1807.11161)]

__Lead Sheet Generation and Arrangement via a Hybrid Generative Model__<br>
Hao-Min Liu\*, Meng-Hsuan Wu\*, and Yi-Hsuan Yang
(\*equal contribution)<br>
in _ISMIR Late-Breaking Demos Session_, 2018.
(non-refereed two-page extended abstract)<br>
[[paper](https://liuhaumin.github.io/LeadsheetArrangement/pdf/ismir2018leadsheetarrangement.pdf)]
[[poster](https://liuhaumin.github.io/LeadsheetArrangement/pdf/ismir-lbd-poster_A0_final.pdf)]

__Lead sheet and Multi-track Piano-roll generation using MuseGAN__<br>
Hao-Min Liu, Hao-Wen Dong, Wen-Yi Hsiao and Yi-Hsuan Yang,
in *GPU Technology Conference* (GTC), 2018.
[[poster](https://liuhaumin.github.io/LeadsheetArrangement/pdf/GTC_poster_HaoMin.pdf)]

## Usage
### Step 0: preprocessing the midi data
1. Go to folder (./data/preprocessing/)
2. Put your input song_name.midi file in folder (./data/preprocessing/mysong_mid_C/)
    * Note that song_name.midi should be in C key and two tracks(melody and chord)
3. Back to folder (./preprocessing/)
4. Run pianoroll_mysong.ipynb
    * Remember to change filenames in code lines into "song_name"
    * Output files will be stored in folder(./data/chord_roll/val/)with name as "x_bar_chroma_song_name.npy" and "y_bar_chroma_song_name.npy"

### Step 1: Loading the data
1. Open file store_sa.py
2. Turn filename in code line # 36 into your filename, i.e. ('y_bar_chroma_song_name.npy')
3. Load the data by running store_sa.py

### Step 2: adjust training or testing modes in main.py
```python
import tensorflow as tf
from musegan.core import MuseGAN
from musegan.components import NowbarHybrid
from config import *

# Initialize a tensorflow session

""" Create TensorFlow Session """
with tf.Session() as sess:
    
    # === Prerequisites ===
    # Step 1 - Initialize the training configuration        
    t_config = TrainingConfig
    t_config.exp_name = 'exps/nowbar_hybrid'        

    # Step 2 - Select the desired model
    model = NowbarHybrid(NowBarHybridConfig)
    
    # Step 3 - Initialize the input data object
    input_data = InputDataNowBarHybrid(model)
    
    # Step 4 - Load training data
    path_x_train_bar = 'tra_X_bars'
    path_y_train_bar = 'tra_y_bars'
    input_data.add_data_sa(path_x_train_bar, path_y_train_bar, 'train') # x: input, y: conditional feature
    
    # Step 5 - Initialize a museGAN object
    musegan = MuseGAN(sess, t_config, model)
    
    # === Training ===
    musegan.train(input_data)

    # === Load a Pretrained Model ===
    musegan.load(musegan.dir_ckpt)

    # === Generate Samples ===
    path_x_test_bar = 'val_X_bars'
    path_y_test_bar = 'val_y_bars'
    input_data.add_data_sa(path_x_test_bar, path_y_test_bar, key='test')
    musegan.gen_test(input_data, is_eval=True)

```
### Step 3: run main.py
1. Training mode
   * Checkpoints are stored in folder (./exps/nowbar_hybrid/checkpoint/)
2. Testing mode
   * the output files are stored in folder (./exps/nowbar_hybrid/gen/)
   
### Step 4: postprocessing the output midi
1. Go to folder (./postprocessing/)
2. Run file npy2mid.ipynb
   * Remember to change filenames in code lines into "song_name"
   * Output files are stored in folder (./exps/nowbar_hybrid/gen_4bar/)


