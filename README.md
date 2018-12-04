## Usage

```python
import tensorflow as tf
from musegan.core import MuseGAN
from musegan.components import NowbarHybrid
from config import *

# Initialize a tensorflow session
with tf.Session() as sess:

    # === Prerequisites ===
    # Step 1 - Initialize the training configuration
    t_config = TrainingConfig

    # Step 2 - Select the desired model
    model = NowbarHybrid(NowBarHybridConfig)

    # Step 3 - Initialize the input data object
    input_data = InputDataNowBarHybrid(model)

    # Step 4 - Load training data
    path_train = 'train.npy'
    input_data.add_data(path_train, key='train')

    # Step 5 - Initialize a museGAN object
    musegan = MuseGAN(sess, t_config, model)

    # === Training ===
    musegan.train(input_data)

    # === Load a Pretrained Model ===
    musegan.load(musegan.dir_ckpt)

    # === Generate Samples ===
    path_test = 'train.npy'
    input_data.add_data(path_test, key='test')
    musegan.gen_test(input_data, is_eval=True)
```
