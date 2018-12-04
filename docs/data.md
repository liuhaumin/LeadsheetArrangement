# Data

## Theorytab Dataset (lead sheet xml format)
  1. Total ~138K Four-Four Time bars
  1. Scale degree (considered as C Major scale)

## [Lakh Pianoroll Dataset](https://salu133445.github.io/lakh-pianoroll-dataset/) (MIDI format)
  1. Total ~160K Four-Four Time bars
  1. Traspose all songs to C Major/A minor scale

## Training Data

- Use *symbolic timing*, which discards tempo information
  (see [here](https://salu133445.github.io/lakh-pianoroll-dataset/representation) for more
  details)
- Discard velocity information (using binary-valued piano-rolls)
- 84 possibilities for note pitch (from C1 to B7)
- Merge tracks into 5 categories: *Bass*, *Drums*, *Guitar*, *Piano* and
  *Strings*
- Consider only songs with an *rock* tag
- Collect musically meaningful 4-bar phrases for the temporal model by
  segmenting the piano-rolls with *structure features* proposed in [1]

Hence, the size of the target output tensor is 4 (bar) &times; 96 (time step)
&times; 84 (pitch) &times; 5 (track).

- [tra_phr.npy](https://drive.google.com/uc?id=1-bQCO6ZxpIgdMM7zXhNJViovHjtBKXde&export=download)
  (7.54 GB) contains 50,266 four-bar phrases. The shape is (50266, 384, 84, 5).
- [tra_bar.npy](https://drive.google.com/uc?id=1Xxj6WU82fcgY9UtBpXJGOspoUkMu58xC&export=download)
  (4.79 GB) contains 127,734 bars. The shape is (127734, 96, 84, 5).

Here are two examples of five-track piano-roll of four-bar long seen in our
training data. The tracks are (from top to bottom): <i>Bass</i>, <i>Drums</i>,
<i>Guitar</i>, <i>Strings</i>, <i>Piano</i>.

<img src="figs/train_samples.png" alt="train_samples" style="max-height:200px; display:block; margin:auto">

## Reference

1. Joan Serrá, Meinard Müller, Peter Grosche and Josep Ll. Arcos,
   "Unsupervised Detection of Music Boundaries by Time Series Structure
   Features,"
   in *AAAI Conference on Artificial Intelligence* (AAAI), 2012
