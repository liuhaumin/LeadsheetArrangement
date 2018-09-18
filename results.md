# Results

## Lead Sheet Generation Demo
### Convolutional GAN version
sample1 {% include audio_player.html filename="gan_sample1.mp3" %}
sample2 {% include audio_player.html filename="gan_sample2.mp3" %}
sample3 {% include audio_player.html filename="gan_sample3.mp3" %}
sample4 {% include audio_player.html filename="gan_sample4.mp3" %}
sample5 {% include audio_player.html filename="gan_sample5.mp3" %}
sample6 {% include audio_player.html filename="gan_sample6.mp3" %}

### Recurrent VAE version
sample {% include audio_player.html filename="vae_TT_Random Sample8bar.mp3" %}

interpolation of two famous songs: from "Hey Jude" to "Some one like you"
{% include audio_player.html filename="jude2some_4barvae.mp3" %}


## Arrangement Generation Demo

| Model          | Sample1 |
|:--------------:|:-------:|
| *chroma-roll*  | {% include audio_player.html filename="sample1_chroma_roll.mp3" %} |
| *chroma-beats* | {% include audio_player.html filename="sample1_chroma_beats.mp3" %} |
| *chord-roll*   | {% include audio_player.html filename="sample1_chord_roll.mp3" %} |

| Model          | Sample2 |
|:--------------:|:-------:|
| *chroma-roll*  | {% include audio_player.html filename="sample2_chroma_roll.mp3" %} |
| *chroma-beats* | {% include audio_player.html filename="sample2_chroma_beats.mp3" %} |
| *chord-roll*   | {% include audio_player.html filename="sample2_chord_roll.mp3" %} |

| Model          | Sample3 |
|:--------------:|:-------:|
| *chroma-roll*  | {% include audio_player.html filename="sample3_chroma_roll.mp3" %} |
| *chroma-beats* | {% include audio_player.html filename="sample3_chroma_beats.mp3" %} |
| *chord-roll*   | {% include audio_player.html filename="sample3_chord_roll.mp3" %} |

| Model          | Sample4 |
|:--------------:|:-------:|
| *chroma-roll*  | {% include audio_player.html filename="sample4_chroma_roll.mp3" %} |
| *chroma-beats* | {% include audio_player.html filename="sample4_chroma_beats.mp3" %} |
| *chord-roll*   | {% include audio_player.html filename="sample4_chord_roll.mp3" %} |


## Piano-roll Visualizations

![evolution](figs/evolution.png)
<p class="caption">Evolution of the generated piano-rolls as a function of update steps</p>

![hybrid](figs/hybrid.png)
<p class="caption">Randomly-picked generation result (piano-rolls), generating from scratch</p>

## Generation From Scratch

> **Lower the volume first!**
No cherry-picking involved for all models. Some might sound unpleasant.

| Model      | Sample |
|:----------:|:-------:|
| *composer* | {% include audio_player.html filename="from_scratch_composer.mp3" %} |
| *jamming*  | {% include audio_player.html filename="from_scratch_jamming.mp3" %} |
| *hybrid*   | {% include audio_player.html filename="from_scratch_hybrid.mp3" %} |

## Track-conditional Generation

> **Lower the volume first!**
No cherry-picking involved for all models. Some might sound unpleasant.

| Model      | Sample |
|:----------:|:------:|
| *composer* | {% include audio_player.html filename="track_conditional_composer.mp3" %} |
| *jamming*  | {% include audio_player.html filename="track_conditional_jamming.mp3" %} |
| *hybrid*   | {% include audio_player.html filename="track_conditional_hybrid.mp3" %} |

