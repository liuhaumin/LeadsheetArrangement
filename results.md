# Results

## Lead Sheet Generation Demo
### Recurrent VAE version
{% include audio_player.html filename="best_samples.mp3" %}
### Convolutional GAN version
{% include audio_player.html filename="best_samples.mp3" %}

## Arrangement Generation Demo
No cherry-picking involved for all models. Some might sound unpleasant.

| Model          | Sample |
|:--------------:|:------:|
| *chroma-roll*  | {% include audio_player.html filename="from_scratch_composer.mp3" %} |
| *chroma-beats* | {% include audio_player.html filename="from_scratch_jamming.mp3" %} |
| *chord-roll*   | {% include audio_player.html filename="from_scratch_hybrid.mp3" %} |


## Piano-roll Visualizations

![evolution](figs/evolution.png)
<p class="caption">Evolution of the generated piano-rolls as a function of update steps</p>

![hybrid](figs/hybrid.png)
<p class="caption">Randomly-picked generation result (piano-rolls), generating from scratch</p>

## Generation From Scratch

> **Lower the volume first!**
No cherry-picking involved for all models. Some might sound unpleasant.

| Model      | Sample |
|:----------:|:------:|
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

