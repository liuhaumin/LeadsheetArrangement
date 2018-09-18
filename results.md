# Results
> **Lower the volume first!**
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

#### interpolation of two famous songs: from "Hey Jude" to "Some one like you"

![interp](figs/interpolation.png)
<p class="caption">interpolate between two pop songs</p>

{% include audio_player.html filename="jude2some_4barvae.mp3" %}


## Arrangement Generation Demo

| Model          | Sample1 |
|:--------------:|:-------:|
| *chord-roll*   | {% include audio_player.html filename="sample1_chord_roll.mp3" %} |
| *chroma-roll*  | {% include audio_player.html filename="sample1_chroma_roll.mp3" %} |
| *chroma-beats* | {% include audio_player.html filename="sample1_chroma_beats.mp3" %} |


| Model          | Sample2 |
|:--------------:|:-------:|
| *chord-roll*   | {% include audio_player.html filename="sample2_chord_roll.mp3" %} |
| *chroma-roll*  | {% include audio_player.html filename="sample2_chroma_roll.mp3" %} |
| *chroma-beats* | {% include audio_player.html filename="sample2_chroma_beats.mp3" %} |


| Model          | Sample3 |
|:--------------:|:-------:|
| *chord-roll*   | {% include audio_player.html filename="sample3_chord_roll.mp3" %} |
| *chroma-roll*  | {% include audio_player.html filename="sample3_chroma_roll.mp3" %} |
| *chroma-beats* | {% include audio_player.html filename="sample3_chroma_beats.mp3" %} |


| Model          | Sample4 |
|:--------------:|:-------:|
| *chord-roll*   | {% include audio_player.html filename="sample4_chord_roll.mp3" %} |
| *chroma-roll*  | {% include audio_player.html filename="sample4_chroma_roll.mp3" %} |
| *chroma-beats* | {% include audio_player.html filename="sample4_chroma_beats.mp3" %} |



## Arrangement Visualizations
![arrangement](figs/Amazing_grace_arrangment.png)
<p class="caption">The music sheet of Amazing grace arrangement</p>


![evolution](figs/evolution.png)
<p class="caption">Evolution of the generated piano-rolls as a function of update steps</p>
