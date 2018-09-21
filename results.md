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
<p class="caption">vae 4-bar samples</p>
sample {% include audio_player.html filename="vae_4bar_samples.mp3" %}

<p class="caption">vae 8-bar samples</p>
sample {% include audio_player.html filename="vae_8bar_samples.mp3" %}

#### interpolation of two pop songs
**from "Hey Jude" to "Some one like you"**
{% include audio_player.html filename="jude2some_4barvae.mp3" %}
**from "Payphone" to "Hey Jude"**
![interp](figs/interpolation.png)
{% include audio_player.html filename="phone2jude_4barvae.mp3" %}



## Arrangement Generation Demo
<p class="caption">Arrangement on theorytab leadsheets</p>

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

<p class="caption">Arrangement on VAE generated leadsheets</p>
{% include audio_player.html filename="vae_arr_4bar_samples.mp3" %}

## Arrangement visualization
{% include audio_player.html filename="amazing_grace_arr.mp3" %}
<p class="caption">The music sheet of Amazing grace arrangement</p>
![arrangement](figs/Amazing_grace_arrangement.png)

