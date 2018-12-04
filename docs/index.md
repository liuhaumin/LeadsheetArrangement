<p style="color:#222;">
  <em>NEWS: Our paper has been accepted by <strong>ICMLA2018</strong>.<span style="color:#727272"> &mdash; Sep 4th, 2018</span></em>
</p>

## Introduction
Nowadays, people commonly used XML format (ex. lead sheet) or MIDI format (multi-instrument) to store musical data. Lead sheet format aims to record the melody line with the chord sequences which give people a brief idea about their harmonic information. On the other hand, MIDI format records all the played notes used by each instrument. Because of different notations of these two datasets, recent works either focus on lead sheet generaton or on multi-instrument generation.

[Lead sheet arrangement](https://liuhaumin.github.io/LeadsheetArrangement/) is a project which aims at solving a new task, which bridges the gap between lead sheet generation and multi-instrument generation. The challenge lies in the situation that lead sheet XML format provides melody information but lacks the performance of chord sequences while MIDI format provides all settings of the played note but seldom mark out where the melody track is. In these project, we try to extract harmonic features from both types of the dataset and link them together. In a nutshell, we want to create music **From Lead Sheet to its Arrangement**. The code could be found in [LeadsheetArrangement Github](https://github.com/liuhaumin/LeadsheetArrangement) (under construction)

## Datasets
1. **Theorytab Dataset** (lead sheet xml format)
    1. Total ~138K Four-Four Time bars
    1. Scale degree (considered as C Major scale)
1. **Lakh Pianoroll Dataset** (MIDI format)
    1. Total ~160K Four-Four Time bars
    1. Traspose all songs to C Major/A minor scale

## Demos are shown [here](https://liuhaumin.github.io/LeadsheetArrangement/results).
