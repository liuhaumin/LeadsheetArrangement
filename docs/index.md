<p style="color:#222;">
  <em>NEWS: Our paper has been accepted by <strong>ICMLA2018</strong>.<span style="color:#727272"> &mdash; Sep 4th, 2018</span></em>
</p>

## Introduction
Nowadays, the common ways people usually adopt for musical data storaging are either lead sheet with XML format or multi-instrument with MIDI format. Lead sheets aim to record the melody line with the accompaniment chord sequences which give people a brief idea about their harmonic information. On the other hand, those music sheets stored in MIDI format record all the played notes of each instrument. Because of the different essence of these two datasets, recent works in deep learning music generaton either focus on lead sheet generaton or multi-instrument generation.

[Lead sheet arrangement](https://liuhaumin.github.io/LeadsheetArrangement/) is a project which aims at solving a new task to bridge the gap between lead sheet generation and multi-instrument generation. The difficulty lies in the situation that lead sheet XML format provides melody information but lacks the settings with its chord sequences, while MIDI format provides all the instrumental settings of the played note but usually does not mark out where the melody track is. In these project, we try to extract some harmonic features from both types of the dataset to achieve the linkage of them. Briefly speaking, we want to create music *'From Lead Sheet to its Arrangement'*. The code could be found in [LeadsheetArrangement Github](https://github.com/liuhaumin/LeadsheetArrangement) (under construction)

## Data Set
1. Theorytab Dataset (lead sheet xml format)
    1. Total ~138K Four-Four Time bars
    1. Scale degree (considered as C Major scale)
1. Lakh Pianoroll Dataset (MIDI format)
    1. Total ~160K Four-Four Time bars
    1. Traspose all songs to C Major/A minor scale

## Demos are shown in result page.
