# MNLP Assignment 2

This contains code for our Guarani ASR experiments using a fork of Fairseq.

First, download Common Voice Guarani data to `cv-corpus/gn`. For multilingual training, download Spanish and Portuguese data to `cv-corpus/es` and `cv-corpus/pt`, respectively.

Note that we have already modified `examples/speech_to_text/prep_librispeech_data.py` to process Common Voice data.

Then, run the following:
```
bash experiment.sh
```
This will run both our experiments with model architectures and our experiments with multilingual training.