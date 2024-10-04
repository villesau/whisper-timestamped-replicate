# whisper-timestamped-replicate
Replicate setup for whisper-timestamped

## Download weights

Downloading the weights makes cold-starting the model faster. COG packages all the repo contents automatically.
You can download the weights by running the following commands:

```shell
cog run scripts/download_whisper_weights
cog run scripts/download_vad_weights
```