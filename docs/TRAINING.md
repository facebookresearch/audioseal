# Training a new watermarking model

This doc shows how to train a new AudioSeal model. The training pipeline was developed using [AudioCraft](https://github.com/facebookresearch/audiocraft) (version 0.1.4 and later). The following example is tested on Pytorch==2.1.0 and, torchaudio==2.1.0:

## Prerequisite

We need AudioCraft >=1.4.0a1. If you want to experiment with different datasets and training recipes, we advise that you download the source code of audiocraft and install directly from source, see [Installation notes](https://github.com/facebookresearch/audiocraft/blob/main/README.md#installation):

```bash
git clone https://github.com/facebookresearch/audiocraft.git
cd audiocraft
pip install -e .

sudo apt-get install ffmpeg
# Or if you are using Anaconda or Miniconda
conda install "ffmpeg<5" -c conda-forge
```

Note that the step of installing ffmpeg (<5.0.0) in the notes is mandatory, otherwise the training loop will fail as our AAC augmentation step depends on it.

## Preparing dataset

The dataset should be processed in AudioCraft [format](https://github.com/facebookresearch/audiocraft/blob/main/docs/DATASETS.md). The first step is to create the manifest for your dataset. For Voxpopuli (which is used in the paper), run the following command:

```bash

# Download the raw audios and segment them
git clone https://github.com/facebookresearch/voxpopuli.git
cd voxpopuli
python -m voxpopuli.download_audios --root [ROOT] --subset 400k
python -m voxpopuli.get_unlabelled_data --root [ROOT] --subset 400k

# Run audiocraft data tool to prepare the manifest
cd [PATH to audiocraft]
python -m audiocraft.data.audio_dataset [ROOT] egs/voxpopuli/data.jsonl.gz
```

Then, prepare the following datasource definition and put it inside the "[audiocraft root]/configs/dset/audio/voxpopuli.yaml":

```yaml
# @package __global__

datasource:
  max_sample_rate: 16000
  max_channels: 1

  train: egs/voxpopuli
  valid: egs/voxpopuli
  evaluate: egs/voxpopuli
  generate: egs/voxpopuli
```

## Training

The training pipeline uses [Dora](https://github.com/facebookresearch/dora) to structure the experiments and perform grid-based paratermeter tuning. It is useful to get yourself familiar with Dora concepts such as dora run, dora grid, etc. before starting.

To test the training pipeline locally, see [this documentation in Audiocraft](https://github.com/facebookresearch/audiocraft/blob/main/docs/WATERMARKING.md). You can replace the example dataset with the above Voxpopuli, e.g. run the following command within the Audiocraft cloned directory:

```bash
dora run solver=watermark/robustness dset=audio/example
```

By default the checkpoints and experiment files are stored in `/tmp/audiocraft_$USER/outputs`. To customize where your own Dora output and experiment folder are, as well as to run in a SLURM cluster, define a config file with the following structure:

```yaml
# File name: my_config.yaml

default:
  dora_dir: [DORA PATH]
  partitions:
    global: your_slurm_partitions
    team: your_slurm_partitions
  reference_dir: /tmp
darwin: # if we detect we are on a Mac, then most likely we are doing unit testing etc.
  dora_dir: [YOUR PATH]
  partitions:
    global: your_slurm_partitions
    team: your_slurm_partitions
  reference_dir: [REFERENCE PATH]
```

where `partitions` indicates the SLURM partitions you are entitled to run your jobs. Then re-run the `dora run` command with the custom config:

```bash
AUDIOCRAFT_CONFIG=my_config.yaml dora run solver=watermark/robustness dset=audio/voxpopuli
```

## Evaluate the checkpoint

If successful, the checkpoints will be stored in an experiment folder in your dora dir, i.e. `[DORA_PATH]/xps/[HASH-ID]/checkpoint_XXX.th` , where `HASH-ID` is the Id of the experiment you will see in the output log when running `dora run`. You can choose to evaluate your checkpoints with diffferent settings for nbits, and choose the ones with lowest losses:

```bash
AUDIOCRAFT_CONFIG=my_config.yaml dora run solver=watermark/robustness execute_only=evaluate dset=audio/voxpopuli continue_from=[PATH_TO_THE_CHECKPOINT_FILE] +dummy_watermarker.nbits=16 seanet.detector.output_dim=32
```

## Postprocessing the checkpoints for inference

The checkpoint contains the jointly-trained generator and detector, so it cannot be used right away in AudioSeal API. To extract the generator and detector, run the conversion script in Audioseal code "src/scripts/checkpoints.py":

```bash
python [AudioSeal path]/src/scripts/checkpoints.py --checkpoint=[PATH TO CHECKPOINT] --outdir=[OUTPUT_DIR] --suffix=[name of the new model]
```

After this step, there will be two checkpoint files named `generator_[suffix].pth` and `detector_[suffix].pth` in the output directory [OUTPUT_DIR]. You can use these new checkpoints directly with AudioSeal API, for instance:

```python

model = AudioSeal.load_generator("[OUTPUT_DIR]/generator_[suffix].pth", nbits=16)
watermark = model.get_watermark(wav, sr)

detector = AudioSeal.load_detector("[OUTPUT_DIR]/detector_[suffix].pth", nbits=16)
result, message = detector(watermarked_audio, sr)
```

## Training the HF AudioSeal model

We also provide the hyperparameter and training config (in Dora term, a "grid") to reproduce our checkpoints for AudioSeal in HuggingFace (which is also the one used to produce the results ported in the ICML paper). To get this, check the AudioCraft's watermarking [grid](https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/grids/watermarking/kbits.py). To reproduce the result, run the `dora grid` command:

```bash
AUDIOCRAFT_CONFIG=my_config.yaml AUDIOCRAFT_DSET=audio/voxpopuli dora grid watermarking.1315_kbits_seeds
```

## Troubleshooting

1. If you encounter the error `Unsupported formats` on Linux, the ffmpeg is not properly installed or superseded by other backends in your system. Try to instruct dora to use the libs you installed in your environment explicitly, i.e. adding them to `LD_LIBRARY_PATH`. If you use Anaconda, you can try:

```bash
LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH AUDIOCRAFT_DORA_DIR=my_config.yaml [dora run/grid command]
```
