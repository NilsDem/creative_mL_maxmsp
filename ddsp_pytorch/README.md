The **Differentiable Digital Signal Processing** (**DDSP**)[1] model was introduced in 2019 by J. Engel et al. from Google Magenta research team. It integrates components from classic signal processing methods into a deep learning framework to enable raw audio waveform synthesis. In particular, a signal model is used to directly control the **pitch** and **loudness** of the generated signal. The code implementation and tutorials are available on their [website](https://magenta.tensorflow.org/ddsp).

As illustrated in the figure bellow (taken from the original paper[1]), the model is composed of DSP components and a deterministic auto-encoder to learn the synthesis parameters through a *latent* representation $\mathbf{z}$. The audio signal is split into successive time frames from which we extract :
- the **pitch**, using a pitch estimation algorithm (in the original paper they rely on **CREPE**[2] which consists of a convolutional neural network),
- the **loudness**, directly computed from the raw waveform using a log-scaled, A-weighted power spectrum,
- the ***latent* representation**, using a Gated Recurrent Unit (GRU) encoder network on the first 30 Mel Frequency Cepstral Coefficients (MFCC) of each frame, that is supposed to capture the remaining signal features associated to timbre.

The sound is generated following Spectral Modelling Synthesis (SMS) which consists of adding the contributions from an additive synthesizer (combination of harmonic oscillators) and a filtered white noise. The extracted features are then fed to the decoder (also a GRU network) to generate the harmonic distribution and the filter coefficients. The extracted pitch is directly used as the fundamental frequency of the harmonic synthesizer. 

The model is trained to encode and reconstruct the signal by minimizing the multiscale spectral distance 

![DDSP archi](https://storage.googleapis.com/ddsp/additive_diagram/ddsp_autoencoder.png)

Thanks to the DSP components, the model is relatively lightweight and fast to train (~2 hours) with a very limited set of parameters compared to large autoregressive or adversarial models such as WaveNet[3] or GANSynth[4]. Moreover, it can be used to generate 48kHZ audio in real-time on a standard laptop CPU. However, the range of sounds that can be generated is restricted especially to monophonic signals.

Today, you will learn to train your own DDSP model in `Pytorch` and use it for real-time synthesis in MaxMSP, with explicit controls on pitch and loudness, using  the `nn~` external ([github](https://github.com/acids-ircam/nn_tilde.git)).

Our code is mostly based on the implementation provided in [https://github.com/acids-ircam/ddsp_pytorch](https://github.com/acids-ircam/ddsp_pytorch.git) which contains the implementation of the DDSP model in `Pytorch` that can be exported into a torchscript model to be used in `PureData`for real-time synthesis. In addition, for this tutorial we:

- added the code implementation to export and use a pretrained DDSP model in MaxMSP using `nn~`,
- implemented a streamable version of `torch.crepe` to extract pitch and loudness signals in real-time in MaxMSP.

### Install

You can use Google colab (we provide a colab notebook with the command lines for each steps of this tutorial in the `ddsp_tutorial.ipynb` file) or you can setup the python environment on your computer. 
For the later, you will need to:

1. Clone this github repository:
```bash
$ git clone https://github.com/NilsDem/creative_mL_maxmsp.git
```

2. Go to the ddsp session folder:
```bash
$ cd ddsp_pytorch
```

3. Setup your Python environment: create a virtual environment with `python 3.12` and install the requirements. For instance, you can install anaconda or [minconda](https://docs.anaconda.com/miniconda/) and then in a terminal:
```bash
$ conda create --name myenv python==3.11
(myenv) $ pip install -r requirements.txt
```

*Note: You can install and use an IDE such as Visual Studio Code for development if you want to edit the code as well.*

Finally, install the `nn~` MaxMSP external. Download the latest release on the [official github repository](https://github.com/acids-ircam/nn_tilde), uncompress the `.tar.gz` archive in the package folder of you Max installation (i.e. in `Documents/Max 8/Packages/`). 

To test the installation open a new patch MaxMSP and instantiate an `nn~` object. Right click on the `nn~` object to open the help patch. 

### Process the data

For this tutorial, we propose to use the *Bach violin dataset* from [5] that you can download [here](https://github.com/salu133445/bach-violin-dataset/releases). It is composed of 6.5 hours of public professional recordings of Bach's sonatas and partitas for solo violin.

1. Download the `bach-violin-dataset-v1.0.zip` file and unzip into a new folder, for instance in `./data/bach_violin/`.

2. Edit the `config.yaml` and specify the data location and preprocess parameters:
```yaml
data:
  data_location: /<replace-with-absolute-filepath>/data/bach_violin
  extensions: ["mp3", "opus", "wav", "flac"]

preprocess:
  sampling_rate: &samplingrate 44100
  signal_length: &signallength 131072
  block_size: &blocksize 512 # must be a power of 2 if using realtime
  oneshot: false # crop every audio file to exactly signal length
  out_dir: /<replace-with-absolute-filepath>/data/bach_violin/ddsp_processed_44100
```

3. In a terminal with your python environment setup, go to the `ddsp_pytorch` directory and launch the preprocessing script:
```bash
(myenv) $ python preprocess.py
```
*Note: It is preferable to use a GPU to accelerate data processing.*

This will store the processed data into 3 files: `loudness.npy`, `pitchs.npy` and `signals.npy`.

To save time, you can download the processed files [here]().

### Train your DDSP

You can then train your model using 

```bash
(myenv) $ python train.py --name ddsp_bach_violin --steps 1000000 --batch 16 
```

This script uses the same `config.yaml` file containing the following model and training parameters configuration:
```yaml
model:
  hidden_size: 1024
  n_harmonic: 100
  n_bands: 65
  sampling_rate: *samplingrate
  block_size: *blocksize

train:
  scales: [4096, 2048, 1024, 512, 256, 128]
  overlap: .75
```

The trained ddsp model is saved in the `state.pth` file located in the `ddsp_pytorch/runs/ddsp_bach_violin/` directory.

*Note: You can use `tensorboard` to monitor the training process:*
```bash
(myenv) $ tensorboard --logdir=./runs --bind_all --port 6008
```

To save time, we provide you with a pretrained ddsp model trained on the Bach Violin dataset that you can download [here]().

### Export your DDSP for real-time audio synthesis in MaxMSP

To use your DDSP model for real-time audio synthesis in MaxMSP, you need to export it using the `nn_tilde` python package that relies on `torchscript`. We also use the [`cached_conv`](https://github.com/acids-ircam/cached_conv.git) python library to make the model streamable. You can find the code implementation in `export_ddsp.py`. Here is a quick overview and explanation of the code:

1. First, make your model streamable and use cached convolution by setting:
```python
import cached_conv as cc

cc.use_cached_conv(True)
```

2. Then, you need to create a class that will have your pretrained model loaded and inherit the `nn_tilde.Module`class. For instance:

```python
class ScriptDDSP(nn_tilde.Module):

    def __init__(self, config):
        super().__init__()
```

3. You need to register the `forward` method as scriptable. For instance, in the class constructor you need to indicate:
```python 
        self.register_method(
            "forward",
            in_channels=2,  # number of input channels, here it is 2 because the user will give pitch and loudness signals in MaxMSP
            in_ratio=BLOCK_SIZE,
            out_channels=1,  # number of output channels, here it is one as we only have the generated audio signal
            out_ratio=1,
            input_labels=[
                f"(signal) Pitch",
                f"(signal) Loudness",
            ],
            output_labels=[f"(signal) audio out"],
            test_buffer_size=512,  
        )
```
and then implement the `forward` method with the decorator `@torch.jit.export` :
```python
    @torch.jit.export
    def forward(self, x):
      ...
      return out
```

4. Finally, save the scripted model into a `torchscript` format:
```python 
scripted_model = ScriptDDSP(...)

scripted_model.export_to_ts("./exports/ddsp.ts")
```

To run the export script and export your ddsp model use:
```bash
(myenv) $ python export_ddsp.py --run ./runs/ddsp_bach_violin
```

Your model is stored as a `.ts` file in the `exports` directory.

### Export a streamable version of the CREPE model for real-time pitch estimation

The CREPE model to extract the pitch from a signal is a pretrained convolutional neural network. Hence, we also need to export a streamable version of this model using the `cached_conv` and `nn_tilde` python modules. We provide a pretrained CREPE model in the `./crepe/assets/tiny.pth`. Then, you can run the `export_crepe.py` python script to export the `torchscript` CREPE model:

```bash
(myenv) $ python export_crepe.py
```

This will save a `crepe_stream.ts` file in the `exports/` directory.

### Inference in MaxMSP

1. Add the `exports` directory to MaxMSP file path in "Options > File Preferences ..." so that you can use your `.ts` models with `nn~`. 

2. You can find an example of a MaxMSP patch to use DDSP for real-time synthesis with explicit control on pitch and loudness in `./patchs/ddsp.maxpat`.


You will also need the [HISSTools Impulse Response Toolbox (HIRT)](https://cycling74.com/packages/hisstools-impulse-response-toolbox-hirt) Max package. Go to the Max Package Manager and install it. Then, create a `hirt.convolver~` object and open it (CMD + double click on the object) and drag and drop the `ddsp_run_44100_impulse.wav` audio file we exported with the ddsp torchscript model.