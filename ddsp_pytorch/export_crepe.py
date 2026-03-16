import torch.nn as nn
import nn_tilde
import os

from crepe.model import Crepev1
import torchaudio
import torch
import librosa


class StreamCrepe(nn_tilde.Module):

    def __init__(self,
                 inference_sampling_rate=44100,
                 hop_length: int = 512,
                 n_fft_loudness=2048,
                 stream: bool = True):
        super(StreamCrepe, self).__init__()

        self.model = Crepev1(model="tiny")
        self.model.load_state_dict(torch.load("./crepe/assets/tiny.pth"))

        self.model.eval()

        self.hop_length = hop_length
        self.stream = stream
        self.register_buffer('audio_buffer',
                             torch.zeros((n_fft_loudness - self.hop_length)))

        self.register_buffer('audio_buffer_spec',
                             torch.zeros((n_fft_loudness - self.hop_length)))

        # Loudness compute
        self.transform = torchaudio.transforms.Spectrogram(
            n_fft=n_fft_loudness,
            win_length=n_fft_loudness,
            hop_length=hop_length,
            center=False)
        # center=not stream)

        f = librosa.fft_frequencies(sr=44100, n_fft=n_fft_loudness)
        self.a_weight = librosa.A_weighting(f)
        self.a_weight = torch.from_numpy(self.a_weight)

        self.n_fft_loudness = n_fft_loudness

        if inference_sampling_rate != 16000:
            self.resampler = torchaudio.transforms.Resample(
                inference_sampling_rate, 16000)
        else:
            self.resampler = None

        self.register_method(
            "forward",
            in_channels=1,
            in_ratio=1,
            out_channels=2,
            out_ratio=hop_length,
            input_labels=[
                f"(signal) Input",
            ],
            output_labels=[f"(signal) Pitch", f"(signal) Loudness"],
            test_buffer_size=2048,
        )

    def to_local_average_cents(self, salience: torch.Tensor):
        """
        find the weighted average cents near the argmax bin
        """

        cents_mapping = torch.linspace(0, 7180, 360) + 1997.3794084376191

        if len(salience.shape) == 1:
            salience = salience.unsqueeze(0)

        out = []

        for s in salience:

            #if center is None:
            center = int(torch.argmax(s))

            start = max(0, center - 4)
            end = min(len(s), center + 5)
            s = s[start:end]
            product_sum = torch.sum(s * cents_mapping[start:end])

            weight_sum = torch.sum(s)
            out.append(product_sum / weight_sum)

        out = torch.stack(out, 0)
        return out

    #@torch.jit.export
    def forward(self, audio: torch.Tensor):
        """
        Perform pitch estimation on given audio

        Parameters
        ----------
        audio : np.ndarray [shape=(N,) or (N, C)]
            The audio samples. Multichannel audio will be downmixed.
        sr : int
            Sample rate of the audio samples. The audio will be resampled if
            the sample rate is not 16 kHz, which is expected by the model.
        model_capacity : 'tiny', 'small', 'medium', 'large', or 'full'
            String specifying the model capacity; see the docstring of
            :func:`~crepe.core.build_and_load_model`
        viterbi : bool
            Apply viterbi smoothing to the estimated pitch curve. False by default.
        center : boolean
            - If `True` (default), the signal `audio` is padded so that frame
            `D[:, t]` is centered at `audio[t * hop_length]`.
            - If `False`, then `D[:, t]` begins at `audio[t * hop_length]`
        step_size : int
            The step size in milliseconds for running pitch estimation.
        verbose : int
            Set the keras verbosity mode: 1 (default) will print out a progress bar
            during prediction, 0 will suppress all non-error printouts.

        Returns
        -------
        A 4-tuple consisting of:

            time: np.ndarray [shape=(T,)]
                The timestamps on which the pitch was estimated
            frequency: np.ndarray [shape=(T,)]
                The predicted pitch values in Hz
            confidence: np.ndarray [shape=(T,)]
                The confidence of voice activity, between 0 and 1
            activation: np.ndarray [shape=(T, 360)]
                The raw activation matrix
        """

        print("audio", audio.shape)
        audio_len = audio.shape[-1]

        ## Compute Loudness

        # Loudness compute
        print(audio.shape)

        audio_spec = audio[:1].reshape(-1).clone()
        audio_spec = torch.cat((self.audio_buffer_spec, audio_spec))

        S = self.transform(audio_spec).float()
        S = torch.log(abs(S) + 1e-7)
        S = S + self.a_weight.reshape(-1, 1)
        S = torch.mean(S, 0)
        S = S.reshape(1, 1, -1).float()
        print("spectrogram", S.shape)

        #if self.stream:
        #    S = S[..., :-self.n_fft_loudness // self.hop_length]
        #    print(S.shape)

        S = torch.nan_to_num(S, nan=0.0)
        #S = torch.nn.functional.interpolate(S, audio_len, mode="linear")

        S = S  # * 1.5 + 10

        self.audio_buffer_spec = audio_spec[..., -(self.n_fft_loudness -
                                                   self.hop_length):]

        ## Compute frequency

        # Audio resampling

        audio = self.resampler(audio)

        print("audio r", audio.shape)

        # make 1024-sample frames of the audio with given hop length
        n = audio.shape[0]
        audio = audio[:1].reshape(-1)
        if self.stream == True:
            audio = torch.cat((self.audio_buffer, audio))

        #x = torch.nn.functional.pad(audio, (512, 512),
        #                            mode='constant',
        #                            value=0.)
        x = audio
        frames = []
        for i in range((x.shape[-1] - 3 * self.hop_length) // self.hop_length):
            frames.append(x[...,
                            i * self.hop_length:i * self.hop_length + 1024])

        frames = torch.stack(frames, dim=0)

        frames -= torch.mean(frames, dim=1)[:, None]
        frames /= torch.std(frames, dim=1)[:, None]
        # pitch inference
        y = self.model(frames)
        cents = self.to_local_average_cents(y)

        frequency = 10 * 2**(cents / 1200)

        print("audio_resam", audio.shape)
        print("frequency", frequency.shape)
        #if self.stream:
        #    frequency = frequency[..., :-self.n_fft_loudness //
        #                          self.hop_length + 4:]

        frequency = frequency.reshape(1, 1, -1)

        print("frequency out", frequency.shape)

        # Rescale and interpolate to audio input size
        frequency = torch.nan_to_num(frequency, nan=0.0)
        frequency = torch.nn.functional.interpolate(frequency,
                                                    S.shape[-1],
                                                    mode="linear")
        print("frequency out resamp", frequency.shape)
        # update buffer
        self.audio_buffer = audio[...,
                                  -(self.n_fft_loudness - self.hop_length):]

        # Cat
        out = torch.cat((frequency, S), 1)
        out = out.repeat(n, 1, 1)
        return out


if __name__ == "__main__":

    model = StreamCrepe()

    print("test with audio buffer of size 2048")
    audio = torch.randn(2048)
    out = model(audio.reshape(1, 1, -1))
    print("Output shape : ", out.shape)

    model.export_to_ts("./exports/crepe_stream" + ".ts")

#scripted = torch.jit.script(model)
