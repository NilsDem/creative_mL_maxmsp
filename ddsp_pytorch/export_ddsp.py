import torch
import yaml
from effortless_config import Config
from os import path, makedirs, system
from ddsp.model import DDSP
import soundfile as sf
import nn_tilde
from preprocess import get_files
import torchaudio
from resampler import Resampler


torch.set_grad_enabled(False)


class args(Config):
    RUN = "runs/run_44100"
    INFERENCE_SAMPLING_RATE = 44100
    MODEL_SAMPLING_RATE = 44100
    DATA = False
    OUT_DIR = "exports"
    REALTIME = True


args.parse_args()
makedirs(args.OUT_DIR, exist_ok=True)


class ScriptDDSP(nn_tilde.Module):

    def __init__(self, config, model_sampling_rate, inference_sampling_rate,
                 mean_loudness: float, std_loudness: float):
        super().__init__()

        ddsp = DDSP(**config["model"])
        pretrained = torch.load(path.join(args.RUN, "state.pth"),
                                map_location="cpu")
        ddsp.load_state_dict(pretrained)
        ddsp.eval()
        self.ddsp = ddsp

        self.ddsp.gru.flatten_parameters()

        self.register_buffer("mean_loudness", torch.tensor(mean_loudness))
        self.register_buffer("std_loudness", torch.tensor(std_loudness))

        if inference_sampling_rate != model_sampling_rate:
            self.resampler = Resampler(target_sr=inference_sampling_rate,
                                       model_sr=model_sampling_rate)
        else:
            self.resampler = None

        self.register_method(
            "forward",
            in_channels=2,
            in_ratio=config["model"]["block_size"],
            out_channels=1,
            out_ratio=1,
            input_labels=[
                f"(signal) Pitch",
                f"(signal) Loudness",
            ],
            output_labels=[f"(signal) audio out"],
            test_buffer_size=512,
        )

    @torch.jit.export
    def forward(self, x):
        n = x.shape[0]
        x = x[:1]

        if self.resampler is not None:
            x = self.resampler.to_model_sampling_rate(x)

        pitch, loudness = torch.split(x, 1, dim=1)
        pitch = pitch.squeeze(1)
        loudness = loudness.squeeze(1)

        pitch = pitch.reshape(1, -1, 1)
        loudness = loudness.reshape(1, -1, 1)
        loudness = (loudness - self.mean_loudness) / self.std_loudness

        out = self.ddsp.realtime_forward(pitch=pitch, loudness=loudness)

        # Consider batch
        out = out.reshape(1, 1, -1)
        out = out.repeat(n, 1, 1)

        if self.resampler is not None:
            out = self.resampler.from_model_sampling_rate(out)

        return out


with open(path.join(args.RUN, "config.yaml"), "r") as config:
    config = yaml.safe_load(config)

name = path.basename(path.normpath(args.RUN))

scripted_model = ScriptDDSP(
    config,
    args.MODEL_SAMPLING_RATE,
    args.INFERENCE_SAMPLING_RATE,
    config["data"]["mean_loudness"],
    config["data"]["std_loudness"],
)

scripted_model.export_to_ts("./exports/ddsp.ts")

### Export impulse
impulse = scripted_model.ddsp.reverb.build_impulse().reshape(-1).numpy()
sf.write(
    path.join(args.OUT_DIR, f"ddsp_{name}_impulse.wav"),
    impulse,
    config["preprocess"]["sampling_rate"],
)
