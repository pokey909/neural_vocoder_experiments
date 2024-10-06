import lightning.pytorch as pl
from lightning.pytorch.cli import LightningCLI, LightningArgumentParser
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import List, Dict, Tuple, Callable, Union
from torchaudio.transforms import MelSpectrogram
import numpy as np
import yaml
from importlib import import_module
from typing import Any, Mapping

from models.utils import get_window_fn
from models.hpn import HarmonicPlusNoiseSynth
from models.sf import SourceFilterSynth
from models.enc import VocoderParameterEncoderInterface
from models.utils import get_f0, freq2cent
from models.audiotensor import AudioTensor


class ScaledLogMelSpectrogram(MelSpectrogram):
    def __init__(self, window: str, **kwargs):
        super().__init__(window_fn=get_window_fn(window), **kwargs)

        self.register_buffer("log_mel_min", torch.tensor(torch.inf))
        self.register_buffer("log_mel_max", torch.tensor(-torch.inf))

    def forward(self, waveform: Tensor) -> Tensor:
        mel = super().forward(waveform).transpose(-1, -2)
        mel = AudioTensor(mel, hop_length=self.hop_length)
        log_mel = torch.log(mel + 1e-8)
        if self.training:
            self.log_mel_min.fill_(min(self.log_mel_min, torch.min(log_mel).item()))
            self.log_mel_max.fill_(max(self.log_mel_max, torch.max(log_mel).item()))
        return (log_mel - self.log_mel_min) / (self.log_mel_max - self.log_mel_min)


class DDSPVocoderCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.link_arguments(
            "model.hop_length", "model.feature_trsfm.init_args.hop_length"
        )

        parser.link_arguments(
            "model.sample_rate", "model.feature_trsfm.init_args.sample_rate"
        )
        parser.link_arguments(
            "model.window",
            "model.feature_trsfm.init_args.window",
        )

        # parser.set_defaults(
        #     {
        #         "model.mel_model": {
        #             "class_path": "models.mel.Mel2Control",
        #         },
        #         "model.glottal": {
        #             "class_path": "models.synth.GlottalFlowTable",
        #         },
        #         "model.mel_trsfm": {
        #             "class_path": "WrappedMelSpectrogram",
        #         },
        #         "model.voice_ttspn": {
        #             "class_path": "models.tspn.TTSPNEncoder",
        #             "init_args": {
        #                 "out_channels": 2,
        #             },
        #         },
        #         "model.noise_ttspn": {
        #             "class_path": "models.tspn.TTSPNEncoder",
        #             "init_args": {
        #                 "out_channels": 2,
        #             },
        #         },
        #     }
        # )


class DDSPVocoder(pl.LightningModule):
    def __init__(
        self,
        decoder: Union[SourceFilterSynth, HarmonicPlusNoiseSynth],
        feature_trsfm: ScaledLogMelSpectrogram,
        criterion: nn.Module,
        encoder_class_path: str = "models.enc.VocoderParameterEncoderInterface",
        encoder_init_args: Dict = {},
        window: str = "hanning",
        sample_rate: int = 24000,
        hop_length: int = 120,
        detach_f0: bool = False,
        detach_voicing: bool = False,
        train_with_true_f0: bool = False,
        l1_loss_weight: float = 0.0,
        f0_loss_weight: float = 1.0,
        voicing_loss_weight: float = 1.0,
        inverse_target: bool = False,
    ):
        super().__init__()

        # self.save_hyperparameters()

        self.decoder = decoder
        self.criterion = criterion
        self.feature_trsfm = feature_trsfm

        module_path, class_name = encoder_class_path.rsplit(".", 1)
        module = import_module(module_path)
        split_sizes, trsfms, args_keys = self.decoder.split_sizes_and_trsfms
        self.encoder = getattr(module, class_name)(
            split_sizes=split_sizes,
            trsfms=trsfms,
            args_keys=args_keys,
            **encoder_init_args,
        )

        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.l1_loss_weight = l1_loss_weight
        self.f0_loss_weight = f0_loss_weight
        self.voicing_loss_weight = voicing_loss_weight
        self.detach_f0 = detach_f0
        self.detach_voicing = detach_voicing
        self.train_with_true_f0 = train_with_true_f0
        self.inverse_target = inverse_target

    def forward(self, feats: torch.Tensor):
        # (f0, *other_params, voicing_logits) = self.encoder(feats)
        params = self.encoder(feats)

        f0 = params.pop("f0")
        params["phase"] = f0 / self.sample_rate

        voicing_logits = params.pop("voicing_logits", None)
        if voicing_logits is not None:
            params["voicing"] = torch.sigmoid(voicing_logits)

        return (
            f0,
            self.decoder(**params),
            params.get("voicing", None),
        )

    def f0_loss(self, f0_hat, f0):
        return F.l1_loss(torch.log(f0_hat + 1e-3), torch.log(f0 + 1e-3))

    def on_train_start(self) -> None:
        if self.logger is not None:
            self.logger.watch(self.encoder, log_freq=1000, log="all", log_graph=False)
            if len(tuple(self.decoder.parameters())) > 0:
                self.logger.watch(
                    self.decoder, log_freq=1000, log="all", log_graph=False
                )

    def on_train_end(self) -> None:
        if self.logger is not None:
            self.logger.experiment.unwatch(self.encoder)
            if len(tuple(self.decoder.parameters())) > 0:
                self.logger.experiment.unwatch(self.decoder)

    def training_step(self, batch, batch_idx):
        x, f0_in_hz = batch
        low_res_f0 = f0_in_hz[:, :: self.hop_length]

        mask = f0_in_hz > 50
        low_res_mask = mask[:, :: self.hop_length]

        feats = self.feature_trsfm(x)
        (f0_hat, *other_params, voicing_logits) = self.encoder(feats)

        minimum_length = min(f0_hat.shape[1], low_res_f0.shape[1])
        low_res_f0 = low_res_f0[:, :minimum_length]
        low_res_mask = low_res_mask[:, :minimum_length]
        f0_hat = f0_hat[:, :minimum_length]

        if voicing_logits is not None:
            voicing_logits = voicing_logits[:, :minimum_length]
            voicing = torch.sigmoid(
                torch.detach(voicing_logits) if self.detach_voicing else voicing_logits
            )
        else:
            voicing = None

        f0_for_decoder = torch.detach(f0_hat) if self.detach_f0 else f0_hat

        if self.train_with_true_f0:
            phase = (
                torch.where(low_res_mask, low_res_f0, f0_for_decoder) / self.sample_rate
            )
        else:
            phase = f0_for_decoder / self.sample_rate

        if self.inverse_target:
            x_hat, invesre_x = self.decoder(
                phase,
                *other_params,
                voicing=voicing,
                target=AudioTensor(x),
            )
            x_hat = x_hat.as_tensor()
            x = invesre_x.as_tensor()
        else:
            x_hat = self.decoder(
                phase,
                *other_params,
                voicing=voicing,
            ).as_tensor()
        # f0_hat = f0_hat.as_tensor().rename(None)

        x = x[..., : x_hat.shape[-1]]
        mask = mask[:, : x_hat.shape[1]]
        loss = self.criterion(x_hat, x)
        l1_loss = torch.sum(mask.float() * (x_hat - x).abs()) / mask.count_nonzero()

        f0_loss = self.f0_loss(f0_hat[low_res_mask], low_res_f0[low_res_mask])

        self.log("train_l1_loss", l1_loss, prog_bar=False, sync_dist=True)
        self.log("train_f0_loss", f0_loss, prog_bar=False, sync_dist=True)
        if self.l1_loss_weight > 0:
            loss = loss + l1_loss * self.l1_loss_weight
        if self.f0_loss_weight > 0:
            loss = loss + f0_loss * self.f0_loss_weight

        if voicing is not None:
            voicing_loss = F.binary_cross_entropy_with_logits(
                voicing_logits, low_res_mask.float()
            )
            self.log("train_voicing_loss", voicing_loss, prog_bar=False, sync_dist=True)
            if self.voicing_loss_weight > 0:
                loss = loss + voicing_loss

        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        self.tmp_val_outputs = []

    def validation_step(self, batch, batch_idx):
        x, f0_in_hz = batch

        mask = f0_in_hz > 50
        num_nonzero = mask.count_nonzero()

        feats = self.feature_trsfm(x)
        f0_hat, x_hat, voicing = self(feats)
        f0_hat = f0_hat.as_tensor()
        x_hat = x_hat.as_tensor()

        x_hat = x_hat[:, : x.shape[-1]]
        x = x[..., : x_hat.shape[-1]]
        mask = mask[:, : x_hat.shape[1]]
        loss = self.criterion(x_hat, x)
        l1_loss = torch.sum(mask.float() * (x_hat - x).abs()) / num_nonzero

        f0_in_hz = f0_in_hz[:, :: self.hop_length]
        f0_mask = mask[:, :: self.hop_length]
        minimum_length = min(f0_hat.shape[1], f0_in_hz.shape[1], f0_mask.shape[1])
        f0_in_hz = f0_in_hz[:, :minimum_length]
        f0_mask = f0_mask[:, :minimum_length]
        f0_hat = f0_hat[:, :minimum_length]
        f0_loss = self.f0_loss(f0_hat[f0_mask], f0_in_hz[f0_mask])

        if self.l1_loss_weight > 0:
            loss = loss + l1_loss * self.l1_loss_weight
        if self.f0_loss_weight > 0:
            loss = loss + f0_loss * self.f0_loss_weight

        if voicing is not None:
            voicing = voicing.as_tensor()[:, :minimum_length]
            voicing_loss = F.binary_cross_entropy(voicing, f0_mask.float())
            if self.voicing_loss_weight > 0:
                loss = loss + voicing_loss

            self.tmp_val_outputs.append((loss, l1_loss, f0_loss, voicing_loss))

            return loss

        self.tmp_val_outputs.append((loss, l1_loss, f0_loss))
        print("YAYAYA")
        return {'loss': loss, 'predictions': x_hat, 'inputs': x}

    def on_validation_epoch_end(self) -> None:
        outputs = self.tmp_val_outputs
        avg_loss = sum(x[0] for x in outputs) / len(outputs)
        avg_l1_loss = sum(x[1] for x in outputs) / len(outputs)
        avg_f0_loss = sum(x[2] for x in outputs) / len(outputs)

        self.log("val_loss", avg_loss, prog_bar=True, sync_dist=True)
        self.log("val_l1_loss", avg_l1_loss, prog_bar=False, sync_dist=True)
        self.log("val_f0_loss", avg_f0_loss, prog_bar=False, sync_dist=True)

        if len(outputs[0]) > 3:
            avg_voicing_loss = sum(x[3] for x in outputs) / len(outputs)
            self.log(
                "val_voicing_loss", avg_voicing_loss, prog_bar=False, sync_dist=True
            )
        delattr(self, "tmp_val_outputs")

    def on_test_start(self) -> None:
        self.tmp_test_outputs = []

        return super().on_test_start()

    def test_step(self, batch, batch_idx):
        x, f0_in_hz = batch
        f0_in_hz = f0_in_hz[:, :: self.hop_length].cpu().numpy()

        feats = self.feature_trsfm(x)
        _, x_hat, _ = self(feats)
        x_hat = x_hat.as_tensor()

        x = x[..., : x_hat.shape[-1]]
        mss_loss = self.criterion(x_hat, x).item()

        x_hat = x_hat.cpu().numpy().astype(np.float64)
        x = x.cpu().numpy().astype(np.float64)
        N = x_hat.shape[0]
        f0_hat_list = []
        for i in range(N):
            f0_hat, _ = get_f0(x_hat[i], self.sample_rate, f0_floor=65)
            f0_hat_list.append(f0_hat)

        f0_hat = np.stack(f0_hat_list, axis=0)
        f0_in_hz = f0_in_hz[:, : f0_hat.shape[1]]
        f0_hat = f0_hat[:, : f0_in_hz.shape[1]]
        f0_in_hz = np.maximum(f0_in_hz, 80)
        f0_hat = np.maximum(f0_hat, 80)
        f0_loss = np.mean(np.abs(freq2cent(f0_hat) - freq2cent(f0_in_hz)))

        self.tmp_test_outputs.append((mss_loss, f0_loss, N))

        return mss_loss, f0_loss, N

    def on_test_epoch_end(self) -> None:
        outputs = self.tmp_test_outputs
        weights = [x[-1] for x in outputs]
        avg_mss_loss = np.average([x[0] for x in outputs], weights=weights)
        avg_f0_loss = np.average([x[1] for x in outputs], weights=weights)

        self.log_dict(
            {
                "avg_mss_loss": avg_mss_loss,
                "avg_f0_loss": avg_f0_loss,
            },
            prog_bar=True,
            sync_dist=True,
        )
        delattr(self, "tmp_test_outputs")
        return

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, _, rel_path = batch

        assert x.shape[0] == 1, "batch size must be 1 for inference"
        # hard coded 6 duration and 0.3 seconds overlap
        frame_length = 6 * self.sample_rate
        hop_length = int(5.7 * self.sample_rate)
        overlap = frame_length - hop_length

        h = F.pad(x, (0, frame_length))

        h = h.unfold(1, frame_length, hop_length).squeeze(0)

        feats = self.feature_trsfm(h)

        _, x_hat, _ = self(feats)

        x_hat = x_hat[:, :frame_length]
        if x_hat.shape[1] < frame_length:
            overlap = x_hat.shape[1] - hop_length
            frame_length = x_hat.shape[1]
        p = torch.arange(overlap, device=x.device) / overlap

        ola = x.new_zeros((1, hop_length * (x_hat.shape[0] - 1) + frame_length))
        for i in range(x_hat.shape[0]):
            addon = x_hat[i].clone()
            if i:
                ola[:, i * hop_length : i * hop_length + overlap] *= 1 - p
                addon[:overlap] *= p
            ola[:, i * hop_length : i * hop_length + frame_length] += addon

        ola = ola[:, : x.shape[1]]

        return AudioTensor(ola), None
