import torch

from ts_benchmark.models.advanced_model_base import Advanced_Model_Base
from .models.patchmlp_model import PatchMLPModel

MODEL_HYPER_PARAMS = {
    "data": "custom",
    "use_amp": 0,
    "label_len": 48,
    "enc_in": 21,
    "dec_in": 21,
    "c_out": 21,
    "e_layers": 1,
    "d_layers": 1,
    "d_model": 1024,
    "d_ff": 2048,
    "freq": "h",
    "factor": 1,
    "distil": 1,
    "embed": "timeF",
    "n_heads": 8,
    "seq_len": 96,
    "activation": "gelu",
    "output_attention": 0,
    "use_norm": 1,
    "dropout": 0.1,
    "moving_avg": 13,
    "batch_size": 32,
    "lradj": "type1",
    "lr": 0.0001,
    "num_epochs": 20,
    "num_workers": 0,
    "loss": "MSE",
    "patience": 5,
}

class PatchMLP(Advanced_Model_Base):
    def __init__(self, **kwargs):
        super(PatchMLP, self).__init__(MODEL_HYPER_PARAMS, **kwargs)

    @property
    def model_name(self):
        return "PatchMLP"

    def _init_model(self):
        return PatchMLPModel(self.config)

    def _process(self, input, target, input_mark, target_mark):
        dec_inp = torch.zeros_like(target[:, -self.config.pred_len:, :]).float()
        dec_inp = (
            torch.cat([target[:, : self.config.label_len, :], dec_inp], dim=1)
            .float()
            .to(input.device)
        )
        # encoder - decoder
        if self.config.use_amp == 1:
            with torch.cuda.amp.autocast():
                if self.config.output_attention == 1:
                    outputs = self.model(
                        input, input_mark, dec_inp, target_mark
                    )[0]
                else:
                    outputs = self.model(
                        input, input_mark, dec_inp, target_mark
                    )

                outputs = outputs[:, -self.config.pred_len:, :]
        else:
            if self.config.output_attention:
                outputs = self.model(input, input_mark, dec_inp, target_mark)[0]
            else:
                outputs = self.model(input, input_mark, dec_inp, target_mark)

        return {"output": outputs}
