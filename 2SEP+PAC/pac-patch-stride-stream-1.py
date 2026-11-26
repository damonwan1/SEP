from pathlib import Path
import sys

from shiying1 import mlpCompressor

sys.path.insert(1, str(Path(__file__).resolve().parents[1]))

from sep_codec import COMMON_DEFAULTS, Codec, main


def build_model(config, device):
    return mlpCompressor(
        config["vocab_size"], config["vocab_dim"], config["hidden_dim"],
        config["n_layers"], config["ffn_dim"], config["n_heads"],
        config["batch_size"], seq_len=config["seq_len"], device=device,
    )


def validate_model(config):
    if config["vocab_dim"] != 16 or config["hidden_dim"] != 256:
        raise ValueError("SEP+PAC requires vocab_dim=16 and hidden_dim=256")
    if config["seq_len"] < 16:
        raise ValueError("seq_len must be at least 16")


CODEC = Codec(
    name="SEP+PAC",
    magic=b"SEPPAC\x01",
    defaults={**COMMON_DEFAULTS, "vocab_dim": 16, "seq_len": 48, "weight_decay": 1e-7},
    build_model=build_model,
    validate_model=validate_model,
    default_input=Path("../../../data/dickens"),
    default_prefix="_dickens",
)


if __name__ == "__main__":
    main(CODEC)
