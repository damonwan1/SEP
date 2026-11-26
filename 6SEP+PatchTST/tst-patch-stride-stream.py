from pathlib import Path
import sys

from mixModel import PatchTST_backbone

sys.path.insert(1, str(Path(__file__).resolve().parents[1]))

from sep_codec import COMMON_DEFAULTS, Codec, main


def build_model(config, device):
    return PatchTST_backbone(
        config["seq_len"], config["patch_len"], config["stride"],
        config["n_heads"], config["vocab_dim"], config["batch_size"],
        n_layers=config["n_layers"], d_model=config["hidden_dim"],
        d_ff=config["ffn_dim"], device=device,
    )


def validate_model(config):
    patch_len = config["patch_len"]
    if not 1 <= patch_len <= config["seq_len"] or config["stride"] < 1:
        raise ValueError("patch_len must be between 1 and seq_len and stride must be positive")
    if config["hidden_dim"] != 256 or patch_len * config["vocab_dim"] != 256:
        raise ValueError("SEP+PatchTST requires hidden_dim=256 and patch_len * vocab_dim=256")
    if config["n_heads"] < 1 or 256 % config["n_heads"]:
        raise ValueError("n_heads must divide 256")
    if config["n_layers"] < 0 or config["ffn_dim"] < 1:
        raise ValueError("n_layers must be nonnegative and ffn_dim must be positive")


CODEC = Codec(
    name="SEP+PatchTST",
    magic=b"SEPPATCHTST\x01",
    defaults={
        **COMMON_DEFAULTS, "vocab_dim": 64, "seq_len": 64,
        "patch_len": 4, "stride": 1, "weight_decay": 0.0,
    },
    build_model=build_model,
    validate_model=validate_model,
    default_input=Path("../../../data/image"),
    default_prefix="_image",
)


if __name__ == "__main__":
    main(CODEC)
