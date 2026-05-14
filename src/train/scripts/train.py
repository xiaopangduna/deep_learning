import torch
from lovely_deep_learning.cli import LovelyLightningCLI

torch.set_float32_matmul_precision("medium")


def cli_main():
    LovelyLightningCLI(
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    cli_main()
