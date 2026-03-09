from lightning.pytorch.cli import LightningCLI


def cli_main():
    LightningCLI(
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    cli_main()
