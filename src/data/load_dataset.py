import logging
import os

import click
import yaml
from dotenv import find_dotenv, load_dotenv
from roboflow import Roboflow


@click.command()
@click.option(
    "-cf",
    "--config_path",
    type=click.Path(exists=True),
    help="Path to config file",
    required=True,
)
def load_dataset(config_path):
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = logging.getLogger(__name__)
    logger.info("Roboflow authentication...")
    ROBOFLOW_KEY = os.environ.get("ROBOFLOW_KEY")
    rf = Roboflow(api_key=ROBOFLOW_KEY)
    project = rf.workspace("plate-tsusp").project("russian-plate")

    logger.info("Start dataset loading...")
    dataset = project.version(3).download(
        model_format=config["data"]["model_format"],
        location=config["data"]["dataset_location"],
    )
    return dataset


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    load_dotenv(find_dotenv())
    load_dataset()
