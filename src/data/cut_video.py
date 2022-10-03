import logging
import os
from pathlib import Path

import click
from moviepy.editor import VideoFileClip

from src.settings import OUTPUT_PATH, VIDEO_PATH


@click.command()
@click.argument("begin", type=click.INT)
@click.argument("end", type=click.INT)
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_folder", type=click.Path())
def make_clip(begin, end, input_filepath=VIDEO_PATH, output_folder=OUTPUT_PATH):
    """Runs movie processing scripts to cut long movie into slices."""
    logger = logging.getLogger(__name__)
    logger.info("Making final data set from raw data")

    file_name = Path(input_filepath).stem
    clip = VideoFileClip(input_filepath).subclip(begin, end)
    clip.write_videofile(
        os.path.join(output_folder, f"{file_name}_cubclip_{begin}_{end}.mp4")
    )


if __name__ == "__main__":
    make_clip(0, 20)
