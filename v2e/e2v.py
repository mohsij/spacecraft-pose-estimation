#!/usr/bin/env python

import glob
import argparse
import importlib
import sys

import os

import numpy as np
import pandas as pd

from tqdm import tqdm

from v2ecore.v2e_args import v2e_check_dvs_exposure_args

from v2ecore.v2e_utils import set_output_dimension
from v2ecore.v2e_utils import set_output_folder
from v2ecore.v2e_utils import v2e_quit

from v2ecore.renderer import EventRenderer, ExposureMode

from pathlib import Path

def makedir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def e2v_args(parser):
    parser.add_argument("--events_file", type=str, help="csv input with each line in the inivation format:\n t(s), x, y, pol")

    parser.add_argument(
        "--delim_whitespace", action="store_true", default=False,
        help="Is the csv file separated by whitespace instead of comma?.")

    parser.add_argument(
        "--swap_xy", action="store_true", default = False,
        help="If this file was output by v2e then the second column in the csv will be y.")

    parser.add_argument(
        "--microseconds_timestamp", action="store_true", default = False,
        help="Prophesee timestamp format is in microseconds whereas inivation format is in seconds.")

    parser.add_argument(
        "--milliseconds_timestamp", action="store_true", default = False,
        help="Prophesee timestamp format is in microseconds whereas inivation format is in seconds.")

    parser.add_argument(
        "--avi_frame_rate", type=int, default=30,
        help="frame rate of output AVI video files; "
             "only affects playback rate. ")

    parser.add_argument(
        "--dvs_vid", type=str, default="dvs-video.avi",
        help="Output DVS events as AVI video at frame_rate. To suppress, supply argument None.")

    parser.add_argument(
        "--dvs_vid_full_scale", type=int, default=2,
        help="Set full scale event count histogram count for DVS videos "
             "to be this many ON or OFF events for full white or black.")

    parser.add_argument(
        "--no_preview", action="store_true", default = False,
        help="disable preview in cv2 windows for faster processing.")

    parser.add_argument(
        "--dvs_exposure", nargs='+', type=str, default=['duration', '0.01'],
        help="R|Mode to finish DVS frame event integration:"
             "\n\tduration time: Use fixed accumulation time in seconds, e.g. "
             "\n\t\t--dvs_exposure duration .005; "
             "\n\tcount n: Count n events per frame,e.g."
             "\n\t\t-dvs_exposure count 5000;"
             "\n\tarea_count M N: frame ends when any area of N x N "
             "pixels fills with M events, e.g."
             "\n\t\t-dvs_exposure area_count 500 64")

    parser.add_argument("--output_folder", type=str, default='.', help="Output folder for the video.")

    parser.add_argument(
        "--output_width", type=int, default=1280, help="Width of output DVS data in pixels. ")

    parser.add_argument(
        "--output_height", type=int, default=720, help="Height of output DVS data in pixels. ")

    return parser

def main():
    parser = argparse.ArgumentParser(
        description='e2v: generate event frames video from DVS events csv.')

    parser = e2v_args(parser)
    args = parser.parse_args()

    avi_frame_rate = args.avi_frame_rate
    dvs_vid = args.dvs_vid
    dvs_vid_full_scale = args.dvs_vid_full_scale
    preview = not args.no_preview

    exposure_mode, exposure_val, area_dimension = \
        v2e_check_dvs_exposure_args(args)

    output_width = args.output_width
    output_height = args.output_height

    input_file = args.events_file

    eventRenderer = EventRenderer(
        output_path=args.output_folder,
        dvs_vid=dvs_vid, preview=preview, full_scale_count=dvs_vid_full_scale,
        exposure_mode=exposure_mode,
        exposure_value=exposure_val,
        area_dimension=area_dimension,
        avi_frame_rate=args.avi_frame_rate)

    makedir(os.path.join(args.output_folder, 'event-frames'))

    column_names = ['t', 'x', 'y', 'p']
    if args.swap_xy:
        column_names = ['t', 'y', 'x', 'p']

    chunksize = 10 ** 16
    first_event_time = None
    with pd.read_csv(input_file, header=None, comment='#', delim_whitespace=args.delim_whitespace, names=column_names, chunksize=chunksize) as reader:
        for chunk in tqdm(reader):
            events = chunk[['t', 'x', 'y', 'p']].values
            events = events.astype(np.int64)

            # Convert 0,1 to expected -1/+1 polarity.
            events[:, -1] = (events[:, -1] * 2) - 1
            # Treat all events the same
            events[:, -1] = 1

            if first_event_time == None:
                first_event_time = events[0, 0]
            #events[:,0] = events[:, 0] - first_event_time
            if args.microseconds_timestamp:
                events[:, 0] = events[:, 0] / 1000000.0
            elif args.milliseconds_timestamp:
                events[:, 0] = events[:, 0] / 1000.0
            print(events[-5:, 0])
            eventRenderer.render_events_to_frames(
                events, height=output_height, width=output_width, output_to_images=True)

if __name__ == "__main__":
    main()
    v2e_quit()