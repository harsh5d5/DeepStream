#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Regression test for nvmultiurisrcbin last-source removal (DeepStream Issue #111).

Modes:
  last   : Add cam-a, wait for frames, remove cam-a (the only source), then add
           cam-b. Expects the pipeline to keep delivering frames. A GPU Xid 31
           MMU fault or zero frames after the re-add indicates the bug is present.
  control: Add cam-a and cam-b, wait for frames, remove cam-a (cam-b stays), then
           add cam-c. Expects frames to keep flowing. This scenario should not
           reproduce the failure.

Exit codes:
  0 : PASS - frames flow after the re-add in both modes as expected.
  1 : FAIL - frames do not resume after re-add, or a REST call returned non-200.
  2 : ERROR - the pipeline never started flowing within the timeout.
"""

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request

from pyservicemaker import BatchMetadataOperator, Pipeline, Probe

PORT = 19556
CONFIG = "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary.yml"
FRAME_TIMEOUT_SECONDS = 120
FRAME_COUNT_TARGET = 60
FRAMES_AFTER_ADD_SECONDS = 15
POST_RETRY_SECONDS = 15
POST_MAX_RETRIES = 1200

frames = 0
frames_lock = threading.Lock()


class Count(BatchMetadataOperator):
    def handle_metadata(self, batch_meta):
        global frames
        with frames_lock:
            frames += len(list(batch_meta.frame_items))


def log(what):
    print(
        f"{time.strftime('%H:%M:%S')}.{int(time.time() * 1000) % 1000:03d} {what} "
        f"(frames so far: {read_frames()})",
        flush=True,
    )


def read_frames():
    with frames_lock:
        return frames


def post(verb, camera, url):
    value = {
        "camera_id": camera,
        "camera_name": camera,
        "camera_url": url,
        "change": f"camera_{verb}",
    }
    body = json.dumps({"key": "sensor", "value": value}).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/api/v1/stream/{verb}",
        data=body,
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for _ in range(POST_MAX_RETRIES):
        try:
            with opener.open(request, timeout=POST_RETRY_SECONDS) as response:
                return response.status
        except urllib.error.HTTPError as error:
            return error.code
        except urllib.error.URLError:
            time.sleep(0.25)
    return -1


def wait_for_frames(count, seconds):
    target, deadline = read_frames() + count, time.monotonic() + seconds
    while read_frames() < target and time.monotonic() < deadline:
        time.sleep(0.2)
    return read_frames() >= target


def build_pipeline():
    unsynced = {"sync": False, "async": False}
    pipeline = Pipeline("repro")
    pipeline.add(
        "nvmultiurisrcbin",
        "src",
        {
            "ip-address": "127.0.0.1",
            "port": str(PORT),
            "max-batch-size": 2,
            "width": 1280,
            "height": 720,
            "live-source": True,
            "drop-pipeline-eos": True,
            "select-rtp-protocol": 4,
            "batched-push-timeout": 33333,
            "rtsp-reconnect-interval": 5,
            "init-rtsp-reconnect-interval": 5,
            "rtsp-reconnect-attempts": -1,
        },
    )
    pipeline.add("nvinfer", "det", {"config-file-path": CONFIG, "batch-size": 2})
    pipeline.add("nvvideoconvert", "conv_rgba", {})
    pipeline.add(
        "capsfilter", "caps_rgba",
        {"caps": "video/x-raw(memory:NVMM),format=RGBA"},
    )
    pipeline.add(
        "nvmultistreamtiler", "tiler",
        {"width": 1280, "height": 720, "rows": 1, "columns": 1},
    )
    pipeline.add("nvvideoconvert", "conv_nv12", {})
    pipeline.add(
        "capsfilter", "caps_nv12",
        {"caps": "video/x-raw(memory:NVMM),format=NV12"},
    )
    pipeline.add("nvv4l2h264enc", "enc", {"bitrate": 4000000})
    pipeline.add("h264parse", "parse", {})
    pipeline.add("mp4mux", "mux", {"fragment-duration": 1000})
    pipeline.add("filesink", "sink", {"location": "/tmp/out.mp4", **unsynced})
    pipeline.link(
        "src", "det", "conv_rgba", "caps_rgba", "tiler", "conv_nv12",
        "caps_nv12", "enc", "parse", "mux", "sink",
    )
    pipeline.attach("det", Probe("count", Count()))
    return pipeline


def run_mode(mode, url):
    global frames
    frames = 0

    pipeline = build_pipeline()
    pipeline.start(lambda message: None)
    threading.Thread(target=pipeline.wait, daemon=True).start()

    log(f"add cam-a: {post('add', 'cam-a', url)}")
    if mode == "control":
        log(f"add cam-b: {post('add', 'cam-b', url)}")

    if not wait_for_frames(FRAME_COUNT_TARGET, FRAME_TIMEOUT_SECONDS):
        log("ERROR: frames never started flowing within timeout")
        return 2

    log(f"frames flowing: True (frames so far: {read_frames()})")
    time.sleep(3)
    log(f"remove cam-a: {post('remove', 'cam-a', url)}")
    time.sleep(5)

    before = read_frames()
    again = "cam-b" if mode == "last" else "cam-c"
    log(f"add {again}: {post('add', again, url)}")
    time.sleep(FRAMES_AFTER_ADD_SECONDS)

    after = read_frames()
    delta = after - before
    log(f"frames in the {FRAMES_AFTER_ADD_SECONDS} s after the add: {delta} "
        f"(frames so far: {after})")

    try:
        pipeline.stop()
    except Exception:
        pass

    if delta <= 0:
        log("FAIL: no frames resumed after re-adding a source")
        return 1
    log("PASS: frames resumed after re-adding a source")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Regression test for nvmultiurisrcbin last-source removal.",
    )
    parser.add_argument(
        "mode",
        choices=["last", "control"],
        help="'last' removes the only source; 'control' keeps another source alive.",
    )
    parser.add_argument(
        "url",
        help="RTSP URL of the camera (e.g. rtsp://127.0.0.1:8620/cam1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=PORT,
        help=f"REST server port (default: {PORT}).",
    )
    parser.add_argument(
        "--config",
        default=CONFIG,
        help="Path to the nvinfer config file.",
    )
    args = parser.parse_args()

    global PORT, CONFIG
    PORT = args.port
    CONFIG = args.config

    sys.exit(run_mode(args.mode, args.url))


if __name__ == "__main__":
    main()
