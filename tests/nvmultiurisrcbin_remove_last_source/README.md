# nvmultiurisrcbin Remove Last Source — Regression Test

This directory contains a regression test for [NVIDIA DeepStream GitHub Issue #111](https://github.com/NVIDIA/DeepStream/issues/111):
removing the only active RTSP source from `nvmultiurisrcbin` through `POST /api/v1/stream/remove` can cause a GPU MMU fault (Xid 31)
about 20 ms after the REST response, followed by `nvvideoconvert` buffer transformation failures and `nvinfer` internal stream errors.
The pipeline does not recover, and a source added afterwards delivers no frames.

## Files

| File | Purpose |
|------|---------|
| `repro.py` | Regression test script. Supports `last` (remove the only source) and `control` (keep another source active) modes. |
| `README.md` | This file — setup, execution, and expected results. |

## Prerequisites

- Docker with the NVIDIA Container Toolkit and a physical GPU (this test cannot run in headless CPU-only CI).
- A host running Ubuntu with Docker and `docker` available to the current user.
- The DeepStream stock container image:
  `ghcr.io/nvidia/deepstream:9.2-triton-multiarch`
- The MediaMTX image:
  `bluenviron/mediamtx:1.21.2-ffmpeg`
- Inside the container, `pyyaml` is required by `pyservicemaker` (installed at runtime via pip).

## Step 1: Extract the sample clip from the DeepStream container

```bash
mkdir -p clip
docker run --rm --entrypoint cat ghcr.io/nvidia/deepstream:9.2-triton-multiarch \
  /opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4 > clip/sample_720p.mp4
```

## Step 2: Start the MediaMTX RTSP camera

```bash
docker run -d --name cam --network host \
  -e MTX_RTSPADDRESS=:8620 \
  -e MTX_MOQ=no \
  -e MTX_PATHS_CAM1_RUNONINIT='ffmpeg -re -stream_loop -1 -i /v/sample_720p.mp4 -c copy -f rtsp rtsp://localhost:8620/cam1' \
  -v "$PWD/clip:/v:ro" \
  bluenviron/mediamtx:1.21.2-ffmpeg
```

This serves `rtsp://127.0.0.1:8620/cam1` looping the DeepStream sample clip.

## Step 3: Run the regression test

Copy `repro.py` into a directory mounted into the container (for example `$PWD:/repro`), then run:

### `last` mode — removes the only source (expected to reproduce the bug)

```bash
docker run --rm --gpus all --network host -v "$PWD:/repro" --entrypoint bash \
  ghcr.io/nvidia/deepstream:9.2-triton-multiarch \
  -c "pip install -q --break-system-packages pyyaml && python3 /repro/repro.py last rtsp://127.0.0.1:8620/cam1"
```

### `control` mode — keeps another source active (expected to pass)

```bash
docker run --rm --gpus all --network host -v "$PWD:/repro" --entrypoint bash \
  ghcr.io/nvidia/deepstream:9.2-triton-multiarch \
  -c "pip install -q --break-system-packages pyyaml && python3 /repro/repro.py control rtsp://127.0.0.1:8620/cam1"
```

## Expected Results

| Scenario | Expected output | Pass / Fail |
|----------|-----------------|-------------|
| `last` | `frames in the 15 s after the add: > 0` | PASS if frames resume; FAIL if `0` |
| `control` | `frames in the 15 s after the add: > 0` | PASS if frames continue flowing |

### Failure signs (bug present)

In `last` mode, the bug is present when:
- The script prints `frames in the 15 s after the add: 0` (or at most 1 frame).
- The kernel log, about 20 ms after the remove response, shows an NVIDIA Xid 31 MMU fault:

```bash
journalctl -k --since "-3 min" | grep -i xid
```

Example:
```
NVRM: Xid (PCI:0000:01:00): 31, pid=..., name=python3, ... MMU Fault: ...
```

And `stderr` contains:
```
WARNING from src_bin_muxer: No Sources found at the input of muxer. Waiting for sources.
ERROR         nvvideoconvert ... buffer transform failed
WARN          nvinfer ... Internal data stream error.
```

### Exit codes

| Code | Meaning |
|------|---------|
| 0 | PASS — frames resumed after re-adding a source. |
| 1 | FAIL — no frames resumed, or a REST call returned non-200. |
| 2 | ERROR — the pipeline never started flowing within the timeout. |

## Cleanup

Stop and remove the MediaMTX container when finished:

```bash
docker stop cam && docker rm cam
```

## Related Files

- Application-level safeguard: `src/apps/sample_apps/deepstream-server/safe_source_manager.py`
- Plugin documentation warning: `src/gst-plugins/gst-nvmultiurisrcbin/README`
