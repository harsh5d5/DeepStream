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
Application-level safeguard for nvmultiurisrcbin source lifecycle.

Removing the only active RTSP source from nvmultiurisrcbin through
POST /api/v1/stream/remove can cause a GPU MMU fault (Xid 31) about
20 ms after the REST response, followed by nvvideoconvert buffer
transformation failures and nvinfer internal stream errors. The
pipeline does not recover.

SafeSourceManager wraps /api/v1/stream/add and /api/v1/stream/remove and
tracks active sources so that removal of the last active source is
rejected. Concurrency-safe via a threading lock.

WARNING: This is a workaround, not a fix. It prevents triggering the bug
but blocks legitimate "remove all cameras" flows until the underlying
plugin issue is resolved. See nvmultiurisrcbin/README for details.
"""

import json
import threading
import time
import urllib.error
import urllib.request


class SafeSourceManager:
    """Thread-safe manager for nvmultiurisrcbin stream add/remove."""

    def __init__(self, base_url="http://127.0.0.1:19556"):
        self.base_url = base_url.rstrip("/")
        self.active = set()
        self.lock = threading.Lock()

    def _post(self, verb, camera_id, camera_name, camera_url):
        value = {
            "camera_id": camera_id,
            "camera_name": camera_name,
            "camera_url": camera_url,
            "change": f"camera_{verb}",
        }
        body = json.dumps({"key": "sensor", "value": value}).encode()
        request = urllib.request.Request(
            f"{self.base_url}/api/v1/stream/{verb}",
            data=body,
            method="POST",
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        for _ in range(1200):
            try:
                with opener.open(request, timeout=15) as response:
                    return response.status
            except urllib.error.HTTPError as error:
                return error.code
            except urllib.error.URLError:
                time.sleep(0.25)
        return -1

    def add(self, camera_id, camera_name=None, camera_url=None):
        """Add a source. Returns (status_code, message)."""
        camera_name = camera_name or camera_id
        if camera_url is None:
            return 400, "camera_url is required"

        with self.lock:
            if camera_id in self.active:
                return 200, "already present"

        status = self._post("add", camera_id, camera_name, camera_url)
        if status == 200:
            with self.lock:
                self.active.add(camera_id)
            return 200, "added"
        return status, f"add failed with HTTP {status}"

    def remove(self, camera_id):
        """Remove a source. Refuses if it is the last active source.

        Returns (status_code, message). 409 means refused by safeguard.
        """
        with self.lock:
            if camera_id not in self.active:
                return 200, "not present"
            if len(self.active) <= 1:
                return 409, (
                    "refusing to remove last active source: "
                    "nvmultiurisrcbin last-source removal may cause Xid 31 "
                    "GPU fault and pipeline failure"
                )

        status = self._post("remove", camera_id, camera_id, "")
        if status == 200:
            with self.lock:
                self.active.discard(camera_id)
            return 200, "removed"
        return status, f"remove failed with HTTP {status}"

    def list_active(self):
        """Return a sorted list of active camera IDs."""
        with self.lock:
            return sorted(self.active)

    def count(self):
        """Return the number of active sources."""
        with self.lock:
            return len(self.active)


__all__ = ["SafeSourceManager"]
