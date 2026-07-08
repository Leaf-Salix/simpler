#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Generate a minimal A3 HCCS rank table for 120.9.10.37 and 120.9.10.35."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEVICE_PORT_BASE = 16666
HOST_PORT_BASE = 17666

NODE37_DEVICE_IPS = {
    0: "10.30.2.1",
    1: "10.30.3.1",
    2: "10.30.2.3",
    3: "10.30.3.3",
    4: "10.30.2.5",
    5: "10.30.3.5",
    6: "10.30.2.7",
    7: "10.30.3.7",
    8: "10.30.2.2",
    9: "10.30.3.2",
    10: "10.30.2.4",
    11: "10.30.3.4",
    12: "10.30.2.6",
    13: "10.30.3.6",
    14: "10.30.2.8",
    15: "10.30.3.8",
}

NODE35_DEVICE_IPS = {
    0: "10.30.0.1",
    1: "10.30.1.1",
    2: "10.30.0.3",
    3: "10.30.1.3",
    4: "10.30.0.5",
    5: "10.30.1.5",
    6: "10.30.0.7",
    7: "10.30.1.7",
    8: "10.30.0.2",
    9: "10.30.1.2",
    10: "10.30.0.4",
    11: "10.30.1.4",
    12: "10.30.0.6",
    13: "10.30.1.6",
    14: "10.30.0.8",
    15: "10.30.1.8",
}


def make_device(device_id: int, rank_id: int, device_ip: str) -> dict[str, str]:
    return {
        "device_id": str(device_id),
        "super_device_id": str(rank_id),
        "device_ip": device_ip,
        "device_port": str(DEVICE_PORT_BASE + device_id),
        "host_port": str(HOST_PORT_BASE + device_id),
        "rank_id": str(rank_id),
    }


def make_server(server_id: str, host_ip: str, base_rank: int, device_ips: dict[int, str]) -> dict[str, object]:
    return {
        "server_id": server_id,
        "host_ip": host_ip,
        "device": [
            make_device(device_id, base_rank + device_id, device_ips[device_id]) for device_id in sorted(device_ips)
        ],
    }


def build_ranktable() -> dict[str, object]:
    return {
        "status": "completed",
        "version": "1.2",
        "server_count": "2",
        "server_list": [
            make_server("host37", "120.9.10.37", 0, NODE37_DEVICE_IPS),
            make_server("host35", "120.9.10.35", 16, NODE35_DEVICE_IPS),
        ],
        "super_pod_list": [
            {
                "super_pod_id": "0",
                "server_list": [
                    {"server_id": "host37"},
                    {"server_id": "host35"},
                ],
            }
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output",
        default="ranktable_a3_2host_32rank.json",
        help="Output rank table path.",
    )
    args = parser.parse_args()

    output = Path(args.output)
    ranktable = build_ranktable()
    output.write_text(json.dumps(ranktable, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {output} with 2 servers and 32 ranks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
