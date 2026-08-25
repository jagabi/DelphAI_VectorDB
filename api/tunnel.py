#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""cloudflared 터널 관리.

사내망에서 인바운드가 막혀 있어도, 안에서 밖으로 나가는 연결(아웃바운드 443)은
대체로 열려 있다. Chrome Remote Desktop 이 되는 것도 같은 이유다. cloudflared 는
그 방향으로 터널을 파서 외부 주소를 붙여준다.

두 가지 모드:

  퀵 터널   cloudflared tunnel --url http://127.0.0.1:8000
            계정 불필요. 대신 주소가 실행할 때마다 바뀐다.

  named     cloudflared tunnel run <이름>
            Cloudflare 계정 + 도메인이 필요하지만 주소가 고정된다.
            사전 준비 (한 번만):
              cloudflared tunnel login
              cloudflared tunnel create openalex
              cloudflared tunnel route dns openalex search.example.com
            그리고 ~/.cloudflared/config.yml 에
              tunnel: openalex
              credentials-file: ...json
              ingress:
                - hostname: search.example.com
                  service: http://127.0.0.1:8000
                - service: http_status:404
"""

from __future__ import annotations

import re
import shutil
import subprocess
import threading
from typing import Callable

from src import config as C

QUICK_URL = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


class Tunnel:
    """cloudflared 를 자식 프로세스로 띄우고 로그에서 URL 을 뽑아낸다."""

    def __init__(self, port: int, *, name: str = "", binary: str = "",
                 on_url: Callable[[str], None] | None = None):
        self.port = port
        self.name = name or C.CLOUDFLARE_TUNNEL_NAME
        self.binary = binary or C.CLOUDFLARED_BIN
        self.on_url = on_url
        self.url: str | None = None
        self.process: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None

    def available(self) -> bool:
        return shutil.which(self.binary) is not None

    def command(self) -> list[str]:
        if self.name:
            return [self.binary, "tunnel", "run", self.name]
        return [self.binary, "tunnel", "--url", f"http://127.0.0.1:{self.port}",
                "--no-autoupdate"]

    def start(self) -> bool:
        if not self.available():
            print(f"[tunnel] '{self.binary}' 를 찾지 못했습니다. 터널 없이 로컬만 서비스합니다.")
            print("         https://developers.cloudflare.com/cloudflare-one/"
                  "connections/connect-networks/downloads/")
            return False

        mode = f"named tunnel '{self.name}'" if self.name else "quick tunnel"
        print(f"[tunnel] {mode} 시작")
        self.process = subprocess.Popen(
            self.command(),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()
        return True

    def _pump(self) -> None:
        """cloudflared 는 URL 을 stderr 로 흘린다. 거기서 주소만 건져낸다."""
        assert self.process is not None and self.process.stdout is not None
        for line in self.process.stdout:
            if self.url is None:
                found = QUICK_URL.search(line)
                if found:
                    self.url = found.group(0)
                    print(f"\n[tunnel] {self.url}\n")
                    if self.on_url:
                        self.on_url(self.url)
            lowered = line.lower()
            if "error" in lowered or "failed" in lowered:
                print(f"[tunnel] {line.rstrip()}")

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None
