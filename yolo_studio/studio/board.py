"""SSH/SFTP operations on the board: capture mode, model upload."""

from __future__ import annotations

import http.client
import shlex
import socket
import time

import paramiko

MODELS_DIR = "/root/models"
STREAM_CTL_DIRS = ("/root/yolo_stream/start_stream.sh", "/root/start_stream.sh")


class BoardError(RuntimeError):
    pass


class Board:
    def __init__(self, ip: str, user: str = "root", password: str = "root",
                 ssh_port: int = 22, stream_port: int = 7777,
                 connect_timeout: float = 10.0) -> None:
        self.ip = (ip or "").strip()
        self.user = user or "root"
        self.password = password or ""
        self.ssh_port = int(ssh_port)
        self.stream_port = int(stream_port)
        self.connect_timeout = float(connect_timeout)

    # ---------- ssh ----------

    def _connect(self) -> paramiko.SSHClient:
        if not self.ip:
            raise BoardError("board IP is not set")
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(
                self.ip, port=self.ssh_port, username=self.user,
                password=self.password, timeout=self.connect_timeout,
                banner_timeout=self.connect_timeout,
                auth_timeout=self.connect_timeout,
                look_for_keys=False, allow_agent=False)
        except (paramiko.SSHException, socket.error, OSError) as exc:
            client.close()
            raise BoardError(f"SSH {self.user}@{self.ip}: {exc}") from exc
        return client

    def run(self, command: str, timeout: float = 90.0) -> tuple[int, str, str]:
        client = self._connect()
        try:
            try:
                _in, out, err = client.exec_command(command, timeout=timeout)
                stdout = out.read().decode("utf-8", "replace")
                stderr = err.read().decode("utf-8", "replace")
                rc = out.channel.recv_exit_status()
            except (socket.timeout, paramiko.SSHException, OSError) as exc:
                raise BoardError(f"command failed: {exc}") from exc
            return rc, stdout, stderr
        finally:
            client.close()

    def check(self) -> dict:
        rc, out, err = self.run(
            "uname -sm; echo '---'; "
            "p=$(cat /tmp/yolo_stream.app.pid 2>/dev/null); "
            "if [ -n \"$p\" ] && [ -d \"/proc/$p\" ]; then echo \"$p\"; "
            "else pidof stream_yolo || true; fi; echo '---'; "
            "uptime", timeout=20)
        parts = [p.strip() for p in out.split("---")]
        yolo = parts[1] if len(parts) > 1 else ""
        return {
            "ssh": rc == 0,
            "uname": parts[0] if parts else "",
            "stream_yolo": yolo or None,
            "uptime": parts[2] if len(parts) > 2 else "",
            "error": err.strip(),
        }

    # ---------- stream modes ----------

    def stop_all(self) -> str:
        """Stops the YOLO stream (the camera must be released)."""
        cmd = (
            "for s in /root/yolo_stream/start_stream.sh /root/start_stream.sh; do "
            "[ -x \"$s\" ] && \"$s\" stop >/dev/null 2>&1; done; "
            "echo stopped; pidof stream_yolo || true"
        )
        rc, out, err = self.run(cmd, timeout=120)
        if rc != 0 and "not found" not in err:
            raise BoardError(f"stream stop failed: {err.strip() or rc}")
        return out.strip()

    @staticmethod
    def stream_cmd(args: str) -> str:
        """Command for start_stream.sh: new path first, then the old one."""
        paths = " ".join(STREAM_CTL_DIRS)
        return (
            f"S=; for p in {paths}; do [ -x \"$p\" ] && {{ S=$p; break; }}; done; "
            "[ -n \"$S\" ] || { echo 'start_stream.sh not found' >&2; exit 127; }; "
            f"\"$S\" {args}"
        )

    def start_clean(self, port: int | None = None) -> None:
        """YOLO stream without detection or boxes — clean frames for the dataset."""
        port = int(port or self.stream_port)
        cmd = (
            "mkdir -p /root/log && cd /root && "
            "SY_DET=0 SY_DRAW=0 SY_HTTP_PORT={port} sh -c {script} "
            ">>/root/log/studio_capture.log 2>&1"
        ).format(port=port, script=shlex.quote(self.stream_cmd("auto")))
        rc, out, err = self.run(cmd, timeout=60)
        if rc != 0:
            raise BoardError(f"clean stream failed to start: {err.strip() or out.strip()}")
        if not self.wait_http(port, timeout=30):
            tail = self.tail("/root/log/studio_capture.log", 12)
            raise BoardError(
                f"stream on port {port} is not responding. Last lines:\n{tail}")

    def start_normal(self) -> None:
        cmd = ("mkdir -p /root/log && cd /root && sh -c {script} "
               ">>/root/log/autostart.log 2>&1"
               .format(script=shlex.quote(self.stream_cmd("auto"))))
        rc, out, err = self.run(cmd, timeout=60)
        if rc != 0:
            raise BoardError(f"stream failed to start: {err.strip() or out.strip()}")
        if not self.wait_http(self.stream_port, timeout=30):
            raise BoardError("normal stream is not responding on " + str(self.stream_port))

    def wait_http(self, port: int | None = None, timeout: float = 30.0) -> bool:
        port = int(port or self.stream_port)
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                conn = http.client.HTTPConnection(self.ip, port, timeout=2.0)
                conn.request("GET", "/stream")
                resp = conn.getresponse()
                ok = resp.status == 200
                resp.close()
                conn.close()
                if ok:
                    return True
            except (OSError, http.client.HTTPException):
                pass
            time.sleep(0.5)
        return False

    def stream_url(self, port: int | None = None) -> str:
        return f"http://{self.ip}:{int(port or self.stream_port)}/stream"

    def tail(self, path: str, lines: int = 15) -> str:
        try:
            _rc, out, _err = self.run(f"tail -n {int(lines)} {shlex.quote(path)} 2>/dev/null",
                                      timeout=15)
            return out.strip()
        except BoardError:
            return ""

    # ---------- models ----------

    def list_models(self) -> list[str]:
        rc, out, _err = self.run(
            "ls -1 /root/yolo_stream/models/*.cvimodel /root/models/*.cvimodel "
            "/mnt/system/usr/bin/ai/models/*.cvimodel 2>/dev/null || true", timeout=20)
        names: list[str] = []
        for line in out.splitlines():
            name = line.strip().rsplit("/", 1)[-1]
            if name and name not in names:
                names.append(name)
        return names

    def upload_model(self, local_path: str, remote_name: str | None = None,
                     overwrite: bool = False) -> str:
        from pathlib import Path

        local = Path(local_path)
        if not local.is_file():
            raise BoardError(f"file not found: {local}")
        name = (remote_name or local.name).strip()
        if not name.endswith(".cvimodel"):
            name += ".cvimodel"
        remote = f"{MODELS_DIR}/{name}"

        client = self._connect()
        try:
            sftp = client.open_sftp()
            try:
                try:
                    sftp.stat(MODELS_DIR)
                except OSError:
                    sftp.mkdir(MODELS_DIR)
                if not overwrite:
                    try:
                        sftp.stat(remote)
                        raise BoardError(f"{name} already exists on the board (overwrite is disabled)")
                    except FileNotFoundError:
                        pass
                sftp.put(str(local), remote)
                sftp.chmod(remote, 0o644)
                # Class names for stream labels: <model>.names
                local_names = local.with_suffix(".names")
                if local_names.is_file():
                    remote_names = remote[:-len(".cvimodel")] + ".names"
                    sftp.put(str(local_names), remote_names)
                    sftp.chmod(remote_names, 0o644)
            finally:
                sftp.close()
        except OSError as exc:
            raise BoardError(f"upload failed: {exc}") from exc
        finally:
            client.close()

        # append to the start_stream.sh menu (order file)
        self.add_to_order(name)
        return remote

    def add_to_order(self, name: str) -> None:
        q = shlex.quote(name)
        cmd = (f"grep -qxF {q} /root/models/models.order 2>/dev/null || "
               f"echo {q} >> /root/models/models.order")
        rc, _out, err = self.run(cmd, timeout=20)
        if rc != 0:
            raise BoardError(f"failed to update models.order: {err.strip()}")

    def restart_with_model(self, substring: str) -> str:
        cmd = ("cd /root && sh -c {script}"
               .format(script=shlex.quote(self.stream_cmd(shlex.quote(substring)))))
        rc, out, err = self.run(cmd, timeout=120)
        if rc != 0:
            raise BoardError(f"model start failed: {err.strip() or out.strip()}")
        return out.strip()
