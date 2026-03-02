import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class RunpodContainerOpensshFlowTests(unittest.TestCase):
    def _write_fake_apt_get(self, fakebin: Path, log_path: Path) -> None:
        id_path = fakebin / "id"
        id_path.write_text(
            "#!/bin/sh\n"
            "if [ \"${1:-}\" = \"-u\" ]; then\n"
            "  echo 0\n"
            "  exit 0\n"
            "fi\n"
            "exec /usr/bin/id \"$@\"\n",
            encoding="utf-8",
        )
        os.chmod(id_path, 0o755)
        apt_path = fakebin / "apt-get"
        apt_path.write_text(
            f"""#!/bin/sh
set -eu
echo "$@" >> "{log_path}"
if [ "${{1:-}}" = "install" ]; then
  cat > "{fakebin / 'ssh'}" <<'EOSSH'
#!/bin/sh
exit 0
EOSSH
  /bin/chmod +x "{fakebin / 'ssh'}"
  cat > "{fakebin / 'ssh-keygen'}" <<'EOKEY'
#!/bin/sh
set -eu
out=""
while [ $# -gt 0 ]; do
  if [ "$1" = "-f" ]; then
    out="${{2:-}}"
    shift 2
    continue
  fi
  shift
done
[ -n "$out" ] || exit 2
echo "private" > "$out"
echo "public" > "$out.pub"
exit 0
EOKEY
  /bin/chmod +x "{fakebin / 'ssh-keygen'}"
fi
exit 0
""",
            encoding="utf-8",
        )
        os.chmod(apt_path, 0o755)

    def test_try_install_openssh_client_runs_in_container_marker_context(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fakebin = root / "bin"
            fakebin.mkdir(parents=True, exist_ok=True)
            marker = root / "containerenv"
            marker.write_text("", encoding="utf-8")
            log_path = root / "apt.log"
            self._write_fake_apt_get(fakebin, log_path)
            proc = subprocess.run(
                [
                    "bash",
                    "-lc",
                    f"source scripts/runpod_cycle_common.sh && "
                    f"PATH='{fakebin}:/bin' runpod_cycle_try_install_openssh_client '{marker}' && "
                    f"PATH='{fakebin}:/bin' command -v ssh >/dev/null && "
                    f"PATH='{fakebin}:/bin' command -v ssh-keygen >/dev/null && "
                    "printf ok",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.stdout, "ok")
            self.assertIn("install -y openssh-client", log_path.read_text(encoding="utf-8"))

    @unittest.skipUnless(Path("/run/.containerenv").exists() or Path("/.containerenv").exists(), "container-specific flow")
    def test_prepare_ssh_client_files_bootstraps_openssh_when_missing(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fakebin = root / "bin"
            fakebin.mkdir(parents=True, exist_ok=True)
            log_path = root / "apt.log"
            self._write_fake_apt_get(fakebin, log_path)
            key_base = root / "managed_key"
            path_env = f"{fakebin}:/bin"
            had_ssh_keygen_before = shutil.which("ssh-keygen", path=path_env) is not None
            proc = subprocess.run(
                [
                    "bash",
                    "-lc",
                    f"source scripts/runpod_cycle_common.sh && "
                    f"export RUNPOD_TEMP_SSH_KEY_BASE='{key_base}' && "
                    f"PATH='{path_env}' runpod_cycle_prepare_ssh_client_files /work && "
                    f"test -f '{key_base}' && test -f '{key_base}.pub' && printf ok",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.stdout, "ok")
            if not had_ssh_keygen_before:
                self.assertIn("install -y openssh-client", log_path.read_text(encoding="utf-8"))
