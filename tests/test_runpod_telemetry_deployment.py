import unittest
from pathlib import Path


class RunpodTelemetryDeploymentTests(unittest.TestCase):
    def test_dockerfile_installs_otel_collector_and_telemetry_utils(self):
        text = Path("deploy/runpod_cloud_training/Dockerfile").read_text(encoding="utf-8")
        self.assertIn("ARG OTELCOL_VERSION", text)
        self.assertIn("otelcol-contrib", text)
        self.assertIn("dnsutils", text)
        self.assertIn("iputils-ping", text)
        self.assertIn("net-tools", text)
        self.assertIn("sysstat", text)
        self.assertIn("healthchecks_ping.sh", text)

    def test_otel_config_has_hostmetrics_and_otlp(self):
        text = Path("deploy/runpod_cloud_training/otel-collector-config.yaml").read_text(
            encoding="utf-8"
        )
        self.assertIn("hostmetrics:", text)
        self.assertIn("otlp:", text)
        self.assertIn("exporters:", text)
        self.assertIn("OTEL_FILE_EXPORT_PATH", text)

    def test_healthchecks_ping_script_exists(self):
        text = Path("deploy/runpod_cloud_training/healthchecks_ping.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("RUNPOD_HEALTHCHECKS_URL", text)
        self.assertIn("HEALTHCHECKS_URL", text)
        self.assertIn("start|success|fail|log", text)
        self.assertIn("curl -fsS -m 10 -X POST", text)


if __name__ == "__main__":
    unittest.main()
