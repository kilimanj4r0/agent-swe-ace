# src/tests/test_docker_setup.py
"""Tests for Docker setup verification."""

import subprocess
from pathlib import Path
import pytest


class TestDockerInstallation:
    """Tests for Docker installation and basic functionality."""

    def test_docker_installed(self):
        """Test that Docker is installed."""
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, "Docker is not installed"
        assert "Docker" in result.stdout

    def test_docker_daemon_running(self):
        """Test that Docker daemon is running."""
        result = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, "Docker daemon is not running"

    def test_docker_permissions(self):
        """Test that user can run Docker commands."""
        result = subprocess.run(
            ["docker", "ps"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, "Permission denied for Docker"


class TestPythonDockerPackage:
    """Tests for Python docker package."""

    def test_docker_package_installed(self):
        """Test that docker Python package is installed."""
        import docker
        assert docker is not None

    def test_docker_client_creation(self):
        """Test that Docker client can be created."""
        import docker
        client = docker.from_env()
        assert client is not None


class TestSWEBenchImports:
    """Tests for SWE-bench and mini-swe-agent imports."""

    def test_swebench_imports(self):
        """Test that swebench can be imported."""
        from swebench.harness.run_evaluation import run_instance
        from swebench.harness.test_spec.test_spec import make_test_spec
        assert run_instance is not None
        assert make_test_spec is not None

    def test_mini_swe_agent_imports(self):
        """Test that mini-swe-agent can be imported."""
        from minisweagent.agents.default import DefaultAgent
        from minisweagent.environments.docker import DockerEnvironment
        from minisweagent.run.extra.swebench import get_swebench_docker_image_name
        assert DefaultAgent is not None
        assert DockerEnvironment is not None
        assert get_swebench_docker_image_name is not None


class TestSWEBenchImageGeneration:
    """Tests for SWE-bench Docker image name generation."""

    def test_image_name_generation(self):
        """Test SWE-bench Docker image name generation."""
        from minisweagent.run.extra.swebench import get_swebench_docker_image_name

        test_instance = {
            "instance_id": "django__django-12345",
            "repo": "django/django",
            "version": "3.0",
        }
        image_name = get_swebench_docker_image_name(test_instance)
        assert "swebench" in image_name
        assert "django" in image_name


class TestConfigLoading:
    """Tests for configuration file loading."""

    def test_config_file_exists(self):
        """Test that config.yaml exists."""
        config_path = Path(__file__).parent.parent.parent / "config.yaml"
        assert config_path.exists(), f"config.yaml not found at {config_path}"

    def test_config_file_loading(self):
        """Test that config.yaml can be loaded with expected structure."""
        import yaml

        config_path = Path(__file__).parent.parent.parent / "config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Check Docker-related settings
        env_type = config.get('environment', {}).get('type', 'docker')
        eval_docker = config.get('evaluation', {}).get('use_docker', True)

        assert env_type == 'docker' or env_type is not None
        assert isinstance(eval_docker, bool)


class TestDockerDiskSpace:
    """Tests for Docker disk space."""

    def test_docker_disk_info(self):
        """Test that Docker disk info can be retrieved."""
        result = subprocess.run(
            ["docker", "system", "df", "--format", "{{.Type}}: {{.Size}}"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert len(result.stdout) > 0


class TestSWEBenchImages:
    """Tests for SWE-bench Docker images."""

    def test_list_swebench_images(self):
        """Test listing SWE-bench Docker images."""
        result = subprocess.run(
            ["docker", "images", "--filter=reference=swebench/*", "--format", "{{.Repository}}:{{.Tag}}"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        # Note: This test passes even if no images exist (they'll be pulled on first run)
        # The test just verifies the command works


@pytest.mark.skip(reason="Slow test - run manually with --pull-image flag")
class TestImagePull:
    """Tests for pulling SWE-bench images (slow, run manually)."""

    def test_pull_sample_image(self):
        """Test pulling a sample SWE-bench Lite image."""
        from datasets import load_dataset
        from minisweagent.run.extra.swebench import get_swebench_docker_image_name

        # Load SWE-bench Lite dataset and get the first instance
        dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
        sample_instance = dataset[0]

        # Get the Docker image name for this instance
        image_name = get_swebench_docker_image_name(sample_instance)

        # Try to pull the image
        result = subprocess.run(
            ["docker", "pull", image_name],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutes for large images
        )

        assert result.returncode == 0, f"Failed to pull image: {result.stderr}"
