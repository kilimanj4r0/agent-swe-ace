#!/usr/bin/env python
"""
Docker Setup Test Script

Verifies that the experiment can run with Docker mode correctly.
Run from: src/tests/

Usage:
    python test_docker_setup.py
    python test_docker_setup.py --pull-image  # Also test pulling a SWE-bench image
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_command(cmd: list, description: str) -> tuple[bool, str]:
    """Run a command and return (success, output)."""
    print(f"\n[TEST] {description}")
    print(f"       Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        success = result.returncode == 0
        output = result.stdout + result.stderr
        return success, output
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except Exception as e:
        return False, str(e)


def test_docker_installed() -> bool:
    """Test 1: Check if Docker is installed."""
    success, output = run_command(["docker", "--version"], "Docker is installed")
    if success:
        print(f"       OK: {output.strip()}")
    else:
        print(f"       FAIL: {output}")
    return success


def test_docker_running() -> bool:
    """Test 2: Check if Docker daemon is running."""
    success, output = run_command(["docker", "info", "--format", "{{.ServerVersion}}"], "Docker daemon is running")
    if success:
        print(f"       OK: Docker server version {output.strip()}")
    else:
        print(f"       FAIL: Docker daemon not running. Start Docker Desktop or dockerd.")
    return success


def test_docker_permissions() -> bool:
    """Test 3: Check if user can run Docker commands."""
    success, output = run_command(["docker", "ps"], "Docker permissions")
    if success:
        print(f"       OK: Can list containers")
    else:
        print(f"       FAIL: Permission denied. Add user to docker group or run with sudo.")
    return success


def test_python_docker_package() -> bool:
    """Test 4: Check if docker Python package is installed."""
    print("\n[TEST] Python docker package")
    try:
        import docker
        client = docker.from_env()
        print(f"       OK: docker package version {docker.__version__}")
        return True
    except ImportError:
        print("       FAIL: docker package not installed. Run: pip install docker")
        return False
    except Exception as e:
        print(f"       FAIL: {e}")
        return False


def test_swebench_imports() -> bool:
    """Test 5: Check if swebench can be imported."""
    print("\n[TEST] SWE-bench imports")
    try:
        from swebench.harness.run_evaluation import run_instance
        from swebench.harness.test_spec.test_spec import make_test_spec
        print("       OK: swebench imports successful")
        return True
    except ImportError as e:
        print(f"       FAIL: {e}")
        print("       Run: pip install swebench")
        return False


def test_mini_swe_agent_imports() -> bool:
    """Test 6: Check if mini-swe-agent can be imported."""
    print("\n[TEST] mini-swe-agent imports")
    try:
        from minisweagent.agents.default import DefaultAgent
        from minisweagent.environments.docker import DockerEnvironment
        from minisweagent.run.extra.swebench import get_swebench_docker_image_name
        print("       OK: mini-swe-agent imports successful")
        return True
    except ImportError as e:
        print(f"       FAIL: {e}")
        print("       Run: pip install git+https://github.com/SWE-agent/mini-swe-agent.git@v1")
        return False


def test_swebench_image_name() -> bool:
    """Test 7: Check SWE-bench Docker image name generation."""
    print("\n[TEST] SWE-bench image name generation")
    try:
        from minisweagent.run.extra.swebench import get_swebench_docker_image_name

        # Test with a sample instance
        test_instance = {
            "instance_id": "django__django-12345",
            "repo": "django/django",
            "version": "3.0",
        }
        image_name = get_swebench_docker_image_name(test_instance)
        print(f"       OK: Generated image name: {image_name}")
        return True
    except Exception as e:
        print(f"       FAIL: {e}")
        return False


def test_list_swebench_images() -> bool:
    """Test 8: List available SWE-bench Docker images."""
    print("\n[TEST] SWE-bench Docker images")

    success, output = run_command(
        ["docker", "images", "--filter=reference=swebench/*", "--format", "{{.Repository}}:{{.Tag}}"],
        "Listing swebench images"
    )

    if success:
        images = [line for line in output.strip().split('\n') if line]
        if images:
            print(f"       OK: Found {len(images)} SWE-bench image(s):")
            for img in images[:5]:  # Show first 5
                print(f"         - {img}")
            if len(images) > 5:
                print(f"         ... and {len(images) - 5} more")
        else:
            print("       WARNING: No SWE-bench images found locally")
            print("       Images will be pulled on first run (this may take time)")
        return True
    return False


def test_pull_sample_image() -> bool:
    """Test 9: Pull a real SWE-bench Lite image (optional, slow)."""
    print("\n[TEST] Pull sample SWE-bench Lite image (this may take a few minutes)")

    try:
        from datasets import load_dataset
        from minisweagent.run.extra.swebench import get_swebench_docker_image_name

        # Load SWE-bench Lite dataset and get the first instance
        print("       Loading SWE-bench Lite dataset...")
        dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
        sample_instance = dataset[0]
        instance_id = sample_instance["instance_id"]

        # Get the Docker image name for this instance
        image_name = get_swebench_docker_image_name(sample_instance)
        print(f"       Selected instance: {instance_id}")
        print(f"       Image: {image_name}")

        # Try to pull the image (increase timeout for large images)
        result = subprocess.run(
            ["docker", "pull", image_name],
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes for large images
        )

        if result.returncode == 0:
            print(f"       OK: Image pulled successfully")
            return True
        else:
            error = result.stderr or result.stdout
            print(f"       FAIL: Could not pull image")
            print(f"       Error: {error[:500]}")
            return False

    except subprocess.TimeoutExpired:
        print("       FAIL: Image pull timed out (10 min limit)")
        return False
    except ImportError as e:
        print(f"       FAIL: Missing dependency - {e}")
        return False
    except Exception as e:
        print(f"       FAIL: {e}")
        return False


def test_evaluation_module() -> bool:
    """Test 10: Test evaluation module with Docker."""
    print("\n[TEST] Evaluation module (Docker validation)")
    try:
        from evaluation import validate_patch, validate_patch_simple

        # Test simple validation (no Docker needed)
        result = validate_patch_simple(
            {"problem_statement": "test"},
            "diff --git a/test.py b/test.py\n--- a/test.py\n+++ b/test.py\n"
        )
        print(f"       OK: Simple validation works (result={result})")
        return True
    except ImportError as e:
        print(f"       FAIL: {e}")
        return False
    except Exception as e:
        print(f"       FAIL: {e}")
        return False


def test_config_loading() -> bool:
    """Test 11: Test config file loading."""
    print("\n[TEST] Config file loading")

    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    if not config_path.exists():
        print(f"       FAIL: config.yaml not found at {config_path}")
        return False

    try:
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Check Docker-related settings
        env_type = config.get('environment', {}).get('type', 'docker')
        eval_docker = config.get('evaluation', {}).get('use_docker', True)

        print(f"       OK: Config loaded")
        print(f"       Environment type: {env_type}")
        print(f"       Evaluation use_docker: {eval_docker}")
        return True
    except Exception as e:
        print(f"       FAIL: {e}")
        return False


def test_docker_disk_space() -> bool:
    """Test 12: Check Docker disk space."""
    print("\n[TEST] Docker disk space")

    success, output = run_command(
        ["docker", "system", "df", "--format", "{{.Type}}: {{.Size}}"],
        "Checking Docker disk usage"
    )

    if success:
        print(f"       OK: Docker disk info:")
        for line in output.strip().split('\n'):
            if line:
                print(f"         {line}")
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Test Docker setup for experiments")
    parser.add_argument(
        "--pull-image",
        action="store_true",
        help="Also test pulling a SWE-bench Docker image (slow)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show verbose output"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Docker Setup Test for ACE + mini-swe-agent Experiment")
    print("=" * 60)

    tests = [
        ("Docker installed", test_docker_installed),
        ("Docker daemon running", test_docker_running),
        ("Docker permissions", test_docker_permissions),
        ("Python docker package", test_python_docker_package),
        ("SWE-bench imports", test_swebench_imports),
        ("mini-swe-agent imports", test_mini_swe_agent_imports),
        ("Image name generation", test_swebench_image_name),
        ("Available images", test_list_swebench_images),
        ("Evaluation module", test_evaluation_module),
        ("Config loading", test_config_loading),
        ("Docker disk space", test_docker_disk_space),
    ]

    if args.pull_image:
        tests.append(("Pull sample image", test_pull_sample_image))

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n       ERROR: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, s in results if s)
    total = len(results)

    for name, success in results:
        status = "OK" if success else "FAIL"
        print(f"  [{status:^4}] {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\nAll tests passed! Ready to run experiments with Docker.")
        return 0
    else:
        print("\nSome tests failed. Fix the issues above before running experiments.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
