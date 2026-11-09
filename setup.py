from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent
requirements_path = ROOT / "requirements.txt"
install_requires = [
    line.strip()
    for line in requirements_path.read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.strip().startswith("#")
]


setup(
    name="universe-video-action",
    version="0.1.0",
    description="UNIVERSE training and inference runtime for autonomous driving.",
    packages=find_packages(include=["diffsynth*"]),
    install_requires=install_requires,
    include_package_data=True,
    python_requires=">=3.10",
)
