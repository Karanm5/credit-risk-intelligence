"""
Setup script for Credit Risk Intelligence Platform.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = (this_directory / "requirements.txt").read_text(encoding="utf-8").splitlines()
requirements = [r.strip() for r in requirements if r.strip() and not r.startswith("#")]

setup(
    name="credit-risk-intelligence",
    version="1.0.0",
    author="Karan",
    author_email="karan@example.com",
    description="Real-time credit risk assessment platform using alternative data signals and graph-based features",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Karanm5/credit-risk-intelligence",
    project_urls={
        "Bug Tracker": "https://github.com/Karanm5/credit-risk-intelligence/issues",
        "Documentation": "https://github.com/Karanm5/credit-risk-intelligence#readme",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial",
    ],
    package_dir={"": "."},
    packages=find_packages(where="."),
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.23.0",
            "black>=24.1.0",
            "isort>=5.13.0",
            "flake8>=7.0.0",
            "mypy>=1.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "credit-risk-train=src.pipelines.training_pipeline:main",
            "credit-risk-serve=src.serving.api:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
