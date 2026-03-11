from setuptools import setup, find_packages

setup(
    name="rte-engine",
    version="0.1.0",
    description="Relational Time Engine for runtime transformer gating",
    author="Athmani Salah",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
    ],
    python_requires=">=3.10",
)
