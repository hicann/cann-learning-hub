from setuptools import find_packages, setup

setup(
    name="rmsnorm_optimized_experiment",
    version="0.1.0",
    description="FP32 RMSNorm optimized custom operator on Ascend CANN",
    packages=find_packages(where="."),
    package_dir={"": "."},
    python_requires=">=3.9",
    install_requires=["torch>=2.0", "numpy"],
)
