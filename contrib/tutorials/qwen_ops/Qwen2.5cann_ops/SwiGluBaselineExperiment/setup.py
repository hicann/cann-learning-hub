from setuptools import setup, find_packages

setup(
    name="swiglu_baseline_experiment",
    version="0.1.0",
    description="FP32 SwiGLU baseline custom operator on Ascend CANN",
    packages=find_packages(where="."),
    package_dir={"": "."},
    python_requires=">=3.9",
    install_requires=["torch>=2.0", "numpy"],
)