from setuptools import setup, find_packages

setup(
    name="swiglu_optimized_experiment",
    version="0.1.0",
    description="FP32 SwiGLU optimized custom operator on Ascend CANN",
    packages=find_packages(where="."),
    package_dir={"": "."},
    python_requires=">=3.9",
    install_requires=["torch>=2.0", "numpy"],
)