from setuptools import setup, find_packages

setup(
    name="gm4ma",
    version="0.1.0",
    description="Generative Models for Multimodal Accessibility",
    author="Aman Grewal",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.2.0",
        "tensorflow>=2.15.0",
        "transformers>=4.40.0",
        "diffusers>=0.27.0",
        "peft>=0.10.0",
        "coremltools>=7.2",
        "Pillow>=10.2.0",
        "pyyaml>=6.0",
        "einops>=0.7.0",
    ],
)
