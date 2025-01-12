import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cp-ai-case",
    version="1.0.0",
    author="Geoffrey Gilles",
    description="cp-ai-case",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dev-geof/cp-ai-case",
    packages=setuptools.find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        "matplotlib>=3.5.1",
        "numpy>=1.24.2",
        "pandas>=2.2.3",
        "PyYAML>=6.0",
        "scikit_learn>=1.2.2",
        "tensorflow>=2.18.0",
        "termcolor>=1.1.0",
        "tqdm>=4.62.3",
        "graphviz>=0.20.1",
        "tf2onnx>=1.12.0",
        "scikit-learn>=1.1.2",
        "spacy>=3.8.3",
        "shap>=0.43.0",
        "seaborn>=0.13.2",
        "imblearn>=0.0",

    ],
    entry_points={
        "console_scripts": [
            "preparation=preparation:main",
            "training=training:main",
            "validation=validation:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Engineering",
    ],
    python_requires=">=3.9.13",
)
