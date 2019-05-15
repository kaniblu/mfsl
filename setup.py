import setuptools

setuptools.setup(
    name="mfsl",
    version="0.1",
    packages=["mfsl"],
    install_requires=[
        "ply",
        "pyyaml",
        "numpy",
        "pandas",
        "stockstats"
    ]
)
