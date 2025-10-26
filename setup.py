from setuptools import setup, find_packages

setup(
    name="upredictor",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "pymatgen",
        "scikit-learn",
        "pandas",
        "numpy",
        "joblib"
    ],
    entry_points={
        "console_scripts": [
            "upredictor = main:main_entry",
        ],
    },
)
