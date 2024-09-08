from setuptools import setup, find_packages  

with open("README.md", "r", encoding="utf-8") as fh:  
    long_description = fh.read()  

setup(  
    name="hdd-predictor",  
    version="0.1.0",  
    author="Jesters",
    author_email="your.email@example.com",
    description="A tool for predicting HDD failures",  
    long_description=long_description,  
    long_description_content_type="text/markdown",  
    url="https://github.com/yourusername/hdd-predictor",  
    packages=find_packages(),  
    classifiers=[  
        "Development Status :: 3 - Alpha",  
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.12",
    ],  
    python_requires=">=3.7",  
    install_requires=[  
        "numpy",  
        "pandas",  
        "scikit-learn",  
        "tensorflow",  
        "matplotlib",  
        "joblib",  
    ],  
    entry_points={  
        "console_scripts": [  
            "hdd-predictor=hdd_predictor.cli:main",  
        ],  
    },  
)