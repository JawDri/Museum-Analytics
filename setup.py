from setuptools import setup, find_packages

setup(
    name="museum-analytics",
    version="0.1.0",
    description="Museum visitor analytics and city population correlation analysis",
    author="Jawher",
    packages=find_packages(),
    install_requires=[
        "requests>=2.31.0",
        "pandas>=2.1.4",
        "numpy>=1.24.3",
        "scikit-learn>=1.3.2",
        "matplotlib>=3.8.2",
        "seaborn>=0.13.0",
        "jupyter>=1.0.0",
        "beautifulsoup4>=4.12.2",
        "lxml>=4.9.3",
    ],
    python_requires=">=3.8",
)