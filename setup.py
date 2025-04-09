from setuptools import setup, find_packages

setup(
    name="robinhood-crypto-analyzer",
    version="0.1.0",
    description="A tool for analyzing leading cryptocurrencies using the Robinhood API",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/bernardogv/robinhood-crypto-analyzer",
    packages=find_packages(),
    install_requires=[
        "pynacl>=1.5.0",
        "requests>=2.28.2",
        "pandas>=2.0.0",
        "numpy>=1.24.3",
        "matplotlib>=3.7.1",
        "seaborn>=0.12.2",
        "plotly>=5.14.0",
        "dash>=2.9.3",
        "python-dotenv>=1.0.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
