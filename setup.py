from setuptools import setup, find_packages
from pathlib import Path
import pkg_resources


setup(
    name="asr_business",
    py_modules=["asr"],
    version="0.1",
    describe="ASR Python package for identifying business audio data",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    readme="README.md",
    python_requires=">=3.8",
    license="MIT Licence",
    url="https://github.com/Tonywu2018/asr_business",
    author="wuwenxiao",
    author_email="wuwenxiao@inke.cn",
    packages=find_packages(),
    include_package_data=True,
    platforms="any",
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            Path(__file__).with_name("requirements.txt").open()
        )
    ]
)
