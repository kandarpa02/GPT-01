from setuptools import setup, find_packages

setup(
    name="GPT01",
    version="0.1.0",
    author="Kandarpa Sarkar",
    author_email="kandarpaexe@gmail.com",
    description="An LLM distillation repo using in TensorFlow",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/kandarpa02/GPT01.git",
    packages=find_packages(),
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    license="MIT",
    zip_safe=False,
    
)