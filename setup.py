# setup.py (نسخه نهایی)

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="sparrow-mlp",  # <-- تغییر نام برای وضوح بیشتر
    version="3.0.2",      # <-- افزایش به نسخه اصلی جدید
    author="AmirReza",
    author_email="amirrezaahali@gmail.com",
    description="A library for building dynamic and self-pruning MLPs with Mixture of Experts and Layer Skipping.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/amirahali2008wp-sudo/sparrow-mlp",
    packages=find_packages(),
    install_requires=[ # نیازمندی‌های اصلی کتابخانه بسیار سبک است
        "torch>=1.9.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.7',
)