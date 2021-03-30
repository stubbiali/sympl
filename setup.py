# -*- coding: utf-8 -*-
import sys
from setuptools import setup

if __name__ == "__main__":
    if sys.version_info.major < 3:
        print("Python 3.x is required.")
        sys.exit(1)

    setup(use_scm_version=True)
