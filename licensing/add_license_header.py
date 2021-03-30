# -*- coding: utf-8 -*-
#
# BSD License
#
# Copyright (c) 2016-2021, Jeremy McGibbon
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice, this
# list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from this
# software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.
#
import glob
import os
import re
import shutil
import tempfile

# path to header files for each programming language
header_names = {
    "bash": "LICENSE_HEADER.sh",
    "cpp": "LICENSE_HEADER.cpp",
    "make": "LICENSE_HEADER.makefile",
    "python": "LICENSE_HEADER.py",
}

# name patterns for each programming language
src_patterns = {
    "bash": "*.sh",
    "cpp": ("*.cpp", "*.hpp"),
    "make": "[M|m]akefile",
    "python": "*.py",
}

# patterns to be ignored
exclude_patterns = (*header_names.values(), "venv/")


def main(root):
    c_exclude_patterns = [
        re.compile(rf"{pattern}") for pattern in exclude_patterns
    ]

    print(">" * 25)

    for language, patterns in src_patterns.items():
        header_name = header_names[language]
        patterns = (patterns,) if isinstance(patterns, str) else patterns

        print(f"Processing {language} files")

        for src_pattern in patterns:
            for src_name in glob.glob(
                os.path.join(root, "**/", src_pattern), recursive=True
            ):
                neglect = (
                    os.path.islink(src_name)
                    or os.path.isdir(src_name)
                    or any(
                        len(cep.findall(src_name)) > 0
                        for cep in c_exclude_patterns
                    )
                )

                if not neglect:
                    with open(src_name, "r") as src:
                        with open(header_name, "r") as header:
                            src = list(src)
                            header = list(header)

                            _, new_src_name = tempfile.mkstemp()
                            with open(new_src_name, "w") as new_src:
                                i = 0
                                while (
                                    i < min(len(src), len(header))
                                    and src[i] == header[i]
                                ):
                                    new_src.write(src[i])
                                    i += 1

                                if i == len(header):
                                    print(f"    {src_name} left unchanged")
                                else:
                                    for j in range(i, len(header)):
                                        new_src.write(header[j])

                                    for j in range(i, len(src)):
                                        new_src.write(src[j])

                                    shutil.move(new_src_name, src_name)

                                    print(f"    {src_name} modified.")

        print(">" * 25)


if __name__ == "__main__":
    main(root="..")
