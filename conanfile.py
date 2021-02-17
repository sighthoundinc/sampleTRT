from conans import ConanFile, CMake, tools
from conans.tools import os_info, SystemPackageTool
import os
import shutil

class SampleConan(ConanFile):
    base_version = "1.0"
    foldername = "sampleTRT"
    license = "3-Clause BSD"
    settings = "os", "compiler", "build_type", "arch"
    options = {
    }
    default_options = tuple([
    ])

    generators = "cmake"

    scm = {
        "type": "git",
        "url": "auto",
        "revision": "auto"
    }


    def build(self):
        makeGen=None

        cmake = CMake(self,generator=makeGen)
        cmake.configure()
        cmake.build()
        cmake.install()


