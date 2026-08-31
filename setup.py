from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'trainit_perception'

setup(
    name=package_name,
    version='0.2.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md']),
        (os.path.join('share', package_name, 'contracts'), glob('contracts/*.md')),
        (os.path.join('share', package_name, 'profiles'), glob('profiles/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='fra',
    maintainer_email='ros.master.ai@gmail.com',
    description='TrainIt perception: detectors emitting vision_msgs/Detection3DArray.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # A console_script, never an install(PROGRAMS) file: entry points are
            # generated executable and cannot hit the permission bug of 2026-08-24.
            'detector_node = trainit_perception.nodes.detector_node:main',
        ],
    },
)
