import os

from setuptools import find_packages, setup
from glob import glob

package_name = 'imagepipe'

# HACK: This is not a perfect data_files load method.
setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*')),
        ('share/' + package_name + '/launch', glob('launch/*')),
        ('share/' + package_name + '/yolo/v10', glob('**/config.json', recursive=True)),
        ('share/' + package_name + '/yolo/v10', glob('**/pytorch_model.bin', recursive=True))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='serianrealm@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
    },
    entry_points={
        'console_scripts': [
            'detector = imagepipe.detector:main',
            'tracker = imagepipe.tracker:main'
        ],
    },
)
