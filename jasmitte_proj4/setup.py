from setuptools import find_packages, setup

package_name = 'jasmitte_proj4'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/localization.launch.py']),
        ('share/' + package_name + '/rviz', ['rviz/proj4.rviz']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='anders',
    maintainer_email='jasmitte@mtu.edu',
    description='A package for performing localization using KF, EKF, and UKF on a differential drive robot.',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'localization_node = jasmitte_proj4.localization_node:main'
        ],
    },
)
