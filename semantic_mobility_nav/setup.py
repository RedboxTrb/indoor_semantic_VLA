from setuptools import setup
package_name = 'semantic_mobility_nav'
setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kushal',
    maintainer_email='gamer3489t@gmail.com',
    description='Semantic mobility-aware navigation',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detector_clip = semantic_mobility_nav.semantic_detector_clip:main',
            'robot_cli = semantic_mobility_nav.robot_cli:main',
            'vla_navigator = semantic_mobility_nav.vla_navigator:main',
        ],
    },
)
