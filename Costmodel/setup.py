from setuptools import setup, find_packages

setup(
    name="AnalyticalModel",
    version="0.1",
    packages=find_packages(),
    package_data={
        '':['chip_configs/*.json','topo_configs/*.json']
    },
    install_requires=[]
)