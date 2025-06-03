from setuptools import setup, find_packages

setup(
    name="piccolo_teatro",
    version="0.1.1-dev",
    description="""Libreria usata dal Piccolo Teatro di Milano per 
                    fare previsioni sull'andamento di acquisto 
                    degli spettacoli""",
    author="Siniscalchi Carlo",
    include_package_data=True,
    package_data={"": ["*.pkl"]},
    packages=find_packages(),
    install_requires=[
        "matplotlib",
        "pandas",
        "pydantic",
        "scikit-learn",
        "xgboost",
    ]

)