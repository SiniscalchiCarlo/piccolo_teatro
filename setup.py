from setuptools import setup, find_packages

setup(
    name="piccolo_teatro",
    varsion="0.1.0",
    description="""Libreria usata dal Piccolo Teatro di Milano per 
                    fare previsioni sull'andamento di acquisto 
                    degli spettacoli""",
    author="Siniscalchi Carlo",
    packages=find_packages(),
    install_requires=[
        "matplotlib",
        "pandas",
        "pydantic",
        "scikit-learn",
        "xgboost",
    ]

)