# RAG-Based LLMs for Code Generation

This repository contains code and instructions for implementing Retrieval-Augmented Generation (RAG) based Language Models (LLMs) for code generation. This project leverages the chromadb library for efficient document storage and retrieval and pythermalcomfort for thermal comfort calculations.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Demo Video](#demo-video)

## Installation

### Prerequisites

- Python 3.8 or higher
- chromadb library
- pythermalcomfort library

### Setup

1. Clone the repository:
    ```sh
    git clone https://github.com/your-username/rag-code-generation.git
    cd rag-code-generation
    ```

2. Install the required libraries:
    ```sh
    pip install chromadb pythermalcomfort
    ```

## Usage

### Querying Documents

Query the stored documents with a specific code snippet:

```python
results = fin_collection.query(query_texts=[f"""
from pythermalcomfort.models import pmv_ppd

results = pmv_ppd(tdb=22, tr=22, vr=0.1, rh=60, met=1, clo=0.9, standard="ASHRAE")
print(results)

I got the below output;
Traceback (most recent call last):
    from pythermalcomfort.models import pmv_ppd
  File "AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\pythermalcomfort\\__init__.py", line 3, in <module>
    from pythermalcomfort.models import *
  File "AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\pythermalcomfort\\models.py", line 28, in <module>
    from pythermalcomfort.jos3_functions import thermoregulation as threg
  File "AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\pythermalcomfort\\jos3_functions\\thermoregulation.py", line 17, in <module>
    from pythermalcomfort.jos3_functions import construction as cons
  File "AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\pythermalcomfort\\jos3_functions\\construction.py", line 16, in <module>
    from pythermalcomfort.jos3_functions.parameters import Default
  File "AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\pythermalcomfort\\jos3_functions\\parameters.py", line 20, in <module>
    @dataclass
     ^^^^^^^^^
  File "AppData\\Local\\Programs\\Python\\Python311\\Lib\\dataclasses.py", line 1230, in dataclass
    return wrap(cls)
           ^^^^^^^^^
  File "AppData\\Local\\Programs\\Python\\Python311\\Lib\\dataclasses.py", line 1220, in wrap
    return _process_class(cls, init, repr, eq, order, unsafe_hash,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "AppData\\Local\\Programs\\Python\\Python311\\Lib\\dataclasses.py", line 958, in _process_class
    cls_fields.append(_get_field(cls, name, type, kw_only))
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "AppData\\Local\\Programs\\Python\\Python311\\Lib\\dataclasses.py", line 815, in _get_field
    raise ValueError(f'mutable default for field '
ValueError: mutable default <class 'numpy.ndarray'> for field local_bsa is not allowed: use default_factory

When pythermalcomfort version is 2.8.4 or 2.8.3 this error occurs. 2.8.2 and 2.8.1 no issue."""], n_results=3)

for i, doc in enumerate(results["documents"][0]):
    print(f"\n\nDocument {i+1}:\n{doc}")
```

## File Structure

```
RAG_Code_Generation/
│
├── data/
│   └── pythermalcomfort/
│
├── assets/
│   └── ppt
│
├── notebooks/
│   └── retrieval_files
│
├── README.md
│
└── rag_llm.py

```


## Demo Video

https://github.com/user-attachments/assets/b1a02c67-9bbe-44d8-aad3-32d048e27af6




