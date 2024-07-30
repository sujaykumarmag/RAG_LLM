import os
import chromadb

DATA_DIR = 'data/pythermalcomfort/'


def get_file_paths(directory):
    file_paths = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path):
            file_paths.append(item_path)
        elif os.path.isdir(item_path):
            file_paths.extend(get_file_paths(item_path))
    return file_paths



def get_file_contents(file_paths):
    file_contents = []
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content_bytes = file.read()
                content_str = content_bytes
                file_contents.append(content_str)
        except Exception as e:
            strs = str(e)+" Sujay"
            print(strs)

    return file_contents


all_file_paths = get_file_paths(DATA_DIR)
text = get_file_contents(all_file_paths)


client = chromadb.Client()
fin_collection = client.get_or_create_collection("rag_doc_llm")

ids= [str(x) for x in range(len(text))]
fin_collection.add(documents=text,ids=ids)



results = fin_collection.query(query_texts=[f"""
from pythermalcomfort.models import pmv_ppd

results = pmv_ppd(tdb=22, tr=22, vr=0.1, rh=60, met=1, clo=0.9, standard="ASHRAE")
print(results)

I got the below output;
Traceback (most recent call last):
    from pythermalcomfort.models import pmv_ppd
  File "AppData\Local\Programs\Python\Python311\Lib\site-packages\pythermalcomfort\__init__.py", line 3, in <module>
    from pythermalcomfort.models import *
  File "AppData\Local\Programs\Python\Python311\Lib\site-packages\pythermalcomfort\models.py", line 28, in <module>
    from pythermalcomfort.jos3_functions import thermoregulation as threg
  File "AppData\Local\Programs\Python\Python311\Lib\site-packages\pythermalcomfort\jos3_functions\thermoregulation.py", line 17, in <module>
    from pythermalcomfort.jos3_functions import construction as cons
  File "AppData\Local\Programs\Python\Python311\Lib\site-packages\pythermalcomfort\jos3_functions\construction.py", line 16, in <module>
    from pythermalcomfort.jos3_functions.parameters import Default
  File "AppData\Local\Programs\Python\Python311\Lib\site-packages\pythermalcomfort\jos3_functions\parameters.py", line 20, in <module>
    @dataclass
     ^^^^^^^^^
  File "AppData\Local\Programs\Python\Python311\Lib\dataclasses.py", line 1230, in dataclass
    return wrap(cls)
           ^^^^^^^^^
  File "AppData\Local\Programs\Python\Python311\Lib\dataclasses.py", line 1220, in wrap
    return _process_class(cls, init, repr, eq, order, unsafe_hash,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "AppData\Local\Programs\Python\Python311\Lib\dataclasses.py", line 958, in _process_class
    cls_fields.append(_get_field(cls, name, type, kw_only))
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "AppData\Local\Programs\Python\Python311\Lib\dataclasses.py", line 815, in _get_field
    raise ValueError(f'mutable default for field '
ValueError: mutable default <class 'numpy.ndarray'> for field local_bsa is not allowed: use default_factory

When pythermalcomfort version is 2.8.4 or 2.8.3 this error occurs. 2.8.2 and 2.8.1 no issue."""],n_results=3)
print(results["documents"][0][1])

print("\n\n"+results["documents"][0][1])

print("\n\n"+results["documents"][0][2])


