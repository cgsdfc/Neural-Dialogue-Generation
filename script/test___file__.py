import os

print(__file__)  # abs
print(os.path.abspath(__file__))
print(os.path.dirname(__file__))
script_dir = os.path.dirname(__file__)
data_dir = os.path.abspath(os.path.join(script_dir, "..", "data"))
print(data_dir)
project_root = os.path.dirname(os.path.dirname(__file__))
print(project_root)
