import tempfile

d = tempfile.mkdtemp()
print(d)
import os

print(os.path.exists(d))
print(os.path.isdir(d))
