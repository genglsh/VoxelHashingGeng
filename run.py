import os
import sys
ins1 = "cd build/"
ins2 = "make clean"
ins3 = "cmake .. && make -j"
ins4 = "cd ../Example/"
ins5 = "./load_frames ../scene0220_02/ ../scene0220_02/scene0220_02.yaml"
os.system(ins1)
try:
    os.system(ins2)
except:
    pass
os.system(ins3)
os.system(ins4)
os.system(ins5)