import pymeshlab

import trimesh
import tqdm
import numpy as np

import os
import sys
import json


mesh_filename = sys.argv[1]

folder = os.path.dirname(mesh_filename)
filename = os.path.basename(mesh_filename).split(".")[0]

output_folder = f"results/{filename}"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    
# load deform file
deform_filename = os.path.join(folder, f"{filename}.deform")
with open(deform_filename, "r") as f:
    deform = json.load(f)


shared_vtx = set(deform[0]["displacements"].keys()).intersection(
    set(deform[1]["displacements"].keys())
)

shared_vtx_deform = {
    str(k): v for k, v in deform[0]["displacements"].items() if str(k) in shared_vtx
}

other_vtx_deform = [{}, {}]

for id in range(2):
    for k, v in deform[id]["displacements"].items():
        if str(k) not in shared_vtx:
            other_vtx_deform[id][str(k)] = np.array(v)

## generate new deform
num0, num1 = 6, 8
scale0 = np.linspace(0, 1, num0)
scale1 = np.linspace(0, 1, num1)

new_deform_list = []
for i in range(num0):
    for j in range(num1):
        new_deform = {}
        new_deform["label"] = f"{filename}_{i}_{j}"
        new_deform["displacements"] = {}
        new_deform["displacements"].update(shared_vtx_deform)
        for k, v in other_vtx_deform[0].items():
            new_deform["displacements"][k] = (v * scale0[i]).tolist()
        for k, v in other_vtx_deform[1].items():
            new_deform["displacements"][k] = (v * scale1[j]).tolist()

        new_deform_list.append(new_deform)

new_deform_filename = os.path.join(output_folder, f"{filename}_new.deform")
with open(new_deform_filename, "w") as f:
    json.dump(new_deform_list, f, indent=4)
    
fk_file = f"{folder}/{filename}.fk"
os.system(f"cp {fk_file} {output_folder}/{filename}_new.fk")

os.system(f"cp {mesh_filename} {output_folder}/{filename}_new.obj")


### launch demo
for i in range(num0):
    for j in range(num1):
        cmd = f"/Users/mhg/Projects/SlippagePreservingReshaping/build/apps/reshaping_demo/reshaping_demo -i {output_folder}/{filename}_new.obj -o {output_folder} -e {filename}_{i}_{j}"
        os.system(cmd)
