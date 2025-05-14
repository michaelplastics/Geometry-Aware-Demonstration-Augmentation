import pymeshlab

import trimesh
import tqdm
import numpy as np

import os
import sys

def mesh_repair_and_clean(input_mesh_path, output_mesh_path, max_iterations=10, targetlen=1):
    """
    Loads a mesh from `input_mesh_path`, then iteratively repairs non-manifold 
    edges and vertices until the mesh is manifold or `max_iterations` is reached.
    Finally, reorients all faces coherently and saves the resulting mesh.
    """
    ms = pymeshlab.MeshSet()
    
    # Load mesh
    ms.load_new_mesh(input_mesh_path)
    ms.apply_filter('meshing_poly_to_tri')
    
    for _ in range(max_iterations):
        # Compute topological measures to detect non-manifold elements
        ms.get_topological_measures()
        ms.meshing_repair_non_manifold_edges()
        
        ms.meshing_repair_non_manifold_vertices()
    
        # Finally, reorient all faces coherently
        ms.meshing_re_orient_faces_coherently()
    ms.meshing_re_orient_faces_by_geometry()
    # isotropic remesh
    ms.meshing_isotropic_explicit_remeshing(targetlen=pymeshlab.PercentageValue(targetlen))

    # Save the repaired mesh
    ms.save_current_mesh(output_mesh_path)
    
    mesh = trimesh.load(output_mesh_path)
    return mesh.vertices, mesh.faces

mesh_filename = sys.argv[1]

folder = os.path.dirname(mesh_filename)
filename = os.path.basename(mesh_filename).split(".")[0]

### remeshing
output_mesh_file = f"{folder}/{filename}_remeshed.obj"

if not os.path.exists(output_mesh_file):
    raw_mesh = trimesh.load_mesh(mesh_filename)
    # center
    raw_mesh.vertices -= raw_mesh.bounding_box.centroid
    # scale to [-1, 1]
    raw_mesh.vertices = (raw_mesh.vertices - raw_mesh.bounding_box.centroid) / \
        (raw_mesh.bounding_box.extents.max() / 2)
    
    norm_mesh_file = f"{folder}/{filename}_norm.obj"
    raw_mesh.export(norm_mesh_file)
    
    mesh_repair_vtx, mesh_repair_face = mesh_repair_and_clean(norm_mesh_file, output_mesh_file, targetlen=2)
    mesh_repair = trimesh.Trimesh(vertices=mesh_repair_vtx, faces=mesh_repair_face)
    mesh_repair.export(output_mesh_file)
    
    
### generate .fk file
fk_file = f"{folder}/{filename}_remeshed.fk"

cmd = f"/Users/mhg/Projects/SlippagePreservingReshaping/apps/libigl/build/bin/MyTest {output_mesh_file} {fk_file}"
os.system(cmd)

### launch demo
cmd = f"/Users/mhg/Projects/SlippagePreservingReshaping/build/apps/reshaping_app/reshaping_app -i {output_mesh_file} -o {folder}"
os.system(cmd)
