import pypgo
import trimesh
import tqdm
import numpy as np
import os
import csv

def remesh_obj(input_file, output_file, faulty_files):
    try:
        raw_mesh = trimesh.load_mesh(input_file)
        raw_v = (raw_mesh.vertices).copy().flatten()
        raw_t = (raw_mesh.faces).copy().flatten()

        raw_mesh_trimeshgeo = pypgo.create_trimeshgeo(raw_v, raw_t)

        trimesh_remeshed = pypgo.mesh_isotropic_remeshing(raw_mesh_trimeshgeo, 0.025, 10, 60)
        tri_v = pypgo.trimeshgeo_get_vertices(trimesh_remeshed)
        tri_t = pypgo.trimeshgeo_get_triangles(trimesh_remeshed)

        trimesh.Trimesh(vertices=tri_v.reshape(-1, 3), faces=tri_t.reshape(-1, 3)).export(output_file)

    except Exception as e:
        faulty_files.append({"file": input_file, "reason": str(e)})
        return

def process_assembly_folder(assembly_folder, output_base_folder, faulty_files):
    if not os.path.exists(assembly_folder):
        print(f"Error: Assembly folder '{assembly_folder}' not found.")
        return

    folder_name = os.path.basename(assembly_folder)
    output_folder = os.path.join(output_base_folder, folder_name + "_remeshed")
    os.makedirs(output_folder, exist_ok=True)

    obj_files = [f for f in os.listdir(assembly_folder) if f.endswith(".obj")]

    for obj_file in tqdm.tqdm(obj_files, desc=f"Remeshing {folder_name}"):
        input_path = os.path.join(assembly_folder, obj_file)
        output_path = os.path.join(output_folder, obj_file)
        remesh_obj(input_path, output_path, faulty_files)

def process_all_assembly_folders(input_base_folder, output_base_folder, faulty_files):
    if not os.path.exists(input_base_folder):
        print(f"Error: Input base folder '{input_base_folder}' not found.")
        return

    assembly_folders = [
        os.path.join(input_base_folder, d)
        for d in os.listdir(input_base_folder)
        if os.path.isdir(os.path.join(input_base_folder, d))
    ]

    for assembly_folder in assembly_folders:
        process_assembly_folder(assembly_folder, output_base_folder, faulty_files)

def output_faulty_files_csv(faulty_files, output_csv_path):
    if not faulty_files:
        return

    with open(output_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['file', 'reason']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for file_info in faulty_files:
            writer.writerow(file_info)

if __name__ == "__main__":
    input_base_folder = "stackables"
    output_base_folder = "stackables_remeshed"
    faulty_files = []

    if not os.path.exists(output_base_folder):
        os.makedirs(output_base_folder)

    process_all_assembly_folders(input_base_folder, output_base_folder, faulty_files)

    output_faulty_files_csv(faulty_files, "faulty_files.csv")

    if faulty_files:
        print("Faulty files found. See faulty_files.csv for details.")
    else:
        print("All files processed successfully.")