import json
import numpy as np

def load_obj_vertices(filename):
    """Loads vertex positions from an OBJ file (Nx3 array)."""
    vertices = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.split()
                x, y, z = map(float, parts[1:4])
                vertices.append([x, y, z])
    return np.array(vertices, dtype=float)

def load_deform(filename):
    """Loads the .deform file (JSON-like)."""
    with open(filename, 'r') as f:
        return json.loads(f.read())

def naive_bounded_z_knn(cyl_pos, mug_vertices, z_tolerance):
    """
    Finds the index of the nearest mug vertex to 'cyl_pos' (in full 3D),
    restricted to mug vertices whose Z is within +/- z_tolerance of cyl_pos.z.
    If no vertices are in-range, falls back to searching the entire mug.
    """
    z_cyl = cyl_pos[2]
    z_min = z_cyl - z_tolerance
    z_max = z_cyl + z_tolerance

    # Filter by Z
    mask = (mug_vertices[:,2] >= z_min) & (mug_vertices[:,2] <= z_max)
    candidate_indices = np.where(mask)[0]

    if len(candidate_indices) == 0:
        # Fallback: no candidate in that Z band; pick the absolute nearest
        diffs = mug_vertices - cyl_pos
        dists_sq = np.sum(diffs*diffs, axis=1)
        return np.argmin(dists_sq)

    # Among those candidates, find the one closest in full 3D
    candidate_points = mug_vertices[candidate_indices]
    diffs = candidate_points - cyl_pos
    dists_sq = np.sum(diffs*diffs, axis=1)
    best_sub = np.argmin(dists_sq)
    return candidate_indices[best_sub]

def reassign_deform_indices(
    deform_data,
    cylinder_vertices,
    mug_vertices,
    z_tolerance
):
    """
    Creates a new .deform where each old cylinder vertex index is replaced
    by the nearest mug vertex index (within +/- z_tolerance in Z).
    The displacement vector remains the same but is assigned to the new key.
    """
    updated_deform = []

    for operation in deform_data:
        new_displacements = {}
        for old_cyl_idx_str, old_disp in operation["displacements"].items():
            # Convert from string to int
            old_cyl_idx = int(old_cyl_idx_str)
            cyl_pos = cylinder_vertices[old_cyl_idx]

            # Find nearest mug vertex with bounded Z
            mug_idx = naive_bounded_z_knn(cyl_pos, mug_vertices, z_tolerance)

            # Keep the same displacement vector but
            # store it under the new mug index
            new_displacements[str(mug_idx)] = old_disp

        updated_deform.append({
            "displacements": new_displacements,
            "label": operation["label"]
        })

    return updated_deform

def save_deform(filename, deform_data):
    """Saves the updated .deform (JSON) data to disk."""
    with open(filename, 'w') as f:
        json.dump(deform_data, f, indent=2)

def main():
    # Adjust file paths as needed
    cylinder_verts = load_obj_vertices('MUG_TEST/mug_1_fit.obj')
    mug_verts = load_obj_vertices('MUG_TEST/mug_1.obj')
    cylinder_deform_data = load_deform('MUG_TEST/mug_1_fit.deform')

    # Example: 2.5% of cylinder's total height
    z_vals = cylinder_verts[:,2]
    cyl_height = z_vals.max() - z_vals.min()
    z_tolerance = 0.025 * cyl_height

    # Reassign each cylinder vertex index -> nearest mug vertex index
    new_deform = reassign_deform_indices(
        deform_data=cylinder_deform_data,
        cylinder_vertices=cylinder_verts,
        mug_vertices=mug_verts,
        z_tolerance=z_tolerance
    )

    save_deform('z_locked.deform', new_deform)

if __name__ == '__main__':
    main()
