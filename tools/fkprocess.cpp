#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <iostream>
#include <fstream>
#include <igl/principal_curvature.h>
#include <filesystem>
#include <fstream> // For file copying

namespace fs = std::filesystem;

int main(int argc, char *argv[]) {
    fs::path input_dir = "stackables_remeshed";
    fs::path output_dir = "stackables_remeshed_fk";

    if (!fs::exists(input_dir) || !fs::is_directory(input_dir)) {
        std::cerr << "Input directory '" << input_dir << "' does not exist or is not a directory." << std::endl;
        return 1;
    }

    if (!fs::exists(output_dir)) {
        fs::create_directory(output_dir);
    }

    for (const auto& entry : fs::recursive_directory_iterator(input_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".obj") {
            fs::path input_path = entry.path();
            fs::path relative_path = fs::relative(input_path, input_dir);
            fs::path output_path = output_dir / relative_path;
            fs::path output_obj_path = output_dir / relative_path; // Path for the copied .obj file
            output_path.replace_extension(".fk");

            fs::create_directories(output_path.parent_path());

            // Copy the .obj file
            try {
                fs::copy_file(input_path, output_obj_path, fs::copy_options::overwrite_existing);
                std::cout << "Copied: " << input_path << " -> " << output_obj_path << std::endl;
            } catch (const fs::filesystem_error& e) {
                std::cerr << "Error copying OBJ file: " << input_path << " " << e.what() << std::endl;
                continue; // Skip to the next file if copying fails
            }

            Eigen::MatrixXd V;
            Eigen::MatrixXi F;
            try {
                igl::readOBJ(input_path.string(), V, F);
            } catch (const std::exception& e) {
                std::cerr << "Error reading OBJ file: " << input_path << " " << e.what() << std::endl;
                continue;
            }

            Eigen::MatrixXd principal_curvatures1, principal_curvatures2;
            Eigen::MatrixXd principal_directions1, principal_directions2;
            try {
                igl::principal_curvature(V, F, principal_directions1, principal_directions2, principal_curvatures1, principal_curvatures2, 5, true);
            } catch (const std::exception& e) {
                std::cerr << "Error computing principal curvatures: " << input_path << " " << e.what() << std::endl;
                continue;
            }

            Eigen::MatrixXd faceCurvatures(F.rows(), 2);
            for (int i = 0; i < F.rows(); i++) {
                int v1 = F(i, 0), v2 = F(i, 1), v3 = F(i, 2);
                faceCurvatures(i, 0) = (principal_curvatures1(v1) + principal_curvatures1(v2) + principal_curvatures1(v3)) / 3.0;
                faceCurvatures(i, 1) = (principal_curvatures2(v1) + principal_curvatures2(v2) + principal_curvatures2(v3)) / 3.0;
            }

            std::ofstream file(output_path.string());
            if (file.is_open()) {
                for (int i = 0; i < faceCurvatures.rows(); i++) {
                    file << faceCurvatures(i, 0) << " " << faceCurvatures(i, 1) << "\n";
                }
                file.close();
                std::cout << "Processed: " << input_path << " -> " << output_path << std::endl;
            } else {
                std::cerr << "Unable to open output file: " << output_path << std::endl;
            }
        }
    }

    return 0;
}