#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <iostream>
#include <fstream>
#include <igl/principal_curvature.h>

int main(int argc, char *argv[])
{   
    std::string inputFile = argv[1];
    std::string outputFile = argv[2];
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    igl::readOBJ(inputFile, V, F);

    Eigen::MatrixXd principal_curvatures1, principal_curvatures2;
    Eigen::MatrixXd principal_directions1, principal_directions2;
    igl::principal_curvature(V, F, principal_directions1, principal_directions2, principal_curvatures1, principal_curvatures2, 5, true);

    Eigen::MatrixXd faceCurvatures(F.rows(), 2);
    for (int i = 0; i < F.rows(); i++)
    {
        int v1 = F(i, 0), v2 = F(i, 1), v3 = F(i, 2);
        // Average the principal curvatures for the face
        faceCurvatures(i, 0) = (principal_curvatures1(v1) + principal_curvatures1(v2) + principal_curvatures1(v3)) / 3.0;
        faceCurvatures(i, 1) = (principal_curvatures2(v1) + principal_curvatures2(v2) + principal_curvatures2(v3)) / 3.0;
    }

    // save to txt file
    std::ofstream file(outputFile);
    if (file.is_open())
    {
        for (int i = 0; i < faceCurvatures.rows(); i++)
        {
            file << faceCurvatures(i, 0) << " " << faceCurvatures(i, 1) << "\n";
        }
        file.close();
    }
    else
    {
        std::cout << "Unable to open file";
    }

    return 0;
}