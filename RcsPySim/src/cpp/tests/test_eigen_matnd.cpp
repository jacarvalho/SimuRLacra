#include <catch2/catch.hpp>

#include <util/eigen_matnd.h>


TEMPLATE_TEST_CASE("Eigen/MatNd conversion", "[matrix]", Eigen::MatrixXd, Eigen::Matrix4d,
                   (Eigen::Matrix<double, 2, 3, Eigen::RowMajor>), Eigen::VectorXd, Eigen::RowVectorXd)
{
    int rows = TestType::RowsAtCompileTime;
    if (rows == -1)
    {
        // arbitrary dynamic value
        rows = 4;
    }
    int cols = TestType::ColsAtCompileTime;
    if (cols == -1)
    {
        // arbitrary dynamic value
        cols = 4;
    }
    
    // create random eigen matrix
    TestType eigen_mat = TestType::Random(rows, cols);
    
    // create MatNd
    MatNd* rcs_mat = NULL;
    MatNd_fromStack(rcs_mat, rows, cols)
    
    SECTION("Eigen to MatNd")
    {
        // perform copy
        Rcs::copyEigen2MatNd(rcs_mat, eigen_mat);
        
        // verify
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                DYNAMIC_SECTION("Entry [" << r << ", " << c << "]")
                {
                    CHECK(eigen_mat(r, c) == MatNd_get2(rcs_mat, r, c));
                }
            }
        }
    }
    
    SECTION("MatNd to Eigen")
    {
        TestType new_eigen_mat;
        
        // perform copy
        Rcs::copyMatNd2Eigen(new_eigen_mat, rcs_mat);
        
        // verify
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                DYNAMIC_SECTION("Entry [" << r << ", " << c << "]")
                {
                    CHECK(MatNd_get2(rcs_mat, r, c) == new_eigen_mat(r, c));
                }
            }
        }
    }
}

TEMPLATE_TEST_CASE("Wrapping Eigen as MatNd", "[matrix]", Eigen::VectorXd, Eigen::Vector4d, Eigen::RowVectorXd,
                   (Eigen::Matrix<double, 2, 3, Eigen::RowMajor>))
{
    int rows = TestType::RowsAtCompileTime;
    if (rows == -1)
    {
        // arbitrary dynamic value
        rows = 4;
    }
    int cols = TestType::ColsAtCompileTime;
    if (cols == -1)
    {
        // arbitrary dynamic value
        cols = 4;
    }
    
    // create random eigen matrix
    TestType eigen_mat = TestType::Random(rows, cols);
    
    // wrap
    MatNd rcs_mat = Rcs::viewEigen2MatNd(eigen_mat);
    
    // verify
    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < cols; c++)
        {
            DYNAMIC_SECTION("Entry [" << r << ", " << c << "]")
            {
                CHECK(eigen_mat(r, c) == MatNd_get2((&rcs_mat), r, c));
            }
        }
    }
}