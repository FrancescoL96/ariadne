
namespace ariadne_cuda {
    void function(const int N, int * h_matrixA, int * h_matrixB, int * h_matrixC);
    float float_approximation (float first_value, float second_value, int operation, int rounding);
    double double_approximation (double first_value, double second_value, int operation, int rounding);
}