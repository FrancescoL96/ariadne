
namespace ariadne_cuda {
    void function(const int N, int * h_matrixA, int * h_matrixB, int * h_matrixC);
    float float_approximation (float first_value, float second_value, int operation, int rounding);
    double double_approximation (double first_value, double second_value, int operation, int rounding);
    double * mallocManagedDouble(int size);
    int * mallocManagedInt(int size);
    void _ifma(int *x_index_vector, double x_value, double x_value_neg, int *y_index_matrix, double *y_value_vector, int ya_len, int y_size, double * error);
}