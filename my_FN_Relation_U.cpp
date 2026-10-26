#include "mex.h"
#include <algorithm> // for std::min
#include <cstring>   // for std::copy>

void my_FN_Relation_U(const mxArray *relationVectorsCell, const mxArray *fuzzyNeighborGroupCell, double *intRelation, mwSize dim, mwSize row, mwSize new_rows, double *index) {
    mwSize new_sample_start = row - new_rows;

    // Iterate over each dimension
    for (mwSize k = 0; k < dim; ++k) {
        double *relationVectors = mxGetPr(mxGetCell(relationVectorsCell, k));
        double *fuzzyNeighborGroup = mxGetPr(mxGetCell(fuzzyNeighborGroupCell, k));

        // 预先分配新的关系矩阵部分
        double *relation_matrix_part = new double[new_rows * row];
        for (mwSize i = 0; i < new_rows; ++i) {
            mwSize sample_idx = new_sample_start + i;
            mwSize idx_i = static_cast<mwSize>(index[sample_idx]);

            for (mwSize j = 0; j < row; ++j) {
                relation_matrix_part[i * row + j] = relationVectors[sample_idx * row + j];
            }

            for (mwSize j = 0; j < row; ++j) {
                if (static_cast<mwSize>(index[j]) == idx_i) {
                    relation_matrix_part[i * row + j] = 1.0;
                }
            }
        }

        // 更新fuzzyNeighborGroup矩阵
        for (mwSize i = 0; i < new_rows; ++i) {
            mwSize sample_idx = new_sample_start + i;

            for (mwSize j = 0; j < row; ++j) {
                fuzzyNeighborGroup[sample_idx * row + j] = relation_matrix_part[i * row + j];
                fuzzyNeighborGroup[j * row + sample_idx] = relation_matrix_part[i * row + j]; // 利用对称性
            }
        }

        delete[] relation_matrix_part;

        // 更新intRelation矩阵
        if (k == 0) {
            std::copy(fuzzyNeighborGroup, fuzzyNeighborGroup + row * row, intRelation);
        } else {
            for (mwSize i = 0; i < row * row; ++i) {
                intRelation[i] = std::min(intRelation[i], fuzzyNeighborGroup[i]);
            }
        }
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 7) {
        mexErrMsgIdAndTxt("MyToolbox:my_FN_Relation_update:nrhs", "Seven inputs required.");
    }
    if (nlhs != 2) {
        mexErrMsgIdAndTxt("MyToolbox:my_FN_Relation_update:nlhs", "Two outputs required.");
    }

    // Validate input types and sizes
    if (!mxIsCell(prhs[0]) || !mxIsCell(prhs[1]) || !mxIsDouble(prhs[2]) || !mxIsDouble(prhs[6])) {
        mexErrMsgIdAndTxt("MyToolbox:my_FN_Relation_update:invalidInputType", "Inputs must be of correct type.");
    }

    const mxArray *relationVectorsCell = prhs[0];
    const mxArray *fuzzyNeighborGroupCell = prhs[1];
    double *intRelation = mxGetPr(prhs[2]);
    mwSize dim = static_cast<mwSize>(mxGetScalar(prhs[3]));
    mwSize row = static_cast<mwSize>(mxGetScalar(prhs[4]));
    mwSize new_rows = static_cast<mwSize>(mxGetScalar(prhs[5]));
    double *index = mxGetPr(prhs[6]);

    // Validate input dimensions
    if (dim <= 0 || row <= 0 || new_rows <= 0) {
        mexErrMsgIdAndTxt("MyToolbox:my_FN_Relation_update:invalidInput", "Dimensions and row sizes must be positive.");
    }
    if (new_rows > row) {
        mexErrMsgIdAndTxt("MyToolbox:my_FN_Relation_update:invalidInput", "New rows cannot be greater than total rows.");
    }

    // Create output matrices
    plhs[0] = mxDuplicateArray(prhs[1]);
    plhs[1] = mxCreateDoubleMatrix(row, row, mxREAL);

    double *intRelation_out = mxGetPr(plhs[1]);

    my_FN_Relation_U(relationVectorsCell, plhs[0], intRelation_out, dim, row, new_rows, index);
}
