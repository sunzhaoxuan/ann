#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <vector>

namespace ann_opq {

static inline std::vector<float> identity_rotation(size_t dim) {
    std::vector<float> rotation(dim * dim, 0.0f);
    for (size_t d = 0; d < dim; ++d) {
        rotation[d * dim + d] = 1.0f;
    }
    return rotation;
}

// Row-major convention: output = rotation * input.
static inline void rotate_vector(
    const float* input,
    const std::vector<float>& rotation,
    size_t dim,
    float* output
) {
    for (size_t row = 0; row < dim; ++row) {
        const float* matrix_row = rotation.data() + row * dim;
        float value = 0.0f;
        for (size_t col = 0; col < dim; ++col) {
            value += matrix_row[col] * input[col];
        }
        output[row] = value;
    }
}

static inline void rotate_matrix(
    const std::vector<float>& input,
    size_t rows,
    size_t dim,
    const std::vector<float>& rotation,
    std::vector<float>& output
) {
    output.resize(rows * dim);
    for (size_t row = 0; row < rows; ++row) {
        rotate_vector(
            input.data() + row * dim,
            rotation,
            dim,
            output.data() + row * dim
        );
    }
}

static inline float squared_l2(
    const float* left,
    const float* right,
    size_t dim
) {
    float distance = 0.0f;
    for (size_t d = 0; d < dim; ++d) {
        const float diff = left[d] - right[d];
        distance += diff * diff;
    }
    return distance;
}

// Trains a temporary global PQ and reconstructs every rotated training vector.
// This PQ is used only by OPQ; the final index still trains one PQ per IVF list.
static inline double pq_reconstruct(
    const std::vector<float>& rotated,
    size_t rows,
    size_t dim,
    size_t subspaces,
    size_t centroids,
    size_t kmeans_iters,
    std::vector<float>& reconstruction
) {
    const size_t subdim = dim / subspaces;
    const size_t effective_centroids = std::min(centroids, rows);
    reconstruction.assign(rows * dim, 0.0f);

    if (rows == 0 || effective_centroids == 0) {
        return 0.0;
    }

    std::vector<float> codebook(effective_centroids * subdim);
    std::vector<float> sums(effective_centroids * subdim);
    std::vector<size_t> counts(effective_centroids);
    std::vector<size_t> assignments(rows);

    for (size_t m = 0; m < subspaces; ++m) {
        const size_t offset = m * subdim;

        for (size_t c = 0; c < effective_centroids; ++c) {
            size_t position = c * rows / effective_centroids;
            if (position >= rows) {
                position = rows - 1;
            }
            const float* source = rotated.data() + position * dim + offset;
            std::copy(
                source,
                source + subdim,
                codebook.data() + c * subdim
            );
        }

        for (size_t iter = 0; iter < kmeans_iters; ++iter) {
            std::fill(sums.begin(), sums.end(), 0.0f);
            std::fill(counts.begin(), counts.end(), 0);

            for (size_t row = 0; row < rows; ++row) {
                const float* subvector = rotated.data() + row * dim + offset;
                size_t best = 0;
                float best_distance = std::numeric_limits<float>::max();
                for (size_t c = 0; c < effective_centroids; ++c) {
                    const float distance = squared_l2(
                        subvector,
                        codebook.data() + c * subdim,
                        subdim
                    );
                    if (distance < best_distance) {
                        best_distance = distance;
                        best = c;
                    }
                }

                assignments[row] = best;
                float* sum = sums.data() + best * subdim;
                for (size_t d = 0; d < subdim; ++d) {
                    sum[d] += subvector[d];
                }
                ++counts[best];
            }

            for (size_t c = 0; c < effective_centroids; ++c) {
                float* centroid = codebook.data() + c * subdim;
                if (counts[c] == 0) {
                    const size_t position = (c + iter + 1) % rows;
                    const float* source = rotated.data() + position * dim + offset;
                    std::copy(source, source + subdim, centroid);
                    continue;
                }
                const float inverse = 1.0f / static_cast<float>(counts[c]);
                const float* sum = sums.data() + c * subdim;
                for (size_t d = 0; d < subdim; ++d) {
                    centroid[d] = sum[d] * inverse;
                }
            }
        }

        // Reassign against the final centroids before constructing X-hat.
        for (size_t row = 0; row < rows; ++row) {
            const float* subvector = rotated.data() + row * dim + offset;
            size_t best = 0;
            float best_distance = std::numeric_limits<float>::max();
            for (size_t c = 0; c < effective_centroids; ++c) {
                const float distance = squared_l2(
                    subvector,
                    codebook.data() + c * subdim,
                    subdim
                );
                if (distance < best_distance) {
                    best_distance = distance;
                    best = c;
                }
            }
            std::copy(
                codebook.data() + best * subdim,
                codebook.data() + (best + 1) * subdim,
                reconstruction.data() + row * dim + offset
            );
        }
    }

    double error = 0.0;
    for (size_t i = 0; i < rotated.size(); ++i) {
        const double diff = static_cast<double>(rotated[i])
            - static_cast<double>(reconstruction[i]);
        error += diff * diff;
    }
    return error / static_cast<double>(rows);
}

// Computes U and V from A = U S V^T using a one-sided Jacobi SVD, then
// returns the Procrustes solution U V^T. A is Y X^T for min ||R X - Y||.
static inline std::vector<float> orthogonal_procrustes(
    const std::vector<float>& source,
    const std::vector<float>& target,
    size_t rows,
    size_t dim
) {
    std::vector<double> matrix(dim * dim, 0.0);
    for (size_t row = 0; row < rows; ++row) {
        const float* x = source.data() + row * dim;
        const float* y = target.data() + row * dim;
        for (size_t i = 0; i < dim; ++i) {
            double* matrix_row = matrix.data() + i * dim;
            const double yi = static_cast<double>(y[i]);
            for (size_t j = 0; j < dim; ++j) {
                matrix_row[j] += yi * static_cast<double>(x[j]);
            }
        }
    }

    std::vector<double> right(dim * dim, 0.0);
    for (size_t d = 0; d < dim; ++d) {
        right[d * dim + d] = 1.0;
    }

    const size_t max_sweeps = 32;
    for (size_t sweep = 0; sweep < max_sweeps; ++sweep) {
        bool changed = false;
        for (size_t p = 0; p < dim; ++p) {
            for (size_t q = p + 1; q < dim; ++q) {
                double alpha = 0.0;
                double beta = 0.0;
                double gamma = 0.0;
                for (size_t row = 0; row < dim; ++row) {
                    const double left = matrix[row * dim + p];
                    const double right_value = matrix[row * dim + q];
                    alpha += left * left;
                    beta += right_value * right_value;
                    gamma += left * right_value;
                }

                if (alpha == 0.0 || beta == 0.0
                    || std::fabs(gamma) <= 1e-7 * std::sqrt(alpha * beta)) {
                    continue;
                }

                const double zeta = (beta - alpha) / (2.0 * gamma);
                const double sign = zeta >= 0.0 ? 1.0 : -1.0;
                const double tangent = sign
                    / (std::fabs(zeta) + std::sqrt(1.0 + zeta * zeta));
                const double cosine = 1.0 / std::sqrt(1.0 + tangent * tangent);
                const double sine = cosine * tangent;

                for (size_t row = 0; row < dim; ++row) {
                    const size_t p_index = row * dim + p;
                    const size_t q_index = row * dim + q;
                    const double old_p = matrix[p_index];
                    const double old_q = matrix[q_index];
                    matrix[p_index] = cosine * old_p - sine * old_q;
                    matrix[q_index] = sine * old_p + cosine * old_q;

                    const double old_vp = right[p_index];
                    const double old_vq = right[q_index];
                    right[p_index] = cosine * old_vp - sine * old_vq;
                    right[q_index] = sine * old_vp + cosine * old_vq;
                }
                changed = true;
            }
        }
        if (!changed) {
            break;
        }
    }

    std::vector<double> left(dim * dim, 0.0);
    for (size_t col = 0; col < dim; ++col) {
        double norm = 0.0;
        for (size_t row = 0; row < dim; ++row) {
            const double value = matrix[row * dim + col];
            norm += value * value;
        }
        norm = std::sqrt(norm);

        std::vector<double> vector(dim, 0.0);
        if (norm > 1e-12) {
            for (size_t row = 0; row < dim; ++row) {
                vector[row] = matrix[row * dim + col] / norm;
            }
        }

        // Explicit MGS keeps U orthogonal even when the cross-covariance is
        // nearly rank deficient (common in tiny smoke datasets).
        for (size_t previous = 0; previous < col; ++previous) {
            double projection = 0.0;
            for (size_t row = 0; row < dim; ++row) {
                projection += vector[row] * left[row * dim + previous];
            }
            for (size_t row = 0; row < dim; ++row) {
                vector[row] -= projection * left[row * dim + previous];
            }
        }

        double vector_norm = 0.0;
        for (size_t row = 0; row < dim; ++row) {
            vector_norm += vector[row] * vector[row];
        }
        vector_norm = std::sqrt(vector_norm);

        if (vector_norm <= 1e-10) {
            for (size_t candidate = 0; candidate < dim; ++candidate) {
                std::fill(vector.begin(), vector.end(), 0.0);
                vector[candidate] = 1.0;
                for (size_t previous = 0; previous < col; ++previous) {
                    double projection = 0.0;
                    for (size_t row = 0; row < dim; ++row) {
                        projection += vector[row] * left[row * dim + previous];
                    }
                    for (size_t row = 0; row < dim; ++row) {
                        vector[row] -= projection * left[row * dim + previous];
                    }
                }
                vector_norm = 0.0;
                for (size_t row = 0; row < dim; ++row) {
                    vector_norm += vector[row] * vector[row];
                }
                vector_norm = std::sqrt(vector_norm);
                if (vector_norm > 1e-10) {
                    break;
                }
            }
        }

        for (size_t row = 0; row < dim; ++row) {
            left[row * dim + col] = vector[row] / vector_norm;
        }
    }

    std::vector<float> rotation(dim * dim, 0.0f);
    for (size_t row = 0; row < dim; ++row) {
        for (size_t col = 0; col < dim; ++col) {
            double value = 0.0;
            for (size_t k = 0; k < dim; ++k) {
                value += left[row * dim + k] * right[col * dim + k];
            }
            rotation[row * dim + col] = static_cast<float>(value);
        }
    }
    return rotation;
}

static inline double orthogonality_error(
    const std::vector<float>& rotation,
    size_t dim
) {
    double max_error = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            double value = 0.0;
            for (size_t k = 0; k < dim; ++k) {
                value += static_cast<double>(rotation[k * dim + i])
                    * static_cast<double>(rotation[k * dim + j]);
            }
            const double expected = i == j ? 1.0 : 0.0;
            max_error = std::max(max_error, std::fabs(value - expected));
        }
    }
    return max_error;
}

static inline std::vector<float> train_rotation(
    const std::vector<float>& training_vectors,
    size_t rows,
    size_t dim,
    size_t subspaces,
    size_t centroids,
    size_t opq_iters,
    size_t kmeans_iters
) {
    std::vector<float> rotation = identity_rotation(dim);
    if (rows == 0 || dim == 0 || subspaces == 0 || opq_iters == 0) {
        return rotation;
    }

    std::vector<float> rotated;
    std::vector<float> reconstruction;
    for (size_t iter = 0; iter < opq_iters; ++iter) {
        rotate_matrix(training_vectors, rows, dim, rotation, rotated);
        const double error = pq_reconstruct(
            rotated,
            rows,
            dim,
            subspaces,
            centroids,
            std::max<size_t>(1, kmeans_iters),
            reconstruction
        );
        rotation = orthogonal_procrustes(
            training_vectors,
            reconstruction,
            rows,
            dim
        );
        std::cerr << "OPQ iteration " << (iter + 1) << "/" << opq_iters
                  << ", temporary PQ error = " << error
                  << ", orthogonality error = "
                  << orthogonality_error(rotation, dim) << "\n";
    }
    return rotation;
}

}  // namespace ann_opq
