#include <vector>
#include <cstdlib>
#include <cmath>
#include <algorithm>

/*
 * brief
 * @x: host memory pointer to @m rows @n columns row-major input matrix
 * @y: host memory pointer to @m rows @n columns row-major output matrix
 * if fabs(v) > zero_abs, treat v as non-zero
 */
template <typename T>
void matrix_shuffle(const T *x,
                    T *y,
                    int m,
                    int n,
                    T zero_abs,
                    double threshold) {
    std::vector<int> nz_rows(m);
    std::vector<int> cut_rows(m);
    int nz_rows_size = 0;
    int cut_rows_size = 0;
    int free_entry_size = 0;
    int cut_items_size = 0;

    for (int i = 0; i < m; ++i) {
        const T *x_row_ptr = x + i * n;
        int nnz = 0;
        for (int j = 0; j < n; ++j) {
            if (std::fabs(x_row_ptr[j] > zero_abs)) {
                ++nnz;
            }
        }

        if (static_cast<double>(nnz) / n < threshold) {
            cut_rows[cut_rows_size] = i;
            ++cut_rows_size;
            cut_items_size += nnz;
        } else {
            nz_rows[nz_rows_size] = i;
            ++nz_rows_size;
            free_entry_size += n - nnz;
        }
    }

    if (cut_items_size < 0) {
        memcpy(y, x, m * n * sizeof(T));
        return;
    }

    std::vector<int> free_entry;

    if (free_entry_size < cut_items_size) {
        int append_free_entry = cut_items_size - free_entry_size;
        int append_free_row = (append_free_entry - 1) / n + 1;
        free_entry.resize(free_entry_size + append_free_row * n);

        std::vector<int> zero_rows(cut_rows);
        int zero_rows_size = cut_rows_size;
        for (int i = 0; i < append_free_row; ++i) {
            int rand_idx = rand() % zero_rows_size;
            int append_row_idx = zero_rows[rand_idx];
            --zero_rows_size;
            zero_rows[rand_idx] = zero_rows[zero_rows_size];

            for (int j = 0; j < n; ++j) {
                free_entry[free_entry_size] = append_row_idx * n + j;
                ++free_entry_size;
            }
        }
    } else {
        free_entry.resize(free_entry_size);
    }

    int free_entry_idx = 0;
    for (int i = 0; i < nz_rows_size; ++i) {
        const T *x_row_ptr = x + nz_rows[i] * n;
        for (int j = 0; j < n; ++j) {
            if (!(std::fabs(x_row_ptr[j] > zero_abs))) {
                free_entry[free_entry_idx] = nz_rows[i] * n + j;
                ++free_entry_idx;
            }
        }
    }

    for (int i = 0; i < nz_rows_size; ++i) {
        memcpy(y + nz_rows[i] * n, x + nz_rows[i] * n, n * sizeof(T));
    }

    for (int i = 0; i < cut_rows_size; ++i) {
        int *x_row_ptr = x + cut_rows[i] * n;
        for (int j = 0; j < n; ++j) {
            if (!(x_row_ptr[j] > zero_abs)) {
                int rand_idx = rand() % free_entry_size;
                y[free_entry[rand_idx]] = x_row_ptr[j];
                --free_entry_size;
                free_entry[rand_idx] = free_entry[free_entry_size];
            }
        }
    }
}

int main() {

}

