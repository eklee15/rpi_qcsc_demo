#include <mpi.h>
#include <fstream>
#include <vector>
#include <iostream>

const uint32_t BITLEN = 10;
const uint32_t MAXVAL = 1 << BITLEN;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    uint32_t total_count = 0;
    std::vector<uint32_t> data;
    printf("DEBUG cpp1");
    if (rank == 0) {
        std::ifstream fin("input.bin", std::ios::binary | std::ios::ate);
        total_count = fin.tellg() / sizeof(uint32_t);
        fin.seekg(0, std::ios::beg);

        data.resize(total_count);
        fin.read(reinterpret_cast<char*>(data.data()), total_count * sizeof(uint32_t));
        fin.close();
    }

    MPI_Bcast(&total_count, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    uint32_t local_n = total_count / size;
    std::vector<uint32_t> local_data(local_n);

    MPI_Scatter(rank == 0 ? data.data() : nullptr, local_n, MPI_UNSIGNED,
                local_data.data(), local_n, MPI_UNSIGNED,
                0, MPI_COMM_WORLD);

    std::vector<int> local_hist(MAXVAL, 0);
    for (auto v : local_data) {
        if (v < MAXVAL) local_hist[v]++;
    }

    std::vector<int> global_hist;
    if (rank == 0) global_hist.resize(MAXVAL, 0);

    MPI_Reduce(local_hist.data(),
               rank == 0 ? global_hist.data() : nullptr,
               MAXVAL, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::ofstream fout("output.json");
        fout << "{";
        bool first = true;
        for (uint32_t i = 0; i < MAXVAL; ++i) {
            if (global_hist[i] > 0) {
                if (!first) fout << ",";
                fout << "\"" << i << "\":" << global_hist[i];
                first = false;
            }
        }
        fout << "}\n";
        fout.close();
    }

    MPI_Finalize();
    return 0;
}
