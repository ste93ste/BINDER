//
// Created by stce on 02/01/25.
//

#ifndef DCT_H
#define DCT_H
# include <fftw3.h>
#include <string>
#include <fstream>
#include <sstream>

class DCT{

public:
    DCT(int N_x_i, int N_y_i, int N_z_i, int N_x_o, int N_y_o, int N_z_o, int threads, int fast_search=0);
    ~DCT();
    void dct() const;
    void idct() const;
    double* data_dct_in;
    double* data_dct_out;
    double* data_idct_in;
    double* data_idct_out;
    static bool wisdom_loaded;
    static std::string wisdom_file;
    static int saved_dims[6];

    static std::string getWisdomFilename(const std::string& base_filename,
                                         const int Nx_i, const int Ny_i, const int Nz_i,
                                         const int Nx_o, const int Ny_o, const int Nz_o) {
        std::stringstream ss;
        ss << base_filename << "_"
           << Nx_i << "x" << Ny_i << "x" << Nz_i << "_"
           << Nx_o << "x" << Ny_o << "x" << Nz_o << ".wisdom";
        return ss.str();
    }

    // Add static method to initialize wisdom
    static void initializeWisdom(const std::string& base_filename,
                                 const int Nx_i, const int Ny_i, const int Nz_i,
                                 const int Nx_o, const int Ny_o, const int Nz_o) {
        wisdom_file = getWisdomFilename(base_filename,
                                      Nx_i, Ny_i, Nz_i,
                                      Nx_o, Ny_o, Nz_o);

        // Store dimensions
        saved_dims[0] = Nx_i; saved_dims[1] = Ny_i; saved_dims[2] = Nz_i;
        saved_dims[3] = Nx_o; saved_dims[4] = Ny_o; saved_dims[5] = Nz_o;

        // Try to load existing wisdom for these dimensions
        if (std::ifstream(wisdom_file).good()) {
            wisdom_loaded = (fftw_import_wisdom_from_filename(wisdom_file.c_str()) != 0);
        } else {
            wisdom_loaded = false;
        }
    }

private:
    fftw_plan plan_forward;
    fftw_plan plan_backward;
    int threads;
    int N_x_i;
    int N_y_i;
    int N_z_i;
    int N_x_o;
    int N_y_o;
    int N_z_o;
    double norm_factor_forward;
    double norm_factor_backward;
};



#endif //DCT_H
