#pragma once
#include <cstdint>
#include <vector>

struct BallisticResult {
    bool success = false;
    double pitch = 0.;
    double t = 0.;
};

class BallisticSolver {
public:
    BallisticSolver() = default;

    bool is_built() const;
    void build(double v, double k);

    BallisticResult query(double x, double y, double z) const;

private:
    size_t step() const;
    size_t length() const;
    size_t pos(size_t i, size_t j) const;

    BallisticResult query2d(double R, double z) const;

private:
    static constexpr double G = 9.79;
    static constexpr double Z_MIN = -2.0;
    static constexpr double Z_MAX =  4.0;
    static constexpr double R_MAX = 20.0;
    static constexpr double INTERP = 0.0025;
    static constexpr size_t NUM_THETA_SAMPLES = 2000;

    bool has_built_LUT;

    std::vector<double> theta_LUT;
    std::vector<double> t_LUT;
};
