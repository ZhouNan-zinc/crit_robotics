#include "basic_predictor/ballestic_utils.hpp"
#include <cmath>
#include <limits>
#include <algorithm>

struct StateR {
    double vx;
    double vz;
    double z;
    double t;
};

StateR deriv_R(const StateR& s, double k, double g) {
    double v = std::hypot(s.vx, s.vz);
    return StateR{
        .vx=-k * v,
        .vz=(-k * v * s.vz - g) / s.vx,
        .z=s.vz/s.vx,
        .t=1.0/s.vx
    };
}

StateR add_scaled(const StateR& a, const StateR& b, double s) {
    return {
        .vx=a.vx + s*b.vx,
        .vz=a.vz + s*b.vz,
        .z=a.z + s*b.z,
        .t=a.t + s*b.t
    };
}

StateR rk4_step_R(const StateR& s, double INTERP, double k, double g) {
    StateR k1 = deriv_R(s, k, g);
    StateR k2 = deriv_R(add_scaled(s, k1, 0.5*INTERP), k, g);
    StateR k3 = deriv_R(add_scaled(s, k2, 0.5*INTERP), k, g);
    StateR k4 = deriv_R(add_scaled(s, k3, INTERP), k, g);

    return StateR{
        .vx=s.vx + (INTERP/6.0) * (k1.vx + 2*k2.vx + 2*k3.vx + k4.vx),
        .vz=s.vz + (INTERP/6.0) * (k1.vz + 2*k2.vz + 2*k3.vz + k4.vz),
        .z=s.z  + (INTERP/6.0) * (k1.z  + 2*k2.z  + 2*k3.z  + k4.z ),
        .t=s.t  + (INTERP/6.0) * (k1.t  + 2*k2.t  + 2*k3.t  + k4.t )
    };
}

size_t BallisticSolver::step() const {
    return (size_t)(llround((Z_MAX-Z_MIN)/INTERP) + 1u);
}

size_t BallisticSolver::length() const {
    return (size_t)(llround(R_MAX/INTERP) + 1u);
}

size_t BallisticSolver::pos(size_t i, size_t j) const {
    return i * step() + j;
}

bool BallisticSolver::is_built() const {
    return has_built_LUT;
}

void BallisticSolver::build(double v, double k) {
    const size_t LUT_size = (size_t)(std::llround(R_MAX / INTERP) + 1u) * step();
    std::vector<double> theta_table(LUT_size);
    std::vector<double> t_table(LUT_size);

    std::vector<double> theta_samples(NUM_THETA_SAMPLES);
    for (int i = 0; i < NUM_THETA_SAMPLES; i++) {
        theta_samples[i] =  M_PI_2*0.83 - M_PI*0.83 * (double)i / (double)(NUM_THETA_SAMPLES - 1);
    }

    for (size_t iter = 0; iter < NUM_THETA_SAMPLES; iter++) {
        StateR s{.vx=v*std::cos(theta_samples[iter]), .vz=v*std::sin(theta_samples[iter]), .z=0., .t=0.};

        for (size_t i = 0; i < length(); i++) {
            size_t j = (size_t)std::floor((s.z - Z_MIN) / INTERP);
            if (0 <= j and j < step()) {
                theta_LUT[pos(i, j)] = theta_samples[iter];
                t_LUT[pos(i, j)] = s.t;
            } else {
                break;
            }
            s = rk4_step_R(s, INTERP, k, G);
        }
    }

    for (size_t i = 0; i < length(); i++) {
        for (size_t j = 1; j < step(); j++) {
            if (t_LUT[pos(i, j)] == 0.) {
                theta_LUT[pos(i, j)] = theta_LUT[pos(i, j-1)];
                t_LUT[pos(i, j)] = t_LUT[pos(i, j-1)];
            }
        }
    }

    has_built_LUT = true;
}

BallisticResult BallisticSolver::query2d(double R, double z) const {
    size_t i = (size_t)std::floor((1.0 / INTERP) * R);
    size_t j = (size_t)std::floor((1.0 / INTERP) * (z - Z_MIN));

    double R0 = (double)i * INTERP;
    double z0 = Z_MIN + (double)j * INTERP;
    double u = (R - R0) / INTERP;
    double v = (z - z0) / INTERP;

    double th00 = theta_LUT[pos(i, j)];
    double th10 = theta_LUT[pos(i+1, j)];
    double th01 = theta_LUT[pos(i, j+1)];
    double th11 = theta_LUT[pos(i+1, j+1)];

    double t00 = t_LUT[pos(i, j)];
    double t10 = t_LUT[pos(i+1, j)];
    double t01 = t_LUT[pos(i, j+1)];
    double t11 = t_LUT[pos(i+1, j+1)];

    return BallisticResult{
        .success=true,
        .pitch=(1 - u) * (1 - v) * th00 + u * (1 - v) * th10 + (1 - u) * v * th01 + u * v * th11,
        .t=(1 - u) * (1 - v) * t00  + u * (1 - v) * t10  + (1 - u) * v * t01  + u * v * t11
    };
}

BallisticResult BallisticSolver::query(double x, double y, double z) const {
    if (not is_built()
        or std::hypot(x, y) - INTERP <= 0 
        or std::hypot(x, y) + INTERP >= R_MAX 
        or z - INTERP <= Z_MIN or z + INTERP >= Z_MAX) {

        return BallisticResult{
            .success=false, 
            .pitch=std::atan2(
                z, 
                std::hypot(x, y)), 
            .t=0.
        };
    }

    return query2d(
        std::hypot(x, y), 
        z
    );
}