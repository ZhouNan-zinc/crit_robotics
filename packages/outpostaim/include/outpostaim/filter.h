#ifndef _FILTER_H
#define _FILTER_H

#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <random>


#define AMeps 1e-2

enum MeanFilterMethod{ 
    ArithmeticMean = 0, 
    GeometricMean = 1, 
    HarmonicMean = 2 
};

class MathFilter {
private:
    double sum;
    std::vector<double> data;
    size_t max_length;
    MeanFilterMethod method;

public:
    
    MathFilter(size_t _max_length = 100, MeanFilterMethod _method = ArithmeticMean)
        : sum(0.), data(), max_length(_max_length), method(_method) {}
    MathFilter(const MathFilter &) = default;

    void set(double _max_length, MeanFilterMethod _method) {
        data.clear();
        max_length = _max_length;
        method = _method;
    }
    
    void reset(){
        data.clear();
        sum = 0.0;
    }

    double get() {
        if (data.empty()) {
            return 0.;
        }
        switch (method) {
            case GeometricMean:
                return std::pow(sum, 1. / data.size());
                break;
            case HarmonicMean:
                return data.size() / sum;
                break;
            case ArithmeticMean:
            default:
                return sum / data.size();
                break;
        }
    }
    void update(const double &item) {
        if (data.size() == max_length) {
            switch (method) {
                case GeometricMean:
                    sum /= data.front();
                    break;
                case HarmonicMean:
                    sum -= 1. / data.front();
                    break;
                case ArithmeticMean:
                default:
                    sum -= data.front();
                    break;
            }
            data.erase(data.begin());
        }
        switch (method) {
            case GeometricMean:
                sum *= item;
                break;
            case HarmonicMean:
                sum += 1. / item;
                break;
            case ArithmeticMean:
            default:
                sum += item;
                break;
        }
        data.push_back(item);
    }
};

template <int NX, int NZ>
class Ekf{
public:
    using Mxx = Eigen::Matrix<double, NX, NX>;
    using Mzx = Eigen::Matrix<double, NZ, NX>;
    using Mxz = Eigen::Matrix<double, NX, NZ>;
    using Mzz = Eigen::Matrix<double, NZ, NZ>;
    using Vx = Eigen::Matrix<double, NX, 1>;
    using Vz = Eigen::Matrix<double, NZ, 1>;
    
    explicit Ekf(Mxx _P = Mxx::Identity(), Mzz _R = Mzz::Identity(), Mxx _Q = Mxx::Identity()):
                                P(_P), R(_R), Q(_Q), H(Mzx::Zero()), F(Mxx::Identity()), last_timestamp_(0.0){}

    // 同时计算F矩阵
    virtual Vx predict(double _dt) = 0;

    virtual Vz h(Vx _X) = 0;
    
    // 预留根据时间自适应计算Q的功能
    virtual void calcQ(double _dt){
        return;
    }

    virtual void update(Vz _Z, double _timestamp){
        Xp = predict(_timestamp - last_timestamp_);
        Zp = h(Xp);
        calcQ(_timestamp - last_timestamp_);
        P = F * P * F.transpose() + Q;
        K = P * H.transpose() * (H * P * H.transpose() + R).inverse();
        Xe = Xp + K * (_Z - Zp);
        P = (Mxx::Identity() - K * H) * P;
        last_timestamp_ = _timestamp;
    }

    inline Vx getX(){
        return Xe;
    }

    inline double get_time_stamp(){
        return last_timestamp_;
    }

    void setQR(Mxx _Q, Mzz _R){
        Q = _Q;
        R = _R;
    }

protected:
    Vx Xe;  // 后验估计状态变量
    Vx Xp;  // 先验估计状态变量
    Mxx F;  // 预测雅克比
    Mzx H;  // 观测雅克比
    Mxx P;  // 状态协方差
    Mxx Q;  // 预测过程协方差
    Mzz R;  // 观测过程协方差
    Mxz K;  // 卡尔曼增益
    Vz Zp;  // 预测观测量
    double last_timestamp_;
};

template<int NX, int NZ>
class AdaptiveEkf : public Ekf<NX, NZ> {
public:

    AdaptiveEkf()=default;



};


#endif // _FILTER_H