#ifndef _OUTPOST_ESTIMATOR_H
#define _OUTPOST_ESTIMATOR_H

#include <Eigen/Dense>
#include <vector>
#include <deque>

#include "outpostaim/filter.h"
#include "outpostaim/datatypes.h"

// OutpostArmorEkf类
class OutpostArmorEkf : public Ekf<6, 3>{ // pitch yaw dis vp vy vd
public:
    OutpostArmorEkf() : Ekf(){ H(0, 0) = H(1, 1) = H(2, 2) = 1; }
    OutpostArmorEkf(Vz _init_Z, Mxx _P = Mxx::Identity(), Mzz _R = Mzz::Identity(), Mxx _Q = Mxx::Identity())
                    :Ekf<6, 3>(_P, _R, _Q){
        H(0, 0) = H(1, 1) = H(2, 2) = 1;
        Xe << _init_Z(0), _init_Z(1), _init_Z(2), 0, 0, 0;
    }

    Vz predict_position(double _timestamp) const{
        Vz res;
        double dt = _timestamp - last_timestamp_;
        res[0] = Xe[0] + Xe[3] * dt;
        res[1] = Xe[1] + Xe[4] * dt;
        res[2] = Xe[2] + Xe[5] * dt;
        return res;
    }
    
    Vx predict(double _dt) override{
        F(0, 3) = F(1, 4) = F(2, 5) = _dt;
        return F * Xe;
    }

    Vz h(Vx _X) override{
        Vz z_ans = H * _X;
        return z_ans;
    }

    void init(Eigen::Vector3d pyd, double _timestamp){
        Xe << pyd[0], pyd[1], pyd[2], 0, 0, 0;
        P = Mxx::Identity();
        Q = config_.vec_Q.asDiagonal();
        R = config_.vec_R.asDiagonal();
        last_timestamp_ = _timestamp;
    }

    struct ArmorEkfConfig{
        Vx vec_Q;
        Vz vec_R;
    };

    // 外部加载静态成员参数
    inline static ArmorEkfConfig config_;
    // Eigen::Vector3d pyd; 
};

// OutpostYawEkf类
class OutpostYawEkf : public Ekf<2, 1>{
public:
    OutpostYawEkf() : Ekf(){ H(0, 0) = 1; }
    OutpostYawEkf(Vz _init_Z, double _sigma2_Q, Mxx _P = Mxx::Identity(), Mzz _R = Mzz::Identity())
                    :Ekf<2, 1>(_P, _R){
        H(0, 0) = 1;
        Xe << _init_Z(0), 0;
        sigma2_Q_ = _sigma2_Q;
    }

    Vx predict(double _dt) override{
        F(0,1) = _dt;
        return F * Xe;
    }

    Vz h(Vx _X) override{
        return H * _X;
    }

    void calcQ(double _dt) override{
        static double dTs[4];
        dTs[0] = _dt;
        for (int i = 1; i < 4; ++i) dTs[i] = dTs[i - 1] * _dt;
        double q_x_x = dTs[3] / 4 * sigma2_Q_, 
               q_x_vx = dTs[2] / 2 * sigma2_Q_, 
               q_vx_vx = dTs[1] * sigma2_Q_;

        Q << q_x_x, q_x_vx, q_x_vx, q_vx_vx;
    }

    // void update(Vz _Z, double _timestamp) override{
    //     // NGXY_DEBUG("YawEkf zzz: %f", _Z[0]);
    //     // NGXY_DEBUG("YawEkf update, yaw: %f", Xe[0]);
    //     Ekf::update(_Z, _timestamp);
    //     // NGXY_DEBUG("YawEkf update, yaw: %f", Xe[0]);
    // }

    void init( Eigen::Matrix<double, 1, 1> _yaw, double _timestamp){
        Xe << _yaw[0], 0;
        P = Mxx::Identity();
        sigma2_Q_ = config_.sigma2_Q;
        R = config_.vec_R.asDiagonal();
        last_timestamp_ = _timestamp;
    }

    struct YawEkfConfig{
        double sigma2_Q;
        Vz vec_R;
    };

    // 外部加载静态成员参数
    inline static YawEkfConfig config_;

private:
    double sigma2_Q_;// 加速度过程噪声方差
};

// OutpostCkf类
const int state_num = 6;
const int measure_num = 4;  // z坐标的处理有点不标准，但是看起来不影响运行
class OutpostCkf{
public:
    // n代表状态维数，m代表输出维数
    using Vx = Eigen::Vector<double, state_num>;
    using Vz = Eigen::Vector<double, measure_num>;
    using Mxx = Eigen::Matrix<double, state_num, state_num>;
    using Mzz = Eigen::Matrix<double, measure_num, measure_num>;
    using Mzx = Eigen::Matrix<double, measure_num, state_num>;
    using Mxz = Eigen::Matrix<double, state_num, measure_num>;

    struct State {
        // x, y, z 中心
        double x, vx, y, vy;
        // const_z均值
        double z;
        // yaw对应id为0的装甲板
        double yaw, omega;

        State(){};
        State(double _x, double _vx, double _y, double _vy, double _z, double _yaw, double _omega) {
            x = _x;
            y = _y;
            z = _z;
            yaw = _yaw;
            vx = _vx;
            vy = _vy;
            omega = _omega;
        }
        State(const Eigen::Ref<const Vx> &_x_vec){
            fromVx(_x_vec);
        }

        Vx toVx() const{
            return Vx(x, vx, y, vy, yaw, omega);
        }

        void fromVx(const Eigen::Ref<const Vx> &_x_vec){
            x = _x_vec[0];
            vx = _x_vec[1];
            y = _x_vec[2];
            vy = _x_vec[3];
            yaw = _x_vec[4];
            omega = _x_vec[5];
        }
    };

    struct Observe {
        // x,y,z 观测装甲板
        double x, y, z;
        // yaw对应id为0的装甲板
        double yaw;
        // int phase_id;
        Observe() {}
        Observe(double _x, double _y, double _z, double _yaw) {
            x = _x;
            y = _y;
            z = _z;
            yaw = _yaw;
        }
        Observe(const Eigen::Ref<const Vz> &_z_vec){
            fromVz(_z_vec);
        }

        Vz toVz() const{
            return Vz(x, y, z, yaw);
        }

        void fromVz(const Eigen::Ref<const Vz> &_z_vec){
            x = _z_vec[0];
            y = _z_vec[1];
            z = _z_vec[2];
            yaw = _z_vec[3];
        }
    };

    inline State getState() const{
        return state_;
    }

    explicit OutpostCkf();

    void reset(const Observe &_observe, int phase_id, int _armor_cnt, double _timestamp, double _z);

    void CKF_update(Observe _observe, double stamp, int phase_id);

    State predictState(double _timestamp);

    static Eigen::Vector3d get_center(State state_);

    Eigen::Vector3d get_center();

    struct OutpostCkfConfig{
        Vx init_P;
        double R_XYZ, R_YAW;
        double Q2_XY, Q2_YAW;
    };
    // 外部加载静态成员参数
    inline static OutpostCkfConfig config_;


    Vx f(const Eigen::Ref<const Vx> &_x, double _dt) const; // 预测函数


    Vz h(const Eigen::Ref<const Vx> &_x, int phase_id, double armor_height) const;

    void SRCR_sampling_3(const Eigen::Ref<const Vx> &_x, const Mxx &_P);  // 3阶球面——径向采样法

    void calcQ(double _dt);

    void calcR(Vz _z);

    void CKF_predict(double _dt);

    void CKF_measure(Vz _z, int phase_id, double armor_height);

    void CKF_correct(Vz _z);

    int sample_num_;
    double const_z_;
    inline static double const_dis_;
    std::vector<Vx> samples_;      // 样本数组
    std::vector<double> weights_;  // 权重数组
    State state_;
    Vx Xe;  // 状态量 x vx y vy yaw omega
    // 自适应参数
    Vx Xp;
    Mxx Pp;
    std::vector<Vx> sample_X;  // 预测
    Mxx Pe;
    Mxx Q;
    Mzz R;
    std::vector<Vz> sample_Z;
    Vz Zp;
    Mzz Pzz;
    Mxz Pxz;
    Mxz K;
    double last_timestamp_;
    double armor_cnt_ = 3.; // 前哨站默认3装甲版
    double angle_dis_;
    
};

// step_fitter类 装甲板过中时间拟合
class step_fitter {
private:
    double sumx;
    double sumy;
    double sumxy;
    double sumx2;
    size_t max_length;
    size_t min_length;

public:
    size_t N;
    std::vector<double> datas;
    double k;
    double b;
    double last_middle_time;

    explicit step_fitter(size_t _max_length = 10000, size_t _min_length = 3): 
        max_length(_max_length), min_length(_min_length) { reset(); }
    
    void reset() {
        N = 0;
        k = 0;
        b = 0;
        sumx = 0;
        sumy = 0;
        sumxy = 0;
        sumx2 = 0;
        datas.clear();
    }

    void update(double y) {
        last_middle_time = y;
        ++N;
        datas.push_back(y);
        double n_double = static_cast<double>(N);
        sumx += n_double;
        sumy += y;
        sumx2 += n_double * n_double;
        sumxy += n_double * y;
        if (N > max_length) {
            pop_front();
        } else if (N >= min_length) {
            solve();
        }
    }

    void pop_front() {
        if (datas.empty()) return;
        // 简单方法：删除数据后重新计算所有统计量
        datas.erase(datas.begin());
        --N;
        
        // 重新计算统计量
        recalculate();
        
        // 重新计算拟合参数
        if (N >= min_length) {
            solve();
        } else {
            k = 0;
            b = 0;
        }
    }

    // 重新计算统计量
    void recalculate() {
        sumx = 0;
        sumy = 0;
        sumxy = 0;
        sumx2 = 0;
        
        for (size_t i = 0; i < N; ++i) {
            double x_val = static_cast<double>(i + 1);  // x从1开始
            double y_val = datas[i];
            sumx += x_val;
            sumy += y_val;
            sumx2 += x_val * x_val;
            sumxy += x_val * y_val;
        }
    }

    inline double fit(double x) { return k * x + b; }
    
    void solve() {
        if (N < min_length) {
            k = 0;
            b = 0;
            return;
        }
        double denominator = sumx2 - sumx * sumx / N;
        if (fabs(denominator) < 1e-10) {
            k = 0;
            b = sumy / N;
            return;
        }
        k = (sumxy - sumx * sumy / N) / denominator;
        b = sumy / N - k * sumx / N;
    }
    
    // 预测第n个周期的过中时间
    double predict_nth_middle_time(int n) {
        if (N < min_length) return -1;
        return fit(n);
    }
    
    // 预测下一次过中时间
    double predict_next_middle_time() {
        if (N < min_length) return -1;
        return fit(N + 1);
    }
    // 获取上一次过中时间
    double get_last_middle_time(){
        return last_middle_time;
    }
};

#endif // _OUTPOST_ESTIMATOR_H