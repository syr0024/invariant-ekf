//
// Created by syr on 8/17/23.
//

#include <Eigen/Dense>
#include <iostream>
#include <unistd.h>
#include <random>
Eigen::Vector3d eInt_;
Eigen::Matrix3d rotation_;
const double imu_Kp = 0.3;
const double imu_Ki = 0;
const double dt_ = 0.001; // 0.0005s
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<float> dis(-0.1, 0.1);

namespace Operation{
Eigen::Vector4d quatProduct(const Eigen::Vector4d &a,
                                       const Eigen::Vector4d &b) {
    Eigen::Vector4d ab;
    ab(0) = a(0)*b(0) - a(1)*b(1) - a(2)*b(2) - a(3)*b(3);
    ab(1) = a(0)*b(1) + a(1)*b(0) + a(2)*b(3) - a(3)*b(2);
    ab(2) = a(0)*b(2) - a(1)*b(3) + a(2)*b(0) + a(3)*b(1);
    ab(3) = a(0)*b(3) + a(1)*b(2) - a(2)*b(1) + a(3)*b(0);

    return ab;
}


    void quatToRotation(const Eigen::Vector4d& q, Eigen::Matrix3d& R) {
        R(0) = q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3];
        R(1) = 2 * q[0] * q[3] + 2 * q[1] * q[2];
        R(2) = 2 * q[1] * q[3] - 2 * q[0] * q[2];

        R(3) = 2 * q[1] * q[2] - 2 * q[0] * q[3];
        R(4) = q[0] * q[0] - q[1] * q[1] + q[2] * q[2] - q[3] * q[3];
        R(5) = 2 * q[0] * q[1] + 2 * q[2] * q[3];

        R(6) = 2 * q[0] * q[2] + 2 * q[1] * q[3];
        R(7) = 2 * q[2] * q[3] - 2 * q[0] * q[1];
        R(8) = q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3];
    }
}

inline void complementaryFilter(Eigen::Matrix3d & rot_mat_,const Eigen::Vector4d & quat_, const Eigen::Vector3d &gyro_,
                                    const Eigen::Vector3d &acc_) {
    if(acc_.norm() < 1e-12)
        return;

    Eigen::Vector4d quat_temp;
    quat_temp.setZero();
    quat_temp = quat_;
    // Estimated direction of gravity;
    Eigen::Vector3d v;
    v.setZero();
    v << 2 * (quat_[1]*quat_[3]-quat_[0]*quat_[2]),
            2 * (quat_[0]*quat_[1]+quat_[2]*quat_[3]),
            quat_[0]*quat_[0]-quat_[1]*quat_[1]-quat_[2]*quat_[2]+quat_[3]*quat_[3];

    Eigen::Vector3d e = acc_.normalized().cross(v);
    eInt_ += e * dt_;
    Eigen::Vector3d gyro_modified = gyro_ + imu_Kp * e + imu_Ki * eInt_;

    Eigen::Vector4d q_dot =
            0.5 * Operation::quatProduct(quat_temp, Eigen::Vector4d(0, gyro_modified(0), gyro_modified(1), gyro_modified(2)));

    quat_temp += q_dot * dt_;
    quat_temp = quat_temp.normalized();

    Operation::quatToRotation(quat_temp,rot_mat_);
}

int main(){

    eInt_.setZero();
    Eigen::Vector4d quat_;
    Eigen::Matrix3d rot_mat_;
    Eigen::Vector3d gyro_;
    Eigen::Vector3d acc_;

    quat_ << 1.0, 0.0, 0.0, 0.0;

for(int i = 0; i < 1000; i++){
    for (int i = 0; i <3; ++i) {
        gyro_(i) = dis(gen);
        acc_(i)  =  dis(gen);
    }

    acc_(2) = -9.81;
    complementaryFilter(rot_mat_,quat_, gyro_, acc_);
    Eigen::Vector4d coeffs_wxyz;
    Eigen::Quaterniond quaternion(rot_mat_);
    quat_ << quaternion.w(), quaternion.x(), quaternion.y(), quaternion.z();


    std::cout << " ----------------------- "  << std::endl;
    std::cout << " count indexs is ..." << i  << std::endl;
    std::cout << " gyro_... " << gyro_.transpose() << std::endl;
    std::cout << " acc_... " << acc_.transpose() << std::endl;
    std::cout << " quat_... " << quat_ << std::endl;
    std::cout << " rot_mat_... " << rot_mat_ << std::endl;

    usleep(10);
}


}