/* ----------------------------------------------------------------------------
 * Copyright 2018, Ross Hartley <m.ross.hartley@gmail.com>
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 *  @file   kinematics.cpp
 *  @author Ross Hartley
 *  @brief  Example of invariant filtering for contact-aided inertial navigation
 *  @date   September 25, 2018
 **/


#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <Eigen/Dense>
#include <boost/algorithm/string.hpp>
#include <vector>
#include "InEKF.h"

#define DT_MIN 1e-6
#define DT_MAX 1

using namespace std;
using namespace inekf;

double stod98(const std::string &s) {
    return atof(s.c_str());
}

int stoi98(const std::string &s) {
    return atoi(s.c_str());
}

int main() {
    //  ---- Initialize invariant extended Kalman filter ----- //
    RobotState initial_state; 

    // Initialize state mean
    Eigen::Matrix3d R0;
    Eigen::Vector3d v0, p0, bg0, ba0;
    R0 << 1, 0, 0, // initial orientation
          0, 1, 0, // IMU frame is rotated 90deg about the x-axis
          0, 0, 1;
    v0 << 0,0,0; // initial velocity
    p0 << 0,0,0; // initial position
    bg0 << 0,0,0; // initial gyroscope bias
    ba0 << 0,0,0; // initial accelerometer bias
    initial_state.setRotation(R0);
    initial_state.setVelocity(v0);
    initial_state.setPosition(p0);
    initial_state.setGyroscopeBias(bg0);
    initial_state.setAccelerometerBias(ba0);

    // Initialize state covariance
    NoiseParams noise_params;
    noise_params.setGyroscopeNoise(0.00);
    noise_params.setAccelerometerNoise(0.0);
    noise_params.setGyroscopeBiasNoise(0.00000);
    noise_params.setAccelerometerBiasNoise(0.0000);
    noise_params.setContactNoise(0.00);

    // Initialize filter
    InEKF filter(initial_state, noise_params);
    cout << "Noise parameters are initialized to: \n";
    cout << filter.getNoiseParams() << endl;
    cout << "Robot's state is initialized to: \n";
    cout << filter.getState() << endl;

    // Open data file
    ifstream infile("../src/data/raisim_test_data.txt"); //imu_kinematic_measurements
    ifstream truefile("../src/data/raisim_true_test_data.txt");
    string line;
    Eigen::Matrix<double,6,1> imu_measurement = Eigen::Matrix<double,6,1>::Zero();
    Eigen::Matrix<double,6,1> imu_measurement_prev = Eigen::Matrix<double,6,1>::Zero();
    Eigen::Matrix<double,5,5> true_X = Eigen::Matrix<double,5,5>::Zero();
    double t = 0;
    double t_prev = 0;
    double dt = 0;
    double v_x = 0;
    double v_x_prev = 0;
    double p_x = 0;
    double u_vx = 0;

    // int NUM_LINES = 638;
    // int count = 0;

    // GNU Plot
    FILE *gnuplotPipe1 = popen("gnuplot -persistent", "w");
    FILE *gnuplotPipe2 = popen("gnuplot -persistent", "w");
    FILE *gnuplotPipe3 = popen("gnuplot -persistent", "w");

    if (!gnuplotPipe1) {
        std::cerr << "Gnuplot could not be opened." << std::endl;
        return -1;
    }

    // ---- Loop through data file and read in measurements line by line ---- //
    while (getline(infile, line)){
        vector<string> measurement;
        boost::split(measurement,line,boost::is_any_of(" "));
        // // Handle measurements
        if (measurement[0].compare("IMU")==0){
//            cout << "Received IMU Data, propagating state\n";
            assert((measurement.size()-2) == 6);
            t = stod98(measurement[1]); 
            // Read in IMU data
            imu_measurement << stod98(measurement[2]),
                               stod98(measurement[3]),
                               stod98(measurement[4]),
                               stod98(measurement[5]),
                               stod98(measurement[6]),
                               stod98(measurement[7]);

            // Propagate using IMU data
            dt = t - t_prev;
            // if (dt > DT_MIN && dt < DT_MAX) {
                filter.Propagate(imu_measurement_prev, dt);
            // }

            // Store previous timestamp
            t_prev = t;
            imu_measurement_prev = imu_measurement;
        }
        else if (measurement[0].compare("CONTACT")==0){
//            cout << "Received CONTACT Data, setting filter's contact state\n";
            assert((measurement.size()-2)%2 == 0);
            vector<pair<int,bool> > contacts;
            int id;
            bool indicator;
            // t = stod98(measurement[1]); 
            // Read in contact data
            for (int i=2; i<measurement.size(); i+=2) {
                id = stoi98(measurement[i]);
                indicator = bool(stod98(measurement[i+1]));
                contacts.push_back(pair<int,bool> (id, indicator));
            }       
            // Set filter's contact state
            filter.setContacts(contacts);
        }
        else if (measurement[0].compare("KINEMATIC")==0){
//            cout << "Received KINEMATIC observation, correcting state\n";
            assert((measurement.size()-2)%44 == 0);
            int id;
            Eigen::Quaternion<double> q;
            Eigen::Vector3d p;
            Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
            Eigen::Matrix<double,6,6> covariance;
            vectorKinematics measured_kinematics;
            // t = stod98(measurement[1]);
            // std::cout << "Kinematic size... " << measurement.size() - 2 << std::endl; // 88
            // Read in kinematic data
            for (int i=2; i<measurement.size(); i+=44) {
                id = stoi98(measurement[i]); 
                q = Eigen::Quaternion<double> (stod98(measurement[i+1]),stod98(measurement[i+2]),stod98(measurement[i+3]),stod98(measurement[i+4]));
                q.normalize();
                p << stod98(measurement[i+5]),stod98(measurement[i+6]),stod98(measurement[i+7]);
                pose.block<3,3>(0,0) = q.toRotationMatrix();
                pose.block<3,1>(0,3) = p;
                for (int j=0; j<6; ++j) {
                    for (int k=0; k<6; ++k) {
                        covariance(j,k) = stod98(measurement[i+8 + j*6+k]); //이게 뭐지?
                    }
                }
                Kinematics frame(id, pose, covariance);
                measured_kinematics.push_back(frame);
            }
            // Correct state using kinematic measurements
            filter.CorrectKinematics(measured_kinematics);
        }
        else if (measurement[0].compare("TRUE_X")==0){
            assert((measurement.size()-1)%25 == 0);

            // Plotting x
            v_x = abs(stod98(measurement[3])); //filter.getState().getVelocity()[0]
            p_x = filter.getState().getPosition()[0];
            Eigen::MatrixXd P = filter.getState().getP();
            u_vx = P(0,0);
//            // Plotting y
//            v_x = abs(stod98(measurement[3+5])); //filter.getState().getVelocity()[1]
//            p_x = filter.getState().getPosition()[1];
//            Eigen::MatrixXd P = filter.getState().getP();
//            u_vx = P(1,1);
//            // Plotting z
//            v_x = abs(stod98(measurement[3+10])); //filter.getState().getVelocity()[2]
//            p_x = filter.getState().getPosition()[2];
//            Eigen::MatrixXd P = filter.getState().getP();
//            u_vx = P(2,2);

            // 첫 번째 그래프 그리기 (v_x)
//            fprintf(gnuplotPipe1, "set yrange [0:10]\n");
            fprintf(gnuplotPipe1, "set grid\n");
            fprintf(gnuplotPipe1, "plot '-' using 1:2 with lines title 'a_x'linecolor rgb 'blue'\n");
            fprintf(gnuplotPipe1, "%lf %lf\n", t, abs(v_x - v_x_prev)/dt);

            // 두 번째 그래프 그리기 (p_x)
            fprintf(gnuplotPipe2, "set grid\n");
            fprintf(gnuplotPipe2, "plot '-' using 1:2 with lines title 'p_x'\n");
            fprintf(gnuplotPipe2, "%lf %lf\n", t, p_x);

            // 세 번째 그래프 그리기 (u_vx)
            fprintf(gnuplotPipe3, "set grid\n");
            fprintf(gnuplotPipe3, "set yrange [0:0.1]\n");
            fprintf(gnuplotPipe3, "plot '-' using 1:2 with lines title 'u_vx'\n");
            fprintf(gnuplotPipe3, "%lf %lf\n", t, u_vx);
        }

        // count++;
        // if (count > NUM_LINES)
        //     break;

//        cout << "t:\n " << t << endl;
//        cout << filter.getState() << endl;
//        cout << "Covariance: \n" << filter.getState().getP() << endl;
//        sleep(1);



    }

    // Print final state
    cout.precision(17);
    cout << "imu:\n " << imu_measurement << endl;
    cout << "t:\n " << t << endl;

    cout << filter.getState() << endl;
    cout << "Covariance: \n" << filter.getState().getP() << endl;

    fprintf(gnuplotPipe1, "e\n"); // 데이터 입력 종료
    fflush(gnuplotPipe1); // 버퍼 비우기
    pclose(gnuplotPipe1); // 파이프 닫기
    fprintf(gnuplotPipe2, "e\n"); // 데이터 입력 종료
    fflush(gnuplotPipe2); // 버퍼 비우기
    pclose(gnuplotPipe2); // 파이프 닫기
    fprintf(gnuplotPipe3, "e\n"); // 데이터 입력 종료
    fflush(gnuplotPipe3); // 버퍼 비우기
    pclose(gnuplotPipe3); // 파이프 닫기
    return 0;
}
