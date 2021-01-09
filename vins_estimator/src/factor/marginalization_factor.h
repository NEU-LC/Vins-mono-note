#pragma once

#include <ros/ros.h>
#include <ros/console.h>
#include <cstdlib>
#include <pthread.h>
#include <ceres/ceres.h>
#include <unordered_map>

#include "../utility/utility.h"
#include "../utility/tic_toc.h"

const int NUM_THREADS = 4;

struct ResidualBlockInfo
{
    ResidualBlockInfo(ceres::CostFunction *_cost_function, ceres::LossFunction *_loss_function, std::vector<double *> _parameter_blocks, std::vector<int> _drop_set)
        : cost_function(_cost_function), loss_function(_loss_function), parameter_blocks(_parameter_blocks), drop_set(_drop_set) {}

    void Evaluate();

    ceres::CostFunction *cost_function; //其中parameter_block_sizes每个优化变量块的变量大小，以IMU残差为例，为[7,9,7,9]
    ceres::LossFunction *loss_function;
    std::vector<double *> parameter_blocks; // 待marg变量,相当于marg掉的顶点
    std::vector<int> drop_set;

    double **raw_jacobians;
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians;
    Eigen::VectorXd residuals;// 残差，IMU 15x1；视觉2x1

    int localSize(int size)
    {
        return size == 7 ? 6 : size;
    }
};

struct ThreadsStruct
{
    std::vector<ResidualBlockInfo *> sub_factors;
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    std::unordered_map<long, int> parameter_block_size; //global size
    std::unordered_map<long, int> parameter_block_idx; //local size
};

// 找出要marg的变量和要保留的变量
// 找到在marg前的H = J^T * thegma * J 和 b = - J^T * thegma * r, 从H矩阵中找出要marg掉和要保留的矩阵块， 然后构建新的 H_p' delta_x = b_p'
// 新的H_p'和 b_p'就是关于marg掉变量的新的先验信息
// 先验残差：我们可以从b_p和H_P中反解出一个一个残差r_p和对应的J_p矩阵，在后续的优化过程中残差r_p或者b_p可能会改变，但是要保证雅克比不能变
class MarginalizationInfo
{
  public:
    ~MarginalizationInfo();
    int localSize(int size) const;
    int globalSize(int size) const;
    //计算每个残差对应的Jacobian，并更新parameter_block_data
    void addResidualBlockInfo(ResidualBlockInfo *residual_block_info);
    void preMarginalize();
    void marginalize();
    std::vector<double *> getParameterBlocks(std::unordered_map<long, double *> &addr_shift);

    std::vector<ResidualBlockInfo *> factors; // 所有观测项
    // m是要marg掉的变量个数，n是要保留下来的变量个数
    int m, n;
    std::unordered_map<long, int> parameter_block_size; //global size
    int sum_block_size;
    std::unordered_map<long, int> parameter_block_idx; //local size
    std::unordered_map<long, double *> parameter_block_data;

    std::vector<int> keep_block_size; //global size
    std::vector<int> keep_block_idx;  //local size
    std::vector<double *> keep_block_data; // 线性化点处的参数值，也就是FEJ

    Eigen::MatrixXd linearized_jacobians;
    Eigen::VectorXd linearized_residuals;
    const double eps = 1e-8;

};
// Marg CostFunction
class MarginalizationFactor : public ceres::CostFunction
{
  public:
    MarginalizationFactor(MarginalizationInfo* _marginalization_info);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

    MarginalizationInfo* marginalization_info;
};
