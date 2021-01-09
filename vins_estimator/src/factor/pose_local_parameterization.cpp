#include "pose_local_parameterization.h"

bool PoseLocalParameterization::Plus(const double *x, const double *delta, double *x_plus_delta) const
{
    Eigen::Map<const Eigen::Vector3d> _p(x);
    Eigen::Map<const Eigen::Quaterniond> _q(x + 3);

    Eigen::Map<const Eigen::Vector3d> dp(delta);

    Eigen::Quaterniond dq = Utility::deltaQ(Eigen::Map<const Eigen::Vector3d>(delta + 3));

    Eigen::Map<Eigen::Vector3d> p(x_plus_delta);
    Eigen::Map<Eigen::Quaterniond> q(x_plus_delta + 3);

    p = _p + dp;
    q = (_q * dq).normalized();

    return true;
}
/*这里将manifold空间到tangent空间的J设置为[I66 O16].transpose()
    ceres关于过参数化的处理机制，当过参数时，为了处理效率的原因，优化的过程是在切空间上进行，实现一个参数关于切空间的雅克比矩阵。
    旋转用四元数，再加上平移向量维度为7，但是旋转的切空间是3维，所以优化的维度是6维。则需要定义一个7*6的雅克比矩阵。在优化的过程中，CostFunction中
J会和这里的J相乘，得到残差关于切空间的J矩阵。
    这里的一个trick是在CostFunction中可以直接计算残差对于切空间上的雅克比，如计算重投影误差对so(3)+t上的雅克比，正常在CostFunction中的雅克比是2*6，
那么再添加一个全为0的列，与此雅克比相乘即可。这样就可以保证该雅克比可以得到四元数的下降方向

**/
bool PoseLocalParameterization::ComputeJacobian(const double *x, double *jacobian) const
{
    Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> j(jacobian);
    j.topRows<6>().setIdentity();
    j.bottomRows<1>().setZero();

    return true;
}
