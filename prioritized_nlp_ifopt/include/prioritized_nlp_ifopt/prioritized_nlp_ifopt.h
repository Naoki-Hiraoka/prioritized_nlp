#ifndef PRIORITIZED_NLP_IFOPT_H
#define PRIORITIZED_NLP_IFOPT_H

#include <Eigen/Eigen>
#include <memory>

namespace prioritized_nlp_ifopt{
  class Task{
  public:
    Eigen::VectorXd lowerBound;
    Eigen::VectorXd upperBound;
    virtual Eigen::VectorXd getValue(const Eigen::VectorXd& x) const = 0;
    virtual Eigen::SparseMatrix<double,Eigen::RowMajor> getJacobian(const Eigen::VectorXd& x) const = 0;
  };

  // 0番目のtaskは解かない
  bool solve(const std::vector<std::shared_ptr<Task> >& tasks, int dim, Eigen::VectorXd& solution);
};

#endif
