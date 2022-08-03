#include <prioritized_nlp_ifopt/prioritized_nlp_ifopt.h>
#include <iostream>
#include <sys/time.h>

class Task0 : public prioritized_nlp_ifopt::Task{
public:
  Task0(){
    lowerBound = Eigen::VectorXd::Ones(1) * -1e20;
    upperBound = Eigen::VectorXd::Ones(1) * 5.0;
  }
  Eigen::VectorXd getValue(const Eigen::VectorXd& x) const override{
    Eigen::VectorXd ret(1);
    ret[0] = x[0] * x[0] + x[1] * x[1];
    return ret;
  }
  Eigen::SparseMatrix<double,Eigen::RowMajor> getJacobian(const Eigen::VectorXd& x) const override{
    Eigen::SparseMatrix<double,Eigen::RowMajor> J(1,2);
    J.coeffRef(0,0) = 2 * x[0];
    J.coeffRef(0,1) = 2 * x[1];
    return J;
  }
};

class Task1 : public prioritized_nlp_ifopt::Task{
public:
  Task1(){
    lowerBound = Eigen::VectorXd::Ones(1) * -1e20;
    upperBound = Eigen::VectorXd::Ones(1) * 0.0;
  }
  Eigen::VectorXd getValue(const Eigen::VectorXd& x) const override{
    Eigen::VectorXd ret(1);
    ret[0] = x[0] * x[0] + 0 - x[1];
    return ret;
  }
  Eigen::SparseMatrix<double,Eigen::RowMajor> getJacobian(const Eigen::VectorXd& x) const override{
    Eigen::SparseMatrix<double,Eigen::RowMajor> J(1,2);
    J.coeffRef(0,0) = 2 * x[0];
    J.coeffRef(0,1) = -1;
    return J;
  }
};

class Task2 : public prioritized_nlp_ifopt::Task{
public:
  Task2(){
    lowerBound = Eigen::VectorXd(2);
    lowerBound << 1.0, 5.0;
    upperBound = Eigen::VectorXd(2);
    upperBound << 1.0, 5.0;
  }
  Eigen::VectorXd getValue(const Eigen::VectorXd& x) const override{
    return x;
  }
  Eigen::SparseMatrix<double,Eigen::RowMajor> getJacobian(const Eigen::VectorXd& x) const override{
    Eigen::SparseMatrix<double,Eigen::RowMajor> J(2,2);
    J.coeffRef(0,0) = 1.0;
    J.coeffRef(1,1) = 1.0;
    return J;
  }
};


int main(void){
  std::vector<std::shared_ptr<prioritized_nlp_ifopt::Task> > tasks;
  tasks.push_back(std::make_shared<Task0>());
  tasks.push_back(std::make_shared<Task1>());
  tasks.push_back(std::make_shared<Task2>());

  Eigen::VectorXd solution = Eigen::VectorXd::Zero(2);
  for(int i=0;i<1;i++){
    struct timeval t0; gettimeofday(&t0, NULL);
    prioritized_nlp_ifopt::solve(tasks, 2, solution);
    struct timeval t1; gettimeofday(&t1, NULL);
    std::cerr << solution.transpose() << std::endl;
    std::cerr << (t1.tv_sec - t0.tv_sec) + (t1.tv_usec - t0.tv_usec) * 1e-6 << std::endl;
  }


  return 0;
}
