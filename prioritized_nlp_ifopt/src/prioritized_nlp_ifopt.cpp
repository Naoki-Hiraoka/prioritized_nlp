#include <prioritized_nlp_ifopt/prioritized_nlp_ifopt.h>

#include <ifopt/problem.h>
#include <ifopt/ipopt_solver.h>
#include <ifopt/constraint_set.h>
#include <ifopt/variable_set.h>
#include <ifopt/cost_term.h>
#include <iostream>

namespace prioritized_nlp_ifopt{
  class MyVariables: public ifopt::VariableSet{
  public:
    MyVariables(int n_var, const Eigen::VectorXd& x) : ifopt::VariableSet(n_var, "var_set1"), x_(x){
      bounds.resize(n_var,ifopt::NoBound);
    }
    void SetVariables(const VectorXd& x) override{
      x_ = x;
    };
    VectorXd GetValues() const override{
      return x_;
    };
    std::vector<ifopt::Bounds> GetBounds() const override{
      return bounds;
    }
  protected:
    Eigen::VectorXd x_;
    std::vector<ifopt::Bounds> bounds;
  };

  class MyConstraint : public ifopt::ConstraintSet {
  public:
    MyConstraint(int n_var, const std::string& name, const std::vector<std::shared_ptr<Task> >& constraints) : ifopt::ConstraintSet(n_var, name), constraints_(constraints) {}

    Eigen::VectorXd GetValues() const override {
      Eigen::VectorXd g(this->GetRows());
      Eigen::VectorXd x = GetVariables()->GetComponent("var_set1")->GetValues();
      int idx = 0;
      for(int i=0;i<this->constraints_.size();i++){
        Eigen::VectorXd value = this->constraints_[i]->getValue(x);
        g.segment(idx,value.rows()) = value;
        idx += value.rows();
      }
      return g;
    };

    VecBound GetBounds() const override {
      std::vector<ifopt::Bounds> b(GetRows());
      int idx = 0;
      for(int i=0;i<this->constraints_.size();i++){
        Eigen::VectorXd lowerBound = this->constraints_[i]->lowerBound;
        Eigen::VectorXd upperBound = this->constraints_[i]->upperBound;
        for(int j=0;j<lowerBound.rows();j++){
          b.at(idx) = ifopt::Bounds(lowerBound[i], upperBound[i]);
          idx++;
        }
      }
      return b;
    }
    void FillJacobianBlock (std::string var_set, Jacobian& jac_block) const override {
      if (var_set == "var_set1") {
        Eigen::VectorXd x = GetVariables()->GetComponent("var_set1")->GetValues();
        int idx = 0;
        for(int i=0;i<this->constraints_.size();i++){
          Eigen::SparseMatrix<double,Eigen::RowMajor> J = this->constraints_[i]->getJacobian(x);
          for (int k=0; k < J.outerSize(); ++k){
            for (Eigen::SparseMatrix<double,Eigen::RowMajor>::InnerIterator it(J,k); it; ++it){
              jac_block.coeffRef(idx + it.row(), it.col()) = it.value();
            }
          }
          idx += J.rows();
        }
      }
    }
  protected:
    std::vector<std::shared_ptr<Task> > constraints_;
  };


  class MyCost: public ifopt::CostTerm {
  public:
    MyCost(const std::string& name, const std::vector<std::shared_ptr<Task> >& costs) : ifopt::CostTerm(name), costs_(costs) {}

    double GetCost() const override {
      Eigen::VectorXd x = GetVariables()->GetComponent("var_set1")->GetValues();
      double cost = 0.0;
      for(int i=0;i<this->costs_.size();i++){
        Eigen::VectorXd value = this->costs_[i]->getValue(x);
        Eigen::VectorXd lowerBound = this->costs_[i]->lowerBound;
        Eigen::VectorXd upperBound = this->costs_[i]->upperBound;
        for(int j=0;j<value.rows();j++){
          if(value[j] > upperBound[j]) cost += 0.5 * std::pow(value[j] - upperBound[j], 2);
          else if(value[j] < lowerBound[j]) cost += 0.5 * std::pow(lowerBound[j] - value[j], 2);
        }
      }
      return cost;
    };

    void FillJacobianBlock (std::string var_set, Jacobian& jac) const override{
      if (var_set == "var_set1") {
        Eigen::VectorXd x = GetVariables()->GetComponent("var_set1")->GetValues();
        int idx = 0;
        for(int i=0;i<this->costs_.size();i++){
          Eigen::VectorXd value = this->costs_[i]->getValue(x);
          Eigen::VectorXd lowerBound = this->costs_[i]->lowerBound;
          Eigen::VectorXd upperBound = this->costs_[i]->upperBound;
          Eigen::SparseMatrix<double,Eigen::RowMajor> error(value.rows(),1);
          for(int j=0;j<value.rows();j++){
            if(value[j] > upperBound[j]) error.insert(j,0) = value[j] - upperBound[j];
            else if(value[j] < lowerBound[j]) error.insert(j,0) = value[j] - lowerBound[j];
            else error.insert(j,0) = 0.0;
          }
          Eigen::SparseMatrix<double,Eigen::RowMajor> J = error.transpose() * this->costs_[i]->getJacobian(x);
          jac += J;
        }
      }
    }
  protected:
    std::vector<std::shared_ptr<Task> > costs_;
  };

  bool solve(const std::vector<std::shared_ptr<Task> >& tasks, int dim, Eigen::VectorXd& solution){
    {
      assert(tasks.size() > 0);
      assert(dim>0);
      assert(solution.rows() == dim);
      Eigen::VectorXd tmpx = Eigen::VectorXd::Zero(dim);
      for(int i=0;i<tasks.size();i++){
        assert(tasks[i]->lowerBound.rows() == tasks[i]->upperBound.rows());
        assert(tasks[i]->lowerBound.rows() == tasks[i]->getValue(tmpx).rows());
        assert(tasks[i]->lowerBound.rows() == tasks[i]->getJacobian(tmpx).rows());
        assert(dim == tasks[i]->getJacobian(tmpx).cols());
      }
    }

    std::vector<Eigen::VectorXd> lowerBoundOrg;
    std::vector<Eigen::VectorXd> upperBoundOrg;
    for(int i=0;i<tasks.size();i++){
      lowerBoundOrg.push_back(tasks[i]->lowerBound);
      upperBoundOrg.push_back(tasks[i]->upperBound);
    }

    Eigen::VectorXd x = solution;
    std::vector<std::shared_ptr<Task> > constraints{tasks[0]};
    double constraints_dim = tasks[0]->lowerBound.rows();

    ifopt::IpoptSolver ipopt;
    ipopt.SetOption("linear_solver", "mumps");
    ipopt.SetOption("jacobian_approximation", "exact");
    ipopt.SetOption("print_level", 0); // supress log. default 5. 0~12
    ipopt.SetOption("nlp_lower_bound_inf", -1e10); // この値より大きいboundはno boundとみなすことで高速化に寄与
    ipopt.SetOption("nlp_upper_bound_inf", 1e10); // この値より大きいboundはno boundとみなすことで高速化に寄与
    ipopt.SetOption("warm_start_init_point", "yes"); // 高速化に寄与

    for(int i=1;i<tasks.size();i++){
      ifopt::Problem nlp;
      nlp.AddVariableSet(std::make_shared<MyVariables>(dim,x));
      nlp.AddConstraintSet(std::make_shared<MyConstraint>(constraints_dim,"constraint0",constraints));
      nlp.AddCostSet(std::make_shared<MyCost>("cost0",std::vector<std::shared_ptr<Task> >{tasks[i]}));
      ipopt.Solve(nlp);
      if(ipopt.GetReturnStatus() != 0){
        return false;
      }
      x = nlp.GetOptVariables()->GetValues();
      constraints_dim += tasks[i]->lowerBound.rows();
      Eigen::VectorXd value = tasks[i]->getValue(x);
      for(int j=0;j<value.rows();j++){
        if(value[j]>tasks[i]->upperBound[j]) tasks[i]->upperBound[j] = value[j];
        else if(value[j]<tasks[i]->lowerBound[j]) tasks[i]->lowerBound[j] = value[j];
      }
      constraints.push_back(tasks[i]);
    }

    for(int i=0;i<tasks.size();i++){
      tasks[i]->lowerBound = lowerBoundOrg[i];
      tasks[i]->upperBound = upperBoundOrg[i];
    }
    solution = x;
    return true;
  }
}
