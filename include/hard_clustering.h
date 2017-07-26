#ifndef HARD_CLUSTERING_H
#define HARD_CLUSTERING_H
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/tmpdir.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/assume_abstract.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/list.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/vector.hpp>
#include "utils.h"

using namespace std;
namespace bserialize = boost::serialization;

class AdaHardCluster {

 public:
  AdaHardCluster(const vector<vector<double>>& data_,
             const vector<unsigned long int>& label_,
             unsigned long int max_round_, unsigned long int k_);
  void initialize_random();
  void initialize_k_plus_plus();
  void fit();
  unsigned long int get_k();
  unsigned long int get_max_round();
  unsigned long int get_n_dims();
  unsigned long int get_n_samples();
  vector<double>& get_kappa();
  vector<double>& get_logliks();
  vector<double>& get_nmis();
  copt::Vector<double>& get_alpha();
  copt::Vector<double>& get_lb();
  copt::Vector<double>& get_ub();
  const vector<vector<double>>& get_data();
  vector<vector<double>>& get_asg();
  vector<vector<double>>& get_mu();
  vector<vector<double>>& get_m2();
  vector<vector<double>>& get_m4();
  virtual double distance(const vector<double>& x,
                          const unsigned long int c);
  virtual double variance(const double x,
                          const double alpha,
                          const unsigned long int dim);
  virtual double diff_variance(const double x,
                               const double alpha,
                               const unsigned long int dim);

 protected:
  unsigned long int n_samples;
  unsigned long int n_dims;
  unsigned long int k;
  unsigned long int max_round;
  const vector<vector<double>>& data;
  const vector<unsigned long int>& label;
  vector<double> kappa;
  vector<double> count;
  vector<vector<double>> mu;
  vector<unsigned long int> asg;
  vector<double> nmis;
  vector<double> logliks;
  copt::Vector<double> alpha;
  copt::Vector<double> lb;
  copt::Vector<double> ub;
  vector<bool> attr_discrete;
  vector<bool> attr_positive;
  vector<bool> attr_nonnegative;
  vector<vector<double>> mu_a;
  vector<vector<double>> mu_b;
  vector<double> kappa_a;
  vector<double> kappa_b;
  vector<vector<double>> m2;
  vector<vector<double>> m4;
  vector<vector<double>> v;
  vector<vector<double>> w_inv;
};

BOOST_SERIALIZATION_ASSUME_ABSTRACT(AdaHardCluster);

class CUGMMLite : public copt::BoundedProblem<double> {
 public:
  CUGMMLite(AdaHardCluster& mdl_, unsigned long int dim_);
  double value(const copt::Vector<double>& x);
  void gradient(const copt::Vector<double>& x, copt::Vector<double>& grad);

 private:
  unsigned long int dim;
  unsigned long int k;
  AdaHardCluster& mdl;
  vector<vector<double>>& mu;
  vector<vector<double>>& m2;
  vector<vector<double>>& m4;
};

#endif  // HARD_CLUSTERING_H
