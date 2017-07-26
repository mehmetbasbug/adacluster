#ifndef SOFT_CLUSTERING_H
#define SOFT_CLUSTERING_H
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

class AdaCluster {
  friend class bserialize::access;
  template <typename Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar& n_samples& n_dims& k& mu& kappa& log_asg;
  }

 public:
  AdaCluster(const vector<vector<double>>& data_,
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
  vector<double>& get_log_pi();
  vector<double>& get_logliks();
  vector<double>& get_nmis();
  copt::Vector<double>& get_alpha();
  copt::Vector<double>& get_lb();
  copt::Vector<double>& get_ub();
  const vector<vector<double>>& get_data();
  vector<vector<double>>& get_log_asg();
  vector<vector<double>>& get_mu();
  virtual double distance(const double x,
                          const double y,
                          const double alpha,
                          const unsigned long int dim);
  virtual double variance(const double x,
                          const double alpha,
                          const unsigned long int dim);
  virtual double diff_distance(const double x,
                               const double y,
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
  vector<double> log_pi;
  vector<double> max_log_pi;
  vector<double> log_asg_sum;
  vector<double> sum_distance;
  vector<vector<double>> mu;
  vector<vector<double>> log_asg;
  vector<vector<double>> log_data;
  vector<vector<double>> max_log_mu;
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
};

BOOST_SERIALIZATION_ASSUME_ABSTRACT(AdaCluster);

// class GMM : public AdaCluster {
//   friend class bserialize::access;
//   template <typename Archive>
//   void serialize(Archive& ar, const unsigned int version) {
//     ar& BOOST_SERIALIZATION_BASE_OBJECT_NVP(AdaCluster);
//   }

//  public:
//   GMM(const vector<vector<double>>& data_,
//       const vector<unsigned long int>& label_, unsigned long int max_round_,
//       unsigned long int k_);
//   double log_base_measure(unsigned long int sample, unsigned long int dim);
//   double distance(const double x, double y, unsigned long int j);
//   void update_hyperparams();
// };

// class BSC : public AdaCluster {
//   friend class bserialize::access;
//   template <typename Archive>
//   void serialize(Archive& ar, const unsigned int version) {
//     ar& BOOST_SERIALIZATION_BASE_OBJECT_NVP(AdaCluster);
//     ar& beta& kappa;
//   }

//  public:
//   BSC(const vector<vector<double>>& data_,
//       const vector<unsigned long int>& label_, unsigned long int max_round_,
//       unsigned long int k_);
//   vector<double>& get_beta();
//   double log_base_measure(unsigned long int sample, unsigned long int dim);
//   double distance(const double x, double y, unsigned long int j);
//   void update_hyperparams();

//  private:
//   vector<double> beta;
//   vector<double> sum_log_data;
// };

class Hyperparam : public copt::BoundedProblem<double> {
 public:
  Hyperparam(AdaCluster& mdl_);
  double value(const copt::Vector<double>& x);
  void gradient(const copt::Vector<double>& x, copt::Vector<double>& grad);

 private:
  unsigned long int k;
  unsigned long int n_dims;
  unsigned long int n_samples;
  AdaCluster& mdl;
  vector<double>& kappa;
  vector<double>& log_pi;
  const vector<vector<double>>& data;
  vector<vector<double>>& log_asg;
  vector<vector<double>>& mu;
  vector<double> buffer;
};

#endif  // SOFT_CLUSTERING_H
