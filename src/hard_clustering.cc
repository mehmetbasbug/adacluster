#include "utils.h"
#include "hard_clustering.h"
#include <boost/random/discrete_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>

using namespace std;

CUGMMLite::CUGMMLite(AdaHardCluster& mdl_,
                     unsigned long int dim_)
    : copt::BoundedProblem<double>(2),
      mdl(mdl_),
      dim(dim_),
      k(mdl_.get_k()),
      mu(mdl_.get_mu()),
      m2(mdl_.get_m2()),
      m4(mdl_.get_m4()){
}

double CUGMMLite::value(const copt::Vector<double>& x) {
  // x = <alpha,kappa>
  double obj = 0.0;
  double alpha = x[0];
  double kappa = x[1];
  double var = 0.0;
  for (size_t c = 0; c < k; ++c) {
    var = mdl.variance(mu[c][dim],alpha,dim);
    obj += (pow(m2[c][dim] - kappa*var,2)
      )/(m4[c][dim] - 2*m2[c][dim]*kappa*var + pow(kappa*var,2));
  }
  return obj;
}

void CUGMMLite::gradient(const copt::Vector<double>& x,
                         copt::Vector<double>& grad) {
  double alpha = x[0];
  double kappa = x[1];
  double var = 0.0;
  double diff_var = 0.0;
  grad[0] = 0.0;
  grad[1] = 0.0;
  double coef = 0.0;
  for (size_t c = 0; c < k; ++c){
    var = mdl.variance(mu[c][dim],alpha,dim);
    diff_var = mdl.diff_variance(mu[c][dim],alpha,dim);
    coef = (2*(pow(m2[c][dim],2)-m4[c][dim])*(m2[c][dim] - kappa*var)
      )/pow(m4[c][dim] - 2*m2[c][dim]*kappa*var + pow(kappa*var,2),2);
    grad[0] += coef*kappa*diff_var;
    grad[1] += coef*var;
  }
}

AdaHardCluster::AdaHardCluster(const vector<vector<double>>& data_,
                       const vector<unsigned long int>& label_,
                       unsigned long int max_round_, unsigned long int k_)
    : data(data_), label(label_) {
  max_round = max_round_;
  k = k_;
  nmis.resize(max_round);
  fill(nmis.begin(), nmis.end(), 0);
  logliks.resize(max_round);
  fill(logliks.begin(), logliks.end(), -numeric_limits<double>::max());
  n_samples = data.size();
  n_dims = data[0].size();
  count.resize(k);
  kappa.resize(n_dims);
  kappa_a.resize(n_dims);
  kappa_b.resize(n_dims);
  mu.resize(k);
  mu_a.resize(k);
  mu_b.resize(k);
  m2.resize(k);
  m4.resize(k);
  v.resize(k);
  w_inv.resize(k);
  for (size_t c = 0; c < k; ++c){
    mu[c].resize(n_dims);
    mu_a[c].resize(n_dims);
    mu_b[c].resize(n_dims);
    m2[c].resize(n_dims);
    m4[c].resize(n_dims);
    v[c].resize(n_dims);
    w_inv[c].resize(n_dims);
  }
  asg.resize(n_samples);
  attr_discrete.resize(n_dims);
  attr_positive.resize(n_dims);
  attr_nonnegative.resize(n_dims);

  alpha.resize(n_dims);
  lb.resize(n_dims);
  ub.resize(n_dims);
  for (unsigned long int j = 0; j < n_dims; ++j) {
    attr_discrete[j] = is_discrete(data, j);
    attr_positive[j] = is_positive(data, j);
    attr_nonnegative[j] = attr_positive[j];
    if (!attr_positive[j]) attr_nonnegative[j] = is_nonnegative(data, j);
  }
  for (unsigned long int j = 0; j < n_dims; ++j) {
    if (attr_discrete[j]) {
      if (attr_nonnegative[j]) {
        alpha[j] = 1;
        lb[j] = 0;
        // ub[j] = numeric_limits<double>::max();
        ub[j] = 10;
      } else {
        alpha[j] = 1;
        lb[j] = 0;
        // ub[j] = numeric_limits<double>::max();
        ub[j] = 10;
      }
    } else {
      if (attr_positive[j]) {
        alpha[j] = 0;
        // lb[j] = -numeric_limits<double>::max();
        lb[j] = -10;
        ub[j] = 2;
      } else if (attr_nonnegative[j]) {
        alpha[j] = 0.5;
        lb[j] = 0;
        ub[j] = 1;
      } else {
        alpha[j] = 1;
        lb[j] = 0;
        // ub[j] = numeric_limits<double>::max();
        ub[j] = 10;
      }
    }
  }
}

double AdaHardCluster::distance(const vector<double>& x,
                                const unsigned long int c) {
  double dist = 0.0;
  for (unsigned long int j = 0; j < n_dims; ++j)
    dist += pow(pow(x[j],2) - kappa[j]*v[c][j],2)*w_inv[c][j];
  return dist;
}

double AdaHardCluster::variance(const double x,
                                const double alpha,
                                const unsigned long int dim){
  // return nnc_variance(x, alpha);
  if (attr_discrete[dim]){
    if (attr_nonnegative[dim])
      return nnd_variance(x, alpha);
    else
      return rc_variance(x, alpha);
  } else {
    if (attr_nonnegative[dim])
      return nnc_variance(x, alpha);
    else
      return rc_variance(x, alpha);
  }
}


double AdaHardCluster::diff_variance(const double x,
                                 const double alpha,
                                 const unsigned long int dim) {
  // return nnc_diff_variance(x, alpha);
  if (attr_discrete[dim]){
    if (attr_nonnegative[dim])
      return nnd_diff_variance(x, alpha);
    else
      return rc_diff_variance(x, alpha);
  } else {
    if (attr_nonnegative[dim])
      return nnc_diff_variance(x, alpha);
    else
      return rc_diff_variance(x, alpha);
  }
}

unsigned long int AdaHardCluster::get_k() { return k; }

unsigned long int AdaHardCluster::get_max_round() { return max_round; }

unsigned long int AdaHardCluster::get_n_dims() { return n_dims; }

unsigned long int AdaHardCluster::get_n_samples() { return n_samples; }

vector<double>& AdaHardCluster::get_kappa() { return kappa; }

vector<double>& AdaHardCluster::get_logliks() { return logliks; }

vector<double>& AdaHardCluster::get_nmis() { return nmis; }

copt::Vector<double>& AdaHardCluster::get_alpha() { return alpha; }

copt::Vector<double>& AdaHardCluster::get_lb() { return lb; }

copt::Vector<double>& AdaHardCluster::get_ub() { return ub; }

const vector<vector<double>>& AdaHardCluster::get_data() { return data; }

vector<vector<double>>& AdaHardCluster::get_mu() { return mu; }

vector<vector<double>>& AdaHardCluster::get_m2() { return m2; }

vector<vector<double>>& AdaHardCluster::get_m4() { return m4; }

void AdaHardCluster::initialize_random() {
  unsigned long int ri;
  for (size_t c = 0; c < k; ++c) {
    ri = rand() % n_samples;
    for (unsigned long int j = 0; j < n_dims; ++j){
      mu[c][j] = data[ri][j];
      mu_a[c][j] = 0;//mu[c][j];
      mu_b[c][j] = 1.0;
    }
  }
  for (unsigned long int j = 0; j < n_dims; ++j){
    kappa[j] = 1.0;
    kappa_a[j] = 1e-9;
    kappa_b[j] = 1.0;
  }
}

void AdaHardCluster::initialize_k_plus_plus() {
  boost::mt19937 gen;
  unsigned long int ri, k_eff;
  ri = rand() % n_samples;
  for (unsigned long int j = 0; j < n_dims; ++j) mu[0][j] = data[ri][j];
  k_eff = 1;
  vector<double> probs(n_samples, 0);
  double dist = 0;
  double best_dist = 0;
  unsigned long int best_asg = -1;
  unsigned long int cur_asg = -1;
  do {
    for (size_t i = 0; i < n_samples; ++i) {
      best_dist = numeric_limits<double>::max();
      for (size_t c = 0; c < k_eff; ++c) {
        dist = 0;
        for (unsigned long int j = 0; j < n_dims; ++j){
          // dist += distance(data[i][j], mu[c][j], alpha[j], j);
          dist += pow(data[i][j] - mu[c][j], 2);
        }
        if (dist < best_dist) best_dist = dist;
      }
      probs[i] = pow(best_dist, 2);
    }
    boost::random::discrete_distribution<> dist(probs.begin(), probs.end());
    ri = dist(gen);
    for (unsigned long int j = 0; j < n_dims; ++j) mu[k_eff][j] = data[ri][j];
    k_eff += 1;
  } while (k_eff != k);

  for (unsigned long int j = 0; j < n_dims; ++j)
    if (attr_nonnegative[j])
      for (size_t c = 0; c < k_eff; ++c)
        if (mu[c][j] == 0)
          mu[c][j] = 1e-9;
  for (unsigned long int j = 0; j < n_dims; ++j){
    kappa[j] = 1.0;
    kappa_a[j] = 1.0;
    kappa_b[j] = 1.0;
  }
  for (size_t c = 0; c < k_eff; ++c)
    for (unsigned long int j = 0; j < n_dims; ++j){
      mu_a[c][j] = mu[c][j];
      mu_b[c][j] = 1.0;
    }
  for (size_t i = 0; i < n_samples; ++i) {
    best_dist = numeric_limits<double>::max();
    best_asg = -1;
    for (size_t c = 0; c < k; ++c) {
      dist = 0.0;
      for (size_t j = 0; j < n_dims; ++j)
        dist += pow(data[i][j] - mu[c][j], 2);
      if (dist < best_dist) {
        best_dist = dist;
        best_asg = c;
      }
    }
    asg[i] = best_asg;
  }
  for (size_t c = 0; c < k; c++){
    fill(mu[c].begin(), mu[c].end(), 1e-9);
    fill(m2[c].begin(), m2[c].end(), 1e-9);
    fill(m4[c].begin(), m4[c].end(), 1e-9);
  }
  fill(count.begin(), count.end(), 0);
  for (size_t i = 0; i < n_samples; ++i) {
    cur_asg = asg[i];
    count[cur_asg] += 1;
    for (size_t j = 0; j < n_dims; ++j){
      mu[cur_asg][j] += data[i][j];
      m2[cur_asg][j] += pow(data[i][j],2);
      m4[cur_asg][j] += pow(data[i][j],4);
    }
  }
  for (size_t c = 0; c < k; c++)
    for (size_t j = 0; j < n_dims; ++j){
      mu[c][j] = (kappa[j]*mu_a[c][j] + mu[c][j]
                  )/(kappa[j]*mu_b[c][j] + count[c]);
      m2[c][j] = (kappa[j]*pow(mu_a[c][j],2) + m2[c][j]
                  )/(kappa[j]*mu_b[c][j] + count[c]);
      m4[c][j] = (kappa[j]*pow(mu_a[c][j],4) + m4[c][j]
                  )/(kappa[j]*mu_b[c][j] + count[c]);
      v[c][j] = variance(mu[c][j],alpha[j],j);
      w_inv[c][j] = (kappa[j]*mu_b[c][j] + count[c]
        ) / (m4[c][j] - 2*m2[c][j]*kappa[j]*v[c][j] + pow(kappa[j]*v[c][j],2));
    }
}

void AdaHardCluster::fit() {
  bool updated = 0;
  double inertia = 0;
  double nmi = 0;
  double dist = 0;
  double best_dist = 0;
  unsigned long int best_asg = -1;
  unsigned long int cur_asg = -1;
  for (size_t r = 0; r < max_round; r++) {
    updated = 0;
    inertia = 0.0;
    for (size_t i = 0; i < n_samples; ++i) {
      best_dist = numeric_limits<double>::max();
      best_asg = -1;
      for (size_t c = 0; c < k; ++c) {
        dist = distance(data[i], c);
        if (dist < best_dist) {
          best_dist = dist;
          best_asg = c;
        }
      }
      if (asg[i] != best_asg) {
        asg[i] = best_asg;
        updated = 1;
      }
      inertia += best_dist;
    }
    nmi = calc_nmi(label, asg);
    nmis[r] = nmi;
    if (!updated) break;

    for (size_t c = 0; c < k; c++){
      fill(mu[c].begin(), mu[c].end(), 1e-9);
      fill(m2[c].begin(), m2[c].end(), 1e-9);
      fill(m4[c].begin(), m4[c].end(), 1e-9);
    }
    fill(count.begin(), count.end(), 0);
    for (size_t i = 0; i < n_samples; ++i) {
      cur_asg = asg[i];
      count[cur_asg] += 1;
      for (size_t j = 0; j < n_dims; ++j){
        mu[cur_asg][j] += data[i][j];
        m2[cur_asg][j] += pow(data[i][j],2);
        m4[cur_asg][j] += pow(data[i][j],4);
      }
    }
    for (size_t c = 0; c < k; c++)
      for (size_t j = 0; j < n_dims; ++j){
        mu[c][j] = (kappa[j]*mu_a[c][j] + mu[c][j]
                    )/(kappa[j]*mu_b[c][j] + count[c]);
        m2[c][j] = (kappa[j]*pow(mu_a[c][j],2) + m2[c][j]
                    )/(kappa[j]*mu_b[c][j] + count[c]);
        m4[c][j] = (kappa[j]*pow(mu_a[c][j],4) + m4[c][j]
                    )/(kappa[j]*mu_b[c][j] + count[c]);
        v[c][j] = variance(mu[c][j],alpha[j],j);
        w_inv[c][j] = (kappa[j]*mu_b[c][j] + count[c]
          ) / (m4[c][j] - 2*m2[c][j]*kappa[j]*v[c][j] + pow(kappa[j]*v[c][j],2));
      }
    double obj = 0;
    for (size_t j = 0; j < n_dims; ++j){
      CUGMMLite prblm(*this, j);
      copt::Vector<double> lbj;
      lbj.resize(2);
      lbj[0] = lb[j];
      lbj[1] = 1e-9;
      prblm.setLowerBound(lbj);
      copt::Vector<double> ubj;
      ubj.resize(2);
      ubj[0] = ub[j];
      ubj[1] = numeric_limits<double>::max();
      prblm.setUpperBound(ubj);
      copt::Vector<double> xj;
      xj.resize(2);
      xj[0] = alpha[j];
      xj[1] = kappa[j];
      copt::LbfgsbSolver<CUGMMLite> solver;
      solver.minimize(prblm, xj);
      obj += prblm.value(xj);
      if (!isnan(xj[0]))
        alpha[j] = xj[0];
      if (!isnan(xj[1]))
        kappa[j] = xj[1];
    }
    cout << "round=" << r << " nmi=" << nmi << " inertia=" << inertia;
    cout << " obj=" << obj << endl;
  }
  cout << "Alpha=";
  for (unsigned long int j = 0; j < n_dims; ++j)
      cout << alpha[j] << " ";
  cout << endl;
  cout << "Kappa=";
  for (unsigned long int j = 0; j < n_dims; ++j)
      cout << kappa[j] << " ";
  cout << endl;
  for (size_t c = 0; c < k; ++c){
    cout << "mu[" << c << "]=";
    for (unsigned long int j = 0; j < n_dims; ++j)
        cout << mu[c][j] << " ";
    cout << endl;
  }
}
