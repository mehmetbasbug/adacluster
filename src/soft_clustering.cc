#include "utils.h"
#include "soft_clustering.h"
#include <boost/random/discrete_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>

using namespace std;

Hyperparam::Hyperparam(AdaCluster& mdl_)
    : copt::BoundedProblem<double>(mdl_.get_n_dims()),
      mdl(mdl_),
      k(mdl_.get_k()),
      n_samples(mdl_.get_n_samples()),
      n_dims(mdl_.get_n_dims()),
      kappa(mdl_.get_kappa()),
      log_pi(mdl_.get_log_pi()),
      data(mdl_.get_data()),
      log_asg(mdl_.get_log_asg()),
      mu(mdl_.get_mu()) {
  buffer.resize(k);
}

double Hyperparam::value(const copt::Vector<double>& x) {
  // x = alpha
  double obj = 0.0;
  for (size_t i = 0; i < n_samples; ++i) {
    for (size_t c = 0; c < k; ++c) {
      buffer[c] = log_pi[c];
      for (unsigned long int j = 0; j < n_dims; ++j){
        if (!isnan(data[i][j])){
          double dist = mdl.distance(data[i][j], mu[c][j], x[j], j) / kappa[j];
          double var = mdl.variance(data[i][j], x[j], j);
          double upd = dist + 0.5 * (log(2 * M_PI) + log(kappa[j]) + log(var));
          // if (isnan(buffer[c])){
          //   cout << "buffer nan is detected ";
          //   cout << m_lowerBound[j] << " " << m_upperBound[j] << " ";
          //   cout << data[i][j] << " " << mu[c][j] << " " << x[j] << " ";
          //   cout << mdl.distance(data[i][j], mu[c][j], x[j], j) / kappa[j] << " ";
          //   cout << mdl.variance(data[i][j], x[j], j) << endl;
          //   break;
          // }
          if (!isnan(upd))
            buffer[c] -= upd;
        }
      }
    }
    obj += logsumexp(buffer);
  }
  return -obj;
}

void Hyperparam::gradient(const copt::Vector<double>& x,
                          copt::Vector<double>& grad) {
  bool terminate = 0;
  double gr = 0.0;
  for (unsigned long int j = 0; j < n_dims; ++j) {
    terminate = 0;
    gr = 0.0;
    for (size_t i = 0; i < n_samples; ++i){
      for (size_t c = 0; c < k; ++c)
        if (!isnan(data[i][j])){
          double diff_dist = mdl.diff_distance(data[i][j], mu[c][j], x[j], j) / kappa[j];
          double diff_var = mdl.diff_variance(data[i][j], x[j], j);
          double var = mdl.variance(data[i][j], x[j], j);
          double pr_asg = exp(log_asg[i][c]);
          double upd = 0.0;
          if (diff_var == 0.0)
            upd = pr_asg * diff_dist;
          else
            upd = pr_asg * (diff_dist + 0.5 * diff_var / var);
          gr += upd;
          // if (isnan(gr)){
          //   cout << "grad nan is detected";
          //   cout << m_lowerBound[j] << " " << m_upperBound[j] << " ";
          //   cout << data[i][j] << " " << mu[c][j] << " " << x[j] << " ";
          //   cout << mdl.diff_distance(data[i][j], mu[c][j], x[j], j) / kappa[j] << " ";
          //   cout << mdl.diff_variance(data[i][j], x[j], j) << " ";
          //   cout << mdl.variance(data[i][j], x[j], j) << endl;
          //   break;
          // }
          if (isnan(upd)){
            terminate = 1;
            break;
          }
        }
      if (terminate)
        break;
    }
    if (terminate)
      grad[j] = 0;
    else
      grad[j] = gr;
  }
}

AdaCluster::AdaCluster(const vector<vector<double>>& data_,
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
  log_pi.resize(k);
  fill(log_pi.begin(), log_pi.end(), -log(k));
  log_asg_sum.resize(k);
  max_log_pi.resize(k);
  kappa.resize(n_dims);
  kappa_a.resize(n_dims);
  kappa_b.resize(n_dims);
  mu.resize(k);
  mu_a.resize(k);
  mu_b.resize(k);
  for (size_t c = 0; c < k; ++c){
    mu[c].resize(n_dims);
    mu_a[c].resize(n_dims);
    mu_b[c].resize(n_dims);
  }
  log_asg.resize(n_samples);
  asg.resize(n_samples);
  for (size_t i = 0; i < n_samples; ++i) log_asg[i].resize(k);
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

double AdaCluster::distance(const double x,
                            const double y,
                            const double alpha,
                            const unsigned long int dim) {
  // return nnc_distance(x, y, alpha);
  if (attr_discrete[dim]){
    if (attr_nonnegative[dim])
      return nnd_distance(x, y, alpha);
    else
      return rc_distance(x, y, alpha);
  } else{
    if (attr_nonnegative[dim])
      return nnc_distance(x, y, alpha);
    else
      return rc_distance(x, y, alpha);
  }
}

double AdaCluster::variance(const double x,
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

double AdaCluster::diff_distance(const double x,
                                 const double y,
                                 const double alpha,
                                 const unsigned long int dim) {
  // return nnc_diff_distance(x, y, alpha);
  if (attr_discrete[dim]){
    if (attr_nonnegative[dim])
      return nnd_diff_distance(x, y, alpha);
    else
      return rc_diff_distance(x, y, alpha);
  } else {
    if (attr_nonnegative[dim])
      return nnc_diff_distance(x, y, alpha);
    else
      return rc_diff_distance(x, y, alpha);
  }
}


double AdaCluster::diff_variance(const double x,
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

unsigned long int AdaCluster::get_k() { return k; }

unsigned long int AdaCluster::get_max_round() { return max_round; }

unsigned long int AdaCluster::get_n_dims() { return n_dims; }

unsigned long int AdaCluster::get_n_samples() { return n_samples; }

vector<double>& AdaCluster::get_kappa() { return kappa; }

vector<double>& AdaCluster::get_log_pi() { return log_pi; }

vector<double>& AdaCluster::get_logliks() { return logliks; }

vector<double>& AdaCluster::get_nmis() { return nmis; }

copt::Vector<double>& AdaCluster::get_alpha() { return alpha; }

copt::Vector<double>& AdaCluster::get_lb() { return lb; }

copt::Vector<double>& AdaCluster::get_ub() { return ub; }

const vector<vector<double>>& AdaCluster::get_data() { return data; }

vector<vector<double>>& AdaCluster::get_log_asg() { return log_asg; }

vector<vector<double>>& AdaCluster::get_mu() { return mu; }

void AdaCluster::initialize_random() {
  unsigned long int ri;
  for (size_t c = 0; c < k; ++c) {
    ri = rand() % n_samples;
    for (unsigned long int j = 0; j < n_dims; ++j){
      mu[c][j] = data[ri][j];
      mu_a[c][j] = mu[c][j];
      mu_b[c][j] = 1.0;
    }
  }
  for (unsigned long int j = 0; j < n_dims; ++j){
    kappa[j] = 1.0;
    kappa_a[j] = 1.0;
    kappa_b[j] = 1e-9;
  }
}

void AdaCluster::initialize_k_plus_plus() {
  boost::mt19937 gen;
  unsigned long int ri, k_eff;
  ri = rand() % n_samples;
  for (unsigned long int j = 0; j < n_dims; ++j) mu[0][j] = data[ri][j];
  k_eff = 1;
  vector<double> probs(n_samples, 0);
  double dist, best_dist;
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
}

void AdaCluster::fit() {
  bool updated = 0;
  double loglik = 0;
  double nmi = 0;
  double dist = 0;
  double best_dist = 0;
  unsigned long int best_asg = -1;
  Hyperparam prblm(*this);
  prblm.setLowerBound(lb);
  prblm.setUpperBound(ub);
  for (size_t r = 0; r < max_round; r++) {
    updated = 0;
    loglik = 0.0;
    fill(max_log_pi.begin(), max_log_pi.end(), -numeric_limits<double>::max());
    for (size_t i = 0; i < n_samples; ++i) {
      for (size_t c = 0; c < k; ++c) {
        log_asg[i][c] = log_pi[c];
        for (unsigned long int j = 0; j < n_dims; ++j)
          if (!isnan(data[i][j])){
            double dist = distance(data[i][j], mu[c][j], alpha[j], j) / kappa[j];
            double var = variance(data[i][j], alpha[j], j);
            log_asg[i][c] -= dist + 0.5 * (log(2 * M_PI) + log(kappa[j]) + log(var));
          }
      }
      double norm = logsumexp(log_asg[i]);
      for (size_t c = 0; c < k; ++c) log_asg[i][c] -= norm;
      loglik += norm;
      for (size_t c = 0; c < k; ++c)
        if (log_asg[i][c] > max_log_pi[c]) max_log_pi[c] = log_asg[i][c];
    }
    for (size_t c = 0; c < k; ++c) {
      log_asg_sum[c] = 0.0;
      for (size_t i = 0; i < n_samples; ++i)
        log_asg_sum[c] += exp(log_asg[i][c] - max_log_pi[c]);
      log_asg_sum[c] = log(log_asg_sum[c]) + max_log_pi[c];
      log_pi[c] = log_asg_sum[c] - log(n_samples);
    }

    for (size_t c = 0; c < k; c++)
      for (unsigned long int j = 0; j < n_dims; ++j) {
        double tmp = 0.0;
        for (size_t i = 0; i < n_samples; ++i)
          if (!isnan(data[i][j]))
            tmp += exp(log_asg[i][c])*data[i][j];
        mu[c][j] = (kappa[j]*mu_a[c][j]*mu_b[c][j] + tmp
          )/(kappa[j]*mu_b[c][j] + exp(log_asg_sum[c]));
        // vector<double> pos(n_samples, 0);
        // vector<double> neg(n_samples, 0);
        // double zero_count = 0;
        // for (size_t i = 0; i < n_samples; ++i){
        //   if (data[i][j] > 0)
        //     pos[i] = log_asg[i][c] + log(data[i][j]);
        //   else if (data[i][j] < 0)
        //     neg[i] = log_asg[i][c] + log(-data[i][j]);
        //   else
        //     zero_count += 1;
        // }
        // mu[c][j] = (kappa[j]*mu_a[c][j] + exp(logsumexp(pos)) - exp(logsumexp(neg))
        //             )/(kappa[j]*mu_b[c][j] + exp(log_asg_sum[c]));
      }

    double pr_asg = 0.0;
    double nom = 0.0;
    double denom = 0.0;
    for (unsigned long int j = 0; j < n_dims; ++j) {
      nom = 0.0;
      denom = 0.0;
      for (size_t i = 0; i < n_samples; ++i)
        for (size_t c = 0; c < k; c++) {
          pr_asg = exp(log_asg[i][c]);
          if (!isnan(data[i][j])) {
            nom += pr_asg * distance(data[i][j], mu[c][j], alpha[j], j);
            denom += pr_asg;
          }
        }
      kappa[j] =  (kappa_b[j] + nom) / (kappa_a[j] + 0.5*denom);
    }

    copt::Vector<double> alpha_prev(alpha);
    // cout << alpha.transpose() << endl;
    copt::LbfgsbSolver<Hyperparam> solver;
    solver.minimize(prblm, alpha);
    // cout << alpha.transpose() << endl;

    for (unsigned long int j = 0; j < n_dims; ++j)
      if (isnan(alpha[j]))
        alpha[j] = alpha_prev[j];

    for (size_t i = 0; i < n_samples; ++i) {
      best_dist = log_asg[i][0];
      best_asg = 0;
      for (size_t c = 1; c < k; ++c)
        if (log_asg[i][c] > best_dist) {
          best_dist = log_asg[i][c];
          best_asg = c;
        }
      if (asg[i] != best_asg) {
        asg[i] = best_asg;
        updated = 1;
      }
    }
    nmi = calc_nmi(label, asg);
    nmis[r] = nmi;
    cout << "round=" << r << " nmi=" << nmi << " loglik=" << loglik << endl;
    if (!updated) break;
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

// GMM::GMM(const vector<vector<double>>& data_,
//          const vector<unsigned long int>& label_, unsigned long int max_round_,
//          unsigned long int k_)
//     : AdaCluster(data_, label_, max_round_, k_) {}

// double GMM::log_base_measure(unsigned long int sample, unsigned long int dim) {
//   return -0.5 * log(M_PI) - 0.5 * log(kappa[dim]);
// }

// double GMM::distance(const double x, double y, unsigned long int dim) {
//   return pow(x - y, 2) / 2.0;
// }

// void GMM::update_hyperparams() {}

// BSC::BSC(const vector<vector<double>>& data_,
//          const vector<unsigned long int>& label_, unsigned long int max_round_,
//          unsigned long int k_)
//     : AdaCluster(data_, label_, max_round_, k_) {
//   beta.resize(n_dims);
//   for (unsigned long int j = 0; j < n_dims; ++j) beta[j] = 2.0;
//   sum_log_data.resize(n_dims);
//   fill(sum_log_data.begin(), sum_log_data.end(), 0);
//   for (unsigned long int j = 0; j < n_dims; ++j)
//     for (size_t i = 0; i < n_samples; ++i) sum_log_data[j] += log_data[i][j];
// }

// vector<double>& BSC::get_beta() { return beta; }

// double BSC::log_base_measure(unsigned long int sample, unsigned long int dim) {
//   return -0.5 * log(M_PI) - 0.5 * log(kappa[dim]) -
//          0.5 * (2.0 - beta[dim]) * log_data[sample][dim];
// }

// double BSC::distance(const double x, double y, unsigned long int dim) {
//   return beta_div(x, y, beta[dim]);
// }

// void BSC::update_hyperparams() {
//   vector<double> x(n_dims);
//   vector<double> lb(n_dims);
//   vector<double> ub(n_dims);
//   for (unsigned long int j = 0; j < n_dims; ++j) {
//     lb[j] = -numeric_limits<double>::max();
//     ub[j] = numeric_limits<double>::max();
//     // lb[j] = -5.0;
//     // ub[j] = 5.0;
//   }
//   Saddle saddle(data, log_data, log_asg, log_pi, mu, kappa);
//   saddle.setLowerBound(lb);
//   saddle.setUpperBound(ub);
//   copt::LbfgsbSolver<double> solver;
//   int max_trial = 1;
//   double estimate = 0.0;
//   bool success = 1;
//   for (int trial = 0; trial < max_trial; trial++) {
//     for (unsigned long int j = 0; j < n_dims; ++j) x[j] = beta[j];
//     solver.minimize(saddle, x);
//     success = 1;
//     for (unsigned long int j = 0; j < n_dims; ++j)
//       if (std::isnan(x[j]) || !std::isfinite(x[j])) {
//         success = 0;
//         break;
//       }
//     if (success) break;
//   }
//   for (unsigned long int j = 0; j < n_dims; ++j) {
//     beta[j] = x[j];
//     // cout << " beta[" << j << "]=" << beta[j];
//   }
//   // cout << endl;
// }
