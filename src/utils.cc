#include "utils.h"

using namespace std;

bool is_pos_inf(double x){
    return ((!std::isnan(x)) && (!std::isfinite(x)) && (!signbit(x)));
}

bool is_neg_inf(double x){
    return ((!std::isnan(x)) && (!std::isfinite(x)) && (signbit(x)));
}

bool is_discrete(const vector<vector<double>>& x, unsigned long int dim) {
  double intpart;
  size_t n_samples = x.size();
  for (size_t i = 0; i < n_samples; ++i)
    if (modf(x[i][dim], &intpart) != 0.0) return 0;
  return 1;
}

bool is_positive(const vector<vector<double>>& x, unsigned long int dim) {
  size_t n_samples = x.size();
  for (size_t i = 0; i < n_samples; ++i)
    if (x[i][dim] <= 0) return 0;
  return 1;
}

bool is_nonnegative(const vector<vector<double>>& x, unsigned long int dim) {
  size_t n_samples = x.size();
  for (size_t i = 0; i < n_samples; ++i)
    if (x[i][dim] < 0) return 0;
  return 1;
}

double calc_nmi(const vector<unsigned long int>& labels,
                vector<unsigned long int>& estimates) {
  outer_map co_occur;
  inner_map label_count;
  inner_map estimate_count;
  unsigned long int label, estimate;
  unsigned long int N = labels.size();
  for (size_t i = 0; i < N; ++i) {
    label = labels[i];
    if (label_count.count(label))
      label_count[label] += 1;
    else
      label_count[label] = 1;
    estimate = estimates[i];
    if (estimate_count.count(estimate))
      estimate_count[estimate] += 1;
    else
      estimate_count[estimate] = 1;
    if (co_occur.count(label)) {
      if (co_occur[label].count(estimate))
        co_occur[label][estimate] += 1;
      else
        co_occur[label][estimate] = 1;
    } else
      co_occur[label][estimate] = 1;
  }
  double entropy = 0.0;
  unsigned long int count = 0;
  double ratio = 0.0;
  for (inner_map::iterator it = label_count.begin(); it != label_count.end();
       it++) {
    count = it->second;
    if (count != 0) {
      ratio = (double)count / N;
      entropy -= ratio * log(ratio);
    }
  }
  for (inner_map::iterator it = estimate_count.begin();
       it != estimate_count.end(); it++) {
    count = it->second;
    if (count != 0) {
      ratio = (double)count / N;
      entropy -= ratio * log(ratio);
    }
  }
  double information = 0.0;
  for (outer_map::iterator it = co_occur.begin(); it != co_occur.end(); it++) {
    for (inner_map::iterator it2 = it->second.begin(); it2 != it->second.end();
         it2++) {
      count = it2->second;
      if (count != 0) {
        ratio = (double)count / N;
        information +=
            ratio * log(ratio * N * N / (double)label_count[it->first] /
                        (double)estimate_count[it2->first]);
      }
    }
  }
  return information * 2 / entropy;
}

double logsumexp(const vector<double>& x, size_t N) {
  double max_exp = x[0], sum = 0.0;
  double out = 0.0;
  size_t i;
  for (i = 1; i < N; ++i)
    if (x[i] > max_exp) max_exp = x[i];
  for (i = 0; i < N; ++i) sum += exp(x[i] - max_exp);
  out = log(sum) + max_exp;
  return out;
}

double logsumexp(const vector<double>& x) { return logsumexp(x, x.size()); }

double logsumexp(const vector<vector<double>>& x) {
  double max_exp = x[0][0], sum = 0.0;
  double out = 0.0;
  unsigned long int dim1 = x.size();
  unsigned long int dim2 = x[0].size();
  for (size_t i = 0; i < dim1; ++i)
    for (size_t j = 0; j < dim2; ++j)
      if (x[i][j] > max_exp) max_exp = x[i][j];
  for (size_t i = 0; i < dim1; ++i)
    for (size_t j = 0; j < dim2; ++j) sum += exp(x[i][j] - max_exp);
  out = log(sum) + max_exp;
  return out;
}

void normalize(vector<double>& x) {
  double sum = 0;
  for (size_t i = 0; i < x.size(); ++i) sum += x[i];
  for (size_t i = 0; i < x.size(); ++i) x[i] = x[i] / sum;
}

void normalize(vector<vector<double>>& x) {
  unsigned long int dim1 = x.size();
  unsigned long int dim2 = x[0].size();
  double sum = 0;
  for (size_t i = 0; i < dim1; ++i)
    for (size_t j = 0; j < dim2; ++j) sum += x[i][j];
  for (size_t i = 0; i < dim1; ++i)
    for (size_t j = 0; j < dim2; ++j) x[i][j] = x[i][j] / sum;
}

void normalize_log(vector<double>& x) {
  double norm = logsumexp(x);
  for (size_t i = 0; i < x.size(); ++i) x[i] -= norm;
}

void normalize_log(vector<vector<double>>& x) {
  unsigned long int dim1 = x.size();
  unsigned long int dim2 = x[0].size();
  double norm = logsumexp(x);
  for (size_t i = 0; i < dim1; ++i)
    for (size_t j = 0; j < dim2; ++j) x[i][j] -= norm;
}

double nnd_distance(const double x, const double y, const double alpha) {
  // non negative discrete
  double out = 0.0;
  if (x == y)
    out = 0.0;
  else if (x == 0) {
    if (alpha == 0)
      out = y;
    else
      out = bmath::log1p(alpha * y) / alpha;
  } else {
    if (alpha == 0)
      out = y - x + x * (log(x) - log(y));
    else
      out = (bmath::log1p(alpha * y) - bmath::log1p(alpha * x)) *
                 (x + 1.0 / alpha) +
             x * (log(x) - log(y));
  }
  if (is_pos_inf(out))
    out = MAX_ALLOWED_DISTANCE;
  if (is_neg_inf(out))
    out = -MAX_ALLOWED_DISTANCE;
  // if (isnan(out))
  //   cout << "nnd  x="<<x<<" y="<<y<<" alpha="<<alpha<<endl;
  return out;
}

double nnd_diff_distance(const double x, const double y, const double alpha) {
  double out = 0.0;
  if (alpha == 0)
    out = -0.5 * pow(x - y, 2);
  else
    out = -(bmath::log1p(alpha * y) - bmath::log1p(alpha * x)) /
               pow(alpha, 2) +
           (y - x) / alpha / (alpha * y + 1);
  if (is_pos_inf(out))
    out = MAX_ALLOWED_DISTANCE;
  if (is_neg_inf(out))
    out = -MAX_ALLOWED_DISTANCE;
  // if (isnan(out))
  //   cout << "nnd_diff  x="<<x<<" y="<<y<<" alpha="<<alpha<<endl;
  return out;
}

double nnd_variance(const double x, const double alpha) {
  double out = 0.0;
  out = x * (1 + alpha * x);
  if (is_pos_inf(out))
    out = MAX_ALLOWED_DISTANCE;
  if (is_neg_inf(out))
    out = -MAX_ALLOWED_DISTANCE;
  if (out == 0.0)
    out = MIN_ALLOWED_VARIANCE;
  // if (isnan(out))
  //   cout << "nnd_variance  x="<< x <<" alpha="<<alpha<<endl;
  return out;
}

double nnd_diff_variance(const double x, const double alpha) {
  double out = 0.0;
  out = pow(x, 2);
  if (is_pos_inf(out))
    out = MAX_ALLOWED_DISTANCE;
  if (is_neg_inf(out))
    out = -MAX_ALLOWED_DISTANCE;
  if (out == 0.0)
    out = MIN_ALLOWED_VARIANCE;
  // if (isnan(out))
  //   cout << "nnd_diff_variance  x="<< x <<" alpha="<<alpha<<endl;
  return out;
}

double rc_distance(const double x, const double y, const double alpha) {
  // real cont
  double out = 0.0;
  if (x == y)
    out = 0.0;
  else if (alpha == 0)
    out = 0.5 * pow(x - y, 2);
  else
    out = 0.5 *
           (bmath::log1p(alpha * pow(y, 2)) - bmath::log1p(alpha * pow(x, 2)) +
            2 * sqrt(alpha) * x *
                (atan(sqrt(alpha) * x) - atan(sqrt(alpha) * y))) /
           alpha;
  if (is_pos_inf(out))
    out = MAX_ALLOWED_DISTANCE;
  if (is_neg_inf(out))
    out = -MAX_ALLOWED_DISTANCE;
  // if (isnan(out))
  //   cout << "rc  x="<<x<<" y="<<y<<" alpha="<<alpha<<endl;
  return out;
}

double rc_diff_distance(const double x, const double y, const double alpha) {
  double out = 0.0;
  if (alpha == 0)
    out = (-pow(x, 4) + 4 * x * pow(y, 3) - 3 * pow(y, 4)) / 12.0;
  else {
    double sqa = sqrt(alpha);
    out = -0.5 *
           (x * (atan(sqa * x) - atan(sqa * y)) / sqa +
            (bmath::log1p(alpha * pow(y, 2)) -
             bmath::log1p(alpha * pow(x, 2))) /
                alpha +
            y * (x - y) / (alpha * pow(y, 2) + 1)) /
           alpha;
  }
  if (is_pos_inf(out))
    out = MAX_ALLOWED_DISTANCE;
  if (is_neg_inf(out))
    out = -MAX_ALLOWED_DISTANCE;
  // if (isnan(out))
  //   cout << "rc_diff  x="<<x<<" y="<<y<<" alpha="<<alpha<<endl;
  return out;
}

double rc_variance(const double x, const double alpha) {
  double out = 0.0;
  out = 1 + alpha * pow(x, 2);
  if (is_pos_inf(out))
    out = MAX_ALLOWED_DISTANCE;
  if (is_neg_inf(out))
    out = -MAX_ALLOWED_DISTANCE;
  if (out == 0.0)
    out = MIN_ALLOWED_VARIANCE;
  // if (isnan(out))
  //   cout << "rc_variance  x="<< x <<" alpha="<<alpha<<endl;
  return out;
}

double rc_diff_variance(const double x, const double alpha) {
  double out = 0.0;
  out = pow(x, 2);
  if (is_pos_inf(out))
    out = MAX_ALLOWED_DISTANCE;
  if (is_neg_inf(out))
    out = -MAX_ALLOWED_DISTANCE;
  if (out == 0.0)
    out = MIN_ALLOWED_VARIANCE;
  // if (isnan(out))
  //   cout << "rc_diff_variance  x="<< x <<" alpha="<<alpha<<endl;
  return out;
}

double nnc_distance(const double x, const double y, const double alpha) {
  double out = 0.0;
  if (x == y)
    out = 0.0;
  else if (alpha == 0)
    out = x / y - log(x) + log(y) - 1;
  else if (alpha == 1){
    if (x == 0)
      out = y;
    else
      out = x * (log(x) - log(y)) - x + y;
  } else
    out = (pow(x, alpha) + ((alpha - 1) * y - alpha * x) * pow(y, alpha - 1)) /
             (alpha * (alpha - 1));
  if (is_pos_inf(out))
    out = MAX_ALLOWED_DISTANCE;
  if (is_neg_inf(out))
    out = -MAX_ALLOWED_DISTANCE;
  // if (isnan(out))
  //   cout << "nnc  x="<<x<<" y="<<y<<" alpha="<<alpha<<endl;
  return out;
}

double nnc_diff_distance(const double x, const double y, const double alpha) {
  double out = 0.0;
  if (x == 0)
    out = pow(y, alpha) * (alpha * log(y) - 1) / pow(alpha, 2);
  else {
    if (alpha == 0)
      out = x * (1 + log(y)) / y - 0.5 * (pow(log(x), 2) - pow(log(y), 2)) -
             log(x) - 1;
    else if (alpha == 1)
      out = x - y + 0.5 * x * (pow(log(x), 2) - pow(log(y), 2)) - x * log(x) +
             y * log(y);
    else {
      double am1 = alpha - 1;
      double asq = pow(alpha, 2);
      double am1sq = pow(am1, 2);
      double xa = pow(x, alpha);
      double ya = pow(y, alpha);
      double xyam1 = x * pow(y, am1);
      out = (ya - xa) / am1 / asq + (ya - xa) / am1sq / alpha +
             xa * log(x) / am1 / alpha + xyam1 / am1sq - ya / am1sq -
             xyam1 * log(y) / am1 + ya * log(y) / alpha;
    }
  }
  if (is_pos_inf(out))
    out = MAX_ALLOWED_DISTANCE;
  if (is_neg_inf(out))
    out = -MAX_ALLOWED_DISTANCE;
  // if (isnan(out))
  //   cout << "nnc_diff  x="<<x<<" y="<<y<<" alpha="<<alpha<<endl;
  return out;
}

double nnc_variance(const double x, const double alpha) {
  double out = 0.0;
  out = pow(x, 2 - alpha);
  if (is_pos_inf(out))
    out = MAX_ALLOWED_DISTANCE;
  if (is_neg_inf(out))
    out = -MAX_ALLOWED_DISTANCE;
  if (out == 0.0)
    out = MIN_ALLOWED_VARIANCE;
  // if (isnan(out))
  //   cout << "nnc_variance  x="<< x <<" alpha="<<alpha<<endl;
  return out;
}

double nnc_diff_variance(const double x, const double alpha) {
  double out = 0.0;
  if (x == 0)
    out = 0;
  else
    out = -pow(x, 2 - alpha) * log(x);
  if (is_pos_inf(out))
    out = MAX_ALLOWED_DISTANCE;
  if (is_neg_inf(out))
    out = -MAX_ALLOWED_DISTANCE;
  if (out == 0.0)
    out = MIN_ALLOWED_VARIANCE;
  // if (isnan(out))
  //   cout << "nnc_diff_variance  x="<< x <<" alpha="<<alpha<<endl;
  return out;
}
