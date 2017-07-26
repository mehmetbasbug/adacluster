#ifndef UTILS_H_
#define UTILS_H_
#include <boost/math/special_functions/log1p.hpp>
#include <cmath>
#include <cstdlib>
#include <map>
#include <vector>
#include <meta.h>
#include <boundedproblem.h>
#include <solver/lbfgsbsolver.h>

using namespace std;
namespace bmath = boost::math;
namespace copt = cppoptlib;

const double MAX_ALLOWED_DISTANCE = 1e9;
const double MIN_ALLOWED_VARIANCE = 1e-9;
typedef map<unsigned long int, unsigned long int> inner_map;
typedef map<unsigned long int, inner_map> outer_map;

bool is_pos_inf(double x);
bool is_neg_inf(double x);
bool is_discrete(const vector<vector<double>>& x, unsigned long int dim);
bool is_positive(const vector<vector<double>>& x, unsigned long int dim);
bool is_nonnegative(const vector<vector<double>>& x, unsigned long int dim);
double calc_nmi(const vector<unsigned long int>&, vector<unsigned long int>&);
double logsumexp(const vector<double>& x, size_t N);
double logsumexp(const vector<double>&);
double logsumexp(const vector<vector<double>>&);
void normalize(vector<double>&);
void normalize(vector<vector<double>>&);
void normalize_log(vector<double>&);
void normalize_log(vector<vector<double>>&);

double nnd_distance(const double x, const double y, const double alpha);
double nnd_diff_distance(const double x, const double y, const double alpha);
double nnd_variance(const double x, const double alpha);
double nnd_diff_variance(const double x, const double alpha);

double rc_distance(const double x, const double y, const double alpha);
double rc_diff_distance(const double x, const double y, const double alpha);
double rc_variance(const double x, const double alpha);
double rc_diff_variance(const double x, const double alpha);

double nnc_distance(const double x, const double y, const double alpha);
double nnc_diff_distance(const double x, const double y, const double alpha);
double nnc_variance(const double x, const double alpha);
double nnc_diff_variance(const double x, const double alpha);

#endif  // UTILS_H
