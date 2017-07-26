#include "utils.h"
#include "soft_clustering.h"
#include "hard_clustering.h"
#include <type_traits>
#include <unistd.h>
#include <iostream>
#include <omp.h>
#include <fstream>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/fusion/adapted.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/iostreams/device/mapped_file.hpp>

using namespace std;
using boost::property_tree::ptree;
namespace qi = boost::spirit::qi;
namespace po = boost::program_options;

const double CUTOFF_LOGLIK = log(1e-6);
const double MAX_ZERO_LOGLIK = log(1e-3);

struct config {
  unsigned long int conf_num;
  string in_file;
  string algorithm;
  unsigned long int max_iter;
} my_config;

void read_config(string& fname, unsigned long int conf_num) {
  my_config.conf_num = conf_num;
  boost::property_tree::ptree pt;
  boost::property_tree::ini_parser::read_ini(fname, pt);
  string strconfnum = to_string(static_cast<long long>(conf_num));
  my_config.in_file = pt.get<string>(strconfnum + ".in_file");
  my_config.algorithm = pt.get<string>(strconfnum + ".algorithm");
  my_config.max_iter = pt.get<unsigned long int>(strconfnum + ".max_iter");
}

int main(int argc, char* argv[]) {
  try {
    po::options_description desc("Configuration");
    desc.add_options()("help", "produce help message")(
        "fname", po::value<string>()->required(), "Conf file")(
        "conf_num", po::value<unsigned long int>()->required(), "Conf num");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
      cout << desc << "\n";
      return 0;
    }
    srand((unsigned)time(0));
    string fname = vm["fname"].as<string>();
    unsigned long int conf_num = vm["conf_num"].as<unsigned long int>();
    read_config(fname, conf_num);
    cout << "Config file is read.\n";

    vector<vector<double>> raw;
    boost::iostreams::mapped_file mmap(my_config.in_file.c_str(),
                                       boost::iostreams::mapped_file::readonly);
    auto f = mmap.const_data();
    auto l = f + mmap.size();
    bool ok =
        qi::phrase_parse(f, l, (qi::double_ % ',') % qi::eol, qi::blank, raw);
    if (ok)
      std::cout << "Input file : parse success\n";
    else
      std::cerr << "Input file : parse failed: '" << std::string(f, l) << "'\n";
    if (f != l)
      std::cerr << "Input file : trailing unparsed: '" << std::string(f, l)
                << "'\n";

    unsigned long int n_samples = raw.size();
    unsigned long int n_dims = raw[0].size();
    vector<unsigned long int> label(n_samples);
    vector<vector<double>> data;
    data.resize(n_samples);
    for (size_t i = 0; i < n_samples; i++) {
      label[i] = raw[i][n_dims - 1];
      data[i].resize(n_dims - 1);
      for (size_t j = 0; j < n_dims - 1; j++) data[i][j] = raw[i][j];
    }
    set<unsigned long int> unique_labels;
    unsigned long int lbl;
    for (size_t i = 0; i < n_samples; i++) unique_labels.insert(label[i]);
    unsigned long int k = unique_labels.size();
    unsigned long int max_iter = my_config.max_iter;
    cout << "Setting k=" << k << endl;
    if (my_config.algorithm.compare("KMeans") == 0) {
      // KMeans hc(data, label, max_iter, k);
      // hc.initialize_k_plus_plus();
      // hc.fit();
      // vector<double>& nmis = hc.get_nmis();
    } else if (my_config.algorithm.compare("AdaHardCluster") == 0) {
      AdaHardCluster hc(data, label, max_iter, k);
      hc.initialize_k_plus_plus();
      hc.fit();
      vector<double>& nmis = hc.get_nmis();
    } else if (my_config.algorithm.compare("GMM") == 0) {
      // for (size_t i = 0; i < n_samples; ++i)
      //   for (unsigned long int j = 0; j < n_dims; ++j) {
      //     if (data[i][j] == 0.0) data[i][j] = 1e-9;
      //     assert(data[i][j] > 0.0);
      //   }
      // GMM sc(data, label, max_iter, k);
      // sc.initialize_k_plus_plus();
      // sc.fit();
      // vector<double>& nmis = sc.get_nmis();
    } else if (my_config.algorithm.compare("AdaCluster") == 0) {
      // for (size_t i = 0; i < n_samples; ++i)
      //   for (unsigned long int j = 0; j < n_dims; ++j) {
      //     if (data[i][j] == 0.0) data[i][j] = 1e-9;
      //     assert(data[i][j] > 0.0);
      //   }
      AdaCluster sc(data, label, max_iter, k);
      sc.initialize_k_plus_plus();
      sc.fit();
      vector<double>& nmis = sc.get_nmis();
    } else {
      cerr << "Unknown model." << endl;
    }

  } catch (exception& e) {
    cerr << "error: " << e.what() << "\n";
    return 1;
  } catch (...) {
    cerr << "Exception of unknown type!\n";
  }
  return 0;
}
