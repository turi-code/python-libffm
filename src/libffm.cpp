/**
 * Copyright (C) 2015 Dato, Inc.
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#include <string>
#include <vector>
#include <graphlab/flexible_type/flexible_type.hpp>
#include <graphlab/sdk/toolkit_class_macros.hpp>
#include <graphlab/sdk/gl_sframe.hpp>
#include <graphlab/logger/logger.hpp>
#include <graphlab/timer/timer.hpp>
#include <../ffm/lib/ffm.h>

using namespace graphlab;
using namespace std;
using namespace ffm;

ffm_problem read_sframe(gl_sframe data, std::string target, 
                        std::vector<std::string> features, 
                        size_t max_field_idx, 
                        size_t max_key_idx)
{
    ffm_problem prob;
    prob.l = data.size();
    prob.n = max_key_idx;
    prob.m = max_field_idx;
    prob.sf = data;
    prob.target_column = target;
    prob.feature_columns = features;
    return prob;
}

gl_sarray predict_sframe(ffm_model *model, gl_sframe data, std::string target_column, std::vector<std::string> feature_columns) 
{
  ffm_double loss = 0;
  vector<ffm_node> x;
  ffm_int i = 0;

  size_t target_col_idx = get_column_index(data, target_column); 
  std::vector<size_t> feature_col_idxs;
  for (auto col : feature_columns) { 
    feature_col_idxs.push_back(get_column_index(data, col));
  }

  gl_sarray_writer f_out(flex_type_enum::FLOAT, 1);
  size_t index = 0;
  auto r = data.range_iterator();
  auto it = r.begin();
  for (; it != r.end(); ++it, ++index) { 

    x.clear();

    const std::vector<flexible_type>& row = *it;
    const auto& yval = row[target_col_idx];
    ffm_float y = (yval.get<flex_int>() > 0) ? 1.0f : -1.0f;

    for (const size_t col_idx : feature_col_idxs) { 
      if (row[col_idx] != FLEX_UNDEFINED) {
        const flex_dict& dv = row[col_idx].get<flex_dict>(); 
        size_t n_values = dv.size(); 

        for(size_t k = 0; k < n_values; ++k) { 
          const std::pair<flexible_type, flexible_type>& kvp = dv[k];


          ffm_node N;
          N.f = col_idx; 
          N.j = kvp.first.get<flex_int>(); 
          N.v = (float) kvp.second;

          x.push_back(N);
        }
      }
    }

    ffm_float y_bar = ffm_predict(x.data(), x.data()+x.size(), model);
    f_out.write(y_bar, 0);

    loss -= y==1? log(y_bar) : log(1-y_bar);
  }

  loss /= i;

  logprogress_stream << "logloss = " << fixed << setprecision(5) << loss << endl;

  return f_out.close();
}

class ffm_py : public toolkit_class_base {

  ffm_parameter param;
  ffm_model* model;
  ffm_problem train;
  ffm_problem valid;
  std::string target;
  std::vector<std::string> features;

  virtual void save_impl(oarchive& oarc) const  {}

  virtual void load_version(iarchive& iarc, size_t version) {}

  // public:
  // ~ffm_py() {
  //   ffm_destroy_model(&model); 
  // }

  std::map<flexible_type, flexible_type> get_params() {
    auto p = std::map<flexible_type, flexible_type>();
    p["eta"] = param.eta;
    p["lambda"] = param.lambda;
    p["nr_iters"] = param.nr_iters;
    p["k"] = param.k;
    p["nr_threads"] = param.nr_threads;
    p["quiet"] = param.quiet;
    p["normalization"] = param.normalization;
    p["random"] = param.random;
    return p;
  }

  void init_model(double eta, double lambda, size_t k) { 
    param = ffm_get_default_param();
    param.eta = eta;
    param.lambda = lambda;
    param.k = k;
  }

  void set_param(size_t nr_iters, size_t nr_threads, size_t quiet) {
    param.nr_iters = nr_iters;
    param.nr_threads = nr_threads;
    param.quiet = quiet;
  }
  
  void load_model(std::string filename) {
    const char * c = filename.c_str();
    ffm_model* m = ffm_load_model(c);
  }

  void fit(gl_sframe trainsf, 
           gl_sframe validsf, 
           std::string _target, 
           std::vector<std::string> _features, 
           size_t max_feature_id) {
    target = _target;
    features = _features;

    // Set max field size to be the number of columns, i.e., each
    // user-provided feature is considered one of the model's "fields".
    // This determines the size of the model. Some of the parameters
    // may go unused if those features aren't chosen.
    size_t F = features.size();

    train = read_sframe(trainsf, target, features, F, max_feature_id);
    valid = read_sframe(validsf, target, features, F, max_feature_id);
    model = train_with_validation(&train, &valid, param);
  }

  gl_sarray predict(gl_sframe testsf) {
    return predict_sframe(model, testsf, target, features);
  }

  BEGIN_CLASS_MEMBER_REGISTRATION("ffm_py")
  REGISTER_CLASS_MEMBER_FUNCTION(ffm_py::get_params);
  REGISTER_CLASS_MEMBER_FUNCTION(ffm_py::init_model, 
                                 "eta", "lambda", "k");
  REGISTER_CLASS_MEMBER_FUNCTION(ffm_py::set_param, 
                                 "nr_iters", "nr_threads", "quiet");
  REGISTER_CLASS_MEMBER_FUNCTION(ffm_py::fit, 
                                 "train", "valid", "target", "features", "max_feature_id");
  REGISTER_CLASS_MEMBER_FUNCTION(ffm_py::predict, 
                                 "test");
  REGISTER_CLASS_MEMBER_FUNCTION(ffm_py::load_model, 
                                 "filename");
  END_CLASS_MEMBER_REGISTRATION
};

BEGIN_CLASS_REGISTRATION
REGISTER_CLASS(ffm_py)
END_CLASS_REGISTRATION

