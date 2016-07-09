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
#include <graphlab/logger/logger.hpp>
#include <graphlab/parallel/lambda_omp.hpp>
#include <graphlab/sdk/gl_sframe.hpp>
#include <graphlab/sdk/toolkit_class_macros.hpp>
#include <graphlab/unity/lib/flex_dict_view.hpp>

#include "libffm.hpp"

using namespace graphlab;
using namespace std;

namespace {

using ffm_node = typename ffm::Model<>::Node;
using ffm_float = typename ffm::Model<>::Value;
using size_t = typename ffm::Model<>::size_t;

size_t get_column_index(const gl_sframe& sf, const std::string& colname) {
  const auto colnames = sf.column_names();
  for (size_t i = 0; i < colnames.size(); ++i) {
    if (colnames[i] == colname) {
      return i;
    }
  }
  return -1;
}

void push_flexible_type(std::vector<ffm_node>& nodes, size_t col,
                        const flexible_type& val) {
  if (val == FLEX_UNDEFINED) return;
  switch (val.get_type()) {
    case flex_type_enum::DICT: {
      const flex_dict& dv = val.get<flex_dict>();
      size_t n_values = dv.size();
      for (size_t k = 0; k < n_values; ++k) {
        const std::pair<flexible_type, flexible_type>& kvp = dv[k];

        ffm_node fv;
        fv.field = col;
        fv.index = (size_t)kvp.first.get<flex_int>();
        fv.value = (ffm_float)kvp.second.get<flex_float>();

        nodes.push_back(fv);
      }
      break;
    }
    case flex_type_enum::VECTOR: {
      const flex_vec& dv = val.get<flex_vec>();
      size_t n_values = dv.size();
      for (size_t k = 0; k < n_values; ++k) {
        ffm_node fv;
        fv.field = col;
        fv.index = (size_t)dv[k];
        fv.value = 1.0;
        nodes.push_back(fv);
      }
      break;
    }
    case flex_type_enum::FLOAT: {
      ffm_node fv;
      fv.field = col;
      fv.index = (size_t)val.get<flex_float>();
      fv.value = 1.0;
      nodes.push_back(fv);
    } break;
    case flex_type_enum::INTEGER: {
      ffm_node fv;
      fv.field = col;
      fv.index = (size_t)val.get<flex_int>();
      fv.value = 1.0;
      nodes.push_back(fv);
    } break;
    default:
      log_and_throw(
          "Feature columns currently must be dict, vec, float, or int.");
  }
}

ffm_float get_label(const flexible_type& val) {
  switch (val.get_type()) {
    case flex_type_enum::INTEGER:
      return (ffm_float)val.get<flex_int>();
      break;
    case flex_type_enum::FLOAT:
      return val.get<flex_float>();
      break;
    default:
      log_and_throw("Response must be float or int type.");
  }
  return 0.0;
}

}  // namespace

class ffm_py : public toolkit_class_base {
 private:
  std::map<flexible_type, flexible_type> get_params() {
    auto p = std::map<flexible_type, flexible_type>();

    p["target"] = target_;
    auto features = flexible_type(flex_type_enum::STRING);
    for (const auto& f : features_) features.push_back(f);
    p["features"] = features;

    p["num_columns"] = params_.num_columns;
    p["num_features"] = params_.num_features;
    p["num_factors"] = params_.num_factors;
    p["eta"] = params_.eta;
    p["lambda"] = params_.lambda;

    p["num_iterations"] = fit_options_.num_iterations;
    p["num_epochs"] = fit_options_.num_epochs;
    p["batch_size"] = fit_options_.batch_size;
    p["num_threads"] = fit_options_.num_threads;
    p["early_stop"] = fit_options_.early_stop;
    p["verbose"] = fit_options_.verbose;
    p["randomize"] = fit_options_.randomize;

    return p;
  }

  void set_params(const flexible_type& kwargs) {
    flex_dict_view dict = kwargs.get<flex_dict>();

    if (dict.has_key("target")) {
      auto it = dict["target"];
      if (it != FLEX_UNDEFINED) {
        target_ = it.get<flex_string>();
      }
    }
    if (dict.has_key("features")) {
      auto it = dict["features"];
      if (it != FLEX_UNDEFINED) {
        features_.clear();
        auto feats = it.get<flex_list>();
        for (size_t i = 0; i < feats.size(); ++i) {
          auto jt = feats[i];
          if (jt != FLEX_UNDEFINED) {
            features_.push_back(jt.get<flex_string>());
          }
        }
      }
    }

    params_.num_columns = features_.size();
    if (dict.has_key("num_features")) {
      auto it = dict["num_features"];
      if (it != FLEX_UNDEFINED) {
        params_.num_features = it.get<flex_int>();
      }
    }
    if (dict.has_key("num_factors")) {
      auto it = dict["num_factors"];
      if (it != FLEX_UNDEFINED) {
        params_.num_factors = it.get<flex_int>();
      }
    }
    if (dict.has_key("eta")) {
      auto it = dict["eta"];
      if (it != FLEX_UNDEFINED) {
        params_.eta = it.get<flex_float>();
      }
    }
    if (dict.has_key("lambda")) {
      auto it = dict["lambda"];
      if (it != FLEX_UNDEFINED) {
        params_.lambda = it.get<flex_float>();
      }
    }

    model_.set_params(params_);

    if (dict.has_key("num_iterations")) {
      auto it = dict["num_iterations"];
      if (it != FLEX_UNDEFINED) {
        fit_options_.num_iterations = it.get<flex_int>();
      }
    }
    if (dict.has_key("num_epochs")) {
      auto it = dict["num_epochs"];
      if (it != FLEX_UNDEFINED) {
        fit_options_.num_epochs = it.get<flex_int>();
      }
    }
    if (dict.has_key("batch_size")) {
      auto it = dict["batch_size"];
      if (it != FLEX_UNDEFINED) {
        fit_options_.batch_size = it.get<flex_int>();
      }
    }
    if (dict.has_key("num_threads")) {
      auto it = dict["num_threads"];
      if (it != FLEX_UNDEFINED) {
        fit_options_.num_threads = it.get<flex_int>();
      }
    }
    if (dict.has_key("early_stop")) {
      auto it = dict["early_stop"];
      if (it != FLEX_UNDEFINED) {
        fit_options_.early_stop = it.get<flex_int>();
      }
    }
    if (dict.has_key("verbose")) {
      auto it = dict["verbose"];
      if (it != FLEX_UNDEFINED) {
        fit_options_.verbose = it.get<flex_int>();
      }
    }
    if (dict.has_key("randomize")) {
      auto it = dict["randomize"];
      if (it != FLEX_UNDEFINED) {
        fit_options_.randomize = it.get<flex_int>();
      }
    }
  }

  template <typename Range>
  class DataRange {
    template <typename>
    class Iterator;

   public:
    using RangeIterator = Iterator<decltype(std::declval<Range>().begin())>;

    DataRange(const Range& rng, size_t size, size_t target_idx,
              const std::vector<size_t>* feature_idxs)
        : rng_(rng),
          size_(size),
          target_idx_(target_idx),
          feature_idxs_(feature_idxs) {}

    RangeIterator begin() {
      return RangeIterator(rng_.begin(), target_idx_, feature_idxs_);
    }

    RangeIterator end() { return RangeIterator(rng_.end(), 0, nullptr); }

    size_t size() const { return size_; }

   private:
    template <typename It>
    class Iterator {
     public:
      Iterator() = default;
      Iterator(const Iterator&) = delete;
      Iterator(Iterator&&) = default;

      Iterator(It&& it, size_t target_idx,
               const std::vector<size_t>* feature_idxs)
          : it_(),
            data_(),
            target_idx_(target_idx),
            feature_idxs_(feature_idxs) {
        using std::swap;
        swap(it_, it);
      }

      Iterator& operator++() {
        if (feature_idxs_ != nullptr) {
          ++it_;
          data_ = {};
        }
        return *this;
      }

      bool operator!=(const Iterator& rhs) const { return it_ != rhs.it_; }

      const std::pair<vector<ffm_node>, ffm_float>* operator->() {
        auto& x = data_.first;
        if (feature_idxs_ != nullptr && x.empty()) {
          const auto& row = *it_;
          for (const size_t col_idx : *feature_idxs_) {
            push_flexible_type(x, col_idx, row[col_idx]);
          }
          data_.second = (*it_)[target_idx_];
        }
        return &data_;
      }

      const std::pair<vector<ffm_node>, ffm_float>& operator*() {
        return *operator->();
      }

     private:
      It it_;
      std::pair<vector<ffm_node>, ffm_float> data_;
      size_t target_idx_;
      const std::vector<size_t>* feature_idxs_;
    };

    Range rng_;
    size_t size_;
    size_t target_idx_;
    const std::vector<size_t>* feature_idxs_;
  };

  template <typename Range>
  DataRange<Range> make_data_range(Range rng, size_t size, size_t tidx,
                                   const std::vector<size_t>* fidxs) {
    return DataRange<Range>(rng, size, tidx, fidxs);
  }

  void fit(const gl_sframe& trainsf, const gl_sframe& validsf) {
    model_.reset();
    fit_partial(trainsf, validsf);
  }

  void fit_partial(const gl_sframe& trainsf, const gl_sframe& validsf) {
    size_t train_target_idx = get_column_index(trainsf, target_);
    std::vector<size_t> train_feature_idxs;
    for (const auto& col : features_) {
      train_feature_idxs.push_back(get_column_index(trainsf, col));
    }

    size_t valid_target_idx = get_column_index(validsf, target_);
    std::vector<size_t> valid_feature_idxs;
    for (const auto& col : features_) {
      valid_feature_idxs.push_back(get_column_index(validsf, col));
    }

    auto train = make_data_range(trainsf.range_iterator(), trainsf.size(),
                                 train_target_idx, &train_feature_idxs);
    auto valid = make_data_range(validsf.range_iterator(), validsf.size(),
                                 valid_target_idx, &valid_feature_idxs);

    model_.fit_partial(fit_options_, train, valid);
  }

  gl_sarray predict(const gl_sframe& testsf) const {
    // get column indexes
    size_t target_col_idx = get_column_index(testsf, target_);
    std::vector<size_t> feature_col_idxs;
    for (const auto& col : features_) {
      feature_col_idxs.push_back(get_column_index(testsf, col));
    }

    // initialize parallelism
    auto& pool = thread_pool::get_instance();
    size_t nworkers = pool.size();
    size_t nlen = testsf.size();  // total range
    // size of range each worker gets, rounded up
    double split_size = (double)(nlen + nworkers - 1) / nworkers;

    gl_sarray_writer f_out(flex_type_enum::FLOAT, nworkers);

    parallel_task_queue threads(pool);
    for (size_t segment = 0; segment < nworkers; ++segment) {
      size_t begin = split_size * segment;      // beginning of this worker's
      size_t end = split_size * (segment + 1);  // end of this worker's range
      if (segment == nworkers - 1) end = nlen;
      threads.launch(
          [&, segment, begin, end, target_col_idx]() {
            vector<ffm_node> x;
            for (const auto& row : testsf.range_iterator(begin, end)) {
              x.clear();

              for (const size_t col_idx : feature_col_idxs) {
                push_flexible_type(x, col_idx, row[col_idx]);
              }

              auto y_bar = model_.predict(x.cbegin(), x.cend());
              f_out.write(y_bar, segment);
            }
          },
          segment);
    }
    threads.join();

    return f_out.close();
  }

  BEGIN_CLASS_MEMBER_REGISTRATION("ffm_py")
  REGISTER_CLASS_MEMBER_FUNCTION(ffm_py::get_params);
  REGISTER_CLASS_MEMBER_FUNCTION(ffm_py::set_params, "params");
  REGISTER_CLASS_MEMBER_FUNCTION(ffm_py::fit, "train", "valid");
  REGISTER_CLASS_MEMBER_FUNCTION(ffm_py::predict, "test");
  END_CLASS_MEMBER_REGISTRATION

  ffm::Model<> model_;
  ffm::Model<>::Parameters params_;
  ffm::Model<>::FitOptions fit_options_;

  std::string target_;
  std::vector<std::string> features_;
};

BEGIN_CLASS_REGISTRATION
REGISTER_CLASS(ffm_py)
END_CLASS_REGISTRATION
