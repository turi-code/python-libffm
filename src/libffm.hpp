// Copyright (c) 2015 The LIBFFM Project.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither name of copyright holders nor the names of its contributors
// may be used to endorse or promote products derived from this software
// without specific prior written permission.
//
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef _LIBFFM_H
#define _LIBFFM_H
#pragma once

#include <pmmintrin.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <random>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#if defined USEOMP
#include <omp.h>
#endif

namespace ffm {

template <typename ValueT = float, typename IndexT = std::size_t,
          typename TValueT = double,
          template <typename> class AllocT = std::allocator,
          template <typename, typename> class VectorT = std::vector>
class Model {
 public:
  template <typename T>
  using Allocator = AllocT<T>;
  using Value = ValueT;
  using TValue = TValueT;
  using size_t = typename Allocator<Value>::size_type;
  using Index = IndexT;
  using Vector = VectorT<Value, Allocator<Value>>;
  using Shape = std::tuple<size_t, size_t, size_t>;

  struct Node {
    // field index, i.e. column index
    Index field;
    // feature index, i.e. categorical value
    Index index;
    // feature value, i.e. 1.0 for categorical or real number for continuous
    Value value;
  };

  struct Parameters {
    // n - number of feature columns, i.e. "fields"
    Index num_columns;
    // m - total number of features, including all one-hot encoded values
    Index num_features;
    // k - number of states per feature value
    Index num_factors;
    // eta - learning rate
    Value eta;
    // lambda - L2 regularization parameter
    Value lambda;
  };

  struct FitOptions {
    // number of times to iterate over training data
    size_t num_iterations;
    // number of times to iterate over training batches
    size_t num_epochs;
    // size of training batches
    size_t batch_size;
    // number of threads to use - defaults to OpenMP value
    size_t num_threads;
    // number of rounds without improvement in loss before stopping
    size_t early_stop;
    // print status messages
    bool verbose;
    // randomize data before training
    bool randomize;
  };

  Model() : Model(default_parameters()) {}

  Model(const Parameters &params) : state_(kNew), data_(), params_(params) {}

  template <typename TrRange, typename VaRange = void *>
  void fit(TrRange tr_range, VaRange va_range = nullptr) {
    fit(default_fit_options(), tr_range, va_range);
  }

  template <typename TrRange, typename VaRange>
  void fit(const FitOptions &options, TrRange tr_range,
           VaRange va_range = nullptr) {
    if (state_ != kInit) {
      init();
    }
    fit_partial(options, tr_range, va_range);
  }

  template <typename TrRange, typename VaRange>
  void fit_partial(TrRange tr_range, VaRange va_range = nullptr) {
    fit_partial(default_fit_options(), tr_range, va_range);
  }

  template <typename TrRange, typename VaRange>
  void fit_partial(const FitOptions &options, TrRange tr_range,
                   VaRange va_range = nullptr) {
    auto &cout = std::cout;
    auto &cerr = std::cerr;

    if (state_ == kNew) {
      init();
      state_ = kTrain;
    }

    size_t num_tr = tr_range.size();
    size_t num_va = va_range.size();

    size_t early_stop = 0;
    if (options.early_stop != 0 && num_va != 0) early_stop = options.early_stop;
    size_t n_early = 0;

    std::vector<Value> prev_W;
    if (early_stop) prev_W.assign(data_.size(), 0.0);
    TValue best_va_loss = std::numeric_limits<TValue>::max();

    if (options.verbose) {
      if (options.early_stop > 0 && num_va == 0) {
        cerr << "ignoring auto-stop because there is no validation set"
             << std::endl;
      }
      cout << std::setw(4) << "iter" << std::setw(13) << "tr_logloss";
      if (num_va != 0) cout << std::setw(17) << "va_logloss";
      cout << std::endl;
    }

    for (size_t iter = 1; iter <= options.num_iterations; ++iter) {
      TValue tr_loss = 0;

      size_t i = 0;
      for (auto it = tr_range.begin(); it != tr_range.end(); ++it, ++i) {
        // get features and label
        auto X = it->first;
        auto y = it->second;
        // matrix mult
        auto t = wTx(X.begin(), X.end());
        // kappa and partial loss
        auto res = kappa_and_partial_loss(y, t);
        auto kappa = res.first;
        tr_loss += res.second;
        // update matrix
        wTx<true>(X.begin(), X.end(), 1.0, kappa, params_.eta, params_.lambda);
      }

      if (options.verbose) {
        tr_loss = loss_from_partial(tr_loss, num_tr);
        cout << std::setw(4) << iter << std::setw(13) << std::fixed
             << std::setprecision(5) << tr_loss;

        if (num_va != 0) {
          TValue va_loss = 0.0;

          size_t i = 0;
          for (auto jt = va_range.begin(); jt != va_range.end(); ++jt, ++i) {
            // get features and label
            auto X = jt->first;
            Value y = jt->second;
            // matrix mult
            Value t = wTx(X.begin(), X.end());
            // partial loss
            auto res = kappa_and_partial_loss(y, t);
            va_loss += res.second;
          }

          va_loss = loss_from_partial(va_loss, num_va);
          cout << std::setw(13) << std::fixed << std::setprecision(5)
               << va_loss;

          if (early_stop != 0) {
            if (va_loss > best_va_loss) {
              if (++n_early >= early_stop) {
                std::swap(data_, prev_W);
                cout << std::endl
                     << "Early stop: Using model from " << iter - 1
                     << "th iteration." << std::endl;
                break;
              }
            } else {
              n_early = 0;
              prev_W.clear();
              prev_W = data_;
              best_va_loss = va_loss;
            }
          }
        }
        cout << std::endl;
      }
    }

#undef cout
#undef cerr
  }

  template <typename RandomAccessIterator>
  Value predict(RandomAccessIterator begin, RandomAccessIterator end) const {
    Value t = wTx(begin, end);
    return value_from_logit(t);
  }

  void reset() {
    data_.clear();
    state_ = kNew;
  }

  const Parameters &default_parameters() const noexcept {
    static const Parameters params = {.num_columns = 0,
                                      .num_features = 0,
                                      .num_factors = 0,
                                      .eta = 0.2,
                                      .lambda = .00002};
    return params;
  }

  const FitOptions &default_fit_options() const noexcept {
    static const FitOptions options = {
        .num_iterations = 15,
        .num_epochs = 1,
        .batch_size = 1,
        .num_threads = 0,
        .early_stop = 0,
        .verbose = true,
        .randomize = false,
    };
    return options;
  }

  const Parameters &get_params() const noexcept { return params_; }
  void set_params(const Parameters &params) {
    if (params_.num_columns != params.num_columns ||
        params_.num_features != params.num_features ||
        params_.num_factors != params.num_factors) {
      std::cerr << "parameter changes forced model reset." << std::endl;
      reset();
    }
    params_ = params;
  }

  const size_t size() const noexcept { return data_.size(); }
  const Shape &shape() const noexcept { return std::make_tuple(n(), m(), k()); }

 private:
  enum State {
    kNew,
    kInit,
    kTrain,
  };

  static constexpr size_t kAlign = 16 / sizeof(Value);

  size_t n() const { return params_.num_columns; }
  size_t m() const { return params_.num_features; }
  size_t k() const { return params_.num_factors; }

  void n(size_t val) { params_.num_columns = val; }
  void m(size_t val) { params_.num_features = val; }
  void k(size_t val) { params_.num_factors = val; }

  size_t k_aligned() const { return k_aligned(k()); }
  size_t k_aligned(size_t k) const {
    size_t k_aligned = k % kAlign;
    k_aligned = k + (k_aligned > 0 ? kAlign - k_aligned : 0);
    return k_aligned;
  }

  void init() {
    size_t n = this->n(), m = this->m(), k = this->k();
    size_t k_aligned = this->k_aligned();

    data_.reserve(n * m * k_aligned * 2);

    Value coef = 1.0 / std::sqrt(k);
    std::default_random_engine generator;
    std::uniform_real_distribution<Value> uniform(0.0, 1.0);

    auto w = back_inserter(data_);
    // for each column
    for (Index i = 0; i < n; i++) {
      // for each feature
      for (Index j = 0; j < m; j++) {
        // for k states
        // initialize weights to uniform(0.0, 1.0)/sqrt(k)
        for (Index d = 0; d < k; d++) {
          *w = coef * uniform(generator);
        }
        // alignment padding
        for (Index d = k; d < k_aligned; d++) {
          *w = 0.0;
        }
        // initialize gradients to 1.0
        for (Index d = k_aligned; d < 2 * k_aligned; d++) {
          *w = 1.0;
        }
      }
    }

    state_ = kInit;
  }

#if 0
  void shrink(size_t k_new) {
    size_t n = this->n(), m = this->m();  //, k = this->k();
    size_t k_aligned = this->k_aligned();
    size_t k_aligned_new = this->k_aligned(k_new);

    auto begin = data_.begin();
    // for each column
    for (size_t i = 0; i < n; i++) {
      // for each feature
      for (size_t j = 0; j < m; j++) {
        // for k states
        // from region of ones...
        auto src = begin + (i * m + j) * k_aligned * 2;
        // ...to states
        auto dst = begin + (i * m + j) * k_aligned_new;
        // set k_new values to ones???
        std::copy(src, src + k_new, dst);
      }
    }

    // save new k
    this->k(k_new);
  }
#endif

  TValue value_from_logit(TValue t) const {
    if (false) {
      // logistic regression
      return 1 / (1 + exp(-t));
    } else {
      // regression
      return t;
    }
  }

  std::pair<TValue, TValue> kappa_and_partial_loss(TValue y, TValue t) const {
    TValue error, loss, kappa;
    if (false) {
      // logistic regression
      error = std::exp(-y * t);
      loss = std::log(1.0 + error);
      // derivative of loss
      // d(log(x)) => 1 / x           [1 / (1 + error)] * d(1 + error)
      // d(exp(x)) => exp(x) * d(x)   d(-y * t) * error / (1 + error)
      //                              -y * error / (1 + error)
      kappa = -y * error / (1.0 + error);
    } else {
      // regression: Root Mean Square Log Error
      y = std::max<TValue>(0.0, y + 1.0);
      t = std::max<TValue>(0.0, t + 1.0);
      // log(t) - log(y) == log(t/y)
      error = std::log(t/y);
      // minimize this
      loss = std::pow(error, 2);
      // derivative of loss
      // d(x**2)   => 2 * x * dx    2 * error * d(error)
      // d(log(x)) => 1 / x         2 * error * y / t
      kappa = 2 * error * y / t;
      // make sure kappa has the right sign
      kappa = std::copysign(kappa, t - y);
      // clip the derivative to prevent NAN
      kappa = std::max<TValue>(-1.0, std::min<TValue>(1.0, kappa));
    }
    return std::make_pair(kappa, loss);
  }

  TValue loss_from_partial(TValue partial_loss, size_t n) const {
    if (false) {
      // logistic regression
      return partial_loss / n;
    } else {
      // regression: Root Mean Square Log Error
      return std::sqrt(partial_loss) / n;
    }
  }

  template <typename ForwardIterator>
  inline Value wTx(ForwardIterator begin, ForwardIterator end, Value r = 1.0,
                   Value kappa = 0.0, Value eta = 0.0,
                   Value lambda = 0.0) const {
    return const_cast<Model *>(this)->wTx<false, ForwardIterator>(
        begin, end, r, kappa, eta, lambda);
  }

  template <bool IsUpdate = false, typename ForwardIterator>
  inline Value wTx(ForwardIterator begin, ForwardIterator end, Value r = 1.0,
                   Value kappa = 0.0, Value eta = 0.0, Value lambda = 0.0) {
    size_t n = this->n(), m = this->m(), k = this->k();
    size_t k_aligned = this->k_aligned();

    // axis=2 has size k_aligned * 2
    // axis=1 has size m * k_aligned * 2
    // axis=0 has size n * m * k_aligned * 2
    size_t axis2size = k_aligned * 2;
    size_t axis1size = m * axis2size;
    auto W = data_.data();

    __m128 XMMkappa = _mm_set1_ps(kappa);
    __m128 XMMeta = _mm_set1_ps(eta);
    __m128 XMMlambda = _mm_set1_ps(lambda);

    __m128 XMMt = _mm_setzero_ps();

    // for each pair x1, x2
    for (auto N1 = begin; N1 != end; N1++) {
      size_t i1 = N1->field;
      size_t j1 = N1->index;
      Value v1 = N1->value;
      if (i1 >= n || j1 >= m) continue;

      for (auto N2 = N1 + 1; N2 != end; N2++) {
        size_t i2 = N2->field;
        size_t j2 = N2->index;
        Value v2 = N2->value;
        if (i2 >= n || j2 >= m) continue;

        // retrieve weight1, weight2
        Value *w1 = W + i1 * axis1size + j2 * axis2size;
        Value *w2 = W + i2 * axis1size + j1 * axis2size;

        // value = value1 * value2 * normalizer
        __m128 XMMv = _mm_set1_ps(v1 * v2 * r);

        if (IsUpdate) {
          // kappa_value = kappa * value
          __m128 XMMkappav = _mm_mul_ps(XMMkappa, XMMv);

          // retrieve gradient1, gradient2
          Value *wg1 = w1 + k_aligned;
          Value *wg2 = w2 + k_aligned;
          for (size_t d = 0; d < k; d += 4) {
            // load weights into CPU registers
            __m128 XMMw1 = _mm_load_ps(w1 + d);
            __m128 XMMw2 = _mm_load_ps(w2 + d);

            // load gradients into CPU registers
            __m128 XMMwg1 = _mm_load_ps(wg1 + d);
            __m128 XMMwg2 = _mm_load_ps(wg2 + d);

            // g1 = lambda * weight1 + kappa_value * weight2
            // g2 = lambda * weight2 + kappa_value * weight1
            __m128 XMMg1 = _mm_add_ps(_mm_mul_ps(XMMlambda, XMMw1),
                                      _mm_mul_ps(XMMkappav, XMMw2));
            __m128 XMMg2 = _mm_add_ps(_mm_mul_ps(XMMlambda, XMMw2),
                                      _mm_mul_ps(XMMkappav, XMMw1));

            // gradient1 += g1^2
            // gradient2 += g2^2
            XMMwg1 = _mm_add_ps(XMMwg1, _mm_mul_ps(XMMg1, XMMg1));
            XMMwg2 = _mm_add_ps(XMMwg2, _mm_mul_ps(XMMg2, XMMg2));

            // weight1 -= eta * sqrt(gradient1) * g1
            // weight2 -= eta * sqrt(gradient2) * g2
            XMMw1 = _mm_sub_ps(
                XMMw1,
                _mm_mul_ps(XMMeta, _mm_mul_ps(_mm_rsqrt_ps(XMMwg1), XMMg1)));
            XMMw2 = _mm_sub_ps(
                XMMw2,
                _mm_mul_ps(XMMeta, _mm_mul_ps(_mm_rsqrt_ps(XMMwg2), XMMg2)));

            // save weight1, weight2
            _mm_store_ps(w1 + d, XMMw1);
            _mm_store_ps(w2 + d, XMMw2);

            // save gradient1, gradient2
            _mm_store_ps(wg1 + d, XMMwg1);
            _mm_store_ps(wg2 + d, XMMwg2);
          }
        } else {
          for (size_t d = 0; d < k; d += 4) {
            // load weight1, weight2
            __m128 XMMw1 = _mm_load_ps(w1 + d);
            __m128 XMMw2 = _mm_load_ps(w2 + d);

            // t += weight1 * weight2 * value
            XMMt = _mm_add_ps(XMMt, _mm_mul_ps(_mm_mul_ps(XMMw1, XMMw2), XMMv));
          }
        }
      }
    }

    if (IsUpdate) return 0;

    // sum everything
    XMMt = _mm_hadd_ps(XMMt, XMMt);
    XMMt = _mm_hadd_ps(XMMt, XMMt);
    Value t;
    _mm_store_ss(&t, XMMt);

    return t;
  }

  State state_;
  Parameters params_;
  Vector data_;
};

}  // namespace ffm

#endif  // _LIBFFM_H
