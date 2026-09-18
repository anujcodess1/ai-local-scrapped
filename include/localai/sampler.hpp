#pragma once

#include <vector>
#include <cstdint>
#include <random>
#include <algorithm>
#include <cmath>

namespace localai {

struct TokenProbability {
    int32_t id;
    float logit;
    float probability;
};

class Sampler {
public:
    Sampler(uint32_t seed = 0);

    void set_temperature(float temp) { temperature_ = temp; }
    void set_top_k(int32_t k) { top_k_ = k; }
    void set_top_p(float p) { top_p_ = p; }
    void set_min_p(float p) { min_p_ = p; }
    void set_repeat_penalty(float penalty) { repeat_penalty_ = penalty; }

    int32_t sample(std::vector<TokenProbability>& candidates) const;

    static void apply_temperature(std::vector<TokenProbability>& candidates, float temperature);
    static void apply_top_k(std::vector<TokenProbability>& candidates, int32_t k);
    static void apply_top_p(std::vector<TokenProbability>& candidates, float p);
    static void apply_min_p(std::vector<TokenProbability>& candidates, float p);
    static void apply_repeat_penalty(std::vector<TokenProbability>& candidates,
                                     const std::vector<int32_t>& recent_tokens,
                                     float penalty);
    static void compute_softmax(std::vector<TokenProbability>& candidates);

private:
    float temperature_ = 0.7f;
    int32_t top_k_ = 40;
    float top_p_ = 0.9f;
    float min_p_ = 0.05f;
    float repeat_penalty_ = 1.1f;
    mutable std::mt19937 rng_;
};

}
