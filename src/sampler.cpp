#include "localai/sampler.hpp"
#include <numeric>
#include <functional>

namespace localai {

Sampler::Sampler(uint32_t seed)
    : rng_(seed == 0 ? std::random_device{}() : seed) {}

void Sampler::apply_temperature(std::vector<TokenProbability>& candidates, float temperature) {
    if (temperature <= 0.0f) temperature = 1.0f;
    for (auto& c : candidates) {
        c.logit /= temperature;
    }
}

void Sampler::apply_top_k(std::vector<TokenProbability>& candidates, int32_t k) {
    if (k <= 0 || k >= static_cast<int32_t>(candidates.size())) return;

    std::partial_sort(candidates.begin(), candidates.begin() + k, candidates.end(),
        [](const TokenProbability& a, const TokenProbability& b) {
            return a.logit > b.logit;
        });

    candidates.resize(k);
}

void Sampler::compute_softmax(std::vector<TokenProbability>& candidates) {
    float max_logit = candidates[0].logit;
    for (const auto& c : candidates) {
        max_logit = std::max(max_logit, c.logit);
    }

    float sum = 0.0f;
    for (auto& c : candidates) {
        c.probability = std::exp(c.logit - max_logit);
        sum += c.probability;
    }

    for (auto& c : candidates) {
        c.probability /= sum;
    }
}

void Sampler::apply_top_p(std::vector<TokenProbability>& candidates, float p) {
    if (p >= 1.0f) return;

    std::sort(candidates.begin(), candidates.end(),
        [](const TokenProbability& a, const TokenProbability& b) {
            return a.probability > b.probability;
        });

    float cumulative = 0.0f;
    size_t cutoff = candidates.size();
    for (size_t i = 0; i < candidates.size(); ++i) {
        cumulative += candidates[i].probability;
        if (cumulative >= p) {
            cutoff = i + 1;
            break;
        }
    }

    candidates.resize(cutoff);
}

void Sampler::apply_min_p(std::vector<TokenProbability>& candidates, float p) {
    if (p <= 0.0f) return;

    float max_prob = 0.0f;
    for (const auto& c : candidates) {
        max_prob = std::max(max_prob, c.probability);
    }

    float threshold = max_prob * p;
    candidates.erase(
        std::remove_if(candidates.begin(), candidates.end(),
            [threshold](const TokenProbability& c) {
                return c.probability < threshold;
            }),
        candidates.end());
}

void Sampler::apply_repeat_penalty(std::vector<TokenProbability>& candidates,
                                   const std::vector<int32_t>& recent_tokens,
                                   float penalty) {
    if (penalty <= 1.0f) return;

    for (auto& c : candidates) {
        for (int32_t recent : recent_tokens) {
            if (c.id == recent) {
                if (c.logit > 0) {
                    c.logit /= penalty;
                } else {
                    c.logit *= penalty;
                }
                break;
            }
        }
    }
}

int32_t Sampler::sample(std::vector<TokenProbability>& candidates) const {
    if (candidates.empty()) return -1;

    apply_temperature(candidates, temperature_);
    apply_top_k(candidates, top_k_);
    compute_softmax(candidates);
    apply_top_p(candidates, top_p_);
    apply_min_p(candidates, min_p_);

    if (candidates.empty()) return -1;

    float total = 0.0f;
    for (const auto& c : candidates) {
        total += c.probability;
    }

    std::uniform_real_distribution<float> dist(0.0f, total);
    float r = dist(rng_);

    float cumulative = 0.0f;
    for (const auto& c : candidates) {
        cumulative += c.probability;
        if (r <= cumulative) return c.id;
    }

    return candidates.back().id;
}

}
