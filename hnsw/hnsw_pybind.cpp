#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <queue>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace py = pybind11;

class HNSW {
public:
    HNSW()
        : dim_(0),
          M_(16),
          maxM_(16),
          maxM0_(32),
          efConstruction_(200),
          enterPoint_(-1),
          maxLevel_(-1),
          rng_(std::random_device{}()),
          uniform_(0.0, 1.0) {}

    void init(int dim, int M = 16, int ef_construction = 200, int random_seed = 42) {
        if (dim <= 0) {
            throw std::invalid_argument("dim must be > 0");
        }
        if (M <= 0) {
            throw std::invalid_argument("M must be > 0");
        }
        if (ef_construction <= 0) {
            throw std::invalid_argument("ef_construction must be > 0");
        }

        dim_ = dim;
        M_ = M;
        maxM_ = M;
        maxM0_ = 2 * M;
        efConstruction_ = ef_construction;
        levelMult_ = 1.0 / std::log(static_cast<double>(M_));

        points_.clear();
        labels_.clear();
        labelToId_.clear();
        enterPoint_ = -1;
        maxLevel_ = -1;

        rng_.seed(random_seed);
    }

    void insert(int label, const std::vector<double>& vec) {
        ensure_initialized();
        ensure_dim(vec);

        if (labelToId_.count(label)) {
            throw std::invalid_argument("label already exists");
        }

        const int nodeLevel = sample_level();
        const int newId = static_cast<int>(points_.size());

        points_.push_back(vec);
        labels_.push_back(label);
        labelToId_[label] = newId;
        levels_.push_back(nodeLevel);
        adj_.push_back(std::vector<std::vector<int>>(nodeLevel + 1));

        if (enterPoint_ == -1) {
            enterPoint_ = newId;
            maxLevel_ = nodeLevel;
            return;
        }

        int ep = enterPoint_;

        // Phase 1: greedy descent above the new node's top layer.
        for (int lc = maxLevel_; lc > nodeLevel; --lc) {
            auto w = search_layer(vec, {ep}, 1, lc);
            ep = w.front();
        }

        // Phase 2: search and connect from min(maxLevel, nodeLevel) down to 0.
        for (int lc = std::min(maxLevel_, nodeLevel); lc >= 0; --lc) {
            auto w = search_layer(vec, {ep}, efConstruction_, lc);
            const int maxConn = (lc == 0) ? maxM0_ : M_;
            auto neighbors = select_neighbors_heuristic(vec, w, maxConn, lc);

            for (int nb : neighbors) {
                connect_bidirectional(newId, nb, lc);
            }

            for (int nb : neighbors) {
                prune_if_needed(nb, lc);
            }

            if (!w.empty()) {
                ep = w.front();
            }
        }

        if (nodeLevel > maxLevel_) {
            enterPoint_ = newId;
            maxLevel_ = nodeLevel;
        }
    }

    std::vector<std::pair<int, double>> knn_search(const std::vector<double>& query, int k, int ef = -1) const {
        ensure_initialized();
        ensure_dim(query);
        if (k <= 0) {
            throw std::invalid_argument("k must be > 0");
        }
        if (points_.empty()) {
            return {};
        }

        int localEf = (ef <= 0) ? std::max(50, k) : std::max(ef, k);
        int ep = enterPoint_;

        // Greedy descent on upper layers.
        for (int lc = maxLevel_; lc >= 1; --lc) {
            auto w = search_layer(query, {ep}, 1, lc);
            ep = w.front();
        }

        // Wider search on layer 0.
        auto w = search_layer(query, {ep}, localEf, 0);

        std::vector<std::pair<int, double>> result;
        result.reserve(w.size());
        for (int id : w) {
            result.push_back({labels_[id], l2(query, points_[id])});
        }

        std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) {
            if (a.second != b.second) return a.second < b.second;
            return a.first < b.first;
        });

        if (static_cast<int>(result.size()) > k) {
            result.resize(k);
        }
        return result;
    }

private:
    int dim_;
    int M_;
    int maxM_;
    int maxM0_;
    int efConstruction_;
    double levelMult_;

    std::vector<std::vector<double>> points_;
    std::vector<int> labels_;
    std::unordered_map<int, int> labelToId_;
    std::vector<int> levels_;
    std::vector<std::vector<std::vector<int>>> adj_; // adj_[node][layer] = neighbors

    int enterPoint_;
    int maxLevel_;

    mutable std::mt19937 rng_;
    mutable std::uniform_real_distribution<double> uniform_;

    void ensure_initialized() const {
        if (dim_ <= 0) {
            throw std::runtime_error("call init() before using HNSW");
        }
    }

    void ensure_dim(const std::vector<double>& vec) const {
        if (static_cast<int>(vec.size()) != dim_) {
            throw std::invalid_argument("vector dimension does not match initialized dim");
        }
    }

    int sample_level() {
        double u = uniform_(rng_);
        u = std::max(u, 1e-12);
        return static_cast<int>(std::floor(-std::log(u) * levelMult_));
    }

    static double l2(const std::vector<double>& a, const std::vector<double>& b) {
        double sum = 0.0;
        const size_t n = a.size();
        for (size_t i = 0; i < n; ++i) {
            const double d = a[i] - b[i];
            sum += d * d;
        }
        return sum;
    }

    bool has_layer(int node, int layer) const {
        return node >= 0 && node < static_cast<int>(adj_.size()) && layer >= 0 && layer <= levels_[node];
    }

    std::vector<int> get_neighbors(int node, int layer) const {
        if (!has_layer(node, layer)) {
            return {};
        }
        return adj_[node][layer];
    }

    void connect_bidirectional(int a, int b, int layer) {
        if (!has_layer(a, layer) || !has_layer(b, layer) || a == b) {
            return;
        }

        auto& na = adj_[a][layer];
        auto& nb = adj_[b][layer];

        if (std::find(na.begin(), na.end(), b) == na.end()) {
            na.push_back(b);
        }
        if (std::find(nb.begin(), nb.end(), a) == nb.end()) {
            nb.push_back(a);
        }
    }

    void prune_if_needed(int node, int layer) {
        auto& nbrs = adj_[node][layer];
        const int limit = (layer == 0) ? maxM0_ : M_;
        if (static_cast<int>(nbrs.size()) <= limit) {
            return;
        }

        std::vector<int> candidates = nbrs;
        auto selected = select_neighbors_heuristic(points_[node], candidates, limit, layer, node);
        nbrs = std::move(selected);
    }

    std::vector<int> search_layer(
        const std::vector<double>& query,
        const std::vector<int>& entryPoints,
        int ef,
        int layer
    ) const {
        struct MinCandidate {
            double dist;
            int id;
        };
        struct MinCmp {
            bool operator()(const MinCandidate& a, const MinCandidate& b) const {
                return a.dist > b.dist;
            }
        };
        struct MaxCmp {
            bool operator()(const MinCandidate& a, const MinCandidate& b) const {
                return a.dist < b.dist;
            }
        };

        std::priority_queue<MinCandidate, std::vector<MinCandidate>, MinCmp> candidates;
        std::priority_queue<MinCandidate, std::vector<MinCandidate>, MaxCmp> best;
        std::unordered_set<int> visited;

        for (int ep : entryPoints) {
            if (!has_layer(ep, layer)) {
                continue;
            }
            const double d = l2(query, points_[ep]);
            candidates.push({d, ep});
            best.push({d, ep});
            visited.insert(ep);
        }

        while (!candidates.empty()) {
            const auto cur = candidates.top();
            candidates.pop();

            const auto farthest = best.top();
            if (cur.dist > farthest.dist) {
                break;
            }

            for (int nb : adj_[cur.id][layer]) {
                if (visited.count(nb)) {
                    continue;
                }
                visited.insert(nb);
                const double d = l2(query, points_[nb]);

                if (static_cast<int>(best.size()) < ef || d < best.top().dist) {
                    candidates.push({d, nb});
                    best.push({d, nb});
                    if (static_cast<int>(best.size()) > ef) {
                        best.pop();
                    }
                }
            }
        }

        std::vector<MinCandidate> ordered;
        ordered.reserve(best.size());
        while (!best.empty()) {
            ordered.push_back(best.top());
            best.pop();
        }

        std::sort(ordered.begin(), ordered.end(), [](const MinCandidate& a, const MinCandidate& b) {
            if (a.dist != b.dist) return a.dist < b.dist;
            return a.id < b.id;
        });

        std::vector<int> result;
        result.reserve(ordered.size());
        for (const auto& x : ordered) {
            result.push_back(x.id);
        }
        return result;
    }

    std::vector<int> select_neighbors_heuristic(
        const std::vector<double>& query,
        const std::vector<int>& candidates,
        int M,
        int layer,
        int excludeId = -1
    ) const {
        std::vector<std::pair<double, int>> ordered;
        ordered.reserve(candidates.size());

        std::unordered_set<int> seen;
        for (int id : candidates) {
            if (id == excludeId) continue;
            if (!has_layer(id, layer)) continue;
            if (seen.insert(id).second) {
                ordered.push_back({l2(query, points_[id]), id});
            }
        }

        std::sort(ordered.begin(), ordered.end(), [](const auto& a, const auto& b) {
            if (a.first != b.first) return a.first < b.first;
            return a.second < b.second;
        });

        std::vector<int> selected;
        selected.reserve(std::min(M, static_cast<int>(ordered.size())));

        for (const auto& [distToQuery, id] : ordered) {
            bool good = true;
            for (int chosen : selected) {
                const double interDist = l2(points_[id], points_[chosen]);
                if (interDist < distToQuery) {
                    good = false;
                    break;
                }
            }
            if (good) {
                selected.push_back(id);
                if (static_cast<int>(selected.size()) == M) {
                    break;
                }
            }
        }

        // Fill remaining slots from closest candidates if heuristic was too strict.
        if (static_cast<int>(selected.size()) < M) {
            for (const auto& [_, id] : ordered) {
                if (std::find(selected.begin(), selected.end(), id) == selected.end()) {
                    selected.push_back(id);
                    if (static_cast<int>(selected.size()) == M) {
                        break;
                    }
                }
            }
        }

        return selected;
    }
};

PYBIND11_MODULE(hnsw_cpp, m) {
    m.doc() = "Minimal HNSW implementation with pybind11 bindings";

    py::class_<HNSW>(m, "HNSW")
        .def(py::init<>())
        .def(
            "init",
            &HNSW::init,
            py::arg("dim"),
            py::arg("M") = 16,
            py::arg("ef_construction") = 200,
            py::arg("random_seed") = 42,
            "Initialize the index"
        )
        .def(
            "insert",
            &HNSW::insert,
            py::arg("label"),
            py::arg("vector"),
            "Insert one vector with an external label"
        )
        .def(
            "knn_search",
            &HNSW::knn_search,
            py::arg("query"),
            py::arg("k"),
            py::arg("ef") = -1,
            "Return [(label, squared_l2_distance), ...]"
        );
}
