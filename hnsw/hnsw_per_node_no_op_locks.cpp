#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <random>
#include <shared_mutex>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace py = pybind11;

#ifndef HNSW_DISABLE_LOCKS
#define HNSW_DISABLE_LOCKS 1
#endif

#if HNSW_DISABLE_LOCKS
struct NoOpMutex {
    void lock() noexcept {}
    void unlock() noexcept {}
    bool try_lock() noexcept { return true; }
};

struct NoOpSharedMutex {
    void lock() noexcept {}
    void unlock() noexcept {}
    bool try_lock() noexcept { return true; }
    void lock_shared() noexcept {}
    void unlock_shared() noexcept {}
    bool try_lock_shared() noexcept { return true; }
};

using Mutex = NoOpMutex;
using SharedMutex = NoOpSharedMutex;
#else
using Mutex = std::mutex;
using SharedMutex = std::shared_mutex;
#endif

class HNSW {
public:
    HNSW()
        : dim_(0),
          M_(16),
          maxM_(16),
          maxM0_(32),
          efConstruction_(200),
          levelMult_(1.0),
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

        std::unique_lock<Mutex> initLock(initMutex_);
        std::unique_lock<SharedMutex> labelsLock(labelMutex_);
        std::unique_lock<SharedMutex> nodesLock(nodesMutex_);
        std::unique_lock<SharedMutex> topLock(topologyMutex_);
        std::unique_lock<Mutex> rngLock(rngMutex_);

        dim_ = dim;
        M_ = M;
        maxM_ = M;
        maxM0_ = 2 * M;
        efConstruction_ = ef_construction;
        levelMult_ = 1.0 / std::log(static_cast<double>(M_));

        nodes_.clear();
        labelToId_.clear();
        enterPoint_ = -1;
        maxLevel_ = -1;

        rng_.seed(random_seed);
    }

    void insert(int label, const std::vector<double>& vec) {
        ensure_initialized();
        ensure_dim(vec);

        const int nodeLevel = sample_level();
        std::shared_ptr<Node> newNode;
        int newId = -1;

        {
            std::unique_lock<SharedMutex> labelLock(labelMutex_);
            if (labelToId_.count(label)) {
                throw std::invalid_argument("label already exists");
            }

            std::unique_lock<SharedMutex> nodesLock(nodesMutex_);
            newId = static_cast<int>(nodes_.size());
            newNode = std::make_shared<Node>(newId, label, nodeLevel, vec);
            nodes_.push_back(newNode);
            labelToId_[label] = newId;
        }

        int ep = -1;
        int topLevel = -1;
        {
            std::shared_lock<SharedMutex> topLock(topologyMutex_);
            ep = enterPoint_;
            topLevel = maxLevel_;
        }

        if (ep == -1) {
            std::unique_lock<SharedMutex> topLock(topologyMutex_);
            if (enterPoint_ == -1) {
                enterPoint_ = newId;
                maxLevel_ = nodeLevel;
                return;
            }
            ep = enterPoint_;
            topLevel = maxLevel_;
        }

        for (int lc = topLevel; lc > nodeLevel; --lc) {
            auto w = search_layer(vec, {ep}, 1, lc);
            if (!w.empty()) {
                ep = w.front();
            }
        }

        for (int lc = std::min(topLevel, nodeLevel); lc >= 0; --lc) {
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

        if (nodeLevel > topLevel) {
            std::unique_lock<SharedMutex> topLock(topologyMutex_);
            if (nodeLevel > maxLevel_) {
                enterPoint_ = newId;
                maxLevel_ = nodeLevel;
            }
        }
    }

    std::vector<std::pair<int, double>> knn_search(const std::vector<double>& query, int k, int ef = -1) const {
        ensure_initialized();
        ensure_dim(query);
        if (k <= 0) {
            throw std::invalid_argument("k must be > 0");
        }

        int ep = -1;
        int topLevel = -1;
        {
            std::shared_lock<SharedMutex> topLock(topologyMutex_);
            ep = enterPoint_;
            topLevel = maxLevel_;
        }

        if (ep == -1) {
            return {};
        }

        int localEf = (ef <= 0) ? std::max(50, k) : std::max(ef, k);

        for (int lc = topLevel; lc >= 1; --lc) {
            auto w = search_layer(query, {ep}, 1, lc);
            if (!w.empty()) {
                ep = w.front();
            }
        }

        auto w = search_layer(query, {ep}, localEf, 0);
        if (w.empty()) {
            return {};
        }

        std::vector<std::pair<int, double>> result;
        result.reserve(w.size());
        for (int id : w) {
            auto node = get_node(id);
            if (!node) {
                continue;
            }
            result.push_back({node->label, l2(query, node->point)});
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
    struct Node {
        Node(int id_, int label_, int level_, const std::vector<double>& point_)
            : id(id_),
              label(label_),
              level(level_),
              point(point_),
              adj(level_ + 1) {}

        int id;
        int label;
        int level;
        std::vector<double> point;
        std::vector<std::vector<int>> adj;
        mutable SharedMutex mutex;
    };

    int dim_;
    int M_;
    int maxM_;
    int maxM0_;
    int efConstruction_;
    double levelMult_;

    std::vector<std::shared_ptr<Node>> nodes_;
    std::unordered_map<int, int> labelToId_;

    int enterPoint_;
    int maxLevel_;

    mutable SharedMutex nodesMutex_;
    mutable SharedMutex labelMutex_;
    mutable SharedMutex topologyMutex_;
    mutable Mutex rngMutex_;
    mutable Mutex initMutex_;
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
        std::lock_guard<Mutex> lock(rngMutex_);
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

    std::shared_ptr<Node> get_node(int id) const {
        std::shared_lock<SharedMutex> lock(nodesMutex_);
        if (id < 0 || id >= static_cast<int>(nodes_.size())) {
            return nullptr;
        }
        return nodes_[id];
    }

    bool has_layer(int nodeId, int layer) const {
        auto node = get_node(nodeId);
        return node && layer >= 0 && layer <= node->level;
    }

    std::vector<int> get_neighbors_copy(int nodeId, int layer) const {
        auto node = get_node(nodeId);
        if (!node || layer < 0 || layer > node->level) {
            return {};
        }

        std::shared_lock<SharedMutex> lock(node->mutex);
        return node->adj[layer];
    }

    void connect_bidirectional(int a, int b, int layer) {
        if (a == b) {
            return;
        }

        auto na = get_node(a);
        auto nb = get_node(b);
        if (!na || !nb || layer < 0 || layer > na->level || layer > nb->level) {
            return;
        }

        if (na->id > nb->id) {
            std::swap(na, nb);
        }

        std::unique_lock<SharedMutex> lockA(na->mutex);
        std::unique_lock<SharedMutex> lockB(nb->mutex);

        auto& aNbrs = na->adj[layer];
        auto& bNbrs = nb->adj[layer];

        if (std::find(aNbrs.begin(), aNbrs.end(), nb->id) == aNbrs.end()) {
            aNbrs.push_back(nb->id);
        }
        if (std::find(bNbrs.begin(), bNbrs.end(), na->id) == bNbrs.end()) {
            bNbrs.push_back(na->id);
        }
    }

    void prune_if_needed(int nodeId, int layer) {
        auto node = get_node(nodeId);
        if (!node || layer < 0 || layer > node->level) {
            return;
        }

        const int limit = (layer == 0) ? maxM0_ : M_;

        std::vector<int> candidates;
        {
            std::unique_lock<SharedMutex> lock(node->mutex);
            if (static_cast<int>(node->adj[layer].size()) <= limit) {
                return;
            }
            candidates = node->adj[layer];
        }

        auto selected = select_neighbors_heuristic(node->point, candidates, limit, layer, nodeId);

        {
            std::unique_lock<SharedMutex> lock(node->mutex);
            if (static_cast<int>(node->adj[layer].size()) > limit) {
                node->adj[layer] = std::move(selected);
            }
        }
    }

    std::vector<int> search_layer(
        const std::vector<double>& query,
        const std::vector<int>& entryPoints,
        int ef,
        int layer
    ) const {
        struct Candidate {
            double dist;
            int id;
        };
        struct MinCmp {
            bool operator()(const Candidate& a, const Candidate& b) const {
                return a.dist > b.dist;
            }
        };
        struct MaxCmp {
            bool operator()(const Candidate& a, const Candidate& b) const {
                return a.dist < b.dist;
            }
        };

        std::priority_queue<Candidate, std::vector<Candidate>, MinCmp> candidates;
        std::priority_queue<Candidate, std::vector<Candidate>, MaxCmp> best;
        std::unordered_set<int> visited;

        for (int ep : entryPoints) {
            auto node = get_node(ep);
            if (!node || layer < 0 || layer > node->level) {
                continue;
            }
            const double d = l2(query, node->point);
            candidates.push({d, ep});
            best.push({d, ep});
            visited.insert(ep);
        }

        if (best.empty()) {
            return {};
        }

        while (!candidates.empty()) {
            const auto cur = candidates.top();
            candidates.pop();

            const auto farthest = best.top();
            if (cur.dist > farthest.dist) {
                break;
            }

            for (int nb : get_neighbors_copy(cur.id, layer)) {
                if (visited.count(nb)) {
                    continue;
                }
                visited.insert(nb);

                auto nbNode = get_node(nb);
                if (!nbNode) {
                    continue;
                }

                const double d = l2(query, nbNode->point);
                if (static_cast<int>(best.size()) < ef || d < best.top().dist) {
                    candidates.push({d, nb});
                    best.push({d, nb});
                    if (static_cast<int>(best.size()) > ef) {
                        best.pop();
                    }
                }
            }
        }

        std::vector<Candidate> ordered;
        ordered.reserve(best.size());
        while (!best.empty()) {
            ordered.push_back(best.top());
            best.pop();
        }

        std::sort(ordered.begin(), ordered.end(), [](const Candidate& a, const Candidate& b) {
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
            if (id == excludeId) {
                continue;
            }

            auto node = get_node(id);
            if (!node || layer < 0 || layer > node->level) {
                continue;
            }

            if (seen.insert(id).second) {
                ordered.push_back({l2(query, node->point), id});
            }
        }

        std::sort(ordered.begin(), ordered.end(), [](const auto& a, const auto& b) {
            if (a.first != b.first) return a.first < b.first;
            return a.second < b.second;
        });

        std::vector<int> selected;
        selected.reserve(std::min(M, static_cast<int>(ordered.size())));

        for (const auto& [distToQuery, id] : ordered) {
            auto node = get_node(id);
            if (!node) {
                continue;
            }

            bool good = true;
            for (int chosen : selected) {
                auto chosenNode = get_node(chosen);
                if (!chosenNode) {
                    continue;
                }
                const double interDist = l2(node->point, chosenNode->point);
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
    m.doc() = "Minimal HNSW implementation with basic concurrent insert/search support";

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
            py::call_guard<py::gil_scoped_release>(),
            "Insert one vector with an external label"
        )
        .def(
            "knn_search",
            &HNSW::knn_search,
            py::arg("query"),
            py::arg("k"),
            py::arg("ef") = -1,
            py::call_guard<py::gil_scoped_release>(),
            "Return [(label, squared_l2_distance), ...]"
        );
}