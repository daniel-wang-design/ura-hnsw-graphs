#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <atomic>
#include <chrono>
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

        std::unique_lock initLock(initMutex_);
        std::unique_lock labelsLock(labelMutex_);
        std::unique_lock nodesLock(nodesMutex_);
        std::unique_lock topLock(topologyMutex_);
        std::unique_lock rngLock(rngMutex_);

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
        reset_node_lock_wait_stats();
    }

    void insert(int label, const std::vector<double>& vec) {
        ensure_initialized();
        ensure_dim(vec);

        const int nodeLevel = sample_level();
        int newId = -1;

        {
            std::unique_lock labelLock(labelMutex_);
            if (labelToId_.find(label) != labelToId_.end()) {
                throw std::invalid_argument("label already exists");
            }

            std::unique_lock nodesLock(nodesMutex_);
            newId = static_cast<int>(nodes_.size());
            nodes_.emplace_back(std::make_shared<Node>(newId, label, nodeLevel, vec, maxM_, maxM0_));
            labelToId_.emplace(label, newId);
        }

        int ep = -1;
        int topLevel = -1;
        {
            std::shared_lock topLock(topologyMutex_);
            ep = enterPoint_;
            topLevel = maxLevel_;
        }

        if (ep == -1) {
            std::unique_lock topLock(topologyMutex_);
            if (enterPoint_ == -1) {
                enterPoint_ = newId;
                maxLevel_ = nodeLevel;
                return;
            }
            ep = enterPoint_;
            topLevel = maxLevel_;
        }

        for (int lc = topLevel; lc > nodeLevel; --lc) {
            auto w = search_layer(vec, ep, 1, lc);
            if (!w.empty()) {
                ep = w.front().id;
            }
        }

        for (int lc = std::min(topLevel, nodeLevel); lc >= 0; --lc) {
            auto w = search_layer(vec, ep, efConstruction_, lc);
            const int maxConn = (lc == 0) ? maxM0_ : M_;
            auto neighbors = select_neighbors_heuristic(vec, w, maxConn, lc);

            for (int nb : neighbors) {
                connect_bidirectional(newId, nb, lc);
            }

            for (int nb : neighbors) {
                prune_if_needed(nb, lc);
            }

            if (!w.empty()) {
                ep = w.front().id;
            }
        }

        if (nodeLevel > topLevel) {
            std::unique_lock topLock(topologyMutex_);
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
            std::shared_lock topLock(topologyMutex_);
            ep = enterPoint_;
            topLevel = maxLevel_;
        }

        if (ep == -1) {
            return {};
        }

        const int localEf = (ef <= 0) ? std::max(50, k) : std::max(ef, k);

        for (int lc = topLevel; lc >= 1; --lc) {
            auto w = search_layer(query, ep, 1, lc);
            if (!w.empty()) {
                ep = w.front().id;
            }
        }

        auto w = search_layer(query, ep, localEf, 0);
        if (w.empty()) {
            return {};
        }

        std::vector<std::pair<int, double>> result;
        result.reserve(w.size());
        for (const auto& cand : w) {
            auto node = get_node(cand.id);
            if (!node) {
                continue;
            }
            result.emplace_back(node->label, cand.dist);
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


    std::pair<std::uint64_t, double> node_lock_wait_stats() const {
        const auto waits = nodeLockWaitCount_.load(std::memory_order_relaxed);
        const auto nanos = nodeLockWaitNanos_.load(std::memory_order_relaxed);
        return {waits, static_cast<double>(nanos) / 1'000'000'000.0};
    }

    void reset_node_lock_wait_stats() {
        nodeLockWaitCount_.store(0, std::memory_order_relaxed);
        nodeLockWaitNanos_.store(0, std::memory_order_relaxed);
    }

private:
    struct Node {
        Node(int id_, int label_, int level_, const std::vector<double>& point_, int maxM, int maxM0)
            : id(id_),
              label(label_),
              level(level_),
              point(point_),
              adj(level_ + 1) {
            if (!adj.empty()) {
                adj[0].reserve(maxM0);
                for (int l = 1; l <= level_; ++l) {
                    adj[l].reserve(maxM);
                }
            }
        }

        int id;
        int label;
        int level;
        std::vector<double> point;
        std::vector<std::vector<int>> adj;
        mutable std::shared_mutex mutex;
    };

    struct Candidate {
        double dist;
        int id;
    };

    struct OrderedCandidate {
        double dist;
        int id;
        std::shared_ptr<Node> node;
    };

    struct MinCmp {
        bool operator()(const Candidate& a, const Candidate& b) const {
            if (a.dist != b.dist) return a.dist > b.dist;
            return a.id > b.id;
        }
    };

    struct MaxCmp {
        bool operator()(const Candidate& a, const Candidate& b) const {
            if (a.dist != b.dist) return a.dist < b.dist;
            return a.id < b.id;
        }
    };

    struct VisitMarks {
        std::vector<uint32_t> marks;
        uint32_t token = 1;

        void reset(size_t n) {
            if (marks.size() < n) {
                marks.resize(n, 0);
            }
            ++token;
            if (token == 0) {
                std::fill(marks.begin(), marks.end(), 0U);
                token = 1;
            }
        }

        bool test_and_set(int id) {
            const size_t idx = static_cast<size_t>(id);
            if (idx >= marks.size()) {
                marks.resize(idx + 1, 0U);
            }
            if (marks[idx] == token) {
                return false;
            }
            marks[idx] = token;
            return true;
        }
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

    mutable std::shared_mutex nodesMutex_;
    mutable std::shared_mutex labelMutex_;
    mutable std::shared_mutex topologyMutex_;
    mutable std::mutex rngMutex_;
    mutable std::mutex initMutex_;
    mutable std::mt19937 rng_;
    mutable std::uniform_real_distribution<double> uniform_;

    mutable std::atomic<std::uint64_t> nodeLockWaitCount_{0};
    mutable std::atomic<std::uint64_t> nodeLockWaitNanos_{0};

    template <typename LockType>
    void lock_node_with_stats(LockType& lock) const {
        if (lock.try_lock()) {
            return;
        }

        const auto start = std::chrono::steady_clock::now();
        lock.lock();
        const auto end = std::chrono::steady_clock::now();

        const auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        nodeLockWaitCount_.fetch_add(1, std::memory_order_relaxed);
        nodeLockWaitNanos_.fetch_add(static_cast<std::uint64_t>(nanos), std::memory_order_relaxed);
    }

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
        std::lock_guard<std::mutex> lock(rngMutex_);
        double u = uniform_(rng_);
        if (u < 1e-12) {
            u = 1e-12;
        }
        return static_cast<int>(std::floor(-std::log(u) * levelMult_));
    }

    static double l2(const std::vector<double>& a, const std::vector<double>& b) {
        const size_t n = a.size();
        const double* pa = a.data();
        const double* pb = b.data();
        double sum = 0.0;
        size_t i = 0;

        for (; i + 3 < n; i += 4) {
            const double d0 = pa[i] - pb[i];
            const double d1 = pa[i + 1] - pb[i + 1];
            const double d2 = pa[i + 2] - pb[i + 2];
            const double d3 = pa[i + 3] - pb[i + 3];
            sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
        }
        for (; i < n; ++i) {
            const double d = pa[i] - pb[i];
            sum += d * d;
        }
        return sum;
    }

    static VisitMarks& visit_marks() {
        static thread_local VisitMarks state;
        return state;
    }

    size_t node_count() const {
        std::shared_lock lock(nodesMutex_);
        return nodes_.size();
    }

    std::shared_ptr<Node> get_node(int id) const {
        std::shared_lock lock(nodesMutex_);
        if (id < 0 || id >= static_cast<int>(nodes_.size())) {
            return nullptr;
        }
        return nodes_[static_cast<size_t>(id)];
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

        std::unique_lock<std::shared_mutex> lockA(na->mutex, std::defer_lock);
        lock_node_with_stats(lockA);
        std::unique_lock<std::shared_mutex> lockB(nb->mutex, std::defer_lock);
        lock_node_with_stats(lockB);

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
            std::unique_lock<std::shared_mutex> lock(node->mutex, std::defer_lock);
            lock_node_with_stats(lock);
            if (static_cast<int>(node->adj[layer].size()) <= limit) {
                return;
            }
            candidates = node->adj[layer];
        }

        auto selected = select_neighbors_heuristic(node->point, candidates, limit, layer, nodeId);

        {
            std::unique_lock<std::shared_mutex> lock(node->mutex, std::defer_lock);
            lock_node_with_stats(lock);
            if (static_cast<int>(node->adj[layer].size()) > limit) {
                node->adj[layer] = std::move(selected);
            }
        }
    }

    std::vector<Candidate> search_layer(
        const std::vector<double>& query,
        int entryPoint,
        int ef,
        int layer
    ) const {
        if (entryPoint < 0 || ef <= 0) {
            return {};
        }

        const size_t nodeCount = node_count();
        if (entryPoint >= static_cast<int>(nodeCount)) {
            return {};
        }

        auto entryNode = get_node(entryPoint);
        if (!entryNode || layer < 0 || layer > entryNode->level) {
            return {};
        }

        auto& visited = visit_marks();
        visited.reset(nodeCount);

        std::vector<Candidate> candStorage;
        candStorage.reserve(static_cast<size_t>(std::max(ef, 8)));
        std::priority_queue<Candidate, std::vector<Candidate>, MinCmp> candidates(MinCmp{}, std::move(candStorage));

        std::vector<Candidate> bestStorage;
        bestStorage.reserve(static_cast<size_t>(std::max(ef, 8)));
        std::priority_queue<Candidate, std::vector<Candidate>, MaxCmp> best(MaxCmp{}, std::move(bestStorage));

        const double entryDist = l2(query, entryNode->point);
        candidates.push({entryDist, entryPoint});
        best.push({entryDist, entryPoint});
        visited.test_and_set(entryPoint);

        while (!candidates.empty()) {
            const Candidate cur = candidates.top();
            candidates.pop();

            const Candidate farthest = best.top();
            if (cur.dist > farthest.dist) {
                break;
            }

            auto curNode = get_node(cur.id);
            if (!curNode || layer > curNode->level) {
                continue;
            }

            std::shared_lock<std::shared_mutex> curLock(curNode->mutex, std::defer_lock);
            lock_node_with_stats(curLock);
            const auto& neighbors = curNode->adj[layer];
            for (int nb : neighbors) {
                if (nb < 0) {
                    continue;
                }
                if (!visited.test_and_set(nb)) {
                    continue;
                }

                auto nbNode = get_node(nb);
                if (!nbNode || layer > nbNode->level) {
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

        return ordered;
    }

    std::vector<int> select_neighbors_from_ordered(
        const std::vector<OrderedCandidate>& ordered,
        int M
    ) const {
        std::vector<int> selected;
        selected.reserve(std::min(M, static_cast<int>(ordered.size())));

        std::vector<const Node*> selectedNodes;
        selectedNodes.reserve(std::min(M, static_cast<int>(ordered.size())));

        for (const auto& cand : ordered) {
            bool good = true;
            for (const Node* chosenNode : selectedNodes) {
                if (l2(cand.node->point, chosenNode->point) < cand.dist) {
                    good = false;
                    break;
                }
            }
            if (good) {
                selected.push_back(cand.id);
                selectedNodes.push_back(cand.node.get());
                if (static_cast<int>(selected.size()) == M) {
                    return selected;
                }
            }
        }

        for (const auto& cand : ordered) {
            bool alreadySelected = false;
            for (int id : selected) {
                if (id == cand.id) {
                    alreadySelected = true;
                    break;
                }
            }
            if (!alreadySelected) {
                selected.push_back(cand.id);
                if (static_cast<int>(selected.size()) == M) {
                    break;
                }
            }
        }

        return selected;
    }

    std::vector<int> select_neighbors_heuristic(
        const std::vector<double>& query,
        const std::vector<Candidate>& candidates,
        int M,
        int layer,
        int excludeId = -1
    ) const {
        const size_t nodeCount = node_count();
        auto& seen = visit_marks();
        seen.reset(nodeCount);

        std::vector<OrderedCandidate> ordered;
        ordered.reserve(candidates.size());

        for (const auto& cand : candidates) {
            const int id = cand.id;
            if (id == excludeId || id < 0) {
                continue;
            }
            if (!seen.test_and_set(id)) {
                continue;
            }

            auto node = get_node(id);
            if (!node || layer < 0 || layer > node->level) {
                continue;
            }

            ordered.push_back({cand.dist, id, std::move(node)});
        }

        return select_neighbors_from_ordered(ordered, M);
    }

    std::vector<int> select_neighbors_heuristic(
        const std::vector<double>& query,
        const std::vector<int>& candidates,
        int M,
        int layer,
        int excludeId = -1
    ) const {
        const size_t nodeCount = node_count();
        auto& seen = visit_marks();
        seen.reset(nodeCount);

        std::vector<OrderedCandidate> ordered;
        ordered.reserve(candidates.size());

        for (int id : candidates) {
            if (id == excludeId || id < 0) {
                continue;
            }
            if (!seen.test_and_set(id)) {
                continue;
            }

            auto node = get_node(id);
            if (!node || layer < 0 || layer > node->level) {
                continue;
            }

            ordered.push_back({l2(query, node->point), id, std::move(node)});
        }

        std::sort(ordered.begin(), ordered.end(), [](const OrderedCandidate& a, const OrderedCandidate& b) {
            if (a.dist != b.dist) return a.dist < b.dist;
            return a.id < b.id;
        });

        return select_neighbors_from_ordered(ordered, M);
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
        )
        .def(
            "node_lock_wait_stats",
            &HNSW::node_lock_wait_stats,
            "Return (wait_count, total_wait_seconds) for contended per-node locks"
        )
        .def(
            "reset_node_lock_wait_stats",
            &HNSW::reset_node_lock_wait_stats,
            "Reset per-node lock wait counters"
        );
}