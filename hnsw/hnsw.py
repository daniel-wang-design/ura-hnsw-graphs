from __future__ import annotations

import math
import heapq
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


def _l2_squared(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(np.dot(diff, diff))


@dataclass
class HNSWParams:
    M: int = 16
    ef_construction: int = 200
    ef_search: int = 50
    random_seed: int = 42
    # Neighbor selection strategy: "simple" or "heuristic"
    select: str = "heuristic"


class SimpleHNSWIndex:
    """
    A minimal HNSW implementation focused on correctness + hackability (not peak perf).
    - space: only 'l2' supported (squared L2 distance).
    - Stores vectors in a fixed-capacity numpy array.
    - Graph is stored as adjacency lists per node per level (internal ids).

    Graph access:
        self.graph[internal_id][level] -> List[int] (neighbors at that level)
    """

    def __init__(self, space: str, dim: int):
        if space != "l2":
            raise ValueError("This minimal version supports only space='l2' (squared L2).")
        if dim <= 0:
            raise ValueError("dim must be positive")

        self.space = space
        self.dim = dim

        # Set by init_index
        self.max_elements: int = 0
        self.params: Optional[HNSWParams] = None
        self.M: int = 0
        self.M0: int = 0
        self.ef_construction: int = 0
        self.ef_search: int = 0
        self._rng: Optional[random.Random] = None
        self._mL: float = 0.0  # level generation factor

        # Storage
        self._data: Optional[np.ndarray] = None  # shape (max_elements, dim), float32
        self._labels: Optional[np.ndarray] = None  # external labels, int64
        self._label_to_internal: Dict[int, int] = {}

        # HNSW graph:
        # graph[internal_id] = [neighbors_level0, neighbors_level1, ...] up to node's max level
        self.graph: List[List[List[int]]] = []
        self._cur: int = 0  # number of inserted points

        # Entry point
        self._entry_point: Optional[int] = None
        self._max_level: int = -1

    # -------------------------
    # Public API (hnswlib-ish)
    # -------------------------

    def init_index(
        self,
        max_elements: int,
        ef_construction: int = 200,
        M: int = 16,
        random_seed: int = 42,
        ef_search: int = 50,
        select: str = "heuristic",
    ) -> None:
        if max_elements <= 0:
            raise ValueError("max_elements must be positive")
        if M < 2:
            raise ValueError("M should be >= 2")
        if ef_construction < 2:
            raise ValueError("ef_construction should be >= 2")
        if ef_search < 2:
            raise ValueError("ef_search should be >= 2")
        if select not in ("simple", "heuristic"):
            raise ValueError("select must be 'simple' or 'heuristic'")

        self.max_elements = max_elements
        self.params = HNSWParams(M=M, ef_construction=ef_construction, ef_search=ef_search,
                                random_seed=random_seed, select=select)

        self.M = M
        self.M0 = 2 * M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self._rng = random.Random(random_seed)

        # HNSW level distribution: level = floor(-ln(U) * mL), where mL = 1 / ln(M)
        self._mL = 1.0 / math.log(M)

        self._data = np.empty((max_elements, self.dim), dtype=np.float32)
        self._labels = np.empty((max_elements,), dtype=np.int64)
        self._label_to_internal.clear()
        self.graph.clear()
        self._cur = 0
        self._entry_point = None
        self._max_level = -1

    def set_ef(self, ef_search: int) -> None:
        if ef_search < 2:
            raise ValueError("ef_search should be >= 2")
        self.ef_search = ef_search

    def add_items(self, data: np.ndarray, ids: Optional[Sequence[int]] = None) -> None:
        self._require_inited()
        data = self._as_2d_float32(data)
        n = data.shape[0]

        if ids is None:
            # mimic common hnswlib usage: if ids not provided, labels are 0..n-1 for that batch
            # but we also keep global uniqueness by defaulting to internal ids.
            ids = list(range(self._cur, self._cur + n))
        if len(ids) != n:
            raise ValueError("len(ids) must match number of vectors")

        for vec, label in zip(data, ids):
            print(f"Inserting index {label}")
            self._add_one(vec, int(label))

    def knn_query(self, data: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        self._require_inited()
        if k <= 0:
            raise ValueError("k must be positive")
        if self._cur == 0:
            raise ValueError("Index is empty")

        data = self._as_2d_float32(data)
        if self.ef_search < k:
            raise ValueError(f"ef_search ({self.ef_search}) must be >= k ({k})")

        out_labels = np.empty((data.shape[0], k), dtype=np.int64)
        out_dists = np.empty((data.shape[0], k), dtype=np.float32)

        for i, q in enumerate(data):
            internal, dists = self._search_knn(q, k)
            out_labels[i, :] = np.array([self._labels[idx] for idx in internal], dtype=np.int64)
            out_dists[i, :] = np.array(dists, dtype=np.float32)
        print("Completed query")
        return out_labels, out_dists

    # -------------------------
    # Introspection / graph access
    # -------------------------

    def __len__(self) -> int:
        return self._cur

    def internal_id(self, label: int) -> int:
        """Map external label -> internal id."""
        return self._label_to_internal[label]

    def label_of(self, internal_id: int) -> int:
        """Map internal id -> external label."""
        assert self._labels is not None
        return int(self._labels[internal_id])

    def node_level(self, internal_id: int) -> int:
        return len(self.graph[internal_id]) - 1

    def neighbors_ref(self, internal_id: int, level: int) -> List[int]:
        """
        Returns a *mutable reference* to the adjacency list at (node, level).
        Useful for future locking/custom graph ops.
        """
        return self.graph[internal_id][level]

    # -------------------------
    # Core HNSW logic
    # -------------------------

    def _add_one(self, vec: np.ndarray, label: int) -> None:
        assert self._data is not None and self._labels is not None and self._rng is not None

        if self._cur >= self.max_elements:
            raise ValueError("Index capacity reached (max_elements).")
        if label in self._label_to_internal:
            raise ValueError(f"Duplicate label: {label}")

        u = self._cur
        self._cur += 1

        self._data[u, :] = vec
        self._labels[u] = label
        self._label_to_internal[label] = u

        level_u = self._random_level()
        self.graph.append([[] for _ in range(level_u + 1)])

        # First element: becomes entry point
        if self._entry_point is None:
            self._entry_point = u
            self._max_level = level_u
            return

        ep = self._entry_point
        assert ep is not None

        # 1) Greedy search from top layer down to level_u + 1
        for l in range(self._max_level, level_u, -1):
            ep = self._greedy_search_layer(vec, ep, l)

        # 2) From min(level_u, max_level) down to 0: search_layer + connect
        max_l_to_connect = min(level_u, self._max_level)
        for l in range(max_l_to_connect, -1, -1):
            candidates = self._search_layer(vec, [ep], ef=self.ef_construction, level=l)
            # candidates is list of (internal_id, dist) sorted ascending by dist
            cand_ids = [cid for cid, _ in candidates if cid != u]

            mmax = self.M0 if l == 0 else self.M
            selected = self._select_neighbors(vec, cand_ids, mmax)

            # set u's neighbor list at this level
            self.graph[u][l] = selected

            # add bidirectional edges and prune if needed
            for v in selected:
                self._link(v, u, l, mmax)

            # update entry for next lower level to be the closest found at this level
            if candidates:
                ep = candidates[0][0]

        # If u has a new highest level, make it the new entry point
        if level_u > self._max_level:
            self._entry_point = u
            self._max_level = level_u

    def _search_knn(self, q: np.ndarray, k: int) -> Tuple[List[int], List[float]]:
        assert self._entry_point is not None
        ep = self._entry_point

        # Greedy descent from top to level 1
        for l in range(self._max_level, 0, -1):
            ep = self._greedy_search_layer(q, ep, l)

        # Best-first search at level 0
        results = self._search_layer(q, [ep], ef=self.ef_search, level=0)
        results = results[:k]
        ids = [cid for cid, _ in results]
        dists = [dist for _, dist in results]
        return ids, dists

    def _greedy_search_layer(self, q: np.ndarray, entry: int, level: int) -> int:
        """
        Greedy walk: repeatedly move to a neighbor that improves distance, until local minimum.
        """
        assert self._data is not None
        cur = entry
        cur_dist = _l2_squared(q, self._data[cur])

        while True:
            improved = False
            for nb in self._neighbors(cur, level):
                d = _l2_squared(q, self._data[nb])
                if d < cur_dist:
                    cur = nb
                    cur_dist = d
                    improved = True
            if not improved:
                return cur

    def _search_layer(self, q: np.ndarray, entry_points: List[int], ef: int, level: int) -> List[Tuple[int, float]]:
        """
        Standard HNSW layer search.
        Returns list of (node_id, dist) sorted ascending by dist, size <= ef.
        """
        assert self._data is not None

        visited = set(entry_points)

        # candidates: min-heap of (dist, node)
        candidates: List[Tuple[float, int]] = []
        # results: max-heap simulated via min-heap on (-dist, node)
        results: List[Tuple[float, int]] = []

        dist_cache: Dict[int, float] = {}

        def get_dist(node: int) -> float:
            if node in dist_cache:
                return dist_cache[node]
            d = _l2_squared(q, self._data[node])
            dist_cache[node] = d
            return d

        for ep in entry_points:
            d = get_dist(ep)
            heapq.heappush(candidates, (d, ep))
            heapq.heappush(results, (-d, ep))

        while candidates:
            cdist, cnode = heapq.heappop(candidates)
            worst = -results[0][0]  # largest dist among results

            if cdist > worst:
                break

            for nb in self._neighbors(cnode, level):
                if nb in visited:
                    continue
                visited.add(nb)

                dnb = get_dist(nb)
                worst = -results[0][0]

                if len(results) < ef or dnb < worst:
                    heapq.heappush(candidates, (dnb, nb))
                    heapq.heappush(results, (-dnb, nb))
                    if len(results) > ef:
                        heapq.heappop(results)  # remove worst

        out = [ (node, -negd) for (negd, node) in results ]
        out.sort(key=lambda x: x[1])
        return out

    def _select_neighbors(self, q: np.ndarray, candidates: List[int], m: int) -> List[int]:
        if not candidates or m <= 0:
            return []

        if self.params is None:
            mode = "heuristic"
        else:
            mode = self.params.select

        if mode == "simple":
            return self._select_simple(q, candidates, m)
        return self._select_heuristic(q, candidates, m)

    def _select_simple(self, q: np.ndarray, candidates: List[int], m: int) -> List[int]:
        assert self._data is not None
        scored = [(cid, _l2_squared(q, self._data[cid])) for cid in candidates]
        scored.sort(key=lambda x: x[1])
        return [cid for cid, _ in scored[:m]]

    def _select_heuristic(self, q: np.ndarray, candidates: List[int], m: int) -> List[int]:
        """
        Classic HNSW heuristic: prefer close neighbors but enforce diversity:
        accept candidate c if for all already selected s: dist(c, s) >= dist(q, c).
        """
        assert self._data is not None

        scored = [(cid, _l2_squared(q, self._data[cid])) for cid in candidates]
        scored.sort(key=lambda x: x[1])

        selected: List[int] = []
        for cid, dq_c in scored:
            good = True
            vc = self._data[cid]
            for sid in selected:
                ds = _l2_squared(vc, self._data[sid])
                if ds < dq_c:
                    good = False
                    break
            if good:
                selected.append(cid)
                if len(selected) >= m:
                    break

        # If heuristic was too strict, fill remaining slots by nearest
        if len(selected) < m:
            selected_set = set(selected)
            for cid, _ in scored:
                if cid not in selected_set:
                    selected.append(cid)
                    selected_set.add(cid)
                    if len(selected) >= m:
                        break

        return selected

    def _link(self, v: int, u: int, level: int, mmax: int) -> None:
        """
        Add u to v's adjacency at this level, then prune v's adjacency if needed.
        """
        # v must have this level to link
        if level >= len(self.graph[v]):
            return

        neigh = self.graph[v][level]
        if u not in neigh:
            neigh.append(u)

        if len(neigh) > mmax:
            # prune v's neighborhood with v as the query point
            assert self._data is not None
            qv = self._data[v]
            pruned = self._select_neighbors(qv, neigh, mmax)
            self.graph[v][level] = pruned

    def _neighbors(self, node: int, level: int) -> List[int]:
        if level >= len(self.graph[node]):
            return []
        return self.graph[node][level]

    def _random_level(self) -> int:
        assert self._rng is not None
        u = max(self._rng.random(), 1e-12)
        return int(-math.log(u) * self._mL)

    # -------------------------
    # Helpers / validation
    # -------------------------

    def _require_inited(self) -> None:
        if self._data is None or self._labels is None or self._rng is None or self.params is None:
            raise ValueError("Call init_index(...) first.")

    def _as_2d_float32(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2:
            raise ValueError("data must be 1D or 2D array-like")
        if x.shape[1] != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {x.shape[1]}")
        if x.dtype != np.float32:
            x = x.astype(np.float32, copy=False)
        return x


