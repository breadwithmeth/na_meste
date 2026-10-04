"""Навигационный граф поверх геометрии помещения.

Узлы (двери, коридоры, комнаты, лестницы, лифты, входы/выходы) и рёбра с
расстоянием и допустимым временем прохода. Дейкстра даёт кратчайший маршрут;
межэтажные переходы разрешены только через узлы stairs/elevator (ТЗ §12):
ребро между узлами разных этажей без такого узла игнорируется.
"""
import heapq
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

# типы узлов, через которые разрешён переход между этажами
FLOOR_TRANSITION_TYPES = ("stairs", "elevator")

# во сколько раз реальный проход дольше идеального, если max_time не задан
DEFAULT_MAX_TIME_FACTOR = 10.0


@dataclass(frozen=True)
class NodeInfo:
    id: int
    floor_id: int
    ntype: str
    name: Optional[str]
    x: float
    y: float


@dataclass(frozen=True)
class EdgeInfo:
    id: int
    from_id: int
    to_id: int
    distance: float
    min_time: Optional[float]
    max_time: Optional[float]


class NavigationGraph:
    """Неизменяемый граф: строится из узлов/рёбер БД при загрузке мира,
    после CRUD пересоздаётся целиком (кеш путей живёт в экземпляре)."""

    def __init__(self, nodes: list[NodeInfo], edges: list[EdgeInfo],
                 max_speed: float = 2.0):
        self._nodes: dict[int, NodeInfo] = {n.id: n for n in nodes}
        self._adj: dict[int, list[tuple[int, EdgeInfo]]] = defaultdict(list)
        skipped = 0
        for e in edges:
            a, b = self._nodes.get(e.from_id), self._nodes.get(e.to_id)
            if a is None or b is None:
                continue
            # межэтажное ребро — только через stairs/elevator
            if (a.floor_id != b.floor_id
                    and a.ntype not in FLOOR_TRANSITION_TYPES
                    and b.ntype not in FLOOR_TRANSITION_TYPES):
                skipped += 1
                continue
            self._adj[e.from_id].append((e.to_id, e))   # рёбра двусторонние
            self._adj[e.to_id].append((e.from_id, e))
        if skipped:
            import logging
            logging.getLogger("app.spatial.navigation").warning(
                "Навигация: проигнорировано %d межэтажных рёбер без "
                "stairs/elevator", skipped)
        self._max_speed = max(0.5, max_speed)
        self._paths: dict[tuple[int, int], Optional[list[int]]] = {}

    # ------------------------------------------------------------ доступ

    @property
    def empty(self) -> bool:
        return not self._nodes

    def node(self, node_id: int) -> Optional[NodeInfo]:
        return self._nodes.get(node_id)

    def nodes(self) -> list[NodeInfo]:
        return list(self._nodes.values())

    def edges(self) -> list[EdgeInfo]:
        seen, out = set(), []
        for lst in self._adj.values():
            for _nid, e in lst:
                if e.id not in seen:
                    seen.add(e.id)
                    out.append(e)
        return out

    def nearest_node(self, x: float, y: float,
                     floor_id: Optional[int]) -> Optional[NodeInfo]:
        """Ближший узел на этаже (евклид). floor_id=None — по всем этажам."""
        best, best_d = None, float("inf")
        for n in self._nodes.values():
            if floor_id is not None and n.floor_id != floor_id:
                continue
            d = math.hypot(n.x - x, n.y - y)
            if d < best_d:
                best, best_d = n, d
        return best

    # ------------------------------------------------------------ маршруты

    def shortest_path(self, from_id: int, to_id: int) -> Optional[list[int]]:
        """Кратчайший по расстоянию маршрут (список id узлов) или None."""
        if from_id not in self._nodes or to_id not in self._nodes:
            return None
        if from_id == to_id:
            return [from_id]
        key = (from_id, to_id)
        if key in self._paths:
            return self._paths[key]
        path = self._dijkstra(from_id, to_id)
        # кешируем в обе стороны (граф неориентированный)
        self._paths[key] = path
        self._paths[(to_id, from_id)] = list(reversed(path)) if path else None
        return path

    def _dijkstra(self, from_id: int, to_id: int) -> Optional[list[int]]:
        dist = {from_id: 0.0}
        prev: dict[int, int] = {}
        heap = [(0.0, from_id)]
        while heap:
            d, cur = heapq.heappop(heap)
            if cur == to_id:
                break
            if d > dist.get(cur, float("inf")):
                continue
            for nxt, edge in self._adj.get(cur, ()):
                nd = d + edge.distance
                if nd < dist.get(nxt, float("inf")):
                    dist[nxt] = nd
                    prev[nxt] = cur
                    heapq.heappush(heap, (nd, nxt))
        if to_id not in dist:
            return None
        path, cur = [to_id], to_id
        while cur != from_id:
            cur = prev[cur]
            path.append(cur)
        path.reverse()
        return path

    def path_length(self, from_id: int, to_id: int) -> Optional[float]:
        """Длина кратчайшего маршрута в метрах (None — маршрута нет)."""
        path = self.shortest_path(from_id, to_id)
        if path is None:
            return None
        return sum(
            edge.distance
            for i in range(len(path) - 1)
            for _nid, edge in self._adj.get(path[i], ())
            if _nid == path[i + 1]
        )

    def travel_window(self, from_id: int, to_id: int
                      ) -> tuple[Optional[float], Optional[float]]:
        """(min_time, max_time) прохода по кратчайшему маршруту, сек.

        min_time ребра = заданное значение либо distance / max_speed.
        Возвращает (None, None), если маршрута нет.
        """
        path = self.shortest_path(from_id, to_id)
        if path is None:
            return None, None
        min_t, max_t = 0.0, 0.0
        for i in range(len(path) - 1):
            edge = next((e for nid, e in self._adj.get(path[i], ())
                         if nid == path[i + 1]), None)
            if edge is None:
                return None, None
            lo = edge.min_time if edge.min_time is not None \
                else edge.distance / self._max_speed
            hi = edge.max_time if edge.max_time is not None \
                else lo * DEFAULT_MAX_TIME_FACTOR
            min_t += lo
            max_t += hi
        return min_t, max_t
