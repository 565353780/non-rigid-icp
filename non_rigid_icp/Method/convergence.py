"""Adaptive plateau detection for the fitting loop.

Watches a scalar error series and reports when its rate of decrease flattens
out (i.e. the fit is near convergence). Used to decide *when* to trigger an
adaptive subdivision round, rather than using a fixed iteration budget.
"""

from collections import deque
from typing import Union


class PlateauMonitor(object):
    """Detect when an error series stops decreasing meaningfully.

    A plateau is declared when the relative improvement between the previous
    `window`-average and the most recent `window`-average stays below
    `rel_tol` for `patience` consecutive checks. Comparing window means (not
    single samples) makes the test robust to per-iteration noise.
    """

    def __init__(
        self,
        window: int = 8,
        rel_tol: float = 2e-3,
        patience: int = 2,
        min_updates: int = 0,
    ) -> None:
        self.window = max(1, int(window))
        self.rel_tol = float(rel_tol)
        self.patience = max(1, int(patience))
        self.min_updates = int(min_updates)
        self._buf = deque(maxlen=2 * self.window)
        self._n = 0
        self._plateau_count = 0
        self.last_rel_drop = float("inf")

    def reset(self) -> None:
        """Forget history (call after a topology change, e.g. subdivision)."""
        self._buf.clear()
        self._n = 0
        self._plateau_count = 0
        self.last_rel_drop = float("inf")

    def update(self, value: Union[float, int]) -> bool:
        """Push a new error value; return True if a plateau is currently held."""
        self._buf.append(float(value))
        self._n += 1
        if self._n < self.min_updates or len(self._buf) < 2 * self.window:
            return False

        buf = list(self._buf)
        prev_mean = sum(buf[: self.window]) / self.window
        cur_mean = sum(buf[self.window :]) / self.window
        denom = abs(prev_mean) + 1e-12
        self.last_rel_drop = (prev_mean - cur_mean) / denom

        if self.last_rel_drop < self.rel_tol:
            self._plateau_count += 1
        else:
            self._plateau_count = 0
        return self._plateau_count >= self.patience
