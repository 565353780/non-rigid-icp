"""Adaptive plateau detection for the fitting loop.

Watches a scalar error series and reports when its rate of decrease flattens
out (i.e. the fit is near convergence). Used to decide *when* to trigger an
adaptive subdivision round, rather than using a fixed iteration budget.
"""

from collections import deque
from typing import Union


class PlateauMonitor(object):
    """Detect when an error series stops decreasing meaningfully.

    A plateau is declared when, for `patience` consecutive checks, the
    improvement between the previous `window`-average and the most recent
    `window`-average is "small". "Small" can be measured two ways and either
    suffices (the union is what makes the test robust across error regimes):

      * RELATIVE: drop / |prev_mean| < `rel_tol`. Natural when the error decays
        toward a non-zero floor (e.g. Adam reaching a residual it cannot beat).
      * ABSOLUTE: drop < `abs_tol` (same units as the fed values; set to None to
        disable). Necessary for a *geometrically* decaying series r_k ~ r0*c^k:
        its relative drop is the ~constant (1-c) forever, so a relative test
        never fires even as the curve visually flattens toward zero -- exactly
        the case for the clamped 0.1*d_i step, where the absolute per-step
        improvement is the meaningful "is it still moving?" signal.

    Comparing window means (not single samples) makes the test robust to
    per-iteration noise.
    """

    def __init__(
        self,
        window: int = 8,
        rel_tol: float = 2e-3,
        patience: int = 2,
        min_updates: int = 0,
        abs_tol: Union[float, None] = None,
    ) -> None:
        self.window = max(1, int(window))
        self.rel_tol = float(rel_tol)
        self.abs_tol = None if abs_tol is None else float(abs_tol)
        self.patience = max(1, int(patience))
        self.min_updates = int(min_updates)
        self._buf = deque(maxlen=2 * self.window)
        self._n = 0
        self._plateau_count = 0
        self.last_rel_drop = float("inf")
        self.last_abs_drop = float("inf")

    def reset(self) -> None:
        """Forget history (call after a topology change, e.g. subdivision)."""
        self._buf.clear()
        self._n = 0
        self._plateau_count = 0
        self.last_rel_drop = float("inf")
        self.last_abs_drop = float("inf")

    def update(self, value: Union[float, int]) -> bool:
        """Push a new error value; return True if a plateau is currently held."""
        self._buf.append(float(value))
        self._n += 1
        if self._n < self.min_updates or len(self._buf) < 2 * self.window:
            return False

        buf = list(self._buf)
        prev_mean = sum(buf[: self.window]) / self.window
        cur_mean = sum(buf[self.window :]) / self.window
        drop = prev_mean - cur_mean
        denom = abs(prev_mean) + 1e-12
        self.last_abs_drop = drop
        self.last_rel_drop = drop / denom

        flat = self.last_rel_drop < self.rel_tol
        if self.abs_tol is not None:
            flat = flat or (self.last_abs_drop < self.abs_tol)
        if flat:
            self._plateau_count += 1
        else:
            self._plateau_count = 0
        return self._plateau_count >= self.patience
