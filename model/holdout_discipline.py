"""Make held-out discipline structural instead of intentional.

THE FAILURE THIS PREVENTS. Every calibrate_* script in this repo
selects on training error -- the argmin lines are honest. But they
also printed each candidate's TEST score on the same line, so whoever
ran them saw the full held-out curve before deciding whether to accept
the argmin, widen the grid, or re-run. Intent is not a control.

It cost something real. model/ratings.py shipped half_life=100 because
the operator saw that the train argmin (a short half-life) scored
worst on the 2023 test set and overrode it. Graded later on 2024-25 --
seasons no calibration had touched -- the short half-life was better
and 100 was the worst of eight candidates
(model/revalidate_half_life_2024_25.py). The train argmin had been
right; the test column is what talked us out of it.

THE CONTROL. A vault holds each candidate's metrics and will not hand
back a held-out number until a selection has been made on training
data alone. Reading it early raises. The scripts therefore cannot
print what they used to print, whoever runs them.

    vault = HoldoutVault(select_by="train_mae", minimize=True)
    for hl in candidates:
        vault.record(hl, train_mae=..., test_mae=..., test_acc=...)
    best = vault.select()          # train-only argmin; unlocks the rest
    print(vault.held_out(best))    # legal only after select()

This does not undo past selections. It stops the next one.
"""


class HoldoutLeak(Exception):
    """Raised when held-out metrics are read before a selection is made."""


class HoldoutVault:
    # Anything whose name starts with one of these is held-out and stays
    # sealed until select() runs.
    SEALED_PREFIXES = ("test_", "holdout_", "held_out_")

    def __init__(self, select_by, minimize=True):
        self.select_by = select_by
        self.minimize = minimize
        self._rows = {}
        self._selected = None

    def record(self, candidate, **metrics):
        if self.select_by not in metrics:
            raise ValueError(
                f"candidate {candidate!r} recorded without {self.select_by!r}, "
                f"which is the field selection reads")
        if self._is_sealed(self.select_by):
            raise HoldoutLeak(
                f"select_by={self.select_by!r} is a held-out metric. Selection "
                f"must read training data only.")
        self._rows[candidate] = dict(metrics)

    def _is_sealed(self, name):
        return name.startswith(self.SEALED_PREFIXES)

    def training_table(self):
        """The only thing safe to look at before selecting."""
        return {c: {k: v for k, v in m.items() if not self._is_sealed(k)}
                for c, m in self._rows.items()}

    def select(self):
        if not self._rows:
            raise ValueError("nothing recorded")
        pick = (min if self.minimize else max)
        self._selected = pick(self._rows, key=lambda c: self._rows[c][self.select_by])
        return self._selected

    def held_out(self, candidate=None):
        """Held-out metrics. Legal only after select(), and by default
        only for the candidate selection actually chose."""
        if self._selected is None:
            raise HoldoutLeak(
                "held-out metrics read before select(). This is the leak that "
                "shipped half_life=100; see the module docstring.")
        candidate = self._selected if candidate is None else candidate
        return {k: v for k, v in self._rows[candidate].items() if self._is_sealed(k)}

    def compare_to_incumbent(self, incumbent):
        """A shipped default may be graded beside the selected candidate --
        that is a comparison, not a search, so it is allowed once a
        selection exists."""
        if self._selected is None:
            raise HoldoutLeak("compare_to_incumbent() before select()")
        return {"selected": self._selected,
                "selected_held_out": self.held_out(self._selected),
                "incumbent": incumbent,
                "incumbent_held_out": self.held_out(incumbent)}
