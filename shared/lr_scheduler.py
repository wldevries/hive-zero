
from typing import List, Optional, Tuple

class LRScheduler:
    def __init__(
        self,
        lr_schedule: Optional[List[Tuple[int, float]]] = None,
        mode: str = "linear",
    ):
        self.lr_schedule = lr_schedule
        self.mode = mode

    def get_scheduled_lr(self, generation: int) -> Optional[float]:
        """Return the LR for this generation.

        In 'linear' mode, linearly interpolate between waypoints.
        In 'step' mode, hold each waypoint's LR until the next waypoint.
        """
        if not self.lr_schedule:
            return None
        # Before first waypoint or at/after last: clamp
        if generation <= self.lr_schedule[0][0]:
            return self.lr_schedule[0][1]
        if generation >= self.lr_schedule[-1][0]:
            return self.lr_schedule[-1][1]
        # Find surrounding waypoints
        for i in range(len(self.lr_schedule) - 1):
            it0, lr0 = self.lr_schedule[i]
            it1, lr1 = self.lr_schedule[i + 1]
            if it0 <= generation < it1:
                if self.mode == "step":
                    return lr0
                # linear interpolation
                t = (generation - it0) / (it1 - it0)
                return lr0 + t * (lr1 - lr0)
        return self.lr_schedule[-1][1]

def parse_lr_schedule(schedule_str: Optional[str]) -> tuple[str, Optional[List[Tuple[int, float]]]]:
    """Parse a LR schedule string into (mode, waypoints).

    Formats:
        "0:0.02,100:0.005"       — linear interpolation (default)
        "s:0:0.02,30:0.01"       — stepped (hold each LR until next waypoint)
    """
    if not schedule_str:
        return "linear", None
    mode = "linear"
    # Check for mode prefix
    if schedule_str.startswith("s:"):
        mode = "step"
        schedule_str = schedule_str[2:]
    elif schedule_str.startswith("l:"):
        mode = "linear"
        schedule_str = schedule_str[2:]
    schedule = []
    for pair in schedule_str.split(","):
        it_str, lr_str = pair.split(":")
        schedule.append((int(it_str), float(lr_str)))
    schedule.sort()
    return mode, schedule

def lr_scheduler_from_string(schedule_str: Optional[str]) -> Optional[LRScheduler]:
    """Convenience to create an LRScheduler directly from a string."""
    mode, lr_schedule = parse_lr_schedule(schedule_str)
    return LRScheduler(lr_schedule=lr_schedule, mode=mode)
