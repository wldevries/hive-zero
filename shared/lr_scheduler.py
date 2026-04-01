
from typing import List, Optional, Tuple

class LRScheduler:
    def __init__(
        self,
        lr_schedule: Optional[List[Tuple[int, float]]] = None,
    ):
        self.lr_schedule = lr_schedule

    def get_scheduled_lr(self, generation: int) -> Optional[float]:
        """Return the LR for this generation by linearly interpolating between waypoints."""
        if not self.lr_schedule:
            return None
        # Before first waypoint or at/after last: clamp
        if generation <= self.lr_schedule[0][0]:
            return self.lr_schedule[0][1]
        if generation >= self.lr_schedule[-1][0]:
            return self.lr_schedule[-1][1]
        # Find surrounding waypoints and interpolate
        for i in range(len(self.lr_schedule) - 1):
            it0, lr0 = self.lr_schedule[i]
            it1, lr1 = self.lr_schedule[i + 1]
            if it0 <= generation <= it1:
                t = (generation - it0) / (it1 - it0)
                return lr0 + t * (lr1 - lr0)
        return self.lr_schedule[-1][1]
    
def parse_lr_schedule(schedule_str: Optional[str]) -> Optional[List[Tuple[int, float]]]:
    """Parse a LR schedule string like "0:0.1,50:0.01" into a list of (generation, lr) tuples."""
    if not schedule_str:
        return None
    schedule = []
    for pair in schedule_str.split(","):
        it_str, lr_str = pair.split(":")
        schedule.append((int(it_str), float(lr_str)))
    schedule.sort()
    return schedule

def lr_scheduler_from_string(schedule_str: Optional[str]) -> Optional[LRScheduler]:
    """Convenience to create an LRScheduler directly from a string."""
    lr_schedule = parse_lr_schedule(schedule_str)
    return LRScheduler(lr_schedule=lr_schedule)
