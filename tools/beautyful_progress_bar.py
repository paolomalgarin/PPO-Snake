import atexit
from rich.text import Text
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    ProgressColumn,
    Task
)

class SpeedColumn(ProgressColumn):
    def __init__(self, unit='it/s', table_column = None):
        super().__init__(table_column)
        self.unit = unit

    def render(self, task: "Task") -> Text:
        speed = task.speed
        if speed is None:
            return Text(f"? {self.unit}", style="progress.data.speed")
        
        return Text(f"{speed:,.0f} {self.unit}", style="progress.data.speed")


class PBar:

    def __init__(self, total, description='', preset=None):
        self.progress = None
        self.task_id = None
        self._closed = False

        match preset:
            case "training":
                self._use_training_preset(total)
            case "eval":
                self._use_eval_preset(total)
            case _:
                # Default case
                self.pbar = Progress()
                self.task_id = self.pbar.add_task("", total=total)
        
        self.set_description(description)

        self.pbar.start()
        atexit.register(self.close)
        

    def update(self, val):
        if not self._closed:
            self.pbar.update(self.task_id, advance=val)
    
    def set_description(self, desc):
        if not self._closed:
            self.pbar.update(self.task_id, description=desc)

    def close(self):
        if not self._closed and self.pbar:
            self.pbar.stop()
            self._closed = True


    def _use_training_preset(self, total):
        colonne = [
            # Description
            TextColumn("{task.description}"),
            # Percentage
            TextColumn("{task.percentage:>3.0f}%"),
            # Progress bar
            BarColumn(
                style="bold #333333", 
                complete_style="bold #2b95fb", 
                finished_style="bold #852eff",
            ),
            # Task count
            TextColumn("{task.completed:,}/{task.total:,}", style="#444444"),
            # Open braket
            TextColumn("["),
            # Time elapsed
            TimeElapsedColumn(),
            # Separator
            TextColumn("<"),
            # Time remaining
            TimeRemainingColumn(),
            # Closed braket
            TextColumn("]"),
            SpeedColumn(),
        ]
        self.pbar = Progress(*colonne)
        self.task_id = self.pbar.add_task("", total=total)
    
    def _use_eval_preset(self, total):
        colonne = [
            # Description
            TextColumn("{task.description}"),
            # Percentage
            TextColumn("{task.percentage:>3.0f}%"),
            # Progress bar
            BarColumn(
                style="bold #333333", 
                complete_style="bold #fbc42b", 
                finished_style="bold #fb2b2e",
            ),
            # Task count
            TextColumn("{task.completed:,}/{task.total:,}", style="#444444"),
            # Open braket
            TextColumn("["),
            # Time elapsed
            TimeElapsedColumn(),
            # Separator
            TextColumn("<"),
            # Time remaining
            TimeRemainingColumn(),
            # Closed braket
            TextColumn("]"),
            SpeedColumn(unit='games/s'),
        ]
        self.pbar = Progress(*colonne)
        self.task_id = self.pbar.add_task("", total=total)
