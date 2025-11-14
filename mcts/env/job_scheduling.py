"""A simple sequential job scheduling environment.

State is the set of unscheduled jobs and the current time. Each decision
chooses the next job to schedule. All jobs must eventually be scheduled.
The reward is returned only at the terminal state (after all jobs are scheduled),
and equals the negative total weighted completion time (since we minimize cost).
"""

from typing import List, Tuple
import copy


class JobSchedulingEnv:
    def __init__(self, jobs: List[Tuple[int, float]]):
        """Create a job scheduling env.

        jobs: list of (duration:int, weight:float)
        """
        self.jobs = list(jobs)
        self.n_jobs = len(jobs)
        self.scheduled: List[int] = []  # indices of jobs already scheduled
        self.legal_actions: List[int] = list(range(self.n_jobs))  # unscheduled job indices
        self.current_time = 0

    def clone(self) -> "JobSchedulingEnv":
        return copy.deepcopy(self)
    
    def get_action(self, i) -> Tuple[int, float]:
        return self.jobs[i]

    def get_schedule(self, nodes) -> List[Tuple[int, float]]:
        return [self.jobs[n.action] for n in nodes]

    def legal_mutation_actions(self, original_schedule, mutate_index):
        """Legal actions = any unscheduled job index."""
        all_jobs = set(range(self.n_jobs))
        used_jobs = set(original_schedule)
        used_jobs.remove(original_schedule[mutate_index])
        return list(all_jobs - used_jobs)

    def update_legal_actions(self):
        """Update which jobs remain unscheduled."""
        self.legal_actions = [i for i in range(self.n_jobs) if i not in self.scheduled]

    def step(self, action_index: int) -> float:
        """Apply action: schedule the chosen job next.

        Returns reward only at terminal state (after all jobs scheduled).
        """
        if action_index not in self.legal_actions:
            raise ValueError("Invalid action: job already scheduled")

        duration, weight = self.jobs[action_index]
        self.current_time += duration
        self.scheduled.append(action_index)
        self.update_legal_actions()

        if self.is_terminal():
            # reward = - total weighted completion time (since we minimize cost)
            reward = -self.total_weighted_completion_time()
            return reward
        return 0.0

    def mutate(self, new_schedule: List[int]) -> float:
        """Mutate the schedule (used in search or evolution)."""
        self.scheduled = new_schedule
        self.current_time = sum(self.jobs[i][0] for i in new_schedule)
        self.update_legal_actions()
        return -self.total_weighted_completion_time()

    def is_terminal(self) -> bool:
        return len(self.scheduled) == self.n_jobs

    def total_weighted_completion_time(self) -> float:
        """Compute total weighted completion time."""
        total_time = 0
        total_cost = 0
        for job_idx in self.scheduled:
            duration, weight = self.jobs[job_idx]
            total_time += duration
            total_cost += weight * total_time
        return total_cost

    def __repr__(self) -> str:
        return (f"JobSchedulingEnv(scheduled={self.scheduled}, "
                f"time={self.current_time}, "
                f"remaining={len(self.legal_actions)})")
