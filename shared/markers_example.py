'''
Authors: Pablo Prietz, Kerstin Pieper
'''

import enum


class Markers(enum.Enum):
    """
    Definition of markers which are set in during experimental presentation.
    They have also be defined for Psychopy lsl outlet.
    """
    block_start = 1
    baseline_high_start = 2
    baseline_high_end = 3
    baseline_low_start = 4
    baseline_low_end = 5
    stimulus_on = 6
    response = 7
    task_start = 8
    task_end = 9
    block_end = 10


class Periods(enum.Enum):
    """Period Definitions

    Each period consists of a list of start and end marker pairs
    """

    block = [(Markers.block_start, Markers.block_end)]
    baseline_h = [(Markers.baseline_high_start, Markers.baseline_high_end)]
    baseline_l = [(Markers.baseline_low_start, Markers.baseline_low_end)]
    task = [(Markers.task_start, Markers.task_end)]


if __name__ == "__main__":
    from pprint import pprint

    pprint(list(Periods))
