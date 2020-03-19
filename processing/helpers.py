'''
Authors: Pablo Prietz, Kerstin Pieper
'''
import itertools
import pathlib
import enum
import typing as T
import pandas as pd

from processing.shared.select_data import select_from_data, split
from processing.shared.markers_example import Periods


# definition of conditions
class Condition(enum.Enum):
    H = "hard"
    C = "control"
    E = "easy"

# order in which condition blocks were presented
class ConditionOrder(T.NamedTuple):
    block0: Condition
    block1: Condition
    block2: Condition


class Helper:

    def get_bads(self):
        """Check for bad channels in lookup file and return them as list

        :param self:
        :return:
        bad_list : list
        bad_list_list : list
        """
        script_parent_folder = pathlib.Path(__file__).parent
        vp_bads_lookup_loc = script_parent_folder / "bads_lookup_example.csv"
        vp_bads_lookup = pd.read_csv(vp_bads_lookup_loc, index_col="VP-Code")
        bad_raw = vp_bads_lookup.loc[self.vp_code]
        bad_list_list = bad_raw.values.tolist()
        if len(bad_list_list) > 1:
            bad_list = list(itertools.chain(*bad_list_list))
            return bad_list
        else:
            return bad_list_list

    def condition_order(self) -> ConditionOrder:
        """Check for order in which different test conditions were applied.

        :return:
        """
        script_parent_folder = pathlib.Path(__file__).parent
        vp_condition_lookup_loc = script_parent_folder / "condition_lookup_example.csv"
        vp_condition_lookup = pd.read_csv(vp_condition_lookup_loc, index_col="VP-Code")
        conditions_raw = vp_condition_lookup.loc[self.vp_code]
        conditions = [Condition(cond) for cond in conditions_raw]
        condition_order = ConditionOrder(*conditions)
        return condition_order

    def extract_periods(block, num_task_subblocks=6):
        data, markers = block
        basline_h = next(select_from_data(data, markers, Periods.baseline_h))
        basline_l = next(select_from_data(data, markers, Periods.baseline_l))

        task, _ = next(select_from_data(data, markers, Periods.task))
        subblocks = split(task, num_periods=num_task_subblocks)
        subblocks = map(lambda groupby: groupby[1], subblocks)

        baselines = (basline_h[0], basline_l[0])
        all_periods = itertools.chain(baselines, subblocks)

        period_names = ["baseline_h", "baseline_l"]
        period_names += [f"task_subblock_{idx}" for idx in range(num_task_subblocks)]
        period_concat = pd.concat(
            all_periods, keys=period_names, names=["period", data.index.name]
        )
        return period_concat