from skchange.costs.l2_cost import init_l2_cost, l2_cost

VALID_COSTS = ["l2"]


def cost_factory(cost_name: str):
    if cost_name == "l2":
        return l2_cost, init_l2_cost
    else:
        raise ValueError(
            f"cost_name={cost_name} not recognized. Must be one of {', '.join(VALID_COSTS)}"
        )
