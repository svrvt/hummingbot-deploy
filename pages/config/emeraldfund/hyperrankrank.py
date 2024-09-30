from pprint import pprint


def hyperrankrank(items, K, sorters, breakdown=False):
    """
    HyperRankRank algorithm.

    Parameters
    ----------
    items : list
        List of items to rank.
    k : int
        Number of items to rank.
    sorters : list
        List of sorters to use.

    Returns
    -------
    list
        List of ranked items.
    """

    k = K * len(sorters)
    if breakdown:
        breakdown_result = []
    for sorter in sorters:
        items = sorted(items, key=sorter)
        items = items[:k]
        if breakdown:
            breakdown_result.append(items)
        k -= K
    if breakdown:
        return breakdown_result
    return items


def test():
    import random
    from functools import cmp_to_key

    demo_items = []
    for i in range(100):
        demo_items.append(
            {
                "profit": i,
                "time_elapsed": 100 - i,
                "max_drawdown": random.random() * 10,
            }
        )

    sorters = [
        cmp_to_key(lambda x, y: y["profit"] - x["profit"]),
        cmp_to_key(lambda x, y: x["time_elapsed"] - y["time_elapsed"]),
        cmp_to_key(lambda x, y: y["max_drawdown"] - x["max_drawdown"]),
    ]
    ranked_items = hyperrankrank(demo_items, 10, sorters)
    print(ranked_items)
    assert (
        len(ranked_items) == 10
    ), f"Wrong number of items in ranked_items: {len(ranked_items)}"

    ranked_items = hyperrankrank(demo_items, 10, sorters, breakdown=True)
    pprint(ranked_items)
    assert (
        len(ranked_items[2]) == 10
    ), f"Wrong number of items in ranked_items: {len(ranked_items)}"

    assert (
        len(ranked_items[0]) == 30
    ), f"Wrong number of items in ranked_items: {len(ranked_items)}"


if __name__ == "__main__":
    test()
