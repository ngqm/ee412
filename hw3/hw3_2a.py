from scipy.special import binom 


def get_s(n, d, t):
    """
    Given n and d, return the greatest int s
    such that 
    n (d choose t) / (n choose t) >= s

    Parameters:
        n: int
        d: int
        t: int
    
    Returns:
        s: int
    """
    return int(n * binom(d, t) / binom(n, t))
    # return int(n * (d/n)**t)


def get_all_pairs(n, d):
    """
    Given n and d, find all pairs (t, s)
    such that s is from get_s(n, d, t)

    Parameters:
        n: int
        d: int
    
    Returns:
        pairs: list[(int, int)]
    """
    pairs = []
    for t in range(1, d + 1):
        s = get_s(n, d, t)
        if s > 0 and t <= s:
            pairs.append((t, s))
    return pairs


print(get_all_pairs(200, 150))