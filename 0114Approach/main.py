from itertools import combinations
from more_itertools import distribute
points=[(1,5)]
well=1


def partition(pred, iterable):
    'Use a predicate to partition entries into false entries and true entries'
    # partition(is_odd, range(10)) --> 0 2 4 6 8   and  1 3 5 7 9
    t1, t2 = tee(iterable)
    return filterfalse(pred, t1), filter(pred, t2)

for i in distribute(2,[0,1,2,3,4]):
    print(list(i))