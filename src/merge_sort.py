import numpy as np


def merge(*args, key=lambda a: a):
    left, right = args[0] if len(args) == 1 else args
    left_length, right_length = len(left), len(right)
    left_index, right_index = 0, 0
    merged = []
    while left_index < left_length and right_index < right_length:
        if key(left[left_index]) <= key(right[right_index]):
            merged.append(left[left_index])
            left_index += 1
        else:
            merged.append(right[right_index])
            right_index += 1
    if left_index == left_length:
        merged.extend(right[right_index:])
    else:
        merged.extend(left[left_index:])
    return merged


def merge_sort(data, key=lambda a: a):
    length = len(data)
    if length <= 1:
        return data
    middle = length // 2
    left = merge_sort(data[:middle], key)
    right = merge_sort(data[middle:], key)
    return merge(left, right, key=key)


if __name__ == '__main__':
    print(merge_sort([3, 2, 1]))
    rand_arr = np.random.rand(100)
    print(sorted(rand_arr) == merge_sort(rand_arr))
