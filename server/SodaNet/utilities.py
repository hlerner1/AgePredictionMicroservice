
def compute_accuracy_labelwise(map_, labels_to_watch):
    ''' Computes the probability of rightly-classified labels'''
    false = 0
    for key in map_.keys():
        if map_.get(key) == labels_to_watch:
            continue
        else:
            false += 1
    return false/len(map_.keys())