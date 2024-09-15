import numpy as np

blocking_params = [
    #To properly block, label these from >10
    #(Label, duration (15 minute increments), ToD)
    (10, 3, 6),
    (11, 4, 7),
    (12, 2, (12, 19)), 
    (13, 1, 16)
]
    

def block(arr):
    """
    Blocks certain parts of the day for tasks.    
    """

    for item in blocking_params:
        if type(item[-1]) == tuple:
            for t in item[-1]:
                loc_arr = np.where(arr[:, 1] == t)[0][0]
                arr[loc_arr:loc_arr+item[1], 0] = item[0]
        
        else:
            loc_arr = np.where(arr[:, -1] == item[-1])[0][0]
            arr[loc_arr:loc_arr+item[1], 0] = item[0]
            

    return arr


def time_arr(arr):
    s = 6; incr = 16/len(arr)
    for i in arr:
        i[1] = s
        s+=incr

    return arr