'''
Calculate the congestion rate.
'''

import pandas as pd
import numpy as np
pd.options.display.max_rows = 50

# %% -----------------------------------------

def congestion_rate(data=None,bldg = "",date=""):
    '''
    '''

    # subset the data to only contain the relevant fields
    tmp = (data.query(f"bldg == '{bldg}'")
           [(date >= d.start_date) & (date <= d.end_date)]
           [['days','start_time','end_time','max_enrl']])

    # All points in time in a day
    tpnts = np.array([[t,0.0,0.0] for t in range(0,24*60 + 1)])

    # Store the average by day
    out = []

    for day in ["M","T","W","R","F","S","U"]:

        # Loop through all classes and calculate congestion at each minute period
        for val in tmp[tmp.days.str.contains(day)].drop('days',axis=1).values:

            start = val[0]
            end = val[1]
            n_in_class = val[2]

            # Count the number of students in class
            tpnts[(tpnts[:,0] >= start) & (tpnts[:,0] <= end),1] += n_in_class

            # Decays for students entering and leaving class
            lead_up = np.array([n_in_class*(t/10) for t in range(1,11)])
            lead_out = np.array([n_in_class*((11-t)/10 ) for t in range(1,11)])

            # Count the number of students out of class
            tpnts[(tpnts[:,0] >= (start - 10)) & (tpnts[:,0] <= (start -1) ),2] += lead_up
            tpnts[(tpnts[:,0] >= (end + 1)) & (tpnts[:,0] <= (end + 10) ),2] += lead_out

        out.append([day,tpnts[:,2].mean()])

    # calculate the average number of students in the hall by minute.
    return out


# %% -----------------------------------------

orig_sched = pd.read_csv("output_data/course_schedule_data.csv")
adj_sched = pd.read_csv("output_data/BSB.csv")

# Congestion rate for current schedule
congestion_rate(orig_sched,bldg="BSB",date="2020-08-26")

(16.14-15.19)/16.14
# Congestion rate for the adjusted schedule
congestion_rate(adj_sched,bldg="BSB",date="2020-08-26")







d.columns
d[['start_date','end_date','start_time','end_time','max_enrl']].values
