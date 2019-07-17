def Evaluate_MOMPTime(raw_data, plotting, verbatim):
    
#    MOMP_data = Evaluate_MOMPTime(Raw_data, plotting, verbatim)

#    MOMP_data.idx_momp.Edge (and MOMP_data.idx_momp.MOMP)
#                    = -1 for too short,
#                     Inf for survivors,
#                  frame# for MOMP at Time(frame#)
#                      -2 for unclassified

#    MOMP_data.idx_momp.lostidx = idx for tracking lost.

#    MOMP_data.Traj.Edge = edge (smoothed)
#    MOMP_data.Traj.Area = area (smoothed)
#    MOMP_data.Traj.Prob = edge/area scoring


    windowSize = round(45/raw_data.Timestep)         # windows size for filtering
    division_time = round(250/raw_data.Timestep)     # assumed division time
    edge_cutoff = 100     # cutoff for the edge criterion
    MOMPchannel_cutoff = 70          # cutoff for the MOMP channel
    margin_delta = .1      # hyteresis for the scoring of the derivate
    p_cutoff_high = 1.1    # lower cutoff for the score to be classified as MOMP
    p_cutoff_low = .5      # upper cutoff for the score to be classified as survival
    MOMP_cutoff = raw_data.Ntimepoints-windowSize
    Delta_Death_MOMP = 4   # number of frame for MOMP to occur before cell change shape

    MOMP_data = raw_data
    MOMP_data.paremeters = {'MOMPwindowSize': [windowSize], 'division_time' : [division_time], 
                           'edge_cutoff' : [edge_cutoff], 'MOMPchannel_cutoff' : [MOMPchannel_cutoff], 
                           'margin_delta' : [margin_delta], 'p_cutoff_high' : [p_cutoff_high], 
                           'p_cutoff_low' : [p_cutoff_low], 'MOMP_cutoff' : [MOMP_cutoff], 
                           'Delta_Death_MOMP' : [Delta_Death_MOMP]}
    
# strcmpi compare strings 
    str = 'MOMP_data.RFPchannel'
    if str.lower().find('MOMP'.lower()):
        useMOMPchannel = True
    else:
        useMOMPchannel = False
    idx_momp = {'MOMPchannel' : {},'Edge':{},'track_lost':{}}
    Traj = {'Prob':{},'Edge':{},'MOMPchannel':{},'Area':{}}
    
    WellLabel = MOMP_data.WellLabel
    Wells = MOMP_data.Wells[:,3]
    rawdata = MOMP_data.rawdata
    parameters = MOMP_data.parameters

