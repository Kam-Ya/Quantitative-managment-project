def cleanData(frame):
    isNan = frame[frame.isna().any(axis=1)]

    # data doesnt need to be cleaned if this is the case
    if (isNan == 0):
        return
    
    frame.dropna()

