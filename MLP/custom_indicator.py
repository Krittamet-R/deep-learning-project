def get_RSI(dataset, interval):
    RSI = []
    for i in range(len(dataset)):
        if i < interval:
            RSI.append([50.0]) 
            continue
            
        gain = 0
        loss = 0
        
        for j in range(i - interval + 1, i + 1):
            change = dataset[j]["Close"] - dataset[j-1]["Close"]
            if change > 0:
                gain += change
            else:
                loss += abs(change)
        
        avg_gain = gain / interval
        avg_loss = loss / interval
        
        if avg_loss == 0:
            value = 100.0  # Price only went up
        else:
            rs = avg_gain / avg_loss
            value = 100 - (100 / (1 + rs))
            
        RSI.append([value])
        
    return RSI

def get_VWAP(dataset, interval):
    VWAP = []
    for i in range(len(dataset)):
        if i < interval:
            VWAP.append([dataset[i]["Close"]])
            continue

        weighted_price_sum = 0.0
        volume_sum = 0.0

        for j in range(i - interval + 1, i + 1):
            high = dataset[j]["High"]
            low = dataset[j]["Low"]
            close = dataset[j]["Close"]
            volume = dataset[j]["Volume"]

            typical_price = (high + low + close) / 3.0
            weighted_price_sum += typical_price * volume
            volume_sum += volume

        if volume_sum == 0:
            value = dataset[i]["Close"]
        else:
            value = weighted_price_sum / volume_sum

        VWAP.append([value])

    return VWAP


