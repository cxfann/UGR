import math
import numpy as np
from sklearn.metrics import roc_auc_score

def get_hit_results(predictions, targets, k):

    bs = len(targets)
    
    processed_preds = []
    for p in predictions:

        if "Response:" in p:
            content = p.split("Response:")[-1]
        else:
            raise ValueError("The prediction does not contain 'Response:' delimiter.")
        processed_preds.append(content.strip().replace(" ", ""))

    results_pos = []
    batch_beams_list = []

    for b in range(bs):

        current_beams = processed_preds[b * k : (b + 1) * k]
        target_item = str(targets[b]).strip()
        
        batch_beams_list.append(current_beams)

        try:
            pos = current_beams.index(target_item)
        except ValueError:
            pos = -1
            
        results_pos.append(pos)

    return results_pos, batch_beams_list

def calculate_global_hit(all_hit_pos, k):

    hits = 0
    total = len(all_hit_pos)
    if total == 0: return 0
    
    for pos in all_hit_pos:
        if pos != -1 and pos < k:
            hits += 1
            
    return hits / total

def calculate_global_ndcg(all_hit_pos, k):

    ndcg_sum = 0.0
    total = len(all_hit_pos)
    if total == 0: return 0

    for pos in all_hit_pos:
        if pos != -1 and pos < k:
            ndcg_sum += 1.0 / math.log(pos + 2, 2)
            
    return ndcg_sum / total
