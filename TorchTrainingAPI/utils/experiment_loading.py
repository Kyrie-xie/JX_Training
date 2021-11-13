import torch as t
import os

def best_model_laoding(result_path):
    model_path = os.path.join(result_path, 'Model_Checkpoint')
    
    # assert os.path.exists(model_path)
    
    res = t.load(os.path.join(model_path, 'profile'))
        
    model = res['model']
    optimizer = res['optimizer']
    
    res =  t.load(os.path.join(model_path, 'best_state_dict'))
    
    model.load_state_dict(res['model_state_dict'])
    optimizer.load_state_dict(res['optimizer_state_dict'])
    
    return {'model': model,
            'optimizer': optimizer}