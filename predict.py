import torch
from lm_explorer.lm.gpt2 import GPT2LanguageModel
model_117M = GPT2LanguageModel(model_name='117M')
model_345M = GPT2LanguageModel(model_name='345M')

def predict(previous: str = None, model_name: str = '345M', next: str = None, topk: int = 10):

    #model_name = "345M"

    if model_name == "117M":
        logits = model_117M.predict(previous, next)
    elif model_name == "345M":
        logits = model_345M.predict(previous, next)

    probabilities = torch.nn.functional.softmax(logits)

    best_logits, best_indices = logits.topk(topk)
    if model_name == "117M":
        best_words = [model_117M[idx.item()] for idx in best_indices]
    elif model_name == "345M":
        best_words = [model_345M[idx.item()] for idx in best_indices]
    best_probabilities = probabilities[best_indices].tolist()
    
    best_words = [word.strip() for word in best_words]
    return dict(zip(best_words, best_probabilities))

    '''
    return {
        'logits': best_logits.tolist(),
        'probabilities': best_probabilities,
        'words': best_words,
        'output': previous + (next or "")
    }
    '''
