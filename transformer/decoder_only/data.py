import torch

token_to_id = {
    "what": 0,
    "is": 1,
    "statquest": 2,
    "awesome": 3,
    "<EOS>": 4,
    "who": 5,
    "oliver": 6,
    "you": 7
}

id_to_token = dict(map(reversed, token_to_id.items()))

inference = torch.tensor([
    token_to_id["what"],
    token_to_id["is"],
    token_to_id["statquest"],
    token_to_id["<EOS>"]
]),

inputs = torch.tensor([[
    token_to_id["what"],
    token_to_id["is"],
    token_to_id["statquest"],
    token_to_id["<EOS>"],
    token_to_id["awesome"]
],
[
    token_to_id["statquest"],
    token_to_id["is"],
    token_to_id["what"],
    token_to_id["<EOS>"],
    token_to_id["awesome"]
],
[
    token_to_id["who"],
    token_to_id["is"],
    token_to_id["oliver"],
    token_to_id["<EOS>"],
    token_to_id["oliver"],
    token_to_id["is"],
    token_to_id["you"]
]
]
)

labels = torch.tensor([[
    token_to_id["is"],
    token_to_id["statquest"],
    token_to_id["<EOS>"],
    token_to_id["awesome"],
    token_to_id["<EOS>"]
],
[
    token_to_id["is"],
    token_to_id["what"],
    token_to_id["<EOS>"],
    token_to_id["awesome"],
    token_to_id["<EOS>"]
],
[
    token_to_id["is"],
    token_to_id["oliver"],
    token_to_id["<EOS>"],
    token_to_id["oliver"],
    token_to_id["is"],
    token_to_id["you"],
    token_to_id["<EOS>"]
]
]
)

def get_inputs():
    return inputs

def get_labels():
    return labels