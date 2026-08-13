import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
import lightning as L


class PositionEncoding(nn.Module):

    def __init__(self, d_model=2, max_len=6):
        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(start=0, end=max_len, step=1).float().unsqueeze(1)
        embedding_index = torch.arange(start=0, end=d_model, step=2).float()

        div_term = 1/torch.tensor(10000)**(embedding_index/d_model)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, word_embeddings):

        return word_embeddings + self.pe[:word_embeddings.size(0), :]


class Attention(nn.Module):

    def __init__(self, d_model ):
        super().__init__()

        self.w_Q = nn.Linear(d_model, d_model, bias=False)
        self.w_K = nn.Linear(d_model, d_model, bias=False)
        self.w_V = nn.Linear(d_model, d_model, bias=False)

        # these variables below gives us flexibility to input training data in sequences or batches
        # We create some variables to keep track of which indices are for rows and columns
        self.row_dim = 0
        self.col_dim = 1

    def forward(self, encodings_for_q, encodings_for_k, encodings_for_v, mask=None):
        """
        For params encodings for q, k, v - we have this to give us flexibility to let us know where the encodings are coming from.
        For example: in encoder decoder model, encodings for q come from the decoder, and use encodings from encoder for k and v.
        """

        Q = self.w_Q(encodings_for_q)
        K = self.w_K(encodings_for_k)
        V = self.w_V(encodings_for_v)

        #Now calculate attention
        sims = torch.matmul(Q, K.transpose(dim0=self.row_dim, dim1=self.col_dim))
        scaled_sims = sims / torch.tensor(K.size(self.col_dim)**0.5)

        if mask is not None:
            scaled_sims = scaled_sims.masked_fill(mask, value=-1e9)

        attention_percents = F.softmax(scaled_sims, dim=self.col_dim)
        attention_scores = torch.matmul(attention_percents, V)

        return attention_scores


class DecoderOnlyTransformer(L.LightningModule):
    # Inherits lightning module 
    def __init__(self, num_tokens, d_model, max_len):
        super().__init__()

        """
        Params:
            num_tokens: the number of tokens in the vocabulary
            d_model: dimension of model embeddings
            max_len: maximum length of the input plus output
        """

        self.we = nn.Embedding(num_embeddings=num_tokens, embedding_dim=d_model)
        self.pe = PositionEncoding(d_model, max_len)
        self.self_attention = Attention(d_model=d_model)
        self.fc_layer = nn.Linear(in_features=d_model, out_features=num_tokens)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, token_ids):
        word_embeddings = self.we(token_ids)
        position_encoded = self.pe(word_embeddings)

        mask = torch.tril(torch.ones((token_ids.size(dim=0), token_ids.size(dim=0))))
        mask = mask == 0

        self_attention_values = self.self_attention(position_encoded,
                                                    position_encoded,
                                                    position_encoded,
                                                    mask)

        residual_connection_values = position_encoded + self_attention_values

        fc_layer_output = self.fc_layer(residual_connection_values)

        return fc_layer_output

    # Wrote the decoder only transformer, but now we need the code to train it
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.1)

    def training_step(self, batch, batch_idx):
        input_tokens, labels = batch
        output = self.forward(input_tokens[0])
        loss = self.loss(output, labels[0])

        return loss


if __name__ == "__main__":

    from data import inference, inputs, labels, token_to_id, id_to_token

    print("Running our model using inference (only one time! and one batch!)")


    model = DecoderOnlyTransformer(num_tokens=len(token_to_id), d_model=2, max_len=6)
    model_input = inference[0].detach().clone()

    #Aside
    print("Input:\n")
    for id in inference[0]:
        print("\t", id_to_token[id.item()])
    
    input_length = model_input.size(dim=0)
    predictions = model(model_input)
    predicted_id = torch.tensor([torch.argmax(predictions[-1:])])
    predicted_ids = predicted_id

    max_length = 6
    for i in range(input_length, max_length):
        if (predicted_id == token_to_id['<EOS>']):
            break

        model_input = torch.cat((model_input, predicted_id))
        predictions = model(model_input)
        predicted_id = torch.tensor([torch.argmax(predictions[-1:])])
        predicted_ids = torch.cat((predicted_ids, predicted_id))

    print(f"\nPredicted tokens:\n")
    for id in predicted_ids:
        print("\t", id_to_token[id.item()])

    print(f"\nNow we can train:\n")

    # ========= Training Time! ===============
    dataset = TensorDataset(inputs, labels)
    dataloader = DataLoader(dataset)

    trainer = L.Trainer(max_epochs=50)
    trainer.fit(model, train_dataloaders=dataloader)

    print(f"\nInference again!\n")

    print("Input:\n")
    for id in inference[0]:
        print("\t", id_to_token[id.item()])

    model_input = inference[0].detach().clone()
    predictions = model(model_input)
    predicted_id = torch.tensor([torch.argmax(predictions[-1:])])
    predicted_ids = predicted_id

    max_length = 6
    for i in range(input_length, max_length):
        if (predicted_id == token_to_id['<EOS>']):
            break

        model_input = torch.cat((model_input, predicted_id))
        predictions = model(model_input)
        predicted_id = torch.tensor([torch.argmax(predictions[-1:])])
        predicted_ids = torch.cat((predicted_ids, predicted_id))

    print(f"\nPredicted tokens:\n")
    for id in predicted_ids:
        print("\t", id_to_token[id.item()])