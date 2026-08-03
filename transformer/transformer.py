import torch
import copy
import torch.nn as nn
from data import get_source_data, ids_to_words

class WordEmbedding(nn.Module):
    def __init__(self, vocab_size: int, max_seq_length: int, d_model: int):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.d_model = d_model

        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: torch.Tensor - shape = (batch, max_seq_length)
        Returns:
            torch.Tensor - shape = (batch, max_seq_length, d_model)
        """
        if x.shape[-1] > self.max_seq_length:
            raise ValueError("Sequence length exceeds max_seq_length")

        embeddings = self.embed(x)
        return embeddings


class PositionalEncoding(nn.Module):
    def __init__(self, max_seq, d_model):
        super().__init__()
        self.max_seq = max_seq
        self.d_model = d_model

        d_model_tensor = torch.arange(0, self.d_model//2)
        pos = torch.arange(0, self.max_seq).unsqueeze(1)

        # Create a single vector for all positionals
        # Then just add it to the x tensor in forward function
        #batch size is not worried here as every positional row can apply to each batch (batches are not unique in positional encoding)
        positional_tensor = torch.zeros((max_seq, d_model)) # shape = max_seq, d_model
        # every even dim
        positional_tensor[:,0::2] = torch.sin(pos/ 10000**(2*d_model_tensor/d_model))
        # every odd dim
        positional_tensor[:,1::2] = torch.cos(pos/ 10000**(2*d_model_tensor/d_model))

        self.positional_tensor = positional_tensor

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: torch.Tensor - shape = (batch, max_seq, d_model)
        """
        seq_len = x.size(1)
        return x + self.positional_tensor[:seq_len, :]


class Attention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, mask = None):
        """
        Args:
            queries: shape = (batch, num_heads, max_seq, d_head)
            keys: shape = (batch, num_heads, max_seq, d_head)
            values: shape = (batch, num_heads, max_seq, d_head)
        Returns:
            torch.tensor : shape = (batch, max_seq, d_model)
        """
        d_head = queries.shape[-1]
        scores: torch.Tensor = torch.matmul(queries, torch.transpose(keys, -2, -1)) / (d_head**(1/2)) # shape = (batch, num_heads, query_len, max_seq_len)

        # Should do masking here (save for later)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Calculate attention
        attention = torch.matmul(torch.softmax(scores, dim=-1), values) # shape = (batch, num_heads, max_seq, d_head)

        #Concatenate
        attention = attention.transpose(1,2).contiguous()
        attention = attention.view(
            attention.size(0),
            attention.size(1),
            attention.size(2) * attention.size(3) # num_heads * d_head = d_model
        )

        return attention # shape = (batch, max_seq, d_model)


class AttentionLayer(nn.Module):

    def __init__(self, num_heads, d_model):
        super().__init__()

        assert d_model % num_heads == 0
        self.d_head = d_model // num_heads
        self.num_heads = num_heads
        self.d_model = d_model

        # Q, K, and V are weights, so make them linear models to make as such
        self.q_weights = nn.Linear(d_model, d_model, bias=False)
        self.k_weights = nn.Linear(d_model, d_model, bias=False)
        self.v_weights = nn.Linear(d_model, d_model, bias=False) # does not define dimensions, means it does math on the final dimension d_model

        #Norm Layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        #Linear output

        self.linear_out = nn.Sequential(
            nn.Linear(self.d_model, self.d_model, bias=False),

            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model, bias=False)
        )

        #Feed forward
        self.feed_forward = nn.Sequential(
            nn.Linear(self.d_model, self.d_model*4, bias=False),
            nn.ReLU(),
            nn.Linear(self.d_model*4, self.d_model, bias=False)
        )


    def forward(self, x, kv = None, mask = None, feedforward = True):
        """
        Args:
            x: torch.Tensor: shape = (batch, max_seq, d_model)
        Returns:
            torch.tensor
        """
        if kv is None:
            kv = x

        # Have to calculate queries, keys, and values

        Q, K, V = self.q_weights(x), self.k_weights(kv), self.v_weights(kv) # shape = 2 I think

        Q = Q.view(x.size(0), x.size(1), self.num_heads, self.d_head).transpose(1,2)
        K = K.view(kv.size(0), kv.size(1), self.num_heads, self.d_head).transpose(1,2)
        V = V.view(kv.size(0), kv.size(1), self.num_heads, self.d_head).transpose(1,2) # shape = batch, num_heads, max_seq, d_head

        att = Attention()

        attention = att(Q, K, V, mask=mask) # includes concatination

        # another linear output
        attention_out = self.linear_out(attention)

        add_norm1 = self.norm1(attention_out + x)
        if not feedforward:
            residual = add_norm1
            return residual

        feed_forward = self.feed_forward(add_norm1)
        residual = self.norm2(add_norm1 + feed_forward)

        return residual


def createPaddedMask(x_input):
    x_mask = copy.copy(x_input)
    x_mask = (x_mask != 0)

    return x_mask

def createCasualMask(seq_len):
    # shape = (seq_len, seq_len); True/1 = allowed, 0 = blocked
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    return mask

def executeEncoder(x_pe, encoder_layer, mask):
    encoderResiduals = encoder_layer(x_pe, mask=mask)
    return encoderResiduals

def executeDecoder(initial_input, max_seq_len, encoder_output, encoder_padding_mask,
                    embedfn, positionfn, nMaskedLayerAttention, nLayerAttention, output_projection):

    outputs = copy.copy(initial_input)

    for _ in range(max_seq_len - outputs.size(1)):
        seq_len = outputs.size(1)
        casual_mask = createCasualMask(seq_len)
        target_padding_mask = createPaddedMask(outputs)
        reshaped_padding = target_padding_mask.view(target_padding_mask.size(0), 1, 1, seq_len)
        combined_mask = casual_mask.view(1, 1, seq_len, seq_len) * reshaped_padding

        embedded = embedfn(outputs)
        pe_x = positionfn(embedded)

        residual = nMaskedLayerAttention(pe_x, mask=combined_mask, feedforward=False)
        cross_residual = nLayerAttention(residual, kv=encoder_output, mask=encoder_padding_mask, feedforward=True)

        logits = output_projection(cross_residual)
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(torch.softmax(next_token_logits, dim=-1), dim=-1, keepdim=True)

        outputs = torch.cat([outputs, next_token], dim=1)

        if (next_token == 1).all():
            break

    return outputs


def infer_transformer(encoder_id_inputs, max_seq, modules):
    batch_size = encoder_id_inputs.size(0)

    encoder_padding_mask = createPaddedMask(encoder_id_inputs)
    encoder_mask = encoder_padding_mask.view(batch_size, 1, 1, encoder_id_inputs.size(1))

    enc_embedded = modules["encoder_embed"](encoder_id_inputs)
    enc_pe = modules["encoder_pos"](enc_embedded)
    encoder_output = executeEncoder(enc_pe, modules["encoder_layer"], encoder_mask)

    initial_decoder_input = torch.zeros((batch_size, 1)).fill_(1).long()

    output = executeDecoder(
        initial_decoder_input, max_seq, encoder_output, encoder_mask,
        modules["decoder_embed"], modules["decoder_pos"],
        modules["decoder_masked_layer"], modules["decoder_cross_layer"],
        modules["output_projection"],
    )
    return output
        

def train_trainsformer(encoder_id_inputs, target_id_inputs, d_model, num_heads,
                        vocab_encoder, vocab_decoder, num_epochs=10, lr=1e-3):
    """
    Args:
        encoder_id_inputs: torch.Tensor - shape (batch, src_seq_len) - source token ids
        target_id_inputs: torch.Tensor - shape (batch, tgt_seq_len) - target token ids,
                           including a leading start token and trailing end token
        vocab_encoder / vocab_decoder: dict/list - used only for vocab sizes here
    """

    batch_size = encoder_id_inputs.size(0)
    src_seq_len = encoder_id_inputs.size(1)
    
    # Teacher forcing: shift target right for decoder input, shift left for labels
    decoder_input = target_id_inputs[:, :-1]   # (batch, tgt_len - 1)
    labels = target_id_inputs[:, 1:]           # (batch, tgt_len - 1)
    tgt_seq_len = decoder_input.size(1)

    # --- Build all the modules once (these hold the trainable weights) ---
    encoder_embed = WordEmbedding(len(vocab_encoder), src_seq_len, d_model)
    encoder_pos = PositionalEncoding(src_seq_len, d_model)
    encoder_layer = AttentionLayer(num_heads, d_model)

    decoder_embed = WordEmbedding(len(vocab_decoder), tgt_seq_len, d_model)
    decoder_pos = PositionalEncoding(tgt_seq_len, d_model)
    decoder_masked_layer = AttentionLayer(num_heads, d_model)
    decoder_cross_layer = AttentionLayer(num_heads, d_model)
    output_projection = nn.Linear(d_model, len(vocab_decoder), bias=False)

    # Collect all parameters across every module for the optimizer
    all_params = (
        list(encoder_embed.parameters()) + list(encoder_layer.parameters()) +
        list(decoder_embed.parameters()) + list(decoder_masked_layer.parameters()) +
        list(decoder_cross_layer.parameters()) + list(output_projection.parameters())
    )
    optimizer = torch.optim.Adam(all_params, lr=lr)

    # padding_idx=0 -> loss ignores predictions where the label itself is a pad token
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    # --- Masks (built once — same shapes every epoch since no generation loop) ---
    encoder_padding_mask = createPaddedMask(encoder_id_inputs)
    encoder_padding_mask = encoder_padding_mask.view(batch_size, 1, 1, src_seq_len)

    causal_mask = createCasualMask(tgt_seq_len).view(1, 1, tgt_seq_len, tgt_seq_len)
    decoder_padding_mask = createPaddedMask(decoder_input).view(batch_size, 1, 1, tgt_seq_len)
    combined_mask = causal_mask * decoder_padding_mask

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # --- Encoder forward pass ---
        enc_embedded = encoder_embed(encoder_id_inputs)
        enc_pe = encoder_pos(enc_embedded)
        encoder_output = encoder_layer(enc_pe, mask=encoder_padding_mask)

        # --- Decoder forward pass (single pass, whole sequence, teacher forcing) ---
        dec_embedded = decoder_embed(decoder_input)
        dec_pe = decoder_pos(dec_embedded)

        residual = decoder_masked_layer(dec_pe, mask=combined_mask, feedforward=False)
        cross_residual = decoder_cross_layer(residual, kv=encoder_output,
                                              mask=encoder_padding_mask, feedforward=True)

        logits = output_projection(cross_residual)   # (batch, tgt_seq_len, vocab_size)

        # --- Loss ---
        # CrossEntropyLoss expects (N, num_classes) and (N,), so flatten batch & seq dims together
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

        # --- Backward pass + optimizer step ---
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs} - loss: {loss.item():.4f}")

    return {
        "encoder_embed": encoder_embed, "encoder_pos": encoder_pos, "encoder_layer": encoder_layer,
        "decoder_embed": decoder_embed, "decoder_pos": decoder_pos,
        "decoder_masked_layer": decoder_masked_layer, "decoder_cross_layer": decoder_cross_layer,
        "output_projection": output_projection,
    }


if __name__ == "__main__":
    data, vocab_encoder, vocab_decoder, _, _ = get_source_data()

    source_input = [seq["source_id"] for seq in data]
    target_input = [seq["target_id"] for seq in data]

    source_id_to_word = {v: k for k, v in vocab_encoder.items()}
    target_id_to_word = {v: k for k, v in vocab_decoder.items()}

    x_input = torch.tensor(source_input)
    target_input = torch.tensor(target_input)

    max_seq = x_input.size(1)
    d_model = 4
    num_heads = 2

    # --- Infer BEFORE training (random weights — expect gibberish) ---
    print("=== Before training ===")
    untrained_modules = {
        "encoder_embed": WordEmbedding(len(vocab_encoder), max_seq, d_model),
        "encoder_pos": PositionalEncoding(max_seq, d_model),
        "encoder_layer": AttentionLayer(num_heads, d_model),
        "decoder_embed": WordEmbedding(len(vocab_decoder), max_seq, d_model),
        "decoder_pos": PositionalEncoding(max_seq, d_model),
        "decoder_masked_layer": AttentionLayer(num_heads, d_model),
        "decoder_cross_layer": AttentionLayer(num_heads, d_model),
        "output_projection": nn.Linear(d_model, len(vocab_decoder), bias=False),
    }
    before_output = infer_transformer(x_input, max_seq, untrained_modules)
    print(before_output)

    inputs = ids_to_words(x_input, source_id_to_word)
    outputs = ids_to_words(before_output, target_id_to_word)

    print(inputs)
    print(outputs)

    # --- Train ---
    print("=== Training ===")
    trained_modules = train_trainsformer(
        x_input, target_input, d_model, num_heads,
        vocab_encoder, vocab_decoder, num_epochs=500, lr=1e-3
    )

    # --- Infer AFTER training ---
    print("=== After training ===")
    after_output = infer_transformer(x_input, max_seq, trained_modules)
    print(after_output)

    print("translation examples:")
    

    inputs = ids_to_words(x_input, source_id_to_word)
    outputs = ids_to_words(after_output, target_id_to_word)

    print(inputs)
    print(outputs)

