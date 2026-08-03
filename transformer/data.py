
data = [
    {
        "source": "Hello my name is Oliver",
        "target": "Dziendobry, mam na imje Oliver"
    },

    {
        "source": "What is your name?",
        "target": "Jak masz na imje?"
    },

    {
        "source": "You have a name?",
        "target": "masz imje?"
    }
]



def get_source_data():

    vocab_encoder = {"PAD": 0, "EOS": 1,}
    vocab_decoder = {"PAD": 0, "EOS": 1,}

    input_vocab = {}
    output_vocab = {}

    max_len = 0
    vocab_id_encoder = 2
    vocab_id_decoder = 2

    for sequence in data:
        source = sequence["source"].lower()
        target = sequence["target"].lower()

        source_id = []
        target_id = []

        for token in source.split():
            if token not in vocab_encoder.keys():
                vocab_encoder[token] = vocab_id_encoder
                input_vocab[token] = vocab_id_encoder
                vocab_id_encoder += 1
            
            source_id.append(vocab_encoder[token])

        for token in target.split():
            if token not in vocab_decoder.keys():
                vocab_decoder[token] = vocab_id_decoder
                output_vocab[token] = vocab_id_decoder
                vocab_id_decoder += 1

            target_id.append(vocab_decoder[token])
        source_id.append(1)
        target_id = [1] + target_id + [1]
        sequence["source_id"] = source_id
        sequence["target_id"] = target_id

        if len(source_id) > max_len:
            max_len = len(source_id)
        if len(target_id) > max_len:
            max_len = len(target_id)

    # Might as well include padding here as well
    for sequence in data:
        source_id = sequence["source_id"]
        target_id = sequence["target_id"]


        if len(source_id) < max_len:
            diff = max_len - len(source_id)
            source_id.extend([0]*diff)
        if len(target_id) < max_len:
            diff = max_len - len(target_id)
            target_id.extend([0]*diff)

    return data, vocab_encoder, vocab_decoder, input_vocab, output_vocab

def ids_to_words(id_array, id_to_word):
    """
    Args:
        id_array: 2D list or tensor of ids, shape (batch, seq_len)
    Returns:
        list of lists of words
    """
    return [
        [id_to_word.get(int(token_id), "<UNK>") for token_id in row]
        for row in id_array
    ]

if __name__ == "__main__":

    data, vocab, input_vocab, output_vocab = get_source_data()
