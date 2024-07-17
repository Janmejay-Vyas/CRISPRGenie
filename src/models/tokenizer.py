"""Custom tokenizer for the CRISPRGenie transformer model"""

class CustomTokenizer:
    def __init__(self):
        # Include all nucleotide bases, special tokens, and necessary characters for gene IDs
        self.token_to_id = {
            '[PAD]': 0, '[ID]': 1, '[SOS]': 2, '[EOS]': 3,
            'A': 4, 'T': 5, 'G': 6, 'C': 7, 
            'E': 8, 'N': 9, 'S': 10, 
            '0': 11, '1': 12, '2': 13, '3': 14, '4': 15,
            '5': 16, '6': 17, '7': 18, '8': 19, '9': 20
        }
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    @property
    def vocab_size(self):
        return len(self.token_to_id)

    def encode(self, text):
        """ Convert text to a list of token IDs, treating each character as a token unless enclosed in []. """
        tokens = []
        i = 0
        while i < len(text):
            if text[i] == '[':  # Start of a special token
                special_token_end = text.find(']', i)
                if special_token_end != -1:
                    tokens.append(text[i:special_token_end+1])
                    i = special_token_end + 1
                else:
                    tokens.append(text[i])  # Fallback if ']' is missing
                    i += 1
            else:
                tokens.append(text[i])
                i += 1
        return [self.token_to_id[token] for token in tokens if token in self.token_to_id]

    def decode(self, token_ids):
        """ Convert a list of token IDs back to a string. """
        return ''.join(self.id_to_token.get(token_id, '') for token_id in token_ids)