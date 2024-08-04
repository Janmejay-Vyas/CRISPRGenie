"""Custom tokenizer for the CRISPRGenie transformer model"""


class CustomTokenizer:
    def __init__(self):
        self.token_to_id = {
            '[PAD]': 0, '[SOS]': 1, '[EOS]': 2,
            'A': 3, 'T': 4, 'G': 5, 'C': 6,
            '[ID]': 7,
            '0': 8, '1': 9, '2': 10, '3': 11, '4': 12,
            '5': 13, '6': 14, '7': 15, '8': 16, '9': 17,
            '[PAD]': 18,
        }
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    def encode(self, text):
        tokens = []
        i = 0
        while i < len(text):
            if text[i:i+4] == 'ENSG':
                tokens.append('[ID]')
                i += 4
            elif text[i] == '[':
                special_token_end = text.find(']', i)
                tokens.append(text[i:special_token_end+1])
                i = special_token_end + 1
            else:
                tokens.append(text[i])
                i += 1
        return [self.token_to_id.get(token, self.token_to_id['[PAD]']) for token in tokens]

    @property
    def vocab_size(self):
        return len(self.token_to_id)

    def decode(self, token_ids):
        return ''.join(self.id_to_token.get(token_id, '') for token_id in token_ids)