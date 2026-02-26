class GPT2TokenizerFast:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def encode(self, text, **kwargs):
        return []
