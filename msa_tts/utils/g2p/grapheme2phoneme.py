from collections import Counter
from .phonemizer_api.phonemize import phonemize
from .char_list import char_list, _punctuations, _pad


class Grapheme2Phoneme():
    def __init__(self):
        super(Grapheme2Phoneme, self).__init__()    
        # Set char list
        self.char_list = char_list
        self.punctutations = _punctuations
        
        # Char to id and id to char conversion
        self.char_to_id = {s: i for i, s in enumerate(self.char_list)}
        self.id_to_char = {i: s for i, s in enumerate(self.char_list)}

    def text_to_phone(self, text, language="en-us"):
        """Converts text to phoneme."""
        ph = phonemize(text, 
                        strip=False, 
                        with_stress=True, 
                        preserve_punctuation=True, 
                        punctuation_marks=self.punctutations,
                        njobs=1, 
                        backend='espeak', 
                        language=language, 
                        language_switch="remove-flags")
        return ph

    def _should_keep_char(self, 
                          p):
        r"""Checks if char is valid and is not pad char."""
        return p in self.char_list and p not in [_pad]

    def phone_to_index_list(self, phones, **kwargs):
        r"""Converts list of phonemes to index list."""
        sequence = [self.char_to_id[s] for s in list(phones) if self._should_keep_char(s)]
        return sequence, phones
        
    def text_to_phone_to_index_list(self, text, **kwargs):
        """Converts text to sequence of indices."""
        sequence = []
        # Get the phoneme for the text
        phones = self.text_to_phone(text, language=kwargs['language'])
        
        # Convert each phone to its corresponding index
        sequence = [self.char_to_id[s] for s in list(phones) if self._should_keep_char(s)]
        if sequence == []:
            print("!! After phoneme conversion the result is None. -- {} ".format(text))
                
        return sequence, phones
    
    def text_to_phone_to_index_list_alignment(self, text, **kwargs):
        r"""Converts text to sequence of indices with alignment of each word."""
        text_ = " ::: ".join(text.split())
        
        # Get the phoneme for the text
        out = self.text_to_phone(text_, language=kwargs['language'])
        out_ = out.split(" ::: ")

        # Extract the alignment between words and their phonetic repr.
        word_to_idx = []
        words = text.split()
        start = 0

        for itr, phone in enumerate(out_):
            end = start + len(phone) - 1
            word_to_idx.append((words[itr], (start, end)))
            start = end + 1
        out_final = ''.join(out_)

        # Convert each phone to its corresponding index
        sequence = []
        sequence = [self.char_to_id[s] for s in list(out_final) if self._should_keep_char(s)]
        if sequence == []:
            print("!! After phoneme conversion the result is None. -- {} ".format(text))
                
        return sequence, word_to_idx

    def convert(self, inp, **kwargs):
        r"""Wrapper function for text to index list."""
        convert_mode = kwargs["convert_mode"]

        if convert_mode == "phone_to_idx":
            return self.phone_to_index_list(inp, **kwargs)
        elif convert_mode == "text_to_phone_to_idx": 
            return self.text_to_phone_to_index_list(inp, **kwargs)
        elif convert_mode == "text_to_phone_to_idx_aligned": 
            return self.text_to_phone_to_index_list_alignment(inp, **kwargs)

    def get_char_list(self):
        return self.char_list
        