from msa_tts.utils.limit_threads import * 
import torch
import argparse
import os
import soundfile as sf
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from ..utils.g2p.grapheme2phoneme import Grapheme2Phoneme


class LJSpeechProcessor():
    def __init__(self, 
                 ds_path, 
                 lang):
        self.ds_path = ds_path
        self.lang = lang 
        self.g2p = Grapheme2Phoneme()

    def get_line_meta(self, spk_id, wav_file, transcript, itr, total_count):
        try:
            print(f"Processing {itr}/{total_count}")
            wav_path = os.path.join(self.ds_path, "wavs", wav_file + ".wav")
            wav, sr = sf.read(wav_path)     
            dur = len(wav) / float(sr)
            
            # if transcript[-1] not in ["!", ".", "?"]:
            #     transcript += "."
            
            phoneme = self.g2p.text_to_phone(transcript, language=self.lang)

            wav_file = "wavs/" + wav_file + ".wav"
            meta_line = f"{spk_id}|{wav_file}|{transcript}|{phoneme}|{dur:#.2}"
            return meta_line
        except:
            return None
            
    def create_metadata(self):
        with open(os.path.join(self.ds_path, "metadata.csv"), "r") as metafile:
            all_lines = metafile.readlines()
        
        all_lines = [l.strip() for l in all_lines]
        all_lines = [l.split("|") for l in all_lines]
        all_lines = [(l[0], l[2]) for l in all_lines]

        executor = ProcessPoolExecutor(max_workers=10)
        meta_lines = []
        spk_id = "lj"
        for itr, line in enumerate(all_lines):
            wav_file, transcript = line
            meta_line = executor.submit(self.get_line_meta, spk_id, wav_file, transcript, 
                                        itr, len(all_lines))
            # meta_line = self.get_line_meta(spk_id, wav_file, transcript, itr, len(all_lines))
            meta_lines.append(meta_line)

        meta_lines = [metaline.result() for metaline in meta_lines if metaline.result() is not None]
        
        # Write metafile
        with open(os.path.join(self.ds_path, "metadata.txt"), "w") as final_meta:
            for l in meta_lines:
                final_meta.write(l + "\n")

        print("Finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds_path", type=str, required=True)
    parser.add_argument("--lang", type=str, required=True)
    args = parser.parse_args()
    
    ds_processor = LJSpeechProcessor(args.ds_path, 
                                     args.lang)
    ds_processor.create_metadata()
