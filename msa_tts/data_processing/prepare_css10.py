from tts.utils.limit_threads import * 
import torch
import argparse
import os
import torchaudio
import soundfile as sf
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import soundfile as sf
from tts.utils.g2p.g2p import Grapheme2Phoneme


class CSS10Processor():
    def __init__(self, 
                 ds_path, 
                 lang,
                 spk_name):
        self.ds_path = ds_path
        self.lang = lang 
        self.spk_name = spk_name
        self.g2p = Grapheme2Phoneme()

    def get_line_meta(self, wav_file, transcript, dur, itr, total_count):
        try:
            print(f"Processing {itr}/{total_count}")

            
            if transcript[-1] not in ["!", ".", "?"]:
                transcript += "."
            
            phoneme = self.g2p.text_to_phone(transcript, language=self.lang)

            meta_line = f"{self.spk_name}|{wav_file}|{transcript}|{phoneme}|{dur}"
            return meta_line
        except:
            return None
            
    def create_metadata(self):
        with open(os.path.join(self.ds_path, "transcript.txt"), "r") as metafile:
            all_lines = metafile.readlines()
        
        all_lines = [l.strip() for l in all_lines]
        all_lines = [l.split("|") for l in all_lines]
        all_lines = [(l[0], l[2], l[3]) for l in all_lines]

        executor = ProcessPoolExecutor(max_workers=10)
        meta_lines = []
        for itr, line in enumerate(all_lines):
            wav_file, transcript, dur = line
            meta_line = executor.submit(self.get_line_meta, wav_file, transcript, dur,  itr, len(all_lines))
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
    parser.add_argument("--spk_name", type=str, required=True)
    args = parser.parse_args()
    
    ds_processor = CSS10Processor(args.ds_path, 
                                  args.lang,
                                  args.spk_name)
    ds_processor.create_metadata()