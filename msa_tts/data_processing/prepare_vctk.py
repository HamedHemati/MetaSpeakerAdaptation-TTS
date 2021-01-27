from msa_tts.utils.limit_threads import * 
import torch
import argparse
import os
import soundfile as sf
import librosa
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from ..utils.g2p.grapheme2phoneme import Grapheme2Phoneme


class ComVoiceProcessor():
    def __init__(self, 
                 ds_path, 
                 lang):
        self.ds_path = ds_path
        self.lang = lang 
        self.g2p = Grapheme2Phoneme()

    def get_line_meta(self, spk_id, wav_file, transcript, itr, total_count):
        try:
            print(f"Processing {itr}/{total_count}")
            wav_path = os.path.join(self.ds_path, "wav48", spk_id, wav_file)
            wav, sr = sf.read(wav_path)     
            
            # Resample
            wav = librosa.resample(wav, sr, 22050)
            sr = 22050

            # Save resampled wav file
            target_wav_path = wav_path.replace("/wav48/", "/wavs/")
            os.makedirs("/".join(target_wav_path.split("/")[:-1]), exist_ok=True)
            sf.write(target_wav_path, wav, sr)
            
            dur = len(wav) / float(sr)
            
            if transcript[-1] not in ["!", ".", "?"]:
                transcript += "."
            
            phoneme = self.g2p.text_to_phone(transcript, language=self.lang)

            meta_line = f"{spk_id}|{wav_file}|{transcript}|{phoneme}|{dur:#.2}"
            return meta_line
        except:
            return None
            
    def read_ds_files(self):
        all_txt_files = glob.glob(os.path.join(self.ds_path, "txt/*/*.txt"))
        all_lines = []
        for txt_file in all_txt_files:
            with open(txt_file, "r") as tfile:
                transcript = tfile.readline().strip()
            spk = txt_file.split("/")[-2]
            wav_file = txt_file.split("/")[-1].replace(".txt", ".wav")
            all_lines.append((spk, wav_file, transcript))
        return all_lines

    def create_metadata(self):
        all_lines = self.read_ds_files()
        # Create ./wavs dir for saving wav files with sr=22K
        os.makedirs(os.path.join(self.ds_path, "wavs"), exist_ok=True)
        
        executor = ProcessPoolExecutor(max_workers=20)
        meta_lines = []
        for itr, line in enumerate(all_lines):
            spk_id, wav_file, transcript = line
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
    
    ds_processor = ComVoiceProcessor(args.ds_path, 
                                     args.lang)
    ds_processor.create_metadata()
