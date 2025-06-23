# 
import torch
torch.cuda.empty_cache()
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

import shutil 
import os
import random
import re

import numpy as np
import torch
import gc

# Collect garbage
gc.collect()
torch.cuda.empty_cache()

import torch.nn as nn


import pandas as pd
import csv
import textwrap


import sent2vec
from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from scipy.spatial.distance import pdist, squareform
from gensim.models import KeyedVectors
import fasttext
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns


SEED = 24
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)


class concept_label_creator(nn.Module):
    def __init__(self,gpu_id:str,
                prompts_csv:str, tok, 
                 #LLMmodel_id="meta-llama/Meta-Llama-3.1-8b-Instruct", 
                 llama_model, 
                encoder_path='/projectnb/seizuredet/Models/BioWordVec_PubMed_MIMICIII_d200.bin', 
                cache_dir='/scratch/mary79/', 
                  ):

        super(concept_label_creator, self).__init__()
        
        # self.llama_tokenizer = AutoTokenizer.from_pretrained(LLMmodel_id, 
        #                                         cache_dir = cache_dir
        #                                         # local_files_only=True
        #                                         )#low_cpu_mem_usage=True)

        # self.llama_model = AutoModelForCausalLM.from_pretrained(LLMmodel_id,
        #                                         cache_dir = cache_dir
        #                                         #    local_files_only=True
        #                                         )#low_cpu_mem_usage=True)
                                                    
        self.llama_tokenizer = tok
        self.llama_model = llama_model
        
        PAD_TOKEN = "<|pad|>"
        self.llama_tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
        self.llama_tokenizer.padding_side = "right"


        self.pipe = pipeline(
            "text-generation",
            model=self.llama_model,
            tokenizer=self.llama_tokenizer,
            torch_dtype=torch.float16,
            max_new_tokens = 128, 
            device = gpu_id,
        # device_map="auto",
        )
        
        self.bw2v = fasttext.load_model(encoder_path)

        # comp_concepts = ["Spike and wave",  "Sharp", "Polyspikes", 
        #             "High frequency oscillations",  "Spikes", "Paroxysmal", 
        #             "Burst suppression", "High amplitude", "rhythmic",  "Fast",   
        #             "artifacts",  "Postictal", "slowing", "evolving frequency",  "noise", 
        #             "Interictal", "epileptiform", "PLED",  "phase reversal", "background"
        #             ]


        comp_concepts = ["Spike and wave",  "Sharp Wave", "Polyspike", 
            "High frequency oscillation",  "Spike", "Paroxysmal", 
            "Burst suppression", "High amplitude", "rhythmic",  "Fast",   
            "artifacts",  "Postictal", "slowing", "evolving frequency",  "noise", 
            "Interictal", "epileptiform", "PLED",  "phase reversal", "background"
            ]
        self.comp_concepts = [concept.lower() for concept in comp_concepts]
        self.all_concepts = (', ').join(comp_concepts)
        self.compconcept_emb = np.array([self.bw2v[word] for word in self.comp_concepts])
        self.compconcept_emb /= np.linalg.norm(self.compconcept_emb, axis=1)[:, None]

        self.data  = pd.read_csv(prompts_csv)
        self.data["input_prompt"] = self.data.apply(self.augment_std_prompts, axis=1)
        self.data["augmented_output"] = self.pipe(list(self.data["input_prompt"]))
        self.data["augmented_prompt"] = self.data.apply(self.get_assistant_text, axis=1)
        self.data["final_prompt"] = self.data.apply(self.create_concept_prompt, axis=1)
        self.data["llama_concepts"] = self.pipe(list(self.data["final_prompt"]))
        self.data["concepts"] = self.data.apply(self.get_concept_phrases, axis=1)  

    def get_data(self):
        return self.data

    def create_concept_prompt(self, row: dict):
    
        prompt = f"""Provide a numbered list of short phrases of all signal features in the EEG of this patient as described in the notes. Wherever possible, choose from {self.all_concepts}. Notes: {row["augmented_prompt"]}"""

        messages = [
            {"role": "system",  "content": "Use only the information to answer the question without description"},
        {"role": "user", "content": prompt}, 
            
        ]
        return self.llama_tokenizer.apply_chat_template(messages, tokenize=False,)# add_generation_prompt=True)

    def augment_std_prompts(self, row: dict):
        
        prompt = f"""Modify the following notes randomly without changing the content using a one of lexical, syntactic, semantic, and/or surface modifications: Notes: {row["characteristics"]}"""

        messages = [
            {"role": "system",  "content": "Use only the information to answer the question without description"},
        {"role": "user", "content": prompt}, 
            
        ]

        return self.llama_tokenizer.apply_chat_template(messages, tokenize=False,)# add_generation_prompt=True)

    def count_tokens(self, row: dict):
        
        return len(self.llama_tokenizer(row["input_prompt"], add_special_tokens=True, return_attention_mask=False)["input_ids"] )

    def get_concept_phrases(self, row):
        text = row["llama_concepts"][0]['generated_text']
        words = text.split()
        for j, word in enumerate(words):
            if  'assistant' in word:
                break
        #print(j, word)
        j += 1
        concept_phrases = []
        while j<len(words):
            if (words[j][:1].isdigit()  and len(words[j])==2) or  words[j]=='-':
                if words[j][1]=='.': 
                    concept_phrases.append('\n')
                    j += 1
                    #continue
                else:
                    concept_phrases.append(words[j])
                    j+= 1
            else:
                concept_phrases.append(words[j])
                j+= 1
        concept_phrases = (' ').join(concept_phrases).split('\n ')[1:]

        return (' ').join(concept_phrases)

    def refine_concept_phrase(self, text:str):
            unwanted = ['left', 'right', 'temporal', 'hemisphere', "and", "seizure", "clinical", "activity", 
                        "dischange","in", "over" , "the", "on" , "onset", "involvement","hz","upto","to", 
                    "Delta", "Theta", "Alpha", "Beta", "Gamma", "phase reversal", 
                        "Prefrontal", "Frontal", "Central", "Temporal", 
                        "Frontotemporal", "Frontocentral", "Parietal", "Occipital", 
                        "Anterior", "Posterior", "Left", "Right", "status", "epilepticus",
                        "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz" , "C4",
                        "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2" , "Synchrony", "Asymmetry", "bilateral", 
                    "generalized", "focal", "mild", "hz", "of", "somewhat"] 
            unwanted = [word.lower() for word in unwanted]
            
            text = text.lower()
            text = text.replace("-", " ")
            text = re.sub(r'[^\w\s]', '', text) #remove puctuations
            text = re.sub(r'\d+', '', text) #remove numbers
            text = " ".join(word for word in text.split() if word.lower() not in unwanted) #remove unwanted
            t = text.split()
            
            if "artifact" in t:
                t.append("artifacts")
                t.remove('artifact')
                
            if "spike" in t and "wave" in t:
                t.append("spike and wave")
                t.remove('spike')
            if "spike" in t or "spikes" in t or "spiky" in t or "spiking" in t:
                while "spike" in t: t.remove("spike")
                while "spikes" in t: t.remove("spikes")
                while "spiky" in t: t.remove("spiky")
                while "spiking" in t: t.remove("spiking")
                t.append("spike")
            if "rhythm" in t:
                t.append("rhythmic")
            if "sharp" in t or "sharps" in t or "sharply" in t: #and "wave" in t:
                t.append("sharp wave")
                while "sharp" in t: t.remove("sharp")
                while "sharps" in t: t.remove("sharps")
            
            while "wave" in t: t.remove("wave")
            while "waves" in t: t.remove("waves")
            
            if "slow" in t:
                t.append("slowing")
                while 'slow' in t:t.remove('slow')         
            if "frequency" in t and ("evolves" in t or "evolving" in t or "increasing" in t or "evolution" in t):
                t.append("evolving frequency")
                
                while "evolves" in t: t.remove("evolves")
                while "evolving" in t: t.remove("evolving")
                while "increasing" in t: t.remove("increasing")
                while "evolution" in t: t.remove("evolution")
            
            if "polyspikes" in t:
                while "polyspikes" in t: t.remove("polyspikes")
                t.append("polyspike")

            if "post" in t and "ictal" in t:
                while "post" in t: t.remove("post")
                t.append("postictal")

            if "inter" in t and "ictal" in t:
                while "inter" in t: t.remove("inter")
                
                t.append("postictal")
            while "ictal" in t: t.remove("ictal")
            
            if "paroxysmal" in t:
                if "fast" in t: t.remove("fast")  #remove just one occurance of fast assoc. with paroxysmal
            if "burst" in t or "suppression" in t or "bursts" in t:
                while "burst" in t: t.remove("burst")
                t.append("burst suppression")
                while "bursts" in t: t.remove("bursts")
                while "suppression" in t: t.remove("suppression")
            if ("high" in t or "increased" in t or "higher" in t)  and "amplitude" in t :
                t.append("high amplitude")
                if "high" in t: t.remove("high")
                while "increased" in t: t.remove("increased")
                while "amplitude" in t: t.remove("amplitude")
            if "phase" in t and "reversal" in t:
                t.append("phase reversal")
                t.remove("phase")
                t.remove("reversal")
            if "high" in t and "frequency" in t and ("oscillations" in t or "oscillation" in t):
                t.append("high frequency oscillation")
                while "high" in t: t.remove("high")
                while "frequency" in t: t.remove("frequency")
                while "oscillations" in t: t.remove("oscillations")
                while "oscillation" in t: t.remove("oscillation")
                
            return t

    def get_assistant_text(self, row):
        text = row['augmented_output'][0]['generated_text']
        words = text.split()
        ret_words= []
        for j, word in enumerate(words):
            if  'assistant' in word:
                break
        j += 1
        while(j<len(words)):
            ret_words.append(words[j])
            j += 1
        return (' ').join(ret_words)

    def get_concept_label(self, row:dict):
        #augment 
        #outputs = self.pipe(prompt)
        #assist_output = self.get_assistant_text(outputs[0]['generated_text'])
        
        #get list of important concepts
        #llama_output = self.pipe(self.create_concept_prompt(assist_output))
        #concepts = self.get_concept_phrases(llama_output[0]['generated_text'])
        concepts = row['concepts'].item()
        concept_list = self.refine_concept_phrase(concepts)

        concept_emb = np.array([self.bw2v[concept] for concept in concept_list])
        concept_emb /= np.linalg.norm(concept_emb, axis=1)[:, None]
        dot = np.corrcoef(self.compconcept_emb, concept_emb)
        dot = dot[ : -concept_emb.shape[0], -concept_emb.shape[0]:]
        concept_label = (dot>0.8).sum(1)>=1


        return concept_label, concept_list
        
    def forward(self, pt_id:int):

        row = self.data[self.data['pt_id']==pt_id]
        #print(row['input_prompt'])
        concept_label, concept_list = self.get_concept_label(row)
        # sz_emb = self.compconcept_emb[:10][concept_label[:10]==1].sum(0)
        # nsz_emb = self.compconcept_emb[10:][concept_label[10:]==1].sum(0)

        sz_emb = self.compconcept_emb[:10][concept_label[:10]==1]
        nsz_emb = self.compconcept_emb[10:][concept_label[10:]==1]
        

        return concept_label, sz_emb, nsz_emb


class label_generation(nn.Module):
    def __init__(self):
        super(label_generation, self).__init__()

        self.encoder_path = '/projectnb/seizuredet/Models/BioWordVec_PubMed_MIMICIII_d200.bin'
        self.bw2v = fasttext.load_model(self.encoder_path)

        comp_concepts = ["Spike and wave",  "Sharp Wave", "Polyspike", 
            "High frequency oscillation",  "Spike", "Paroxysmal", 
            "Burst suppression", "High amplitude", "rhythmic",  "Fast",   
            "artifacts",  "Postictal", "slowing", "evolving frequency",  "noise", 
            "Interictal", "epileptiform", "PLED",  "phase reversal", "background"
            ]
                   
        self.comp_concepts = [concept.lower() for concept in comp_concepts]
        self.all_concepts = (', ').join(comp_concepts)
        self.compconcept_emb = np.array([self.bw2v[word] for word in self.comp_concepts])
        self.compconcept_emb /= np.linalg.norm(self.compconcept_emb, axis=1)[:, None]
    def refine_concept_phrase(self, text:str):
            unwanted = ['left', 'right', 'temporal', 'hemisphere', "and", "seizure", "clinical", "activity", 
                        "dischange","in", "over" , "the", "on" , "onset", "involvement","hz","upto","to", 
                    "Delta", "Theta", "Alpha", "Beta", "Gamma", "phase reversal", 
                        "Prefrontal", "Frontal", "Central", "Temporal", 
                        "Frontotemporal", "Frontocentral", "Parietal", "Occipital", 
                        "Anterior", "Posterior", "Left", "Right", "status", "epilepticus",
                        "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz" , "C4",
                        "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2" , "Synchrony", "Asymmetry", "bilateral", 
                    "generalized", "focal", "mild", "hz", "of", "somewhat"] 
            unwanted = [word.lower() for word in unwanted]
            
            text = text.lower()
            text = text.replace("-", " ")
            text = re.sub(r'[^\w\s]', '', text) #remove puctuations
            text = re.sub(r'\d+', '', text) #remove numbers
            text = " ".join(word for word in text.split() if word.lower() not in unwanted) #remove unwanted
            t = text.split()
            
            if "artifact" in t:
                t.append("artifacts")
                t.remove('artifact')
                
            if "spike" in t and "wave" in t:
                t.append("spike and wave")
                t.remove('spike')
            if "spike" in t or "spikes" in t or "spiky" in t or "spiking" in t:
                while "spike" in t: t.remove("spike")
                while "spikes" in t: t.remove("spikes")
                while "spiky" in t: t.remove("spiky")
                while "spiking" in t: t.remove("spiking")
                t.append("spike")
            if "rhythm" in t:
                t.append("rhythmic")
            if "sharp" in t or "sharps" in t or "sharply" in t: #and "wave" in t:
                t.append("sharp wave")
                while "sharp" in t: t.remove("sharp")
                while "sharps" in t: t.remove("sharps")
            
            while "wave" in t: t.remove("wave")
            while "waves" in t: t.remove("waves")
            
            if "slow" in t:
                t.append("slowing")
                while 'slow' in t:t.remove('slow')         
            if "frequency" in t and ("evolves" in t or "evolving" in t or "increasing" in t or "evolution" in t):
                t.append("evolving frequency")
                
                while "evolves" in t: t.remove("evolves")
                while "evolving" in t: t.remove("evolving")
                while "increasing" in t: t.remove("increasing")
                while "evolution" in t: t.remove("evolution")
            
            if "polyspikes" in t:
                while "polyspikes" in t: t.remove("polyspikes")
                t.append("polyspike")

            if "post" in t and "ictal" in t:
                while "post" in t: t.remove("post")
                t.append("postictal")

            if "inter" in t and "ictal" in t:
                while "inter" in t: t.remove("inter")
                
                t.append("postictal")
            while "ictal" in t: t.remove("ictal")
            
            if "paroxysmal" in t:
                if "fast" in t: t.remove("fast")  #remove just one occurance of fast assoc. with paroxysmal
            if "burst" in t or "suppression" in t or "bursts" in t:
                while "burst" in t: t.remove("burst")
                t.append("burst suppression")
                while "bursts" in t: t.remove("bursts")
                while "suppression" in t: t.remove("suppression")
            if ("high" in t or "increased" in t or "higher" in t)  and "amplitude" in t :
                t.append("high amplitude")
                if "high" in t: t.remove("high")
                while "increased" in t: t.remove("increased")
                while "amplitude" in t: t.remove("amplitude")
            if "phase" in t and "reversal" in t:
                t.append("phase reversal")
                t.remove("phase")
                t.remove("reversal")
            if "high" in t and "frequency" in t and ("oscillations" in t or "oscillation" in t):
                t.append("high frequency oscillation")
                while "high" in t: t.remove("high")
                while "frequency" in t: t.remove("frequency")
                while "oscillations" in t: t.remove("oscillations")
                while "oscillation" in t: t.remove("oscillation")
                
            return t


    def get_concept_label(self, row:dict):
        concepts = row['concepts'].item()
        concept_list = self.refine_concept_phrase(concepts)

        concept_emb = np.array([self.bw2v[concept] for concept in concept_list])
        concept_emb /= np.linalg.norm(concept_emb, axis=1)[:, None]
        dot = np.corrcoef(self.compconcept_emb, concept_emb)
        dot = dot[ : -concept_emb.shape[0], -concept_emb.shape[0]:]
        concept_label = (dot>0.8).sum(1)>=1
        return concept_label, concept_list

    def forward(self, data, pt_id:int):

        row = data[data['pt_id']==pt_id]
        concept_label, concept_list = self.get_concept_label(row)
        sz_emb = self.compconcept_emb[:10][concept_label[:10]==1].sum(0)
        nsz_emb = self.compconcept_emb[10:][concept_label[10:]==1].sum(0)

        return concept_label


