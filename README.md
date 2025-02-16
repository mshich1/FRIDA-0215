# FRIDA to the Rescue! Analyzing Synthetic Data Effectiveness in Common Sense Reasoning for Disaster Response

## Directory
**Assets:** all the charts made for the paper

**Gemini:** code for generating synthetic data with gemini, as well as querying its API for evaluation purposes

**Gemini_Results:** The resulting synthetic datasets and subsets. It also includes the baseline results for the gemini and llama 8b models

**LlaMa:** The code for getting answers from the fine-tuned and base models LLaMa models, as well as Gemini, and evaluating the responses

**LLaMa_Results:** All results for all LLaMa based FRIDA models. sm refers to FRIDA 1B, md refers to FRIDA 3B, no marking refers to 8B. 
The name refers to the ablation model in question. em means exact match, sm means semscore.

**Mistral:** The code for getting answers from the fine-tuned and base Ministral 8B model and evaluating the responses

**Mistral_Results:** All results for the Mistral based FRIDA model. 
The name refers to the ablation model in question. em means exact match, sem means semscore.

## Walk through of the pipeline
### Part 1: Creating the seed sentences
1. An expert identifies a concrete saw as an important object in earthquake search and rescue. The expert also identifies that concrete saws are used to cut through large pieces of rubble to be trucked away in order to gain access to survivors. In our paper, an author researched the Turkiye-Syria earthquake and completed this step, but in future work this information could come from an expert in earthquake search and rescue.
2. A linguistic experts adds the concrete saw's affordances to an object affordance ontology, as well as to a vocabulary bank of pertinent disaster terms
3. The liguistic expert starts to fill in the blanks of the templates to make seed sentences for synthetic data generation. She gets to the Equipment in Task template, which reads *"What role does [object] play in [disaster subtask]"*
4. The expert reads the instruction for filling in the first blank in the template, which requires the object to be disaster related. She chooses concrete saw as her object. The seed instruction now reads *"What role does a concrete saw play in [disaster subtask]"*
5. The expert follows instruction for filling in the second blank with the corresponding subtask to concrete saw. The seed instruction now reads *"What role does a concrete saw play in removing rubble?"*
6. The expert generates the answer choices. The correct answer is a summarization of the use case the disaster expert gave that also involves the concrete saw's affordance in the affordance ontology. The incorrect answers are affordances of different earthquake search and rescue objects in the ontology, such as shovels, hydrualic lifts, and dump trucks. The final seed instruction reads:

*"What role does a concrete saw play in removing rubble? A) accesses debris in tight spaces B) lifts large debris to access trapped victims C) breaks large pieces down for easier transport offsite D)lifts heavy pieces into trucks E) moves small pieces of rubble by hand*

*Answer: C) breaks large pieces down for easier transport offsite"*

9. The linguistic experts converts this into a json object for easier parsing.
10. The linguist creates 5 seed instructions from each template, following the instructions for filling in the template blanks and generating the answers. All templates can be seen in `templates.csv` and all instructions for filling in these templates can be seen in `task_descriptions.csv`.

### Part 2: Generating the seed questions
1. Gemini is sent a system instruction to multiple choice generate instructions about object affordances and earthquake search and rescue and return them as a specifically formatted json object.
2. Gemini is then sent the 5-shot seed examples preceded by the instruction chosen specifically for them. For our concrete saw example above, this instruction reads *Create 40 unique multiple choice questions about how an object is used in a task. The tasks and objects should be related to earthquakes. The answer choices should be brief descriptions of potential ways to use the object in the task. These questions must be multiple choice and they must have 5 options with 1 correct answer. Make sure each answer option is unique.*
It is followed by the 5 shot seed examples, one of which is the instruction generated in part 1.
3. Gemini returns its response, which is then filtered for objects that do not follow the json object format requested and for instructions including words that indicate the instruction likely has a visual component which our LLMs cannot access.
4. The generated instructions are then compared to the existing seed and synthetic instructions using ROUGE score, and are then discarded if the ROUGE score is above a certain threshold. This threshold was determined experimentally by how quickly synthetic instructions were being generated. For most categories, including the Equipment in Task template, is 0.8.
5. Steps 2-4 are continued until 1000 synthetic instructions are generated for the Equipment in Task template. They are saved together in the file `gemini_results/equip_in_task.json`, along with the corresponding ROUGE scores.
### Part 3: Fine tuning
We used the TRL library and example scripts accessible by cloning the TRL github. We used their accelerate config and their supervised fine-tuning script, `sft.py`. 

In addition to fine-tuning on all data, we ran an ablation study by fine-tuning on subsets of the data distinguished by subject matter. Our example above falls into the Specialized Equipment category. Thus only full FRIDA models and the aFRIDA: Specialized Equipment model would be fine-tuned using the synthetic data generated from the equipment in task seed instructions. 

We had to modify TRL's fine-tuning script within their repo to clean up our custom datasets, convert them into the chat trl template, and use them for fine-tuning. We also modified it to automatically go through all custom datasets we made for the ablation studies. You can find our modified script at `llama/sft.py `. 

After modifying TRL's sft.py, we train a model on all of our data, as well as all subsets, on 2 A100 GPUs using the following command within the `trl` directory.  
`accelerate launch\`  
 `--config_file=examples/accelerate_configs/multi_gpu.yaml\`  
 `--num_processes 2 \`  
`examples/scripts/sft.py \`  
    `--model_name_or_path YOUR_BASE_MODEL_HERE\`  
    `--learning_rate 2.0e-4 \`  
    `--num_train_epochs 3 \`  
    `--per_device_train_batch_size 2 \`  
    `--gradient_accumulation_steps 8 \`  
    `--logging_steps 25 \`  
    `--eval_strategy steps \`  
    `--eval_steps 100 \`  
    `--use_peft \`  
    `--lora_r 32 \`  
    `--lora_alpha 16 \`  
    `--dataset_name / \` *this line was made redundant by our modifications to sft.py*  
    `--output_dir ../YOUR_OUTPUT_DIRECTORY_HERE`  
### Part 4: Evaluating
We evaluated on a separate dataset created in the same way as in Part 1. A step by step outline of our data analysis is as follows.

1. We collected responses from Gemini, LLaMa 1B, 3B, and 8B as our baselines.
2. We collected responses from FRIDA 1B, 3B, and 8B as well as their corresponding ablation models
3. We ran a script using Huggingface's evaluation tools to get an exact match score between the gold standard and model responses for every model.
4. We ran a script calculating the SemScore (embedding-space cosine distance) between the gold-standard and model response embeddings for every model.

## Installation
Everything here was installed with conda, pip, homebrew, and/or huggingface.

## Acknowledgments
Thank you to the reviewers for taking the time to look through this resource and examples# Gemini-Llama-Test
