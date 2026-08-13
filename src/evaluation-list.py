import json
import evaluate
import os
import csv

# read the json file
def read_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f'File not found: {file_path}')
        return None

# read the txt file
def read_txt_file(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    return data

# define the type and prediction
task = 'event-reconstruction' # 'suspicious-keyword' # 'grep'   
prediction = 'chatgpt-no-knowledge-multiple' # 'chatgpt-with-knowledge-multiple' # 'chatgpt-no-knowledge' #  #  # 'chatgpt-with-knowledge' #  

# define the directories
groundtruth_dir = f'./{task}/ground-truth-multiple/' # ground-truth-multiple
llm_dir = f"./{task}/{prediction}"

# get the list of files in the directories
groundtruth_files = os.listdir(groundtruth_dir)
llm_files = os.listdir(llm_dir)

# create a list to store the results
groundtruth = []
llm = []

# load the bleu and rouge metrics
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")

# create a csv file to store the results
csv_file = f'./{task}/{prediction}-results.csv'
print(csv_file)

# open the csv file
f = open(csv_file, 'w', newline='')

# Save the results to a CSV file
writer = csv.writer(f)

# Write the header
writer.writerow(['file', 'blue', 'brevity_penalty', 'length_ratio', 'translation_length', 'reference_length', 'rouge1', 'rouge2', 'rougeL', 'rougeLsum'])  

# loop through the files
for groundtruth_file in groundtruth_files:
    if groundtruth_file.endswith('.json') or groundtruth_file.endswith('.txt'):
        # Create a list to store the results
        results = []
        llm_data_zero_len = False

        # get the file path
        groundtruth_file_path = os.path.join(groundtruth_dir, groundtruth_file)
        llm_file_path = os.path.join(llm_dir, groundtruth_file)
        print(groundtruth_file_path, llm_file_path)
        
        if groundtruth_file.endswith('.json'):
            # Read the json files
            groundtruth_data = read_json_file(groundtruth_file_path)
            llm_data = read_json_file(llm_file_path)

            if llm_data is None:
                continue

            # check if it is a list or dict
            if isinstance(groundtruth_data, dict):
                groundtruth_data = [groundtruth_data]
                llm_data = [llm_data]

            # get the length of the groundtruth data
            groundtruth_data_len = len(groundtruth_data)

            # get the length of the llm data
            llm_data_len = len(llm_data)

            # if the length of the llm data is zero
            if llm_data_len == 0:
                # skip the file
                llm_data_zero_len = True

            # if the length of the llm data is greater than the groundtruth data
            elif llm_data_len > groundtruth_data_len:
                # truncate the llm data
                llm_data = llm_data[:groundtruth_data_len]
            
            # if the length of the llm data is less than the groundtruth data
            elif llm_data_len < groundtruth_data_len:
                # truncate the groundtruth data
                groundtruth_data = groundtruth_data[:llm_data_len]
            
            # convert each dictionary to a string
            if llm_data_zero_len == False:
                groundtruth_data = [json.dumps(data) for data in groundtruth_data]
                llm_data = [json.dumps(data) for data in llm_data]
        
        elif groundtruth_file.endswith('.txt'):
            # read the txt files
            groundtruth_data = read_txt_file(groundtruth_file_path)
            llm_data = read_txt_file(llm_file_path)

            # get the length of the groundtruth data
            groundtruth_data_list = groundtruth_data.splitlines()
            groundtruth_data_len = len(groundtruth_data_list)

            # get the length of the llm data
            llm_data_list = llm_data.splitlines()
            llm_data_len = len(llm_data_list)

            # if the length of the llm data is greater than the groundtruth data
            if llm_data_len > groundtruth_data_len:
                # truncate the llm data
                llm_data = llm_data_list[:groundtruth_data_len]

                # get the list of the groundtruth data
                groundtruth_data = groundtruth_data_list

            # if the length of the llm data is less than the groundtruth data
            elif llm_data_len < groundtruth_data_len:
                # truncate the groundtruth data
                groundtruth_data = groundtruth_data_list[:llm_data_len]

                # get the list of the llm data
                llm_data = llm_data_list
            
            # get the list of the groundtruth data and llm data
            else:
                groundtruth_data = groundtruth_data_list
                llm_data = llm_data_list

        if llm_data_zero_len == False:
            # compute the bleu and rouge scores
            bleu_results = bleu.compute(predictions=llm_data, references=groundtruth_data)
            rouge_results = rouge.compute(predictions=llm_data, references=groundtruth_data)

            # convert bleu_results and rouge_results from dictionary to list
            bleu_results = [bleu_results['bleu'], bleu_results['brevity_penalty'], bleu_results['length_ratio'], bleu_results['translation_length'], bleu_results['reference_length']]
            rouge_results = [rouge_results['rouge1'], rouge_results['rouge2'], rouge_results['rougeL'], rouge_results['rougeLsum']]
        
        else:
            bleu_results = [0, 0, 0, 0, 0]
            rouge_results = [0, 0, 0, 0]

        # Append the results to the list
        results.append(groundtruth_file)
        results.extend(bleu_results)
        results.extend(rouge_results)

        print(results)
        writer.writerow(results)  # Write the results

f.close()