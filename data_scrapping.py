import os
import csv
import pickle
import pandas as pd
from time import sleep
from tqdm import tqdm
from googletrans import Translator
from indicnlp.tokenize import indic_tokenize

# function for generating dictionary containing input and target
def generate_data_dictionary(paths, destination='data.pickle'):
    
    # to be saved
    return_dict = {'tokens': []}
    
    # iterating over each file corresponding to the path
    for path in tqdm(paths):
        if path.endswith('.txt'):
            
            # reading the file
            file = open(path, 'r', encoding='utf-8')
            text = file.read().replace('\n', '').replace('·', '')
            
            # tokenizing the text
            token_list = indic_tokenize.trivial_tokenize(text, lang='hi')
            
            # committing the new tokens to the return dict
            return_dict['tokens'].append(token_list)
    
    # pickling
    with open(destination, 'wb') as pickle_file:
        pickle.dump(return_dict, pickle_file)
            

# function for pickling data for language model
def pickle_hindi_data_for_LM(data_dir):
    
    # directories containing the text files
    train_dir = os.path.join(data_dir, 'train', 'train')
    val_dir = os.path.join(data_dir, 'valid', 'valid')
    
    # getting the paths of all the training files and validation files
    train_paths = [os.path.join(train_dir, file) for file in os.listdir(train_dir)]
    val_paths = [os.path.join(val_dir, file) for file in os.listdir(val_dir)]
    
    # getting the results
    train_result_dict = generate_data_dictionary(train_paths, 'wiki-hindi-train.pickle')
    val_result_dict = generate_data_dictionary(val_paths, 'wiki-hindi-val.pickle')
    
    
# splitting the data into multiple csv files
def split_csv(csv_file_name, num_rows=1000):
    
    # retrieving the length of the file
    total_rows = sum(0 for row in open(csv_file_name, encoding='ISO-8859-1'))
    base_name = os.path.splitext(os.path.basename(csv_file_name))[0]
    
    # splitting begins
    for i in range(1, total_rows, num_rows):
        df = pd.read_csv(csv_file_name, nrows=num_rows, skiprows=i, encoding='ISO-8859-1')
        df.to_csv(base_name + '_' + str(i // num_rows) + '.csv')
        
    
# function for translating the text into hindi
def generate_cross_lingual_summary_dataset(csv_file_name, target_file_name):

    # translator object initialization
    translator = Translator()

    # to be converted to csv file
    result_dict = {'text': [], 'summary': []}

    # opening the csv file
    with open(csv_file_name, 'r', encoding='ISO-8859-1') as input_csv_file:
        input_csv_reader = csv.DictReader(input_csv_file)
        
        # iterating over lines
        buffer_text = []
        buffer_summary = []
        index = 0
        for line in tqdm(input_csv_reader):
            
            # extracting information from the line
            input_text = line['text']
            input_summary = line['headlines']
            
            # conditioning for appending to buffer
            if len(buffer_text) < 199:
                buffer_text.append(input_text)
                buffer_summary.append(input_summary)
                
            else:
                
                # translate everything to hindi in buffer, 
                translated_output = translator.translate(buffer_summary, src='en', dest='hi')
                result_dict['text'] += buffer_text
                buffer_text = []
                buffer_summary = []
                result_dict['summary'] += [item.text for item in translated_output]
                sleep(5)
            
            # updating the index
            index += 1
                
        if len(buffer_text) != 0:
            translated_output = translator.translate(buffer_summary, src='en', dest='hi')
            result_dict['text'] += buffer_text
            buffer_text = []
            buffer_summary = []
            result_dict['summary'] += [item.text for item in translated_output]
            sleep(5)
                
    # saving the result dict
    with open(target_file_name, 'w', newline='', encoding='utf-8') as target_csv_file:
        
        # writer object initialization
        target_csv_writer = csv.DictWriter(target_csv_file, ['text', 'summary'])
        target_csv_writer.writeheader()
        
        for (text, summary) in zip(result_dict['text'], result_dict['summary']):
            target_csv_writer.writerow({'text': text, 'summary': summary})


if __name__ == '__main__':
	pickle_hindi_data_for_LM('hindi_wiki/')
	generate_cross_lingual_summary_dataset('daily-news/news_summary.csv', 'test.csv')