import csv
import random

# This script is used by the annotator while filling in templates. 
# It randomly samples terms from the ontology, functions in the ontology,
# terms that have a given function, or terms from a list you give yourself
prop_dict = {}
with open('../vocab/vocab_propbank.csv') as prop_csv:
    csv_reader = csv.reader(prop_csv,delimiter=',')
    for line in csv_reader:
        key = ' '.join(line[1:])
        if key in prop_dict:
            prop_dict[key].append(line[0])
        else:
            prop_dict[key] = [line[0]]

vocab = open('../vocab/vocab.txt').read().split('\n')

prompt = input('do you want a random sample of (1) general terms, (2) function specific terms, (3) functions, or (4) choose random elements from a given list? ')

if prompt == '1':
    num_samples = int(input('how many samples do you need? '))
    num = int(input('how many terms do you need? '))
    if num <= len(vocab):
        for i in range(num_samples):
            print(random.sample(vocab,k=num))
elif prompt == '2':
    func = input('what function do you need? ')
    func_terms = prop_dict[func]
    num_samples = int(input('how many random samples do you need? '))
    num = int(input('how many terms per sample do you need? '))
    if num <= len(func_terms):
        for i in range(num_samples):
            print(random.sample(func_terms,k=num))
elif prompt == '3':
    num = int(input('how many terms do you need? '))
    mems = int(input('how many members does the function group need to have? '))
    if num <= len(list(prop_dict.keys())):
        print(random.sample([w for w in list(prop_dict.keys())if len(prop_dict[w]) >= mems],k=num))
elif prompt == '4':
    string_to_list = input('enter the string you want to convert to a list. Please separate entries with commas: ')
    to_sample = string_to_list.split(',')
    num = int(input('how many terms do you need? '))
    if num <= len(to_sample):
        print(random.sample(to_sample,k=num))