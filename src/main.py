import os
import json
import pandas as pd
from coreset import Coreset
from model_train import train_classifier
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def println(line_length=75, num=1):
    for i in range(num):
        print(f'\n{"_"*line_length}\n')


# Read config
with open(os.path.join(ROOT_DIR, '..', 'config', 'conf.json')) as f:
    config = json.load(f)

data_dir = config['dirs']['data_dir']
models_dir = config['dirs']['models_dir']
results_dir = config['dirs']['results_dir']
dataset_name = config['dataset_name']
techniques = config['techniques']
sizes = config['sizes']
dataset_path = os.path.join(data_dir, dataset_name)

# Read in the full dataset
# This can be optimized for larger datasets using Spark
#print('\nLoading dataset...')
full_dataset = pd.read_csv(dataset_path)

# coreset = Coreset(dataset=full_dataset, technique='uniform')
# coreset.populate(0.1)
# coreset.save(data_dir, 'mnist_sampled_0.1.csv')

print('| CORESET CREATION EXPERIMENT |')
# Create and train coresets
training_stats = []
for tech in techniques:
    coreset = Coreset(dataset=full_dataset, technique=tech)
    for size in sizes:
        # gen = input(f'Generate a {size*100}% coreset using {tech}?')
        # if gen.lower() in ['yes', 'y']:
        coreset_name = f'coreset_{tech}_{str(size)}'
        println()
        print('Building and training: ', coreset_name)
        cs_creation_time = coreset.populate(size)
        print(f'Coreset shape: {coreset.coreset.shape})')
        # coreset.save(data_dir, f'{coreset_name}.csv')
        training_stat = train_classifier(coreset.coreset, coreset_name, cs_creation_time)
        training_stats.append(training_stat)

println()
training_stats.append(train_classifier(full_dataset))

print('Saving stats to disk...')
with open(os.path.join(results_dir, 'results.json')) as f:
    json.dump(training_stats, f)
println()



