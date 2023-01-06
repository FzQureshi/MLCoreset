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
num_sims = config['defaults']['num_sims']
dataset_path = os.path.join(data_dir, dataset_name)

for directory in [models_dir, results_dir]:
    if not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)


# Read in the full dataset
# This can be optimized for larger datasets using Spark
full_dataset = pd.read_csv(dataset_path)


print('| CORESET CREATION COMPARISON |')


def create_and_train():
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
            training_stat = train_classifier(coreset.coreset, coreset_name, cs_creation_time, models_dir)
            training_stats.append(training_stat)
    return training_stats


def holdout_test():
    pass


'''SCRIPT BEGINS HERE'''
println()

full_training_stats = {}
for i in range(num_sims):
    full_training_stats[f'Simulation_{i}'] = create_and_train()

if config['defaults']['train_full_dataset']:
    println()
    print('Training on full dataset (this might take a while)...')
    full_training_stats['Full_dataset'] = train_classifier(full_dataset, 'full_data', 0, models_dir)
    println()

print(stats_json_str := json.dumps(full_training_stats, indent=4))
print('Saving stats to disk...')

with open(os.path.join(results_dir, 'results.json'), 'w+') as f:
    f.write(stats_json_str)

println()
print('Done.')



