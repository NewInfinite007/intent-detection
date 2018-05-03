from fountain import data_generator

dg = data_generator.DataGenerator()
dg.parse('template.yaml')
dg.to_json('train_set.json')