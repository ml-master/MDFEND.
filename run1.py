from faknow.run import run

model = 'mdfend'  # lowercase short name of models
kargs = {'train_path': 'train.json', 'test_path': 'test.json'}  # dict arguments
run(model, **kargs)