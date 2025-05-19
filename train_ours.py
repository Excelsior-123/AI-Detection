from simcse.models import LongformerTextTrainer

train_path = './SentEval/data/downstream/AI_detection/raid_train.csv'
val_paths = ['./SentEval/data/downstream/AI_detection/raid_test.csv',
             './SentEval/data/downstream/AI_detection/raid_evaluate.csv',
             './SentEval/data/downstream/AI_detection/evaluate_coling_new.csv']

classifier = LongformerTextTrainer(train_path=train_path, val_paths=val_paths)
classifier.train()
