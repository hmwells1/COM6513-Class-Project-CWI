from utils.dataset import Dataset
from utils.Model import MyLine
from utils.baseline import Baseline
from utils.scorer import report_score


def execute_demo(language):
    data = Dataset(language)
    
    if test == True:
        print("{}: {} training - {} dev".format(language, len(data.trainset), len(data.testset)))
    else:
        print("{}: {} training - {} dev".format(language, len(data.trainset), len(data.devset)))

    if Base == True:
        baseline = Baseline(language)
    else:
        baseline = MyLine(language)

    baseline.train(data.trainset)
    
    if test == True:
        predictions = baseline.test(data.testset)
        gold_labels = [sent['gold_label'] for sent in data.testset]

    else:
        predictions = baseline.test(data.devset)
        gold_labels = [sent['gold_label'] for sent in data.devset]

    report_score(gold_labels, predictions)


if __name__ == '__main__':
    test = True
    Base = False
    execute_demo('english')
    execute_demo('spanish')
