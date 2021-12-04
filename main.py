# import
from src.project_parameters import ProjectParameters
from src.train import Train
from src.predict import Predict
from src.tuning import Tuning


# def
def main():
    # project parameters
    project_parameters = ProjectParameters().parse()

    assert project_parameters.mode in [
        'train', 'predict', 'predict_gui', 'tuning'
    ], 'please check the mode argument.\nmode: {}\nvalid: {}'.format(
        project_parameters.mode, ['train', 'predict', 'predict_gui', 'tuning'])

    if project_parameters.mode == 'train':
        result = Train(project_parameters=project_parameters).train()
    elif project_parameters.mode == 'predict':
        result = Predict(project_parameters=project_parameters)
        result = Predict(project_parameters=project_parameters).predict(
            filepath=project_parameters.root)
    elif project_parameters.mode == 'predict_gui':
        from src.predict_gui import PredictGUI
        result = PredictGUI(project_parameters=project_parameters).run()
    elif project_parameters.mode == 'tuning':
        result = Tuning(project_parameters=project_parameters,
                        train_class=Train).tuning()
    return result


if __name__ == '__main__':
    main()