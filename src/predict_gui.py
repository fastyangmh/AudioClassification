# import
from src.project_parameters import ProjectParameters
from DeepLearningTemplate.predict_gui import BasePredictGUI
from src.predict import Predict
from DeepLearningTemplate.data_preparation import AudioLoader, parse_transforms
from tkinter import Button, messagebox
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from playsound import playsound
import tkinter as tk
import gradio as gr


# class
class PredictGUI(BasePredictGUI):
    def __init__(self, project_parameters) -> None:
        super().__init__(extensions=('.wav'))
        self.predictor = Predict(project_parameters=project_parameters)
        self.classes = project_parameters.classes
        self.loader = AudioLoader(sample_rate=project_parameters.sample_rate)
        self.transform = parse_transforms(
            transforms_config=project_parameters.transforms_config)['predict']
        self.sample_rate = project_parameters.sample_rate
        self.web_interface = project_parameters.web_interface
        self.examples = project_parameters.examples if len(
            project_parameters.examples) else None

        # button
        self.play_button = Button(master=self.window,
                                  text='Play',
                                  command=self.play)

        # matplotlib canvas
        # this is Tkinter default background-color
        facecolor = (0.9254760742, 0.9254760742, 0.9254760742)
        figsize = np.array([12, 4]) * project_parameters.in_chans
        self.image_canvas = FigureCanvasTkAgg(Figure(figsize=figsize,
                                                     facecolor=facecolor),
                                              master=self.window)

    def reset_widget(self):
        super().reset_widget()
        self.image_canvas.figure.clear()

    def display(self):
        waveform = self.loader(path=self.filepath)
        # the transformed sample dimension is (in_chans, freq, time)
        sample = self.transform(waveform)
        sample = sample.cpu().data.numpy()
        # invert the freq axis so that the frequency axis of the spectrogram is displayed correctly
        sample = sample[:, ::-1, :]
        rows, cols = len(sample), 2
        for idx in range(1, rows * cols + 1):
            subplot = self.image_canvas.figure.add_subplot(rows, cols, idx)
            if idx % cols == 1:
                # plot waveform
                subplot.title.set_text(
                    'channel {} waveform'.format((idx - 1) // cols + 1))
                subplot.set_xlabel('time')
                subplot.set_ylabel('amplitude')
                time = np.linspace(
                    0, len(waveform[(idx - 1) // cols]),
                    len(waveform[(idx - 1) // cols])) / self.sample_rate
                subplot.plot(time, waveform[(idx - 1) // cols])
            else:
                # plot spectrogram
                # TODO: display frequency and time.
                subplot.title.set_text(
                    'channel {} spectrogram'.format((idx - 1) // cols + 1))
                subplot.imshow(sample[(idx - 1) // cols])
                subplot.axis('off')
        self.image_canvas.draw()

    def open_file(self):
        super().open_file()
        self.display()

    def recognize(self):
        if self.filepath is not None:
            predicted = self.predictor.predict(inputs=self.filepath)
            text = ''
            for idx, (c, p) in enumerate(zip(self.classes, predicted)):
                text += '{}: {}, '.format(c, p.round(3))
                if (idx + 1) < len(self.classes) and (idx + 1) % 5 == 0:
                    text += '\n'
            # remove last commas and space
            text = text[:-2]
            self.predicted_label.config(text='probability:\n{}'.format(text))
            self.result_label.config(text=self.classes[predicted.argmax(-1)])
        else:
            messagebox.showerror(title='Error!', message='please open a file!')

    def play(self):
        if self.filepath is not None:
            playsound(sound=self.filepath, block=True)
        else:
            messagebox.showerror(title='Error!', message='please open a file!')

    def inference(self, inputs):
        prediction = self.predictor.predict(inputs=inputs)
        result = {c: p for c, p in zip(self.classes, prediction)}
        return result

    def run(self):
        if self.web_interface:
            gr.Interface(fn=self.inference,
                         inputs=gr.inputs.Audio(source='microphone',
                                                type='filepath'),
                         outputs='label',
                         examples=self.examples,
                         interpretation="default").launch(share=True,
                                                          inbrowser=True)
        else:
            # NW
            self.open_file_button.pack(anchor=tk.NW)
            self.recognize_button.pack(anchor=tk.NW)
            self.play_button.pack(anchor=tk.NW)

            # N
            self.filepath_label.pack(anchor=tk.N)
            self.image_canvas.get_tk_widget().pack(anchor=tk.N)
            self.predicted_label.pack(anchor=tk.N)
            self.result_label.pack(anchor=tk.N)

            # run
            super().run()


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # launch prediction gui
    PredictGUI(project_parameters=project_parameters).run()
