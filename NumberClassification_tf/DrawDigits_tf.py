import numpy as np
from NeuralNet_tf import *
from tkinter import *
import tensorflow as tf


class CreateGUI(object):
    """docstring for CreateGUI."""

    def __init__(self, master, sq_length, squares, UI_width, Model):
        self.master = master
        self.sq_length = sq_length
        self.squares = squares
        self.UI_width = UI_width
        self.Model = Model
        self.drawn_digit = np.zeros(784)
        self.master.minsize(width = self.sq_length * self.squares + self.UI_width, height = self.sq_length*self.squares)

        # Create teh drawing and response frames
        self.draw_frame = Frame(master)
        self.response_frame = Frame(master)
        self.draw_frame.pack(expand=True,fill=BOTH,side=LEFT)
        self.response_frame.pack(expand=True,fill=BOTH,side=LEFT)

        # Create the frames used to make the drawing grid
        self.pixels = [] # Create a
        for i in range(self.squares**2):
            # Make the rows and columns change size based on the window size
            self.draw_frame.rowconfigure(int(i/self.squares),weight=1)
            self.draw_frame.columnconfigure(int(i/self.squares),weight=1)
            # Create the frames for each pixel and pack them into a grid
            frame = Frame(self.draw_frame, width=self.sq_length, height=self.sq_length, bg='black')
            frame.grid(row=int(i/self.squares),column=int(i%self.squares), sticky='nsew')

            frame.bind('<Motion>',lambda event, widget=frame: self.draw(event,widget)) # Bind the frames to change color and value when the mouse moves over them
            self.pixels.append(frame) # Add the frames to a list to reference later

        # Create the buttons on the right frame
        self.clear_btn = Button(self.response_frame, text='Clear', command=lambda : self.clear())
        self.guess_label = Label(self.response_frame, text='Hi')
        self.guess_btn = Button(self.response_frame, text='Guess', command=lambda : self.guess())

        self.clear_btn.pack(side=TOP)
        self.guess_btn.pack(side=TOP)
        self.guess_label.pack(side=TOP)

    # Draw on the grid on the left side of the screen
    def draw(self, event, frame):
        if frame['bg'] != 'white':
            frame.config(bg='white')
            point = self.pixels.index(frame)
            self.drawn_digit[point] = 1

    # Reset the drawing board
    def clear(self):
        for i in range(len(self.pixels)):
            self.pixels[i].config(bg='black')
            self.drawn_digit[i] = 0

    # Guess the digit from the drawing on the screen
    def guess(self):
        result = np.argmax(self.Model.predict(np.array([self.drawn_digit,])))
        self.guess_label.config(text=result)


def main():
    # Create input parameters for the window
    sq_length = 15
    squares = 28
    UI_width = 150

    try:
        # Load in the model if it available
        Model = tf.keras.models.load_model('num_classification.model')
    except:
        # If the model is not available, train a new model and load it
        train()
        Model = tf.keras.models.load_model('num_classification.model')

    # create the tkinter window
    root = Tk()
    CreateGUI(root, sq_length, squares, UI_width, Model)
    root.mainloop()

if __name__ == "__main__":
    main()
