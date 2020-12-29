import numpy as np
from tkinter import *
from PIL import Image, ImageTk
from NeuralNet_tf import *

class ImgGUI():
    def __init__(self, root, x_test, y_test, guesses, Model):
        self.Test = [x_test, y_test, guesses]
        self.Model = Model
        self.width, self.height = 400, 200
        self.result = np.random.randint(len(self.Test[0]))
        root.title('Handwriting')
        # root.minsize(width,height)

        # Create the left and right frames foror the images and buttons
        self.left_frame = Frame(master=root,bg='red',width=self.width/2,height=self.height)
        self.right_frame = Frame(master=root,bg='blue',width=self.width/2,height=self.height)

        self.left_frame.grid(row=0,column=0,sticky="nsew")
        self.right_frame.grid(row=0,column=1)

        # Canvas used to hold image
        self.pic_canvas = Canvas(master=self.left_frame,width=28,height=28, bg='white')
        self.pic_canvas.place(relx=.5,rely=.5,anchor=CENTER)

        # Create the first picture
        self.pic = np.zeros((28,28))
        for i in range(28):
            for j in range(28):
                self.pic[i][j] = self.Test[0][self.result][i][j]

        self.image = Image.fromarray(self.pic)
        self.image = ImageTk.PhotoImage(self.image)

        self.imageFrame = Label(master=self.pic_canvas, image=self.image)
        self.imageFrame.image = self.image
        self.imageFrame.pack()

        # Create the labels and buttons in the right frame.
        self.text = Label(master=self.right_frame, text="Take a Guess")
        self.guess = Label(master=self.right_frame, text="Guess will appear here")
        self.btn = Button(master=self.right_frame,text="Next Image",command= lambda: self.update())

        # Allow the rows/columns to fill the entire right frame.
        for r in range(3):
            self.right_frame.rowconfigure(r, minsize=self.width/6)
            self.right_frame.columnconfigure(r, minsize=self.height/3)

        self.text.grid(row=0,column=0,columnspan=3,sticky="NSEW")
        self.guess.grid(row=1,column=0,columnspan=3,sticky="NSEW")
        self.btn.grid(row=2,column=0,columnspan=3,sticky="NSEW")

        root.mainloop()

    # This function is used to show the next image and guess.  Called on every button click
    def update(self):
        # Arrange the values to show the image
        for i in range(28):
            for j in range(28):
                self.pic[i][j] = self.Test[0][self.result][i][j]

        # Create the image.  Prepare it to fit into the tkinter window
        self.image = Image.fromarray(self.pic)
        self.image = ImageTk.PhotoImage(self.image)

        self.imageFrame.configure(image=self.image)
        self.imageFrame.image = self.image

        # Set the label text.  Show the guess, the actual answer, and the image number.
        self.guess['text'] = "Guess: " + str(np.argmax(self.Test[2][self.result])) + ", Actual: " + str(self.Test[1][self.result]) + ", Result No.: " + str(self.result)

        # Set the result to change for the next image
        self.result += 1
        if self.result > len(self.Test[0]):
            self.result = 0

if __name__ == "__main__":
    # Grab the mnist data for testing
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    try:
        # Load in the model if it available
        new_model = tf.keras.models.load_model('num_classification.model')
    except:
        # If the model is not available, train a new model and load it
        x_test, y_test = train()
        new_model = tf.keras.models.load_model('num_classification.model')

    # Use the model to guess the digit from the image data
    guesses = new_model.predict(x_test)

    # Create the tkinter window
    root = Tk()
    ImgGUI(root, x_test,  y_test, guesses, new_model)
