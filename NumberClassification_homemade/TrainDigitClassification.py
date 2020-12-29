import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from NeuralNetwork import *
import cv2
from tkinter import *
from PIL import Image, ImageTk

class ImgGUI():
    def __init__(self, root, Trial, Test):
        self.Test = Test
        self.Trial = Trial
        self.width, self.height = 400, 200
        self.result = np.random.randint(len(self.Test))
        root.title('Handwriting')
        # root.minsize(width,height)

        self.left_frame = Frame(master=root,bg='red',width=self.width/2,height=self.height)
        self.right_frame = Frame(master=root,bg='blue',width=self.width/2,height=self.height)

        self.left_frame.grid(row=0,column=0,sticky="nsew")
        self.right_frame.grid(row=0,column=1)

        self.pic_canvas = Canvas(master=self.left_frame,width=28,height=28, bg='white')
        self.pic_canvas.place(relx=.5,rely=.5,anchor=CENTER)

        self.pic = np.zeros((28,28))
        for i in range(784):
            self.pic[int(i/28)][i%28] = self.Test[self.result][i] * 255

        self.image = Image.fromarray(self.pic)
        self.image = ImageTk.PhotoImage(self.image)

        self.imageFrame = Label(master=self.pic_canvas, image=self.image)
        self.imageFrame.image = self.image
        self.imageFrame.pack()

        self.text = Label(master=self.right_frame, text="Take a Guess")
        self.guess = Label(master=self.right_frame, text="Guess will appear here")
        self.btn = Button(master=self.right_frame,text="Next Image",command= lambda: self.update())

        for r in range(3):
            self.right_frame.rowconfigure(r, minsize=self.width/6)
            self.right_frame.columnconfigure(r, minsize=self.height/3)

        self.text.grid(row=0,column=0,columnspan=3,sticky="NSEW")
        self.guess.grid(row=1,column=0,columnspan=3,sticky="NSEW")
        self.btn.grid(row=2,column=0,columnspan=3,sticky="NSEW")

        root.mainloop()

    def update(self):
        for i in range(784):
            self.pic[int(i/28)][i%28] = self.Test[self.result][i] * 255

        self.image = Image.fromarray(self.pic)
        self.image = ImageTk.PhotoImage(self.image)

        self.imageFrame.configure(image=self.image)
        self.imageFrame.image = self.image
        guess = self.Trial.run(self.Test[self.result])
        self.guess['text'] = "Guess: " + str(np.argmax(guess)) + ", Result No.: " + str(self.result)
        self.result += 1
        if self.result > len(self.Test):
            self.result = 0
        print(guess)


def get_CSV(url):
    return pd.read_csv(url, header=0)

def train_network(Trial, X_train, Y_train, epochs):
    # epochs = 1000
    for i in range(epochs):
        for j in range(X_train.shape[0]-1):
            MSE = 0.0
            output = np.zeros(Trial.layers[-1])
            output[Y_train[j]] = 1
            MSE += Trial.backpropagation(X_train[j], output)
        print("Epoch: ", i)
        # Trial.printWeights()
    print("MSE: ", MSE)

def test_network(Trial, X_test, Y_test):
    count = 0
    score = 0
    for i in range(X_test.shape[0]-1):
        print(Trial.run(X_test[i]),Y_test[i])
        if np.argmax(Trial.run(X_test[i])) == Y_test[i]:
            score += 1
        count += 1
    print(score,'/',count)
    print('Results: ', score/count)

def getColorData(img, pix):
    digit = cv2.imread(img)
    gray_digit = cv2.cvtColor(digit, cv2.COLOR_BGR2GRAY)
    color_data = np.zeros((pix, pix))
    for i in range(pix):
        for j in range(pix):
            color_data[i][j] = gray_digit[i*pix+pix/2][j*pix+pix/2]
    print(color_data)

if __name__ == "__main__":
    Trial = Multilayer_Perceptron(layers = [784, 10, 10])

    data = get_CSV('train.csv')
    epochs = 3
    test_data = data.values
    _X_train = np.array(test_data[:2,1:])
    Y_train = np.array(test_data[:2,0:1])
    X_train = _X_train / 255

    data = get_CSV('test.csv')
    test_data = data.values
    _Test = np.array(test_data[:,:])
    Test = _Test / 255

    try:
        saved_weights = Trial.readWeights('weights.txt')
        Trial.setWeights(saved_weights)
    except:
        print("There were no saved weights.")

    train_network(Trial, X_train, Y_train, epochs)

    print(Trial.network[1][0].weights[0])

    Trial.saveWeights('weights.txt')

    root = Tk()
    ImgGUI(root, Trial, Test)


    # The output is not the same as what is running maybe
