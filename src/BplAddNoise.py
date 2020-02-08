import BplGlobal as g
import BplFileAccess
import random

# Create a dummy tk environment in order to use BplFileAccess.load_data_file
import tkinter as tk
gui = tk.Tk()
g.numRows = tk.StringVar()
g.numCols = tk.StringVar()
g.outputLayerSize = tk.StringVar()


inFile = "../Alphabet/alfabet_gray.txt"
outFile = "../Alphabet/alfabet_gray_gray_noise_50.txt"

BplFileAccess.load_data_file(inFile, g.allTrainingPattern)

generated_file = open(outFile, 'w')

header = "application=BackPropLab type=data_gray version=0 input_rows=" \
         + str(g.numRows.get()) + " input_columns=" + str(g.numCols.get()) + "\n"
generated_file.write(header)

second = "patterns: "
for value in g.outputLayerLabels:
    second += value + " "
generated_file.write(second + "\n")

rows = int(g.numRows.get())
columns = int(g.numCols.get())

for p in g.allTrainingPattern:
    index, data = p
    data = data.reshape(rows, columns)
    generated_file.write(g.outputLayerLabels[index] + "\n")
    for row in range(rows):
        for col in range(columns):
            value = data[row][col]
            noise = random.random() * 0.5   # NOISE LEVEL HERE
            value += noise
            value = min(1.0, value)
            value = max(0.0, value)
            value = int(value * 255.0)
            generated_file.write(str(value) + " ")
        generated_file.write("\n")
