input_file = "mnist_train.csv"
output_file = "mnist_train.txt"

file_in = open(input_file)
file_out = open(output_file, 'w')

header = "application=BackPropLab version=0 type=data_gray input_rows=28" \
          " input_columns=28"
file_out.write(header + '\n')

second = "patterns: 0 1 2 3 4 5 6 7 8 9"
file_out.write(second + '\n')

line_count = 0
for line in file_in:
    line_count += 1
    print(line_count)
    line = line[:-1] # Remove \n
    pixel = line.split(',')
    file_out.write(pixel[0] + '\n')
    pixel = pixel[1:]
    for i in range(28):
        out_string = ""
        for j in range(28):
            out_string += pixel[28 * i + j] + ' '
        file_out.write(out_string + '\n')
