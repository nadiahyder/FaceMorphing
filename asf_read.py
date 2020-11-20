def read_asf(file_name):
    asf = open(file_name, "r")
    lines = asf.readlines()
    num_info = int(lines[9])
    output_lines = []
    for i in range(16, num_info + 16):
        output_lines.append(lines[i])
    return output_lines