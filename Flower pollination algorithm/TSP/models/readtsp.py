def read(filename):
    data_pos = []
    with open(filename, 'r') as file:
        # 跳过注释行和元数据
        for line in file:
            if line.startswith("NODE_COORD_SECTION"):
                break

        # 读取坐标
        for line in file:
            values = line.strip().split()
            if len(values) == 3:
                x = float(values[1])
                y = float(values[2])
                data_pos.append((x, y))
    return data_pos