from PIL import Image

# 设置每个图像的大小（假设宽度和高度相同）
image_width = 480
image_height = 360

# 设置网格的大小
grid_width = 5 * image_width
grid_height = 6 * image_height

# 创建一个新的空白图像，作为最终的拼接结果
grid_image = Image.new('RGB', (grid_width, grid_height))

# 循环遍历每个图像并将其粘贴到网格中
for i in range(30):
    # 构建图像文件名
    image_filename = f"CEC2005 fun1_{i + 1}.jpg"

    try:
        # 打开图像
        image = Image.open(image_filename)

        # 调整图像大小以适应网格单元格的大小
        image = image.resize((image_width, image_height))

        # 计算当前图像的行和列索引
        row = i // 5
        col = i % 5

        # 计算当前图像在网格中的位置
        x = col * image_width
        y = row * image_height

        # 将图像粘贴到网格中的指定位置
        grid_image.paste(image, (x, y))

    except FileNotFoundError:
        print(f"无法找到图像文件：{image_filename}")

# 保存拼接后的图像
grid_image.save("grid_image.jpg")