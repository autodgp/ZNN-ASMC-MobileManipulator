from PIL import Image
save_path = 'planning_mat2py_floating/floating_trajectory'
img_path = save_path + '.png'
img = Image.open(img_path)

# 手动裁掉上面 80 像素（根据实际调整）
left, top, right, bottom = 0, 150, img.width, img.height
cropped = img.crop((left, top, right, bottom))
cropped.save(save_path + '_cropped.png')
