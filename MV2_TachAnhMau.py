import cv2    #sử dngj thư viện python
import numpy as np 

#đọc ảnh màu dùng thư viện open cv
img = cv2.imread('cogailenaxauxi.jpg', cv2.IMREAD_COLOR )

#lấy kích thước của ảnh
height = len(img[0])
width = len(img[1])

#khai báo 3 biến để chứa hình của 3 kênh màu R, G, B
red = np.zeros((width,height,3) ,np.uint8)  #số 3 là 3 kênh, mỗi kênh là 8bit
green = np.zeros((width,height,3) ,np.uint8)
blue = np.zeros((width,height,3) ,np.uint8)

#ban đầu set zero cho tất cả các ảnh
red[:] = [0,0,0]
green[:] = [0,0,0]
blue[:] = [0,0,0]

#mỗi hình là 1 ma trận 2 chiều nên rta dùng 2 vòng for để đọc các giá trị pixel trong hình
for x in range(width):
    for y in range(height):
        #lấy giá trị điểm ảnh tại vị trí x, y
        R = img[x,y,2]
        G = img[x,y,1]
        B = img[x,y,0]

        #thiết lập màu cho các kênh
        red[x,y,2] = R
        green[x,y,1] = G
        blue[x,y,0] = B

#hiển thị hình dùng thư viện OpenCv
cv2.imshow('Hinh Co Gai Xinh Dep', img)
cv2.imshow('kenh mau do',red)
cv2.imshow('kenh mau xanh la',green)
cv2.imshow('kenh mau xanh bien',blue)


#bấm phím bất kì để đóng cửa sổ hiển thị
cv2.waitKey(0)

#giải ohings bộ nhớ cho cửa sổ đã hiển thị
cv2.destroyAllWindows()
