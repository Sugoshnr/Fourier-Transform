import numpy as np
import math
from cv2 import *
import cmath
from matplotlib import pyplot as plt

path="Lenna.png"


def mse(img1,img2):
    rows,cols=img1.shape
    mse_val=0;
    for i in range(rows):
        for j in range(cols):
            mse_val+=(img1[i][j]-img2[i][j])**2
    return mse_val



def dft_calc(img):
    rows,cols = img.shape
    out_img=[[0.0 for i in range(rows)] for j in range(cols)]

    for k in range(rows):
        for l in range(cols):
            sum_pixel = 0.0
            for m in range(rows):
                for n in range(cols):
                    value = img[n][m];
                    complex_part=cmath.exp(-1j*float(cmath.pi*2*float(float(k* m)/rows + float(l*n)/cols)))
                    sum_pixel+=value*complex_part;
            out_img[l][k] = sum_pixel/(rows*cols)

    return (out_img)



def inverse_dft(img):
    
    image = img.copy()
    rows,cols=img.shape
    for m in range(rows):
        for n in range(cols):
            sum_pixel=0.0
            for k in range(rows):
                for l in range(cols):
                    complex_part =cmath.exp(1j* float(cmath.pi*2* float(float(k*m)/rows + float(l*n)/cols)))
                    #print complex_part
                    sum_pixel += img[l][k] * complex_part
            pixel= cmath.polar(sum_pixel)[0]
            image[n][m] = pixel
    return image


img=imread(path,0);
## Using resize only to reduce computational time. Works equaly the same for any size but computationally expensive
img=resize(img,(50,50))
n=len(img)
#imshow("1",img)
out_img_all=dft_calc(img)
out_img=np.asarray((out_img_all))
out_img=np.abs(out_img)
#img=copyMakeBorder(img,1,1,1,1,BORDER_CONSTANT,value=0)
#out_img=np.array(out_img,dtype='uint8')
#imshow("out_img",out_img)



out1=out_img[0:n/2, 0:n/2]#2nd Quad
out2=out_img[0:n/2, n/2:n]#1st Quad
out3=out_img[n/2:n, 0:n/2]#3rd Quad
out4=out_img[n/2:n, n/2:n]#4th Quad



out=out_img.copy()
for i in range(n/2):
    for j in range(n/2):
        out[i][j]=(out4[i][j]);#4->2
for i in range(n/2,n):
    for j in range(n/2,n):
        out[i][j]=(out1[i-n/2][j-n/2]);#2->4
for i in range(n/2):
    for j in range(n/2,n):
        out[i][j]=(out3[i][j-n/2]);#3->1
for i in range(n/2,n):
    for j in range(n/2):
        out[i][j]=(out2[i-n/2][j]);#1->3

#img1=imread("Lenna.png",0)
#img1=resize(img1,(50,50))
#fft_val = np.fft.fft2(img1)
#ffshift = np.fft.fftshift(f)
#val1 = 20*np.log(np.abs(ffshift))
log_val = 20*np.log(np.abs(out))
plt.subplot(131),plt.imshow(img, cmap = 'gray')
plt.title("Original Image");
plt.subplot(132),plt.imshow(log_val, cmap = 'gray')
plt.title("Fourier Transform");
x=np.asarray(out_img_all)
inverse_img=inverse_dft(x)
inverse_img1=np.asarray(np.abs(inverse_img))
#inverse_img1=np.array(np.abs(inverse_img),dtype='uint8')
#imshow("Inverse",inverse_img1)
plt.subplot(133),plt.imshow(inverse_img1, cmap = 'gray')
plt.title("Reconstructed Image");

mse_val=mse(img,inverse_img1);
print "Mean Square Error= %d"%int(mse_val)


plt.show()



waitKey(0)
destroyAllWindows() 
