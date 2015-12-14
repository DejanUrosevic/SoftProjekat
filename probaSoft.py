# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from Tkinter import*
import cv2
import matplotlib.pyplot as plt
import ttk
import PIL
from PIL import Image, ImageTk
import numpy as np
import collections





#--------------------------------------------------------
def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
def image_bin(image_gs):
    ret,image_bin = cv2.threshold(image_gs, 21, 255, cv2.THRESH_BINARY)
    return image_bin

def dilate(image):
    kernel = np.ones((2,2)) # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=1)
def erode(image):
    kernel = np.ones((3,3)) # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=1)


def resize_region(region):
    resized = cv2.resize(region,(28,28), interpolation = cv2.INTER_NEAREST)
    return resized

def select_roi(image_orig, image_bin):
    '''
    Funkcija kao u vežbi 2, iscrtava pravougaonike na originalnoj slici, pronalazi sortiran niz regiona sa slike,
    i dodatno treba da sačuva rastojanja između susednih regiona.
    '''
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #Način određivanja kontura je promenjen na spoljašnje konture: cv2.RETR_EXTERNAL
    regions_dict = {}
    for contour in contours: 
        x,y,w,h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area > 100 and h < 100 and h > 9 and w > 7:
            region = image_bin[y:y+h+1,x:x+w+1];
            # Proširiti regions_dict elemente sa vrednostima boundingRect-a ili samim konturama
            regions_dict[x] = [resize_region(region), (x,y,w,h)]
            cropp = (x, y, x+w, y+h)
            cv2.rectangle(image_orig,(x,y),(x+w,y+h),(0,255,0),2)
            
            
    sorted_regions_dict = collections.OrderedDict(sorted(regions_dict.items()))
    sorted_regions = np.array(sorted_regions_dict.values())
    
    sorted_rectangles = sorted_regions[:,1]
    region_distances = [-sorted_rectangles[0][0]-sorted_rectangles[0][2]]
    # Izdvojiti sortirane parametre opisujućih pravougaonika
    # Izračunati rastojanja između svih susednih regiona po x osi i dodati ih u region_distances niz
    for x,y,w,h in sorted_regions[1:-1, 1]:
        region_distances[-1] += x
        region_distances.append(-x-w)
    region_distances[-1] += sorted_rectangles[-1][0]
    
    return image_orig, sorted_regions[:, 0], region_distances, cropp
    


def tablica():
    image_color = load_image('resized.jpg')
    img2 = image_bin(image_gray(image_color))
    img2_erode = erode(erode(erode(img2)))
    selected_regions2, letters2, distances2, cropp = select_roi(image_color.copy(), img2_erode)
    
    return cropp
#-------------------------------------------------------








    
root = Tk()
root.resizable(0,0)
root.title("My first GUI app")
root.geometry('1050x800+200+200')

fontBold = ('Calibri', 12, 'bold')
fontSerial = ('Calibri', 12)
fontTextField = ('Calibri', 16)



#PannedWindow


#Donji deo, prvi




labelIspis = Label(root, text="Ispis tablice: ", height=5)
labelIspis.config(font = fontBold)
labelIspis.place(x=450, y=635)



unosText = StringVar()
tablicaText = Entry(root, textvariable=unosText, width=21)
tablicaText.config(font = fontTextField)
tablicaText.place(x=620, y=665)
tablicaText.config(state='readonly')


ispisi1 = Button(root, text="Ispisi", width=20, height=2)
ispisi1.config(font = fontSerial)
ispisi1.place(x=850, y=650)



#donji deo, drugi
labelRegion = Label(root, text="Region registracije: ", height=5)
labelRegion.config(font = fontBold)
labelRegion.place(x=390, y=700)

unosRegionText = StringVar()
regionText = Entry(root, textvariable=unosRegionText, width=21)
regionText.config(font = fontTextField)
regionText.place(x=620, y=730)
regionText.config(state='readonly')


ispisi2 = Button(root, text="Ispisi region", width=20, height=2)
ispisi2.config(font = fontSerial)
ispisi2.place(x=850, y=720)

#Ucitaj dugme levo
dugmeUcitaj = Button(root, text="Ucitaj sliku", width=25, height=5)

dugmeUcitaj.place(x=30,y=650)


#gornji deo

baseheight = 900
img = Image.open('Slike/ford1.JPG')

hpercent = (baseheight / float(img.size[1]))
wsize = int((float(img.size[0]) * float(hpercent)))
img = img.resize((wsize, baseheight), PIL.Image.ANTIALIAS)
img.save('resized.jpg')

basewidth = 900
img = Image.open('resized.jpg')
wpercent = (basewidth / float(img.size[0]))
hsize = int((float(img.size[1]) * float(wpercent)))
img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
img.save('resized.jpg')

w, h = img.size
print w
print h 
#img.crop((359, 501, w-314, h-47)).save('resized_crop.jpg')
#img.crop((h/3, h/3-30, w-h/3, h-h/10)).save('resized_crop.jpg')
img.crop((w/4-50, h/4-210, 3*w/4-95, 3*h/4-55)).save('resized_crop.jpg')

#img.crop(tablica()).save('tablica_crop.jpg')


photo = ImageTk.PhotoImage(Image.open('resized_crop.jpg')) 
labelSlika = Label(root, image=photo)
labelSlika.pack()


root.mainloop()

