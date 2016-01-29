# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from Tkinter import*
from tkFileDialog import askopenfilename

import cv2
import matplotlib.pyplot as plt
import ttk
import PIL
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import collections

from sklearn.cluster import KMeans
import scipy as sc
from scipy.spatial import distance

# keras
from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.optimizers import SGD

import matplotlib.pylab as pylab


#--------------------------------------------------------
def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
def image_bin(image_gs):
    ret,image_bin = cv2.threshold(image_gs, 108, 255, cv2.THRESH_BINARY)
    return image_bin

def image_bin_adaptive(image_gs):
    image_bin = cv2.adaptiveThreshold(image_gs, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, 10)
    return image_bin
    
def invert(image):
    return 255-image

def dilate(image):
    kernel = np.ones((2,2)) # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=1)
def erode(image):
    kernel = np.ones((3,3)) # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=1)

def erode2(image):
    kernel = np.ones((2,2)) # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=1)

def display_image(image, color= False):
    if color:
        plt.imshow(image)
        plt.figure()
    else:
        plt.imshow(image, 'gray')
        plt.figure()
        

def resize_region(region):
    resized = cv2.resize(region,(28,28), interpolation = cv2.INTER_NEAREST)
    return resized


def scale_to_range(image):
    return image / 255
def matrix_to_vector(image):
    return image.flatten()
def prepare_for_ann(regions):
    ready_for_ann = []
    for region in regions:
        ready_for_ann.append(matrix_to_vector(scale_to_range(region)))
    return ready_for_ann
def convert_output(outputs):
    return np.eye(len(outputs))
def winner(output):
    return max(enumerate(output), key=lambda x: x[1])[0]


def rotate_regions(contours,angles,centers,sizes):
    '''Funkcija koja vrši rotiranje regiona oko njihovih centralnih tačaka
    Args:
        contours: skup svih kontura [kontura1, kontura2, ..., konturaN]
        angles:   skup svih uglova nagiba kontura [nagib1, nagib2, ..., nagibN]
        centers:  skup svih centara minimalnih pravougaonika koji su opisani 
                  oko kontura [centar1, centar2, ..., centarN]
        sizes:    skup parova (height,width) koji predstavljaju duzine stranica minimalnog
                  pravougaonika koji je opisan oko konture [(h1,w1), (h2,w2), ...,(hN,wN)]
    Return:
        ret_val: rotirane konture'''
    ret_val = []
    for idx, contour in enumerate(contours):
                
        angle = angles[idx]
        cx,cy = centers[idx]
        height, width = sizes[idx]
        if width<height:
            angle+=90
            
        # Rotiranje svake tačke regiona oko centra rotacije
        alpha = np.pi/2 - abs(np.radians(angle))
        region_points_rotated = np.ndarray((len(contour), 2), dtype=np.int16)
        for i, point in enumerate(contour):
            x = point[0]
            y = point[1]
            
            #TODO 1 - izračunati koordinate tačke nakon rotacije
            rx = np.sin(alpha)*(x-cx) - np.cos(alpha)*(y-cy) + cx
            ry = np.cos(alpha)*(x-cx) + np.sin(alpha)*(y-cy) + cy
            
            
            region_points_rotated[i] = [rx,ry]
        ret_val.append(region_points_rotated)
        

    return ret_val

# TODO 2
def merge_regions(contours):
    '''Funkcija koja vrši spajanje kukica i kvačica sa osnovnim karakterima
    Args:
        contours: skup svih kontura (kontura - niz tacaka bele boje)
    Return:
        ret_val: skup kontura sa spojenim kukicama i kvacicama'''
    ret_val = []
    merged_index = [] #lista indeksa kontura koje su već spojene sa nekim

    for i,contour1 in enumerate(contours): #slova
        if i in merged_index:
            continue
        min_x1 = min(contour1[:,0])
        max_x1 = max(contour1[:,0])
        min_y1 = min(contour1[:,1])
        max_y1 = max(contour1[:,1])
        for j,contour2 in enumerate(contours): #kukice
            if j in merged_index or i == j:
                continue
            min_x2 = min(contour2[:,0])
            max_x2 = max(contour2[:,0])
            min_y2 = min(contour2[:,1])
            max_y2 = max(contour2[:,1])
            
            #TODO 2 - izvršiti spajanje kukica iznad slova
            #spajanje dva niza je moguće obaviti funkcijom np.concatenate((contour1,contour2))
            
            if len(contour1)/2>len(contour2): #provera pretpostavke da je contour1 slovo
                
                if (min_y1-max_y2)<max(max_y1-min_y1,max_y2-min_y2)/2 \
                and (min_x2>min_x1-5 and max_x2<max_x1+5):
                    #spajanje kontura
                    ret_val.append(np.concatenate((contour1,contour2)))
                    merged_index.append(i)
                    merged_index.append(j)
                '''
                elif (max(max_y2, max_y1) > min(min_y1, min_y2)):
                   ret_val.append(np.concatenate((contour1,contour2)))
                   merged_index.append(i)
                   merged_index.append(j)
              '''  
                    
            
                   
    #svi regioni koji se nisu ni sa kim spojili idu u listu kontura, bez spajanja
    for idx,contour in enumerate(contours):
        if idx not in merged_index:
            ret_val.append(contour)
        
    '''
    for aa in range(0, len(ret_val)):
        if aa == (len(ret_val) - 2):
            contour4 = ret_val[aa]
            contour5 = ret_val[aa+1]
            ret_val.append(np.concatenate((contour4, contour5)))
    '''
   
    return ret_val


def select_roi_mreza(image_orig, image_bin):
    
    img, contours_borders, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    contours = []
    contour_angles = []
    contour_centers = []
    contour_sizes = []
    for contour in contours_borders:
        center, size, angle = cv2.minAreaRect(contour)
        xt,yt,h,w = cv2.boundingRect(contour)
       
        region_points = []
        for i in range (xt,xt+h):
            for j in range(yt,yt+w):
                dist = cv2.pointPolygonTest(contour,(i,j),False)
                if dist>=0 and image_bin[j,i]==255: # da li se tacka nalazi unutar konture?
                    region_points.append([i,j])
        contour_centers.append(center)
        contour_angles.append(angle)
        contour_sizes.append(size)
        contours.append(region_points)
    
    #Postavljanje kontura u vertikalan polozaj
    contours = rotate_regions(contours, contour_angles, contour_centers, contour_sizes)
    
    #spajanje kukica i kvacica
    contours = merge_regions(contours)
    
    regions_dict = {}
    for contour in contours:
    
        min_x = min(contour[:,0])
        max_x = max(contour[:,0])
        min_y = min(contour[:,1])
        max_y = max(contour[:,1])

        region = np.zeros((max_y-min_y+1,max_x-min_x+1), dtype=np.int16)
        for point in contour:
            x = point[0]
            y = point[1]
            
             # TODO 3 - koordinate tacaka regiona prebaciti u relativne koordinate
            '''Pretpostavimo da gornja leva tačka regiona ima apsolutne koordinate (100,100).
            Ako uzmemo tačku sa koordinatama unutar regiona, recimo (105,105), nakon
            prebacivanja u relativne koordinate tačka bi trebala imati koorinate (5,5) unutar
            samog regiona.
            '''
            region[y-min_y,x-min_x] = 255

        
        regions_dict[min_x] = [resize_region(region), (min_x,min_y,max_x-min_x,max_y-min_y)]
        
    sorted_regions_dict = collections.OrderedDict(sorted(regions_dict.items()))
    sorted_regions = np.array(sorted_regions_dict.values())
    
    sorted_rectangles = sorted_regions[:,1]
    region_distances = [-sorted_rectangles[0][0]-sorted_rectangles[0][2]]
    for x,y,w,h in sorted_regions[1:-1, 1]:
        region_distances[-1] += x
        region_distances.append(-x-w)
    region_distances[-1] += sorted_rectangles[-1][0]
    
    return image_orig, sorted_regions[:, 0], region_distances


def select_roi2(image_orig, image_bin):
    
    img, contours_borders, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    contours = []
    contour_angles = []
    contour_centers = []
    contour_sizes = []
    for contour in contours_borders:
        center, size, angle = cv2.minAreaRect(contour)
        xt,yt,h,w = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)      
        if area > 100 and area < 1500 and w < 90:
       # if h < 70 and h > 15 and w > 8 and w < 40:
       # if h < 70 and h > 41 and w > 12 and w < 40:
        #if area > 220 and h > 20 and w > 11:
            region_points = []
            for i in range (xt,xt+h):
                for j in range(yt,yt+w):
                    dist = cv2.pointPolygonTest(contour,(i,j),False)
                    if dist>=0 and image_bin[j,i]==255: # da li se tacka nalazi unutar konture?
                        region_points.append([i,j])
            contour_centers.append(center)
            contour_angles.append(angle)
            contour_sizes.append(size)
            contours.append(region_points)
    
    #Postavljanje kontura u vertikalan polozaj
    contours = rotate_regions(contours, contour_angles, contour_centers, contour_sizes)
    
    #spajanje kukica i kvacica
    contours = merge_regions(contours)
    
    regions_dict = {}
    for contour in contours:
    
        min_x = min(contour[:,0])
        max_x = max(contour[:,0])
        min_y = min(contour[:,1])
        max_y = max(contour[:,1])

        region = np.zeros((max_y-min_y+1,max_x-min_x+1), dtype=np.int16)
        for point in contour:
            x = point[0]
            y = point[1]
            
             # TODO 3 - koordinate tacaka regiona prebaciti u relativne koordinate
            '''Pretpostavimo da gornja leva tačka regiona ima apsolutne koordinate (100,100).
            Ako uzmemo tačku sa koordinatama unutar regiona, recimo (105,105), nakon
            prebacivanja u relativne koordinate tačka bi trebala imati koorinate (5,5) unutar
            samog regiona.
            '''
            region[y-min_y,x-min_x] = 255

        
        regions_dict[min_x] = [resize_region(region), (min_x,min_y,max_x-min_x,max_y-min_y)]
        
    sorted_regions_dict = collections.OrderedDict(sorted(regions_dict.items()))
    sorted_regions = np.array(sorted_regions_dict.values())
    
    sorted_rectangles = sorted_regions[:,1]
    region_distances = [-sorted_rectangles[0][0]-sorted_rectangles[0][2]]
    for x,y,w,h in sorted_regions[1:-1, 1]:
        region_distances[-1] += x
        region_distances.append(-x-w)
    region_distances[-1] += sorted_rectangles[-1][0]
    
    return image_orig, sorted_regions[:, 0], region_distances


def create_ann():
    
    ann = Sequential()
    # Postavljanje slojeva neurona mreže 'ann'
    ann.add(Dense(input_dim=784, output_dim=128,init="glorot_uniform"))
    ann.add(Activation("sigmoid"))
    ann.add(Dense(input_dim=128, output_dim=40,init="glorot_uniform"))
    ann.add(Activation("sigmoid"))
    return ann
    
def train_ann(ann, X_train, y_train):
    X_train = np.array(X_train, np.float32)
    y_train = np.array(y_train, np.float32)
   
    # definisanje parametra algoritma za obucavanje
    sgd = SGD(lr=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)

    # obucavanje neuronske mreze
    ann.fit(X_train, y_train, nb_epoch=3000, batch_size=1, verbose = 0, shuffle=False, show_accuracy = False) 
      
    return ann



def display_result(outputs, alphabet):
    '''
    Funkcija određuje koja od grupa predstavlja razmak između reči, a koja između slova, i na osnovu
    toga formira string od elemenata pronađenih sa slike.
    Args:
        outputs: niz izlaza iz neuronske mreže.
        alphabet: niz karaktera koje je potrebno prepoznati
        kmeans: obučen kmeans objekat
    Return:
        Vraća formatiran string
    '''
    # Odrediti indeks grupe koja odgovara rastojanju između reči, pomoću vrednosti iz k_means.cluster_centers_
    #w_space_group = max(enumerate(k_means.cluster_centers_), key = lambda x: x[1])[0]
    result = alphabet[winner(outputs[0])]
    for idx, output in enumerate(outputs[1:,:]):
        # Iterativno dodavati prepoznate elemente kao u vežbi 2, alphabet[winner(output)]
        # Dodati space karakter u slučaju da odgovarajuće rastojanje između dva slova odgovara razmaku između reči.
        # U ovu svrhu, koristiti atribut niz k_means.labels_ koji sadrži sortirana rastojanja između susednih slova.
        #if (k_means.labels_[idx] == w_space_group):
            #result += ' '
        result += alphabet[winner(output)] 
    return result



    



#--------------------------- dole je za tablice, gore je za mrezu


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
        #if area > 10000 and area < 15000 and h < 100 and h > 12 and w > 50:
        #if area > 8000 and h > 50 and h < 70 and w > 250 and w < 290:
        if area > 8000 and h > 40 and h < 105 and w > 180 and w < 350:
            print area
            
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
    img2_erode = erode((img2))
    selected_regions2, letters2, distances2, cropp = select_roi(image_color.copy(), img2_erode)
    #display_image(selected_regions2)
    
    return cropp
    
def openFile():
    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    location = askopenfilename() # show an "Open" dialog box and return the path to the selected file
    return location
    
    
def set_Pic_path(text):
    img = Image.open(text)
    img = img.resize((900, 600), PIL.Image.ANTIALIAS)
    dr = ImageDraw.Draw(img)

    
    img.save('resized.jpg')
    dr.rectangle(tablica(), outline = "red")
    img.save('resized.jpg')
    
    
    img_resize_crop = Image.open('resized.jpg')
    img_resize_crop.crop(tablica()).save('tablica_crop.jpg')

    
    
    tablica_1 = Image.open('tablica_crop.jpg')
    tablica_2 = Image.open('tablica_crop.jpg')
    tablica_3 = Image.open('tablica_crop.jpg')
    
    wid = tablica_1.size[0]
    hei = tablica_1.size[1]
   
    krop1 = (0, 0, wid-3*wid/4+15, hei)
    print krop1
    tablica_1.crop(krop1).save('tablica_1.jpg')
    
    krop2 = (wid-wid/4, 0, wid, hei)
    tablica_2.crop(krop2).save('tablica_2.jpg')
    
    krop3 = (wid-3*wid/4+28, 0, 3*wid/4, hei)
    tablica_3.crop(krop3).save('tablica_3.jpg')
    
    
    photo = ImageTk.PhotoImage(Image.open('resized.jpg')) 
    labelSlika = Label(root, image=photo)
    labelSlika.image = photo
    labelSlika.place(x=66, y=20)
        
   
    return
    
    
def test_tablica():
    
    image_test_original = load_image('tablica_crop.jpg')
    image_test = erode2(invert(image_bin(image_gray(image_test_original))))
    display_image(image_test)
    selected_regions_test, letters_test, region_distances_test = select_roi2(image_test_original.copy(), image_test)
    print 'Broj prepoznatih regiona:', len(letters_test)
    '''
    for aa in letters_test:
        plt.imshow(aa)
        plt.figure()
    '''
    region_distances_test = np.array(region_distances_test).reshape(len(region_distances_test), 1)
    #k_means_test = KMeans(n_clusters=2, max_iter=2000, tol=0.00001, n_init=10)
    #k_means_test.fit(region_distances_test)
    inputs_test = prepare_for_ann(letters_test)
    results_test = ann.predict(np.array(inputs_test, np.float32))
    
    ispis = display_result(results_test, alphabet)
    
    tablicaText.delete(0,END)
    tablicaText.insert(0,ispis)
    
    

    ispis = ispis[0:2]
    if(ispis == 'NS'):
        regionText.delete(0,END)
        regionText.insert(0, 'Novi Sad')
    elif(ispis == 'BG'):
        regionText.delete(0,END)
        regionText.insert(0, 'Beograd')
    elif(ispis == 'ZR'):
        regionText.delete(0,END)
        regionText.insert(0, 'Zrenjanin')
    elif(ispis == 'RU'):
        regionText.delete(0,END)
        regionText.insert(0, 'Ruma')
    else:
        regionText.delete(0,END)
        regionText.insert(0, 'Nepoznato')
    
    return
    
    
    

#-------------------------------------------------------






location = ''

    
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
#tablicaText.config(state='readonly')


ispisi1 = Button(root, text="Ispisi", width=20, height=2, command=lambda:test_tablica())
ispisi1.config(font = fontSerial)
ispisi1.place(x=850, y=682)



#donji deo, drugi
labelRegion = Label(root, text="Region registracije: ", height=5)
labelRegion.config(font = fontBold)
labelRegion.place(x=390, y=700)

unosRegionText = StringVar()
regionText = Entry(root, textvariable=unosRegionText, width=21)
regionText.config(font = fontTextField)
regionText.place(x=620, y=730)


'''
ispisi2 = Button(root, text="Ispisi region", width=20, height=2)
ispisi2.config(font = fontSerial)
ispisi2.place(x=850, y=720)
'''
#Ucitaj dugme levo
dugmeUcitaj = Button(root, text="Ucitaj sliku", width=25, height=5, command=lambda:set_Pic_path(openFile()))

dugmeUcitaj.place(x=30,y=650)





#gornji deo
'''
baseheight = 900
if location != '':
    img = Image.open(location)

    hpercent = (baseheight / float(img.size[1]))
    wsize = int((float(img.size[0]) * float(hpercent)))
    img = img.resize((900, 600), PIL.Image.ANTIALIAS)
    img.save('resized.jpg')
 
    w, h = img.size
    print w
    print h
'''
#img.crop((359, 501, w-314, h-47)).save('resized_crop.jpg')
#img.crop((h/4, w/3-30, w-h/3, h-h/10)).save('resized_crop.jpg')

#img.crop((w/4, h/4, 3*w/4, 3*h/4)).save('resized_crop.jpg')


image_test_original_obucavanje = load_image('images/noviSkup.png')
image_test_obucavanje = erode2(invert(image_bin(image_gray(image_test_original_obucavanje))))
display_image(image_test_obucavanje)
               
        
selected_test_obucavanje, letters_obucavanje, region_distances_obucavanje = select_roi_mreza(image_test_original_obucavanje.copy(), image_test_obucavanje)

'''
#--------------------- konkatenacija nule
contour4 = letters_obucavanje[40]
contour5 = letters_obucavanje[39]
contour6 = np.concatenate((contour5, contour4))

contour6 = resize_region(contour6)

letters_obucavanje = letters_obucavanje[0:40]
letters_obucavanje[39] = contour6
#-------------------------
    '''
letters_obucavanje = letters_obucavanje[0:40]


inputs_obucavanje = prepare_for_ann(letters_obucavanje)
alphabet = ['A', 'B', 'C', 'Ć', 'Č', 'D', 'Đ', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'Š', 'T', 'U', 'V', 'Z', 'Ž', 'W', 'X', 'Y', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']


outputs_obucavanje = convert_output(alphabet)
ann = create_ann()
#ann = train_ann(ann, inputs_obucavanje, outputs_obucavanje)






root.mainloop()

